import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat
import math

from causal_conv1d import causal_conv1d_fn
import selective_scan_cuda

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd()
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

class SimpleAwareStateSpaceModel(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank="auto", conv_bias=True, device=None, dtype=None,
                 polarity_write_mode="none"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.conv_bias = conv_bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.device = device
        self.dtype = dtype
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.dim = d_model
        self.d_inner = int(self.expand * self.d_model)
        self.norm = nn.LayerNorm(d_model)
        # Event-relation conditioning changes state writing, not output gating.
        modes = {"none", "identity", "geometry", "polarity", "content", "polarity_state"}
        if polarity_write_mode not in modes:
            raise ValueError(f"Unknown polarity_write_mode: {polarity_write_mode}")
        if polarity_write_mode == "polarity_state" and (d_state < 2 or d_state % 2):
            raise ValueError("polarity_state requires a positive, even d_state >= 2")
        self.polarity_write_mode = polarity_write_mode
        self.write_gate = None
        if polarity_write_mode not in {"none", "polarity_state"}:
            input_dim = d_model if polarity_write_mode == "content" else 8
            self.write_gate = nn.Sequential(
                nn.Linear(input_dim, 8), nn.SiLU(), nn.Linear(8, 1)
            )
            # g = 2 * sigmoid(0) = 1: preserve the baseline recurrence initially.
            nn.init.zeros_(self.write_gate[-1].weight)
            nn.init.zeros_(self.write_gate[-1].bias)
        self.act = nn.SiLU()
        self.multi_act = nn.Sigmoid()
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False, **self.factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv // 2,
            **self.factory_kwargs,
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **self.factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **self.factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **self.factory_kwargs)

        self.dt_init()
        self.A_init()
        self.D_init()

    def A_init(self):
        # S4D real initialization
        if self.polarity_write_mode == "polarity_state":
            # Give positive/negative banks the same initial decay spectrum.
            # Splitting arange(1, 17) directly would make one polarity decay faster.
            rates = torch.arange(1, self.d_state // 2 + 1, dtype=torch.float32, device=self.device)
            rates = torch.cat([rates, rates])
        else:
            rates = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.device)
        A = repeat(
            rates, "n -> d n", d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

    def dt_init(self, dt_init="random", dt_scale=1.0, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **self.factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

    def D_init(self):
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=self.device))  # Keep in fp32
        self.D._no_weight_decay = True

    def get_write_gate(self, normalized_x, valid_mask, relation):
        if self.polarity_write_mode in {"none", "polarity_state"}:
            return None
        if valid_mask is None:
            raise ValueError("Explicit writing requires a [batch, length] valid_mask")
        if self.polarity_write_mode == "identity":
            return valid_mask.unsqueeze(-1).to(normalized_x.dtype)
        if self.polarity_write_mode == "content":
            gate_input = normalized_x
        else:
            if relation is None or relation.shape != (*normalized_x.shape[:2], 8):
                raise ValueError("Expected current-path relations of shape [batch, length, 8]")
            gate_input = relation.to(normalized_x.dtype)
            if self.polarity_write_mode == "geometry":
                gate_input = gate_input.clone()
                gate_input[..., 4:7] = 0  # Remove only the explicit polarity inputs.
        gate = 2.0 * torch.sigmoid(self.write_gate(gate_input))
        pair_valid = torch.zeros_like(valid_mask)
        pair_valid[:, 1:] = valid_mask[:, 1:] & valid_mask[:, :-1]
        # No previous event exists at a sequence start: use neutral writing.
        gate = torch.where(pair_valid.unsqueeze(-1), gate, torch.ones_like(gate))
        return gate * valid_mask.unsqueeze(-1).to(gate.dtype)

    @torch.no_grad()
    def build_polarity_state_mask(self, polarity, valid_mask):
        """Route observed event features into +/- halves of one recurrent state.

        polarity: signed metadata [batch, length, 1], serialized in the SAME
        order as the features. Padding is inactive regardless of its value.
        Returns [batch, state, length]. Both banks retain their normal dynamics;
        this mask affects only the input term B*u, never the state transition A.
        """
        if self.d_state < 2 or self.d_state % 2:
            raise ValueError("Polarity state routing requires an even state dimension")
        if polarity is None or valid_mask is None:
            raise ValueError("polarity_state requires signed polarity and valid_mask")
        if polarity.shape != (*valid_mask.shape, 1):
            raise ValueError("polarity must have shape [batch, length, 1]")
        p = polarity[..., 0]
        half = self.d_state // 2
        positive = ((p > 0) & valid_mask).unsqueeze(1).expand(-1, half, -1)
        negative = ((p < 0) & valid_mask).unsqueeze(1).expand(-1, half, -1)
        return torch.cat([positive, negative], dim=1).to(dtype=torch.float32)

    def ssm(self, conv1d_out, z,
            x_proj_weight, delta_proj_weight,
            A, B=None, C=None, D=None,
            delta_bias=None, B_proj_bias=None, C_proj_bias=None, delta_softplus=True,
            write_gate=None, state_write_mask=None):
        L = conv1d_out.shape[-1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        delta_rank = self.dt_proj.weight.shape[1]

        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)

        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None: D = D.contiguous()

        if write_gate is not None:
            # write_gate [batch, length, 1], variable B [batch, group, state, length].
            gate = write_gate.permute(0, 2, 1).to(B.dtype)
            if B.dim() == 4:
                gate = gate.unsqueeze(1)
            elif B.dim() != 3:
                raise ValueError("Polarity writing expects token-dependent B")
            B = (B * gate).contiguous()

        if state_write_mask is not None:
            if write_gate is not None:
                raise ValueError("Use either relation-gated writing or polarity state routing")
            # First half of B writes positive evidence; second half writes negative.
            # C still reads all state dimensions jointly in the original CUDA scan.
            mask = state_write_mask.to(B.dtype)
            if B.dim() == 4:
                mask = mask.unsqueeze(1)
            elif B.dim() != 3:
                raise ValueError("Polarity routing expects token-dependent B")
            B = (B * mask).contiguous()

        out = selective_scan_fn(conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus)

        # return F.linear(rearrange(out, "b d l -> b l d"), out_proj_weight, out_proj_bias)
        return out

        
    @autocast(enabled=False)
    def forward(self, multi_x, valid_mask=None, relation=None, polarity=None):
        if multi_x.dtype in {torch.float16, torch.bfloat16}:
            multi_x = multi_x.float()
        multi_batch, multi_seqlen, channels = multi_x.shape
        if channels != self.d_model:
            raise ValueError("Input channel count does not match d_model")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=multi_x.device, dtype=torch.bool)
            if valid_mask.shape != multi_x.shape[:2]:
                raise ValueError("valid_mask must have shape [batch, length]")
            multi_x = multi_x * valid_mask.unsqueeze(-1).to(multi_x.dtype)
        multi_hidden_states = self.norm(multi_x)
        if valid_mask is not None:
            # LayerNorm has an affine bias: mask again before the noncausal conv.
            multi_hidden_states = multi_hidden_states * valid_mask.unsqueeze(-1).to(multi_hidden_states.dtype)
        write_gate = self.get_write_gate(multi_hidden_states, valid_mask, relation)
        state_write_mask = None
        if self.polarity_write_mode == "polarity_state":
            state_write_mask = self.build_polarity_state_mask(polarity, valid_mask)
        multix_z = rearrange(
            self.in_proj.weight @ rearrange(multi_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=multi_seqlen,
        )
        A = -torch.exp(self.A_log.float())
        multix_z = multix_z.contiguous()
        multix, z = multix_z.chunk(2, dim=1)
        mgafeat = self.act(self.conv1d(multix))
        if valid_mask is not None:
            mgafeat = mgafeat * valid_mask.unsqueeze(1).to(mgafeat.dtype)
        ssm_out = self.ssm(
            conv1d_out=mgafeat, z=z,
            x_proj_weight=self.x_proj.weight,
            delta_proj_weight=self.dt_proj.weight,
            A=A, B=None, C=None, D=self.D.float(),
            delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            write_gate=write_gate,
            state_write_mask=state_write_mask,
        )
        mamba_out = self.out_proj(rearrange(ssm_out, "b d l -> b l d"))
        if valid_mask is not None:
            mamba_out = mamba_out * valid_mask.unsqueeze(-1).to(mamba_out.dtype)
        return mamba_out
