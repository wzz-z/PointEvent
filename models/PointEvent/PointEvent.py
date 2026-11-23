"""
Checkpoint Utils for Models

Author: Zongze Wu (zongze0907@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

from models.builder import MODELS
from models.utils.misc import offset2bincount
from models.utils.structure import Point, Point_V2
from models.modules import PointModule, PointSequential
from torch.cuda.amp import autocast
from mamba_ssm.modules.mamba_simple import Mamba
from .mssm import SimpleAwareStateSpaceModel
import numpy as np
import torch.nn.functional as F

class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point

class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factorf
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, n_tokens, C = x.shape
        assert C == self.dim

        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)

        return x_mamba
    
class SimpleStateSpaceModelBlock(PointModule):
    def __init__(
        self,
        channels,
        gather_num,
        order_index=0,
        polarity_write_mode="none",
        relation_scale=(0.03, 0.03, 0.03),
    ):
        super().__init__()
        self.channels = channels
        self.order_index = order_index
        self.gather_num = gather_num
        self.polarity_write_mode = polarity_write_mode
        scale = torch.as_tensor(relation_scale, dtype=torch.float32)
        if scale.shape != (3,) or not bool((scale > 0).all()):
            raise ValueError("relation_scale must contain three positive values")
        self.register_buffer("relation_scale", scale, persistent=False)
        self.mamba = SimpleAwareStateSpaceModel(
            d_model=channels, d_conv=5, polarity_write_mode=polarity_write_mode
        )
        
    @torch.no_grad()
    def get_padding(self, point):
        offset = point.offset
        b_bincount = offset2bincount(offset)
        B = b_bincount.shape[0]
        
        max_bincount = torch.max(b_bincount).item()

        max_indices = b_bincount.unsqueeze(1)
        indices = torch.arange(max_bincount, device=b_bincount.device).unsqueeze(0).expand(B, -1)
        multi_mask = indices < max_indices
        multi_mask = multi_mask.view(-1)
        return multi_mask

    def get_inverse(self, feat, B, L, mask):
        # reverse
        feat = feat.reshape(B * L, -1)
        reverse_feat = feat[mask.view(-1)]
        return reverse_feat

    @torch.no_grad()
    def serialized(self, point):
        sp_inds = point.serialized_order[self.order_index, :]
        sp_inverse = point.serialized_inverse[self.order_index, :]
        return sp_inds, sp_inverse
    
    @torch.no_grad()
    def build_scan_relation(self, coord, polarity, valid_mask):
        # Inputs have already been serialized using THIS block's order.
        delta = torch.zeros_like(coord, dtype=torch.float32)
        delta[:, 1:] = (coord[:, 1:].float() - coord[:, :-1].float()) / self.relation_scale.view(1, 1, 3)
        signed_log_delta = torch.sign(delta) * torch.log1p(delta.abs())
        abs_log_dt = torch.log1p(delta[..., 2:3].abs())
        p_prev = torch.zeros_like(polarity)
        p_prev[:, 1:] = polarity[:, :-1]
        pair_valid = torch.zeros_like(valid_mask)
        pair_valid[:, 1:] = valid_mask[:, 1:] & valid_mask[:, :-1]
        relation = torch.cat([
            signed_log_delta, abs_log_dt,
            polarity, p_prev, polarity * p_prev,
            pair_valid.unsqueeze(-1).to(coord.dtype),
        ], dim=-1)
        return relation * pair_valid.unsqueeze(-1).to(relation.dtype)

    def mssm_forward(self, point, return_feat = False):
        B = point.offset.shape[0]
        C = point.feat.shape[1]
        b_bincount = offset2bincount(point.offset)
        max_bincount = int(torch.max(b_bincount).item())
        
        # serialized
        sp_inds, sp_inverse = self.serialized(point)
        multi_serialized_feat = point.feat[sp_inds]

        # pad
        multi_feat_pad = torch.zeros(B * max_bincount, C, dtype=point.feat.dtype, device=multi_serialized_feat.device)
        
        multi_mask = self.get_padding(point)
        multi_feat_pad[multi_mask] = multi_serialized_feat
        # reshape
        L = max_bincount
        multi_feat_pad = multi_feat_pad.reshape(B, L, -1)

        valid_mask = multi_mask.reshape(B, L)
        relation = None
        polarity_pad = None
        if self.polarity_write_mode in {"polarity", "geometry", "polarity_state"}:
            if "event_polarity" not in point.keys():
                raise ValueError("Save event_polarity before ECE; do not infer it from learned features")
            polarity_pad = point.coord.new_zeros((B * L, 1), dtype=torch.float32)
            polarity_pad[multi_mask] = point.event_polarity[sp_inds].float()
            polarity_pad = polarity_pad.reshape(B, L, 1)
        if self.polarity_write_mode in {"polarity", "geometry"}:
            coord_pad = point.coord.new_zeros((B * L, 3), dtype=torch.float32)
            coord_pad[multi_mask] = point.coord[sp_inds].float()
            relation = self.build_scan_relation(
                coord_pad.reshape(B, L, 3), polarity_pad, valid_mask
            )
        feat = self.mamba(multi_feat_pad, valid_mask=valid_mask, relation=relation,
                          polarity=polarity_pad)

        if return_feat:
            return self.get_inverse(feat, B, L, multi_mask)[sp_inverse]
        
        # pad inverse
        # feat = self.get_inverse(feat, B, L, multi_mask)[sp_inverse] 
        point.feat = self.get_inverse(feat, B, L, multi_mask)
        
        # space inverse
        point.feat = point.feat[sp_inverse]        # L, D
        # point.feat.shape ==torch.Size([319468, 16]) torch.Size([405421, 16])
        return point
    
    def forward(self, point, return_feat = False):
        return self.mssm_forward(point, return_feat=return_feat)

class SimpleStateSpaceModelBlock_TScan(SimpleStateSpaceModelBlock):
    def __init__(self, channels, gather_num, order_index=0,
                 polarity_write_mode="none", relation_scale=(0.03, 0.03, 0.03)):
        super().__init__(channels, gather_num, order_index,
                         polarity_write_mode=polarity_write_mode,
                         relation_scale=relation_scale)
        
    @torch.no_grad()
    def serialized(self, point):
        sp_inds = point.t_serialized_order[self.order_index, :]
        sp_inverse = point.t_serialized_inverse[self.order_index, :]
    
        return sp_inds, sp_inverse



class Block(PointModule):
    def __init__(
        self,
        channels,
        gather_num,
        mlp_ratio=4.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        scan_mode="spatial",
        order_index=0,
        cpe_indice_key=None,
        polarity_write_mode="none",
        relation_scale=(0.03, 0.03, 0.03),
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.scan_mode = scan_mode
        self.order_index = order_index

        self.cpe = ACPE(
            channels=channels,
            norm_layer=norm_layer,
            act_layer=act_layer,
            indice_key=cpe_indice_key,
            temporal_kernel=5,
            use_gate=True,
        )

        self.norm1 = PointSequential(norm_layer(channels))

        # IMPORTANT:
        # scan_mode and order_index are decoupled.
        # order_index only selects the order within spatial or temporal order lists.
        if scan_mode == "spatial":
            self.attn = SimpleStateSpaceModelBlock(
                channels=channels,
                gather_num=gather_num,
                order_index=order_index,
                polarity_write_mode=polarity_write_mode,
                relation_scale=relation_scale,
            )
        elif scan_mode == "temporal":
            self.attn = SimpleStateSpaceModelBlock_TScan(
                channels=channels,
                gather_num=gather_num,
                order_index=order_index,
                polarity_write_mode=polarity_write_mode,
                relation_scale=relation_scale,
            )
        else:
            raise ValueError(f"Unknown scan_mode: {scan_mode}")

        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        # ----- Local curve positional encoding -----
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        # ----- Spatial or temporal serialized SSM -----
        shortcut = point.feat
        point = self.norm1(point)

        feat_scan = self.attn(point, return_feat=True)
        point.feat = shortcut + self.drop_path(feat_scan)

        # ----- FFN -----
        shortcut = point.feat
        point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        # self.n_proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())
        
    @torch.no_grad()
    def batch_time_order(self, coord, batch, descending=False):
        """
        对每个 batch 内部单独按照时间 coord[:, 2] 排序。
        避免不同 batch 之间因为 coord[:,2] 范围过大而混排。

        coord: [N, 3]
        batch: [N]
        return: [N]
        """
        orders = []

        for b in batch.unique(sorted=True):
            idx = torch.nonzero(batch == b, as_tuple=False).reshape(-1)

            if idx.numel() == 0:
                continue

            local_t = coord[idx, 2]
            local_order = torch.argsort(local_t, descending=descending)
            orders.append(idx[local_order])

        return torch.cat(orders, dim=0)
            
    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
            "t_serialized_order",
            "t_serialized_inverse",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders and self.training:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]
        
        coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
        )
        batch = point.batch[head_indices]
        order_ = self.batch_time_order(coord, batch, descending=False)
        order_descend = self.batch_time_order(coord, batch, descending=True)

        t_order = torch.stack([order_, order_descend], dim=0)
        t_inverse = torch.zeros_like(t_order).scatter_(
            dim=1,         
            index=t_order,     
            src=torch.arange(coord.size(0), device=t_order.device).repeat(
                t_order.shape[0], 1
            ),
        )
        
        proj_feat = self.proj(point.feat)

        feat_max = torch_scatter.segment_csr(
            proj_feat[indices],
            idx_ptr,
            reduce="max",
        )

        feat_mean = torch_scatter.segment_csr(
            proj_feat[indices],
            idx_ptr,
            reduce="mean",
        )

        feat_pool = 0.5 * feat_max + 0.5 * feat_mean
        
        # collect information
        point_dict = Dict(
            feat=feat_pool,
            coord=coord,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            t_serialized_order = t_order,
            t_serialized_inverse = t_inverse,
            batch=batch,
        )

        for key in ("segment", "origin_t"):
            if key in point.keys():
                point_dict[key] = point[key][head_indices]

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point_V2(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
        use_gate=True,
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))
        self.use_gate = use_gate

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        if self.use_gate:
            hidden_channels = max(out_channels // 2, 16)
            self.gate = nn.Sequential(
                nn.Linear(out_channels * 2, hidden_channels),
                act_layer() if act_layer is not None else nn.GELU(),
                nn.Linear(hidden_channels, out_channels),
                nn.Sigmoid(),
            )

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")

        point = self.proj(point)
        parent = self.proj_skip(parent)

        ctx_feat = point.feat[inverse]
        if self.use_gate:
            gate = self.gate(torch.cat([parent.feat, ctx_feat], dim=1))
            parent.feat = parent.feat + gate * ctx_feat
        else:
            parent.feat = parent.feat + ctx_feat

        if "odl_loss" in point.keys():
            parent.odl_loss = point.odl_loss

        if self.traceable:
            parent["unpooling_parent"] = point
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)
        return parent

class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

class ACPE(PointModule):
    """
    Anisotropic Curve Positional Encoding.

    The module keeps the original spatial/temporal sparse convolution design,
    but uses a channel-wise gate instead of a two-scalar gate. This makes the
    spatial-temporal preference more flexible under complex backgrounds.

    Note:
        This implementation assumes that the third dimension of grid_coord is
        the temporal dimension. If the sparse tensor coordinate order is not
        (x, y, t), the kernel layout should be adjusted accordingly.
    """

    def __init__(
        self,
        channels,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        indice_key=None,
        temporal_kernel=5,
        use_gate=True,
    ):
        super().__init__()
        self.channels = channels
        self.use_gate = use_gate

        assert temporal_kernel % 2 == 1, "temporal_kernel should be odd."
        temporal_pad = temporal_kernel // 2

        base_key = "acpe" if indice_key is None else indice_key

        self.spatial_conv = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=(3, 3, 1),
            padding=(1, 1, 0),
            bias=True,
            indice_key=f"{base_key}_spatial",
        )

        self.temporal_conv = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=(1, 1, temporal_kernel),
            padding=(0, 0, temporal_pad),
            bias=True,
            indice_key=f"{base_key}_temporal",
        )

        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(channels, channels),
                act_layer(),
                nn.Linear(channels, 2 * channels),
            )
        else:
            self.alpha = nn.Parameter(torch.zeros(2, channels))

        self.proj = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            norm_layer(channels),
        )

    def forward(self, point: Point):
        identity_feat = point.feat

        spatial_feat = self.spatial_conv(point.sparse_conv_feat).features
        temporal_feat = self.temporal_conv(point.sparse_conv_feat).features

        if self.use_gate:
            gate = self.gate(identity_feat).view(-1, 2, self.channels)
            gate = torch.softmax(gate, dim=1)
            feat = gate[:, 0, :] * spatial_feat + gate[:, 1, :] * temporal_feat
        else:
            weight = torch.softmax(self.alpha, dim=0)
            feat = weight[0] * spatial_feat + weight[1] * temporal_feat

        feat = self.proj(feat)

        point.feat = feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(feat)
        return point

class ECE(PointModule):
    """
    Reliability-aware Event Coordinate Embedding.

    Compared with the original xyz/xyp embedding, this version adds a lightweight
    reliability branch computed on the serialized event sequence:
        delta_t: temporal interval to the previous event in the sequence;
        polarity_consistency: polarity relation to the previous event;
        temporal_continuity: exp(-normalized delta_t).

    This branch is intended to preserve weak but temporally continuous target
    events and suppress isolated/noisy responses in complex backgrounds.
    """

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        drop=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        act = act_layer if act_layer is not None else nn.GELU
        norm = norm_layer if norm_layer is not None else nn.BatchNorm1d

        self.stem_xyz = nn.Sequential(
            nn.Conv1d(3, embed_channels, kernel_size=3, padding=1, bias=False),
            act(),
        )
        self.stem_xyp = nn.Sequential(
            nn.Conv1d(3, embed_channels, kernel_size=3, padding=1, bias=False),
            act(),
        )
        self.stem_rel = nn.Sequential(
            nn.Conv1d(3, embed_channels, kernel_size=3, padding=1, bias=False),
            act(),
        )

        self.stem_final = nn.Sequential(
            nn.Conv1d(embed_channels * 3, embed_channels, kernel_size=3, padding=1, bias=False),
            norm(embed_channels),
            act(),
        )

        self.fuse_gate = nn.Sequential(
            nn.Conv1d(embed_channels * 3, embed_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    @torch.no_grad()
    def serialized(self, point):
        sp_inds = point.serialized_order[0, :]
        sp_inverse = point.serialized_inverse[0, :]
        return sp_inds, sp_inverse

    @torch.no_grad()
    def get_padding(self, point):
        offset = point.offset
        b_bincount = offset2bincount(offset)
        B = b_bincount.shape[0]
        max_bincount = int(torch.max(b_bincount).item())

        max_indices = b_bincount.unsqueeze(1)
        indices = torch.arange(max_bincount, device=b_bincount.device).unsqueeze(0).expand(B, -1)
        multi_mask = indices < max_indices
        multi_mask = multi_mask.view(-1)
        return multi_mask

    def get_inverse(self, feat, B, L, mask):
        feat = feat.reshape(B * L, -1)
        reverse_feat = feat[mask.view(-1)]
        return reverse_feat

    def build_reliability_feature(self, pad_feat_xyz, pad_feat_xyp):
        """
        pad_feat_xyz: [B, 3, L], containing x, y, t
        pad_feat_xyp: [B, 3, L], containing x, y, p
        return: [B, 3, L], containing normalized delta_t, polarity consistency,
                and temporal continuity.
        """
        t = pad_feat_xyz[:, 2:3, :]
        p = pad_feat_xyp[:, 2:3, :]

        dt = torch.zeros_like(t)
        dt[:, :, 1:] = t[:, :, 1:] - t[:, :, :-1]
        dt = torch.clamp(dt, min=0.0)

        # Normalize delta_t per sample to avoid numerical underflow when timestamps
        # are in microseconds.
        dt_mean = dt.mean(dim=2, keepdim=True).clamp_min(1e-6)
        dt_norm = dt / dt_mean

        p_prev = torch.zeros_like(p)
        p_prev[:, :, 1:] = p[:, :, :-1]

        polarity_consistency = p * p_prev
        temporal_continuity = torch.exp(-dt_norm).clamp(0.0, 1.0)

        rel_feat = torch.cat([dt_norm, polarity_consistency, temporal_continuity], dim=1)
        return rel_feat

    def forward(self, point: Point):
        sp_inds, sp_inverse = self.serialized(point)
        feat = point.sparse_conv_feat.features[sp_inds]

        B = point.offset.shape[0]
        C_raw = self.in_channels
        b_bincount = offset2bincount(point.offset)
        max_bincount = int(torch.max(b_bincount).item())

        mask = self.get_padding(point)

        pad_feat_xyz = torch.zeros(
            B * max_bincount,
            3,
            dtype=feat.dtype,
            device=feat.device,
        )
        pad_feat_xyp = torch.zeros(
            B * max_bincount,
            3,
            dtype=feat.dtype,
            device=feat.device,
        )

        # Expected raw event feature layout: [x, y, t, p].
        assert C_raw >= 4, "ECE expects at least four input channels: x, y, t, p."
        pad_feat_xyz[mask] = feat[:, :3]
        pad_feat_xyp[mask] = torch.cat((feat[:, 0:1], feat[:, 1:2], feat[:, 3:4]), dim=1)

        pad_feat_xyz = pad_feat_xyz.reshape(B, max_bincount, 3).permute(0, 2, 1)
        pad_feat_xyp = pad_feat_xyp.reshape(B, max_bincount, 3).permute(0, 2, 1)

        pad_feat_rel = self.build_reliability_feature(pad_feat_xyz, pad_feat_xyp)

        feat_xyz = self.stem_xyz(pad_feat_xyz)
        feat_xyp = self.stem_xyp(pad_feat_xyp)
        feat_rel = self.stem_rel(pad_feat_rel)

        feat_cat = torch.cat((feat_xyz, feat_xyp, feat_rel), dim=1)
        feat_fused = self.stem_final(feat_cat)
        gate = self.fuse_gate(feat_cat)

        # Preserve geometry-dominant information while injecting reliability cues.
        pad_feat = feat_xyz + gate * feat_fused
        pad_feat = self.drop(pad_feat)

        pad_feat = self.get_inverse(
            pad_feat.permute(0, 2, 1),
            B,
            max_bincount,
            mask,
        )
        pad_feat = pad_feat[sp_inverse]

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(pad_feat)
        point.feat = pad_feat
        return point


@MODELS.register_module("PointEvent")
class PointEvent(PointModule):
    """
    High-resolution context-guided PointEvent.

    This version is intentionally different from the original encoder-decoder
    design. It keeps a high-resolution event stream as the main path and uses
    a single context branch for background-aware feature modulation:

        ECE -> HR Blocks -> Context Down -> Context Blocks -> Gated Injection
            -> HR Refinement Blocks -> event-wise output

    Compared with the previous enc-pool-enc-unpool-dec pipeline, weak target
    events are preserved in the high-resolution stream, while pooled context is
    injected only through a gate. This is designed for complex scenes where
    coarse background context is useful but should not overwrite tiny targets.
    """

    def __init__(
        self,
        in_channels=4,
        gather_num=4,
        order=("z", "z_trans"),
        stride=(2,),
        enc_depths=(2, 4),
        enc_channels=(32, 64),
        dec_depths=(2,),
        dec_channels=(32,),
        mlp_ratio=4,
        proj_drop=0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        use_context_branch=True,
        polarity_write_mode="none",
        relation_scale=(0.03, 0.03, 0.03),
        polarity_encoding="zero_one",
    ):
        super().__init__()
        self.t_order_len = 2
        self.order = [order] if isinstance(order, str) else list(order)
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.use_context_branch = use_context_branch
        self.polarity_write_mode = polarity_write_mode
        self.relation_scale = relation_scale
        if polarity_encoding not in {"zero_one", "signed"}:
            raise ValueError("polarity_encoding must be 'zero_one' or 'signed'")
        self.polarity_encoding = polarity_encoding

        stride = [stride] if isinstance(stride, int) else list(stride)
        enc_depths = [enc_depths] if isinstance(enc_depths, int) else list(enc_depths)
        enc_channels = [enc_channels] if isinstance(enc_channels, int) else list(enc_channels)
        dec_depths = [dec_depths] if isinstance(dec_depths, int) else list(dec_depths)
        dec_channels = [dec_channels] if isinstance(dec_channels, int) else list(dec_channels)

        assert len(enc_channels) >= 1, "enc_channels must contain at least one channel value."
        self.hr_channels = enc_channels[0]
        self.context_channels = enc_channels[1] if len(enc_channels) > 1 else enc_channels[0]
        self.context_stride = stride[0] if len(stride) > 0 else 2

        # Interpret the original config in a new architecture:
        # enc_depths[0] -> high-resolution pre-context depth
        # enc_depths[1] -> context branch depth
        # dec_depths[0] -> high-resolution refinement depth
        self.hr_pre_depth = enc_depths[0] if len(enc_depths) > 0 else 2
        self.context_depth = enc_depths[1] if len(enc_depths) > 1 else max(2, self.hr_pre_depth)
        self.hr_post_depth = dec_depths[0] if len(dec_depths) > 0 else 2

        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm

        act_layer = nn.GELU
        total_depth = self.hr_pre_depth + self.context_depth + self.hr_post_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, total_depth)]
        cursor = 0

        self.embedding = ECE(
            in_channels=in_channels,
            embed_channels=self.hr_channels,
            norm_layer=bn_layer,
            act_layer=act_layer,
            drop=0.1,
        )

        self.hr_pre = self._make_blocks(
            channels=self.hr_channels,
            depth=self.hr_pre_depth,
            drop_paths=dpr[cursor: cursor + self.hr_pre_depth],
            gather_num=gather_num,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop,
            norm_layer=ln_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
            name_prefix="hr_pre",
        )
        cursor += self.hr_pre_depth

        if self.use_context_branch:
            self.context_down = SerializedPooling(
                in_channels=self.hr_channels,
                out_channels=self.context_channels,
                stride=self.context_stride,
                norm_layer=bn_layer,
                act_layer=act_layer,
                shuffle_orders=shuffle_orders,
            )
            self.context_blocks = self._make_blocks(
                channels=self.context_channels,
                depth=self.context_depth,
                drop_paths=dpr[cursor: cursor + self.context_depth],
                gather_num=gather_num,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop,
                norm_layer=ln_layer,
                act_layer=act_layer,
                pre_norm=pre_norm,
                name_prefix="ctx",
            )
            self.context_up = SerializedUnpooling(
                in_channels=self.context_channels,
                skip_channels=self.hr_channels,
                out_channels=self.hr_channels,
                norm_layer=bn_layer,
                act_layer=act_layer,
                use_gate=True,
            )
        else:
            self.context_down = None
            self.context_blocks = None
            self.context_up = None
        cursor += self.context_depth

        self.hr_post = self._make_blocks(
            channels=self.hr_channels,
            depth=self.hr_post_depth,
            drop_paths=dpr[cursor: cursor + self.hr_post_depth],
            gather_num=gather_num,
            mlp_ratio=mlp_ratio,
            proj_drop=proj_drop,
            norm_layer=ln_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
            name_prefix="hr_post",
        )

    def _make_blocks(
        self,
        channels,
        depth,
        drop_paths,
        gather_num,
        mlp_ratio,
        proj_drop,
        norm_layer,
        act_layer,
        pre_norm,
        name_prefix,
    ):
        seq = PointSequential()
        for i in range(depth):
            if i % 2 == 0:
                scan_mode = "spatial"
                order_idx = (i // 2) % len(self.order)
            else:
                scan_mode = "temporal"
                order_idx = (i // 2) % self.t_order_len

            seq.add(
                Block(
                    channels=channels,
                    gather_num=gather_num,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    drop_path=drop_paths[i] if i < len(drop_paths) else 0.0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    scan_mode=scan_mode,
                    order_index=order_idx,
                    cpe_indice_key=f"{name_prefix}_block{i}",
                    polarity_write_mode=(self.polarity_write_mode
                        if name_prefix in {"hr_pre", "hr_post"} else "none"),
                    relation_scale=self.relation_scale,
                ),
                name=f"block{i}",
            )
        return seq

    def forward(self, data_dict):
        point = Point_V2(data_dict)
        # Preserve raw metadata before ECE replaces point.feat.
        raw_polarity = point.feat[:, 3:4].detach()
        point["event_polarity"] = (2.0 * raw_polarity - 1.0
            if self.polarity_encoding == "zero_one" else raw_polarity)
        point.serialization(
            order=self.order,
            shuffle_orders=(self.shuffle_orders and self.training),
        )
        point.time_serialization()
        point.sparsify(wtn=False)

        point = self.embedding(point)
        point = self.hr_pre(point)

        if self.use_context_branch:
            context = self.context_down(point)
            context = self.context_blocks(context)
            point = self.context_up(context)

        point = self.hr_post(point)
        return point
