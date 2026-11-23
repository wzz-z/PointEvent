
import torch.optim.lr_scheduler as lr_scheduler
from .registry import Registry

SCHEDULERS = Registry("schedulers")


@SCHEDULERS.register_module()
class MultiStepLR(lr_scheduler.MultiStepLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            milestones=[rate * total_steps for rate in milestones],
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class MultiStepWithWarmupLR(lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        milestones,
        total_steps,
        gamma=0.1,
        warmup_rate=0.05,
        warmup_scale=1e-6,
        last_epoch=-1,
        verbose=False,
    ):
        milestones = [rate * total_steps for rate in milestones]

        def multi_step_with_warmup(s):
            factor = 1.0
            for i in range(len(milestones)):
                if s < milestones[i]:
                    break
                factor *= gamma

            if s <= warmup_rate * total_steps:
                warmup_coefficient = 1 - (1 - s / warmup_rate / total_steps) * (
                    1 - warmup_scale
                )
            else:
                warmup_coefficient = 1.0
            return warmup_coefficient * factor

        super().__init__(
            optimizer=optimizer,
            lr_lambda=multi_step_with_warmup,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class PolyLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps, power=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: (1 - s / (total_steps + 1)) ** power,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class ExpLR(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, total_steps, gamma=0.9, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=lambda s: gamma ** (s / total_steps),
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, total_steps, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
            verbose=verbose,
        )


@SCHEDULERS.register_module()
class OneCycleLR(lr_scheduler.OneCycleLR):
    r"""
    torch.optim.lr_scheduler.OneCycleLR, Block total_steps
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_epoch,
            verbose=verbose,
        )
        
from torch.optim.lr_scheduler import _LRScheduler
import math

@SCHEDULERS.register_module()
class WarmupConstLR(_LRScheduler):
    r"""
    Warmup + Const 形式的学习率调度器。

    - 在 warmup_iters 步内，学习率线性上升从 warmup_ratio * base_lr 到 base_lr；
    - warmup 结束后，学习率恒定保持为 base_lr，直到训练结束。

    参数：
        optimizer (Optimizer): 被调度的优化器。
        warmup_iters (int): warmup 的迭代次数。
        warmup_ratio (float): 初始学习率相对于 base_lr 的比例。
        total_iters (int): 总训练步数。
        last_epoch (int): 上一次的 epoch，默认 -1。
        verbose (bool): 是否输出调试信息。
    """

    def __init__(
        self,
        optimizer,
        warmup_iters=500,
        warmup_ratio=0.1,
        total_steps=10000,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.total_iters = total_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # 当前训练步数
        step = self.last_epoch + 1
        if step < self.warmup_iters:
            # warmup 阶段，线性增长
            return [
                base_lr * (self.warmup_ratio + (1 - self.warmup_ratio) * step / self.warmup_iters)
                for base_lr in self.base_lrs
            ]
        else:
            # 常数阶段
            return [base_lr for base_lr in self.base_lrs]



from torch.optim.lr_scheduler import ReduceLROnPlateau

@SCHEDULERS.register_module()
class ReduceLROnPlateauScheduler(ReduceLROnPlateau):
    def __init__(self,
                 optimizer,
                 mode='max',
                 factor=0.1,
                 patience=3,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=1,
                 min_lr=0,
                 eps=1e-8,
                 verbose=False):
        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose
        )

@SCHEDULERS.register_module()
class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self,
                 optimizer,
                 warmup_epochs=5,
                 base_lr=1e-3,
                 warmup_start_lr=None,
                 mode='max',
                 factor=0.1,
                 patience=10,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8,
                 verbose=False):
        # 先初始化父类
        super().__init__(optimizer=optimizer,
                         mode=mode,
                         factor=factor,
                         patience=patience,
                         threshold=threshold,
                         threshold_mode=threshold_mode,
                         cooldown=cooldown,
                         min_lr=min_lr,
                         eps=eps,
                         verbose=verbose)
        
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else base_lr * 0.1
        self.current_epoch = 0
        self._finished_warmup = False

        # 初始化为 warmup_start_lr
        for group in self.optimizer.param_groups:
            group['lr'] = self.warmup_start_lr

    def step(self, metrics, epoch=None):
        if self.current_epoch < self.warmup_epochs:
            # warmup 阶段线性上升
            warmup_lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * ((self.current_epoch + 1) / self.warmup_epochs)
            for group in self.optimizer.param_groups:
                group['lr'] = warmup_lr
            if epoch is None:
                print(f"[Warmup] Epoch {self.current_epoch + 1}, LR = {warmup_lr:.6f}")
        else:
            # warmup 结束，使用 ReduceLROnPlateau 逻辑
            if not self._finished_warmup:
                print(f"[Warmup] Finished at Epoch {self.current_epoch}, base_lr = {self.base_lr:.6f}")
                self._finished_warmup = True
            super().step(metrics, epoch)

        self.current_epoch += 1
        
        
def build_scheduler(cfg, optimizer):
    cfg.optimizer = optimizer
    return SCHEDULERS.build(cfg=cfg)
