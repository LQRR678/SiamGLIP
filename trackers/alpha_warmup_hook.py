from typing import Optional

import torch
# from mmengine.hooks import Hook, HOOKS
from mmengine.hooks import Hook
from mmengine.registry import HOOKS 


@HOOKS.register_module()
class FusionAlphaWarmupHook(Hook):
    """
    Warm up fusion alpha for gated fusion modules.

    alpha(epoch) =
      0                                  , epoch < start_epoch
      alpha_max * (epoch - start_epoch + 1) / warmup_epochs  , warmup window
      alpha_max                         , otherwise

    Args:
        alpha_max (float): Target alpha after warmup.
        warmup_epochs (int): Number of warmup epochs.
        start_epoch (int): Epoch to start warmup (1-based, matches runner).
        module_attr (str): Attribute path from model to fusion module.
            Default: 'fusion_module'.
    """

    def __init__(self,
                 alpha_max: float = 0.1,
                 warmup_epochs: int = 4,
                 start_epoch: int = 1,
                 module_attr: str = 'fusion_module') -> None:
        self.alpha_max = float(alpha_max)
        self.warmup_epochs = int(warmup_epochs)
        self.start_epoch = int(start_epoch)
        self.module_attr = module_attr

    def _get_module(self, model) -> Optional[torch.nn.Module]:
        # unwrap DDP if needed
        if hasattr(model, 'module') and getattr(model, 'module') is not None:
            model = model.module
        mod = getattr(model, self.module_attr, None)
        return mod

    def before_train_epoch(self, runner) -> None:
        cur_epoch = runner.epoch + 1  # runner is 0-based internally
        if cur_epoch < self.start_epoch:
            target_alpha = 0.0
        elif self.warmup_epochs <= 0:
            target_alpha = self.alpha_max
        else:
            k = cur_epoch - self.start_epoch + 1
            if k <= 0:
                target_alpha = 0.0
            elif k >= self.warmup_epochs:
                target_alpha = self.alpha_max
            else:
                target_alpha = self.alpha_max * (k / self.warmup_epochs)

        module = self._get_module(runner.model)
        if module is not None and hasattr(module, 'set_alpha'):
            try:
                module.set_alpha(float(target_alpha))
                runner.logger.info(f'[FusionAlphaWarmupHook] epoch={cur_epoch} set alpha={target_alpha:.4f}')
            except Exception as e:
                runner.logger.warning(f'[FusionAlphaWarmupHook] set_alpha failed: {e}')
        else:
            runner.logger.debug('[FusionAlphaWarmupHook] fusion module not found or set_alpha missing; skipped')


