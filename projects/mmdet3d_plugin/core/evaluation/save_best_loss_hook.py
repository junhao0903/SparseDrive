import os.path as osp
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SaveBestLossHook(Hook):
    """Save checkpoint when training loss reaches a new minimum."""

    def __init__(self, key='loss', filename='best_loss.pth'):
        self.key = key
        self.filename = filename
        self.best_loss = float('inf')

    def after_train_iter(self, runner):
        history = runner.log_buffer.val_history
        if self.key not in history:
            return
        values = history[self.key]
        if not values:
            return
        import torch
        val = values[-1]
        if torch.is_tensor(val):
            val = val.item()
        current = float(val)
        if current < self.best_loss:
            self.best_loss = current
            runner.save_checkpoint(
                runner.work_dir, filename_tmpl=self.filename,
                create_symlink=False,
            )
