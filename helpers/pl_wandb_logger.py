import os
import pathlib
from typing import Optional
from weakref import proxy

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class CustomModelCheckpoint(ModelCheckpoint):
    def save_checkpoint(self, trainer: 'pl.Trainer', unused: Optional['pl.LightningModule'] = None) -> None:
        super().save_checkpoint(trainer, unused)
        # notify loggers
        if trainer.is_global_zero and trainer.logger:
            if hasattr(trainer.logger, 'after_save_checkpoint'):
                trainer.logger.after_save_checkpoint(proxy(self))

class CustomWandbLogger(WandbLogger):
    # This flag controls whether we log the model files.
    save_all_checkpoints = False

    def after_save_checkpoint(self, checkpoint_callback):
        # Only proceed if model logging is enabled
        if not self._log_model:
            return

        # Build a set with the file paths for the last and best checkpoints.
        ckpt_paths = set()
        last_path = checkpoint_callback.last_model_path
        best_path = checkpoint_callback.best_model_path

        if last_path and os.path.isfile(last_path):
            ckpt_paths.add(last_path)
        if best_path and os.path.isfile(best_path):
            ckpt_paths.add(best_path)

        # Upload any checkpoint that hasn’t been uploaded yet.
        for path in ckpt_paths:
            self.experiment.save(path)
