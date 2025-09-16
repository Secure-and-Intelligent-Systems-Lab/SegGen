# src/callbacks/checkpoint.py

from .base import Callback
import os
import torch

class CheckpointCallback(Callback):
    def __init__(self, monitor='mIoU', mode='max'):
        self.monitor = monitor
        self.mode = mode
        self.best_metric = None

    def on_epoch_end(self, trainer, epoch):
        current_metric = getattr(trainer, f'best_{self.monitor}', None)
        if current_metric is None:
            return

        if self.best_metric is None or \
           (self.mode == 'max' and current_metric > self.best_metric) or \
           (self.mode == 'min' and current_metric < self.best_metric):
            self.best_metric = current_metric
            path = self.save_checkpoint(trainer, epoch)

            # Update EVAL.MODEL_PATH in config
            trainer.config_manager.set("EVAL.MODEL_PATH", path)
            trainer.config_manager.save()  # Save to original config
            if trainer.experiment_manager:
                trainer.config_manager.save_to_experiment_dir(trainer.experiment_manager.get_experiment_dir())

    def save_checkpoint(self, trainer, epoch):
        experiment_dir = trainer.experiment_manager.get_experiment_dir()
        for fname in os.listdir(experiment_dir):
            if fname.startswith('best_model') and fname.endswith('.pth'):
                os.remove(os.path.join(experiment_dir, fname))
        save_path = os.path.join(experiment_dir, f'best_model_epoch{epoch+1}_{self.monitor}{self.best_metric:.4f}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            f'best_{self.monitor}': self.best_metric
        }, save_path)
        trainer.logger.info(f"Checkpoint saved at {save_path}")
        return save_path
