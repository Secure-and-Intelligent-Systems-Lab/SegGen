from components.callbacks.base import Callback
from components.utils import print_results
class EvalCallback(Callback):
    def __init__(self, eval_interval=1, start_epoch=1):
        self.eval_interval = eval_interval
        self.start_epoch = start_epoch

    def on_epoch_end(self, trainer, epoch):
        if epoch >= self.start_epoch:
            if (epoch + 1) % self.eval_interval == 0:
                acc, macc, f1, mf1, ious, miou, avg_loss = trainer.evaluate(epoch)
                trainer.logger.info(f"\nEpoch {epoch}\nValidation Results:")
                trainer.logger.info(print_results(epoch, ious, miou, acc, macc, trainer.trainset.CLASSES))
                trainer.logger.info(f"Epoch {epoch+1} Validation loss: {avg_loss:.4f}\n")
                setattr(trainer, 'best_mIoU', max(getattr(trainer, 'best_mIoU', 0), miou))
