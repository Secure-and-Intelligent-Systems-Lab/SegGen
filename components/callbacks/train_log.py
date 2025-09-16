from components.callbacks.base import Callback
from components.utils import print_results
class TrainLogCallback(Callback):
    def __init__(self, eval_interval=5):
        self.eval_interval = eval_interval

    def on_train_batch_end(self, trainer, inputs, batch_idx, loss):
        if (batch_idx + 1) % self.eval_interval == 0:
            trainer.logger.info(f"Batch {batch_idx + 1}/{trainer.iters_per_epoch} loss: {loss}")
