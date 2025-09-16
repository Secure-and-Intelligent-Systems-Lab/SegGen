
class Callback:
    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, trainer, epoch):
        """Called at the end of an epoch."""
        pass

    def on_train_batch_begin(self, trainer, batch, batch_idx):
        """Called at the beginning of a training batch."""
        pass

    def on_train_batch_end(self, trainer, batch, batch_idx, loss):
        """Called at the end of a training batch."""
        pass

    def on_validation_begin(self, trainer):
        """Called before validation starts."""
        pass

    def on_validation_end(self, trainer):
        """Called after validation ends."""
        pass
