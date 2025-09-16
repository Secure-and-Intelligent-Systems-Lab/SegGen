import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from components.factory.config_manager import ConfigManager
from components.factory.logger_factory import LoggerFactory
from components.factory.experiment_manager import ExperimentManager
from components.factory.factory import MODELS, DATASETS, LOSSES, SCHEDULERS
from components.callbacks.base import Callback
from components.utils import setup_ddp, cleanup_ddp, fix_seeds, setup_cudnn
from utils.metrics import Metrics



class Trainer:
    def __init__(self, cfg_path: str, experiment_name: str, callbacks: list[Callback] = [], rank: int = 0, world_size: int = 1):
        # Setup configuration
        fix_seeds()
        setup_cudnn()
        self.rank = rank
        self.world_size = world_size
        self.config_manager = ConfigManager(cfg_path)
        self.ddp_enabled = self.config_manager.get('TRAIN.DDP', False)

        # Setup DDP
        if self.ddp_enabled:
            setup_ddp(rank, world_size)
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging & experiment tracking (only from rank 0)
        if self.rank == 0:
            self.experiment_manager = ExperimentManager(self.config_manager.get('SAVE_DIR'), experiment_name)
            self.logger = LoggerFactory.create_logger(self.experiment_manager.get_experiment_dir())
        else:
            self.experiment_manager = None
            self.logger = None

        # Setup model, dataset, and loss function
        dataset_cls = DATASETS.get(self.config_manager.get('DATASET.NAME'))
        model_cls = MODELS.get(self.config_manager.get('MODEL.NAME'))
        loss_cls = LOSSES.get(self.config_manager.get('LOSS.NAME'))
        scheduler_cls = SCHEDULERS.get(self.config_manager.get('SCHEDULER.NAME'))

        self.trainset = dataset_cls(self.config_manager.get('DATASET.ROOT'), 'train', modals=self.config_manager.get('DATASET.MODALS'))
        self.valset = dataset_cls(self.config_manager.get('DATASET.ROOT'), 'val', modals=self.config_manager.get('DATASET.MODALS'))

        self.model = model_cls(self.config_manager.get('MODEL.BACKBONE'), self.trainset.n_classes,
                               self.config_manager.get('DATASET.MODALS'), self.config_manager.get('MODEL.PRETRAINED')).to(self.device)
        if self.ddp_enabled:
            self.model = DDP(self.model, device_ids=[rank])

        self.loss_fn = loss_cls().to(self.device)

        # Setup optimizer and scheduler
        optimizer_cls = getattr(torch.optim, self.config_manager.get('OPTIMIZER.NAME'))
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.config_manager.get('OPTIMIZER.LR'))
        iters_per_epoch = len(self.trainset) // int(self.config_manager.get('TRAIN.BATCH_SIZE'))
        self.iters_per_epoch = iters_per_epoch
        self.scheduler = scheduler_cls(self.optimizer, (int(self.config_manager.get('TRAIN.EPOCHS'))+1)*iters_per_epoch,
                                       self.config_manager.get('SCHEDULER.POWER'), iters_per_epoch * self.config_manager.get('SCHEDULER.WARMUP'),
                                       self.config_manager.get('SCHEDULER.WARMUP_RATIO'))

        # Setup data loaders
        if self.ddp_enabled:
            sampler = DistributedSampler(self.trainset, num_replicas=world_size, rank=rank, shuffle=True)
        else:
            sampler = RandomSampler(self.trainset)

        self.trainloader = DataLoader(self.trainset, batch_size=self.config_manager.get('TRAIN.BATCH_SIZE'),
                                      sampler=sampler, num_workers= 8, prefetch_factor=16)
        self.valloader = DataLoader(self.valset, batch_size=self.config_manager.get('EVAL.BATCH_SIZE'),
                                    num_workers=8, prefetch_factor=16)

        self.scaler = GradScaler(enabled=self.config_manager.get('TRAIN.AMP'))

        # Callbacks
        self.callbacks = callbacks

    def train(self):
        epochs = self.config_manager.get('TRAIN.EPOCHS')
        if self.rank == 0:
            for callback in self.callbacks:
                callback.on_train_begin(self)

        for epoch in range(epochs):
            if self.ddp_enabled:
                self.trainloader.sampler.set_epoch(epoch)

            if self.rank == 0:
                for callback in self.callbacks:
                    callback.on_epoch_begin(self, epoch)

            self.model.train()
            epoch_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if self.rank == 0:
                    for callback in self.callbacks:
                        callback.on_train_batch_begin(self, (inputs, targets), batch_idx)
                inputs = [x.to(self.device, non_blocking=True) for x in inputs] if type(inputs) is list else inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                with autocast(enabled=self.config_manager.get('TRAIN.AMP'), device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_loss += loss.item()

                if self.rank == 0:
                    for callback in self.callbacks:
                        callback.on_train_batch_end(self, (inputs, targets), batch_idx, loss.item())

            avg_loss = epoch_loss / len(self.trainloader)
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
                self.scheduler.step()
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch)

        if self.rank == 0:
            for callback in self.callbacks:
                callback.on_train_end(self)

        if self.ddp_enabled:
            cleanup_ddp()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.valloader)
        if self.rank == 0:
            self.logger.info(f"Validation Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        n_classes = self.trainset.n_classes
        metrics = Metrics(n_classes, self.trainset.ignore_label, self.device)
        total_loss = 0
        for images, targets in self.valloader:
            images = [x.to(self.device, non_blocking = True) for x in images] if type(images) is list else images.to(self.device, non_blocking = True)
            targets = targets.to(self.device, non_blocking = True)
            outputs = self.model(images).softmax(dim=1)
            loss = self.loss_fn(outputs, targets)
            metrics.update(outputs, targets)
            total_loss += loss.item()

        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()
        avg_loss = total_loss / len(self.valloader)

        return acc, macc, f1, mf1, ious, miou, avg_loss