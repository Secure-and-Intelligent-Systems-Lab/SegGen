import argparse
import torch
import torch.multiprocessing as mp
from components.trainer import Trainer
from components.callbacks.checkpoint import CheckpointCallback
from components.callbacks.eval import EvalCallback
from components.callbacks.train_log import TrainLogCallback
from components.factory.config_manager import ConfigManager
from components.inference import InferenceRunner
from components.factory.auto_register import auto_register_modules



def ddp_worker(rank, world_size, cfg_path, experiment_name):
    auto_register_modules("datasets")
    auto_register_modules("models")
    auto_register_modules("losses")
    auto_register_modules("schedulers")
    callbacks = [
        EvalCallback(eval_interval=1, start_epoch=25),
        CheckpointCallback(monitor='mIoU', mode='max'),
        TrainLogCallback(eval_interval=10)
    ]
    trainer = Trainer(cfg_path=cfg_path, experiment_name=experiment_name, callbacks=callbacks, rank=rank, world_size=world_size)
    trainer.train()
    if rank == 0:
        runner = InferenceRunner(trainer)
        runner.run()


def main():
    parser = argparse.ArgumentParser(description='Run training experiments.')
    parser.add_argument('--cfg', type=str, default="/home/justin/PycharmProjects/UE5-Semantic-Segmentation/configs/ue5cmx.yaml", help='Path to the configuration file')
    parser.add_argument('--exp-name', type=str, default=None, help='Name for the experiment')
    args = parser.parse_args()

    config = ConfigManager(args.cfg)
    use_ddp = config.get('TRAIN.DDP', False)
    world_size = torch.cuda.device_count() if use_ddp else 1

    if use_ddp:
        mp.spawn(ddp_worker, args=(world_size, args.cfg, args.exp_name), nprocs=world_size, join=True)
    else:
        ddp_worker(0, 1, args.cfg, args.exp_name)


if __name__ == '__main__':
    main()
