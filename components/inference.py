import os
import torch
from pathlib import Path
from components.factory.auto_register import auto_register_modules
auto_register_modules("datasets")
auto_register_modules("models")
from components.factory.factory import MODELS, DATASETS
from components.factory.logger_factory import LoggerFactory

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap



class SemSeg:
    def __init__(self, trainer, logger):
        self.config_manager = trainer.config_manager
        self.logger = logger
        self.device = torch.device(self.config_manager.get('DEVICE', 'cuda'))

        dataset_cls = DATASETS.get(self.config_manager.get('DATASET.NAME'))
        self.dataset = dataset_cls(self.config_manager.get('DATASET.ROOT'), 'val', modals=self.config_manager.get('DATASET.MODALS'))
        self.labels = self.dataset.CLASSES
        self.palette = self.dataset.PALETTE.numpy() / 255.0

        model_cls = MODELS.get(self.config_manager.get('MODEL.NAME'))
        self.model = model_cls(self.config_manager.get('MODEL.BACKBONE'), self.dataset.n_classes, self.config_manager.get('DATASET.MODALS')).to(self.device)

        model_path = self.config_manager.get('EVAL.MODEL_PATH')
        state_dict = torch.load(model_path, map_location='cpu')['model_state_dict']
        msg = self.model.load_state_dict(state_dict, strict=True)
        self.logger.info(f"Loaded model from {model_path} with message: {msg}")
        self.model.eval()

    def predict(self, sample):
        with torch.inference_mode():
            output = self.model(sample)
        return output.softmax(dim=1).argmax(dim=1).cpu().to(torch.uint8)

    def save_prediction(self, pred_map, save_path):
        color_map = ListedColormap(self.palette)
        plt.imshow(pred_map.squeeze(0), cmap=color_map)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()


class InferenceRunner:
    def __init__(self, trainer):
        self.cfg = trainer.config_manager.cfg
        self.logger = LoggerFactory.create_logger(os.path.join(trainer.experiment_manager.get_experiment_dir(), "inference.log"))
        self.semseg = SemSeg(trainer, self.logger)
        self.experiment_dir = trainer.experiment_manager.get_experiment_dir()

    def run(self):
        dataset = self.semseg.dataset
        save_dir = Path(self.experiment_dir) / 'inference_results'
        pred_save_dir = save_dir / 'inference_predictions'
        label_dir = save_dir / 'labels'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(pred_save_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for i, (sample, label) in enumerate(dataset):
            sample = [s.to(self.semseg.device).unsqueeze(0) for s in sample]
            prediction = self.semseg.predict(sample)

            pred_path = save_dir / f"pred_{i}.png"
            label_path = label_dir / f"label_{i}.png"

            self.semseg.save_prediction(prediction, pred_path)
            self.semseg.save_prediction(label.unsqueeze(0), label_path)

            self.logger.info(f"Saved prediction: {pred_path}")
            self.logger.info(f"Saved label: {label_path}")
