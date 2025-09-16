# src/factory/experiment_manager.py

import os
import time
import shutil
from typing import Optional


class ExperimentManager:
    def __init__(self, base_dir: str, experiment_name: Optional[str] = None):
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        if experiment_name is None:
            experiment_name = timestamp
        else:
            experiment_name = f"{experiment_name}_{timestamp}"

        self.experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

    def get_experiment_dir(self) -> str:
        return self.experiment_dir

    def save_config(self, cfg_path: str):
        shutil.copy2(cfg_path, os.path.join(self.experiment_dir, 'config.yaml'))
