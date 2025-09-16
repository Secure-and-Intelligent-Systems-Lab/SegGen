# src/factory/config_manager.py

import yaml
from typing import Any, Dict
import os


class ConfigManager:
    def __init__(self, cfg_path: str):
        self.cfg_path = cfg_path
        self.cfg = self.load_config(cfg_path)

    @staticmethod
    def load_config(cfg_path: str) -> Dict[str, Any]:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        val = self.cfg
        for k in keys:
            val = val.get(k, default)
            if val is default:
                break
        return val

    def set(self, key: str, value: Any):
        keys = key.split('.')
        cfg_section = self.cfg
        for k in keys[:-1]:
            cfg_section = cfg_section.setdefault(k, {})
        cfg_section[keys[-1]] = value

    def save(self, path: str = None):
        save_path = path if path else self.cfg_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.cfg, f, default_flow_style=False)

    def save_to_experiment_dir(self, experiment_dir: str):
        save_path = os.path.join(experiment_dir, 'config.yaml')
        self.save(save_path)
        return save_path