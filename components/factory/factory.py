# src/factory/factory.py

from typing import Type, Dict


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type:
        if name not in self._registry:
            raise ValueError(f"{name} is not registered in the registry.")
        return self._registry[name]


# Create global registries for models, datasets, and losses
MODELS = Registry()
DATASETS = Registry()
LOSSES = Registry()
SCHEDULERS = Registry()

# Usage example:
#
# @MODELS.register('FuseForm')
# class FuseForm(nn.Module):
#     ...