# src/factory/auto_register.py

import importlib
import os
import pkgutil
import sys


def auto_register_modules(base_package):
    """
    Automatically imports all modules in the given base_package.
    Ensures all decorators for registration are triggered.
    """
    package = importlib.import_module(base_package)
    package_path = package.__path__

    for _, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if not is_pkg:
            full_module_name = f"{base_package}.{module_name}"
            importlib.import_module(full_module_name)

# Example usage:
# auto_register_modules("datasets")
# auto_register_modules("models")
# auto_register_modules("losses")
