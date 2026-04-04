"""Models module for RaCo."""

import importlib

from .base_model import BaseModel


def get_model(name):
    """Get model class by name."""
    module_name = f"{__name__}.extractors.{name}"
    try:
        module = importlib.import_module(module_name)
        return module.__model__
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Model {name} not found: {e}")


__all__ = ["BaseModel", "get_model"]
