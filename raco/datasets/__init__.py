"""Datasets module for RaCo."""

from .base_dataset import BaseDataset, collate
from .oxford_paris import OxfordParisDataset
from .hpatches import HPatchesDataset


DATASET_REGISTRY = {
    "oxford_paris": OxfordParisDataset,
    "hpatches": HPatchesDataset,
}


def get_dataset(name):
    """Get dataset class by name."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    raise ValueError(f"Dataset {name} not found. Available: {list(DATASET_REGISTRY.keys())}")


__all__ = ["BaseDataset", "get_dataset", "collate", "OxfordParisDataset", "HPatchesDataset"]
