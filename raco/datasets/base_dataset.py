"""Base class for datasets."""

import collections
from abc import ABCMeta, abstractmethod

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler


def collate(batch):
    """Collate function that can handle dicts."""
    if not isinstance(batch, list):
        return batch
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, dim=0)
    elif isinstance(elem, (float, int)):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        return [collate(samples) for samples in zip(*batch)]
    else:
        return torch.stack(batch, 0) if hasattr(batch[0], '__torch_function__') else batch


class BaseDataset(metaclass=ABCMeta):
    """Base class for all datasets."""

    base_default_conf = {
        "name": "???",
        "batch_size": 1,
        "num_workers": 4,
        "seed": 0,
        "pin_memory": True,
        "persistent_workers": False,
    }
    default_conf = {}

    def __init__(self, conf):
        default_conf = OmegaConf.merge(
            OmegaConf.create(self.base_default_conf),
            OmegaConf.create(self.default_conf),
        )
        OmegaConf.set_struct(default_conf, True)
        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(self.conf, True)
        self._init(self.conf)

    @abstractmethod
    def _init(self, conf):
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, split):
        raise NotImplementedError

    def get_data_loader(self, split, shuffle=None):
        dataset = self.get_dataset(split)
        batch_size = self.conf.batch_size
        num_workers = self.conf.num_workers
        pin_memory = self.conf.get("pin_memory", True)
        persistent_workers = self.conf.get("persistent_workers", False)

        if shuffle is None:
            shuffle = split == "train"

        # Persistent workers only when num_workers > 0
        if persistent_workers and num_workers == 0:
            persistent_workers = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=(split == "train"),
        )
