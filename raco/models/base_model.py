"""
Base class for trainable models.
Adapted from glue-factory.
"""

from abc import ABCMeta, abstractmethod
from copy import copy

import omegaconf
from omegaconf import OmegaConf
from torch import nn


class MetaModel(ABCMeta):
    def __prepare__(name, bases, **kwds):
        total_conf = OmegaConf.create()
        for base in bases:
            for key in ("base_default_conf", "default_conf"):
                update = getattr(base, key, {})
                if isinstance(update, dict):
                    update = OmegaConf.create(update)
                total_conf = OmegaConf.merge(total_conf, update)
        return dict(base_default_conf=total_conf)


class BaseModel(nn.Module, metaclass=MetaModel):
    """
    Base class for all trainable models.

    Child classes should declare:
        - default_conf: dict of default configuration
        - required_data_keys: list of expected input data keys
        - _init(self, conf): initialization
        - _forward(self, data): forward pass returning predictions
        - loss(self, pred, data): return dict of losses
        - metrics(self, pred, data): return dict of metrics
    """

    default_conf = {
        "name": None,
        "trainable": True,
        "freeze_batch_normalization": False,
    }
    required_data_keys = []
    strict_conf = False

    def __init__(self, conf):
        super().__init__()
        default_conf = OmegaConf.merge(
            self.base_default_conf, OmegaConf.create(self.default_conf)
        )
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        if self.conf.freeze_batch_normalization:
            self.apply(freeze_bn)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    def forward(self, data):
        """Check inputs and call _forward."""
        for key in self.required_data_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        return self._forward(data)

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def loss(self, pred, data):
        """
        Compute losses. Returns dict of losses, each a tensor of shape (B,).
        The total loss to optimize should have key 'total'.
        """
        raise NotImplementedError

    def metrics(self, pred, data):
        """Compute metrics. Returns dict of metrics."""
        return {}
