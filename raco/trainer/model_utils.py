"""
Model utilities for trainer.
Handles parameter freezing, gradient requirements, etc.
"""

import torch
from typing import List, Optional


def set_stage_require_grad(model: torch.nn.Module, stage: str) -> None:
    """
    Set requires_grad for parameters based on training stage.

    Args:
        model: RaCo model
        stage: Training stage (detector/ranker/covariance/ranker_covariance)

    Note:
        Detector params: encoder (block1-4, conv1-4, pool2, pool4, gate) + score_head
        Ranker params: ranker_head
        Covariance params: covariance_estimator_head
        Ranker+Covariance params: ranker_head + covariance_estimator_head
    """
    detector_names = [
        "block1", "block2", "block3", "block4",
        "conv1", "conv2", "conv3", "conv4",
        "pool2", "pool4", "gate", "normalizer",
        "score_head"
    ]

    for name, param in model.named_parameters():
        if stage == "detector":
            # Train everything
            param.requires_grad = True
        elif stage == "ranker":
            # Only ranker_head
            param.requires_grad = "ranker_head" in name
        elif stage == "covariance":
            # Only covariance_estimator_head
            param.requires_grad = ("covariance_estimator_head" in name) or ("var_activation" in name)
        elif stage == "ranker_covariance":
            # Both ranker_head and covariance_estimator_head
            param.requires_grad = ("ranker_head" in name) or ("covariance_estimator_head" in name) or ("var_activation" in name)
        else:
            param.requires_grad = False


__all__ = ['set_stage_require_grad']
