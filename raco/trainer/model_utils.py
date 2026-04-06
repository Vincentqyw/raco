"""
Model utilities for trainer.
Handles parameter freezing, gradient requirements, etc.
"""

import torch
from torch import nn
from loguru import logger

def set_stage_require_grad(model: torch.nn.Module, stage: str) -> None:
    """
    Set requires_grad for parameters and selectively freeze BN layers based on training stage.

    Args:
        model: RaCo model
        stage: Training stage (detector/ranker/covariance/ranker_covariance)

    Note:
        Detector params: encoder (block1-4, conv1-4, pool2, pool4, gate) + score_head
        Ranker params: ranker_head
        Covariance params: covariance_estimator_head
        Ranker+Covariance params: ranker_head + covariance_estimator_head

    BatchNorm Freezing Strategy:
        - detector stage: All BN can train (everything is learning)
        - non-detector stages:
            * encoder/score_head BN: freeze (already trained, keep keypoint stable)
            * ranker_head/covariance_head BN: train (learning new features from scratch)
    """

    # 1. Define sets of names for easier matching
    detector_names = [
        "block1", "block2", "block3", "block4",
        "conv1", "conv2", "conv3", "conv4",
        "pool2", "pool4", "gate", "normalizer",
        "score_head"
    ]
    covariance_names = ["covariance_estimator_head", "var_activation"]
    ranker_names = ["ranker_head"]
    
    # 2. Determine which components should be TRAINABLE based on the stage
    if stage == "detector":
        trainable_keys = detector_names
    elif stage == "ranker":
        trainable_keys = ranker_names
    elif stage == "covariance":
        trainable_keys = covariance_names
    elif stage == "ranker_covariance":
        trainable_keys = ranker_names + covariance_names
    else:
        raise ValueError(f"Invalid stage: {stage}")

    # 3. Iterate through modules to handle both Params and BN Buffers
    for name, module in model.named_modules():
        # Check if this module belongs to a trainable component
        is_trainable = any(key in name for key in trainable_keys)
        
        # Handle Parameters (Weights/Biases)
        for param in module.parameters(recurse=False):
            param.requires_grad = is_trainable
            
        # Handle BatchNorm Strategy
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if is_trainable:
                # If it's part of the active head, let it train/update normally
                module.train()
                # Reset the train method in case it was previously "stubbed"
                if hasattr(module, '_original_train'):
                    module.train = module._original_train
            else:
                # If it's part of the frozen encoder/score_head
                module.eval()
                # "Stub" the train method so model.train() won't reactivate it
                if not hasattr(module, '_original_train'):
                    module._original_train = module.train
                module.train = lambda mode=True: None


__all__ = ['set_stage_require_grad']
