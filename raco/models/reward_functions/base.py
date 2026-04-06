"""Base class and interfaces for reward functions."""

import torch
import torch.nn as nn
from typing import Optional


class RewardFunction(nn.Module):
    """Base class for reward functions used in policy gradient losses.

    Subclasses should implement `forward` to compute rewards from
    task-specific signals (e.g., reprojection distances, matching scores).

    Expected interface:
        reward_fn(distances, valid_mask) -> torch.Tensor of shape (B, N)
    """

    def forward(
        self,
        distances: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-keypoint rewards.

        Args:
            distances: (B, N) task-specific distances or errors.
            valid_mask: (B, N) optional boolean mask.

        Returns:
            rewards: (B, N) reward values.
        """
        raise NotImplementedError
