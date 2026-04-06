"""Reward function for keypoint detector (based on DaD paper)."""

import torch
from typing import Optional
from .base import RewardFunction


class ConstantReward(RewardFunction):
    """Policy gradient reward for keypoint detection.

    Uses thresholded reprojection distance to assign positive/negative
    rewards, then normalizes per-sample (DaD paper convention).
    """

    def __init__(
        self,
        d_max: float = 1.2,
        rho_pos: float = 1.0,
        rho_neg_max: float = 1e-2,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.d_max = d_max
        self.rho_pos = rho_pos
        self.rho_neg_max = rho_neg_max
        self.epsilon = epsilon
        self.step = 0

    def forward(
        self,
        distances: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward from distances.

        Returns:
            (B, N) tensor of rewards (before normalization).
        """
        rho_neg = -min(self.rho_neg_max, max(1e-6, self.step * 1e-7))
        rewards = torch.where(
            distances <= self.d_max,
            torch.full_like(distances, self.rho_pos),
            torch.full_like(distances, rho_neg),
        )
        if valid_mask is not None:
            rewards = rewards * valid_mask.float()
        return rewards

    def set_step(self, step: int):
        """Update training step for dynamic rho_neg."""
        self.step = step
