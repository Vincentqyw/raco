"""
Loss computation utilities for training stages.
Extracted from train.py for better modularity.
"""

import torch
from loguru import logger
from typing import Tuple, Dict, Optional

from raco.geometry.homography import compute_homography_jacobian


def compute_detector_loss(
    pred: Dict[str, torch.Tensor],
    det_loss_fn,
    distances_0_to_1: torch.Tensor,
    distances_1_to_0: torch.Tensor,
    valid_0_to_1: torch.Tensor,
    valid_1_to_0: torch.Tensor,
    iteration: int
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute detector loss for sparse keypoints.

    Args:
        pred: Model predictions dict
        det_loss_fn: DetectorLoss instance
        distances_0_to_1: Reprojection distances (B, N)
        distances_1_to_0: Reprojection distances (B, N)
        valid_0_to_1: Valid mask (B, N)
        valid_1_to_0: Valid mask (B, N)
        iteration: Current training iteration

    Returns:
        Tuple of (loss, metrics_dict)
    """
    prob_0_sparse = pred["image0"]["keypoint_scores"]
    prob_1_sparse = pred["image1"]["keypoint_scores"]

    # Compute loss on sparse samples
    loss_0, loss_0_details = det_loss_fn.forward_sparse(prob_0_sparse, distances_0_to_1, valid_0_to_1)
    loss_1, loss_1_details = det_loss_fn.forward_sparse(prob_1_sparse, distances_1_to_0, valid_1_to_0)
    loss = (loss_0 + loss_1) / 2.0

    # Debug: check if loss is 0 and why
    if loss.item() == 0 and iteration % 10 == 0:
        distances0 = loss_0_details['distances']
        distances1 = loss_1_details['distances']
        logger.info(
            f"Zero loss at iter {iteration}: valid_ratio={valid_0_to_1.float().mean():.3f}, "
            f"prob_mean={prob_0_sparse.mean():.6f}, "
            f"distances0={distances0.mean():.6f}, "
            f"distances1={distances1.mean():.6f}"
        )

    # Update step for dynamic negative reward
    det_loss_fn.set_step(iteration)

    metrics = {
        'loss_0': loss_0.item(),
        'loss_1': loss_1.item(),
        'rewards_0': loss_0_details['rewards'].mean().item(),
        'rewards_1': loss_1_details['rewards'].mean().item(),
    }

    return loss, metrics


def compute_ranker_loss(
    pred: Dict[str, torch.Tensor],
    rank_loss_fn,
    mutual_mask_0_to_1: torch.Tensor,
    mutual_mask_1_to_0: torch.Tensor,
    nearest_idx_0_to_1: torch.Tensor,
    nearest_idx_1_to_0: torch.Tensor
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute ranking loss using soft ranking.

    Args:
        pred: Model predictions dict
        rank_loss_fn: RankingLoss instance
        mutual_mask_0_to_1: Mutual match mask (B, N)
        mutual_mask_1_to_0: Mutual match mask (B, N)
        nearest_idx_0_to_1: Nearest neighbor indices (B, N)
        nearest_idx_1_to_0: Nearest neighbor indices (B, N)

    Returns:
        Tuple of (loss, metrics_dict)
    """
    ranker_scores_0 = pred["ranker_scores_0"]  # (B, N)
    ranker_scores_1 = pred["ranker_scores_1"]

    # Bidirectional ranking loss
    loss_0_to_1, details_0_to_1 = rank_loss_fn(
        ranker_scores_0, ranker_scores_1,
        mutual_mask_a=mutual_mask_0_to_1,
        nearest_idx_a=nearest_idx_0_to_1
    )
    loss_1_to_0, details_1_to_0 = rank_loss_fn(
        ranker_scores_1, ranker_scores_0,
        mutual_mask_a=mutual_mask_1_to_0,
        nearest_idx_a=nearest_idx_1_to_0
    )
    loss = (loss_0_to_1 + loss_1_to_0) / 2.0

    metrics = {
        'loss_0_to_1': loss_0_to_1.item(),
        'loss_1_to_0': loss_1_to_0.item(),
        'spearman_loss': (details_0_to_1.get('spearman_loss', 0) + details_1_to_0.get('spearman_loss', 0)) / 2,
        'pull_loss': (details_0_to_1.get('pull_loss', 0) + details_1_to_0.get('pull_loss', 0)) / 2,
    }

    return loss, metrics


def compute_covariance_loss(
    pred: Dict[str, torch.Tensor],
    cov_loss_fn,
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    kpts0_in_1: torch.Tensor,
    kpts1_in_0: torch.Tensor,
    H_0to1: torch.Tensor,
    mutual_mask_0_to_1: torch.Tensor,
    mutual_mask_1_to_0: torch.Tensor,
    nearest_idx_0_to_1: torch.Tensor,
    nearest_idx_1_to_0: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute covariance loss using reprojection error.

    Args:
        pred: Model predictions dict
        cov_loss_fn: CovarianceLoss instance
        kpts0: Keypoints in view 0 (B, N, 2)
        kpts1: Keypoints in view 1 (B, N, 2)
        kpts0_in_1: Keypoints 0 projected to view 1 (B, N, 2)
        kpts1_in_0: Keypoints 1 projected to view 0 (B, N, 2)
        H_0to1: Homography from 0 to 1 (3, 3)
        mutual_mask_0_to_1: Mutual match mask (B, N)
        mutual_mask_1_to_0: Mutual match mask (B, N)
        nearest_idx_0_to_1: Nearest neighbor indices (B, N)
        nearest_idx_1_to_0: Nearest neighbor indices (B, N)

    Returns:
        Tuple of (loss, metrics_dict)
    """
    covariances_0 = pred["covariances_0"]  # (B, N, 2, 2)
    covariances_1 = pred["covariances_1"]

    B = covariances_0.shape[0]

    # Compute Jacobians
    jacobian_0_to_1 = compute_homography_jacobian(H_0to1, kpts0)
    jacobian_1_to_0 = compute_homography_jacobian(torch.inverse(H_0to1), kpts1)

    # Compute reprojection error vectors
    batch_idx = torch.arange(B, device=kpts0.device).unsqueeze(1).expand(-1, kpts0.shape[1])
    nearest_kpts1 = kpts1[batch_idx, nearest_idx_0_to_1]  # (B, N, 2)
    nearest_kpts0 = kpts0[batch_idx, nearest_idx_1_to_0]  # (B, N, 2)

    # Error vectors (B, N, 2)
    errors_0_to_1 = kpts0_in_1 - nearest_kpts1
    errors_1_to_0 = kpts1_in_0 - nearest_kpts0

    # Compute bidirectional loss with separate masks
    loss, details = cov_loss_fn.forward_bidirectional(
        covariances_0, covariances_1,
        errors_0_to_1, errors_1_to_0,
        jacobian_0_to_1, jacobian_1_to_0,
        valid_mask_a_to_b=mutual_mask_0_to_1,
        valid_mask_b_to_a=mutual_mask_1_to_0,
    )

    metrics = {
        'cov_nll_0_to_1': details.get('cov_nll_a_to_b', 0),
        'cov_nll_1_to_0': details.get('cov_nll_b_to_a', 0),
        'total_loss': details.get('cov_total', 0),
    }

    return loss, metrics


__all__ = [
    'compute_detector_loss',
    'compute_ranker_loss',
    'compute_covariance_loss',
]
