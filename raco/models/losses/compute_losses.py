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
    mask_0_to_1: torch.Tensor,
    mask_1_to_0: torch.Tensor,
    nearest_idx_0_to_1: torch.Tensor,
    nearest_idx_1_to_0: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute covariance loss - OPTIMIZED with early masking.

    Key improvement: Extract matched keypoints BEFORE expensive operations.
    This avoids computing eigendecomposition, Cholesky, etc. on unmatched keypoints.

    Args:
        pred: Model predictions dict
        cov_loss_fn: CovarianceLoss instance
        kpts0: Keypoints in view 0 (B, N, 2)
        kpts1: Keypoints in view 1 (B, N, 2)
        kpts0_in_1: Keypoints 0 projected to view 1 (B, N, 2)
        kpts1_in_0: Keypoints 1 projected to view 0 (B, N, 2)
        H_0to1: Homography from 0 to 1 (3, 3)
        mask_0_to_1: Mutual match mask (B, N)
        mask_1_to_0: Mutual match mask (B, N)
        nearest_idx_0_to_1: Nearest neighbor indices (B, N)
        nearest_idx_1_to_0: Nearest neighbor indices (B, N)

    Returns:
        Tuple of (loss, metrics_dict)
    """
    covariances_0 = pred["covariances_0"]  # (B, N, 2, 2)
    covariances_1 = pred["covariances_1"]
    B, N = covariances_0.shape[:2]
    device = covariances_0.device

    # ========== Compute Jacobians for ALL keypoints first ==========
    # (This is still O(N), but much cheaper than eigendecomposition/Cholesky)

    # Handle different H_0to1 shapes:
    # - (3, 3): single homography
    # - (B, 3, 3): batched homographies
    # - (1, B, 3, 3): dataloader format with outer dimension
    if H_0to1.dim() == 4:
        # Dataloader format: (1, B, 3, 3) -> (B, 3, 3)
        H_0to1_batched = H_0to1.squeeze(0)  # (B, 3, 3)
    elif H_0to1.dim() == 2:
        # Single homography: (3, 3) -> (B, 3, 3)
        H_0to1_batched = H_0to1.unsqueeze(0).expand(B, -1, -1)
    else:
        # Already batched: (B, 3, 3)
        H_0to1_batched = H_0to1

    H_1to0_batched = torch.inverse(H_0to1_batched)  # (B, 3, 3)

    jacobian_0_to_1_all = compute_homography_jacobian(H_0to1_batched, kpts0)  # (B, N, 2, 2)
    jacobian_1_to_0_all = compute_homography_jacobian(H_1to0_batched, kpts1)  # (B, N, 2, 2)

    # ========== Direction: View 0 → View 1 ==========
    loss_0_to_1, loss_0_to_1_metrics = _extract_matched_keypoints_for_covariance(
        mask=mask_0_to_1,
        covariances_src=covariances_0,
        kpts_src_in_tgt=kpts0_in_1,
        covariances_tgt=covariances_1,
        kpts_tgt=kpts1,
        nearest_idx=nearest_idx_0_to_1,
        jacobian_all=jacobian_0_to_1_all,
        cov_loss_fn=cov_loss_fn,
        N=N,
    )

    # ========== Direction: View 1 → View 0 ==========
    loss_1_to_0, loss_1_to_0_metrics = _extract_matched_keypoints_for_covariance(
        mask=mask_1_to_0,
        covariances_src=covariances_1,
        kpts_src_in_tgt=kpts1_in_0,
        covariances_tgt=covariances_0,
        kpts_tgt=kpts0,
        nearest_idx=nearest_idx_1_to_0,
        jacobian_all=jacobian_1_to_0_all,
        cov_loss_fn=cov_loss_fn,
        N=N,
    )

    # ========== Combine bidirectional losses ==========
    total_loss = (loss_0_to_1 + loss_1_to_0) / 2.0

    metrics = {
        'total_loss': total_loss.item(),
        'cov_nll_0_to_1': loss_0_to_1.item(),
        'cov_nll_1_to_0': loss_1_to_0.item(),
        "covariances_0": loss_0_to_1_metrics["covariances_src"],
        "covariances_1": loss_1_to_0_metrics["covariances_src"],
        "errors_0_to_1": loss_0_to_1_metrics["matched_errors"],
        "errors_1_to_0": loss_1_to_0_metrics["matched_errors"],
        "jacobian_0_to_1": loss_0_to_1_metrics["jacobian"],
        "jacobian_1_to_0": loss_1_to_0_metrics["jacobian"],
    }

    return total_loss, metrics


def _extract_matched_keypoints_for_covariance(
    mask: torch.Tensor,
    covariances_src: torch.Tensor,
    kpts_src_in_tgt: torch.Tensor,
    covariances_tgt: torch.Tensor,
    kpts_tgt: torch.Tensor,
    nearest_idx: torch.Tensor,
    jacobian_all: torch.Tensor,
    cov_loss_fn,
    N: int,
) -> Tuple[torch.Tensor, int]:
    """
    Extract matched keypoints and compute covariance loss for one direction.

    This helper function avoids code duplication between the two bidirectional directions.

    Args:
        mask: (B, N) boolean mask indicating matched keypoints in source view
        covariances_src: (B, N, 2, 2) covariances for source view
        kpts_src_in_tgt: (B, N, 2) source keypoints projected to target view
        covariances_tgt: (B, N, 2, 2) covariances for target view
        kpts_tgt: (B, N, 2) keypoints in target view
        nearest_idx: (B, N) nearest neighbor indices from source to target
        jacobian_all: (B, N, 2, 2) Jacobians for all source keypoints
        cov_loss_fn: CovarianceLoss instance
        N: Number of keypoints

    Returns:
        loss: Scalar covariance loss for this direction
        num_matches: Number of matched keypoints
    """
    device = covariances_src.device

    # Extract ALL matched keypoints across batches using mask
    matched_idx = mask.flatten().nonzero(as_tuple=True)[0]  # (TotalM,)

    if len(matched_idx) > 0:
        # Convert flat indices to (batch, keypoint) indices
        batch_idx = matched_idx // N  # (TotalM,)

        # Extract matched keypoints' data - NOW WITH FEWER POINTS!
        matched_cov_src = covariances_src.reshape(-1, 2, 2)[matched_idx]  # (TotalM, 2, 2)

        # Get nearest neighbors for matched keypoints
        nearest_idx_matched = nearest_idx.reshape(-1)[matched_idx]  # (TotalM,)

        # Project keypoint indices to flat indexing for target view
        matched_nearest_kpts_tgt_flat_idx = batch_idx * N + nearest_idx_matched
        matched_nearest_kpts_tgt = kpts_tgt.reshape(-1, 2)[matched_nearest_kpts_tgt_flat_idx]  # (TotalM, 2)

        # Compute errors for matched keypoints only
        matched_kpts_src_in_tgt = kpts_src_in_tgt.reshape(-1, 2)[matched_idx]  # (TotalM, 2)
        matched_errors = matched_kpts_src_in_tgt - matched_nearest_kpts_tgt  # (TotalM, 2)

        # Get matched covariances from target view
        matched_cov_tgt = covariances_tgt.reshape(-1, 2, 2)[matched_nearest_kpts_tgt_flat_idx]  # (TotalM, 2, 2)

        # Extract Jacobians ONLY for matched keypoints
        matched_jacobian = jacobian_all.reshape(-1, 2, 2)[matched_idx]  # (TotalM, 2, 2)

        # Compute loss for this direction
        loss, loss_metrics = cov_loss_fn.forward_on_flattened(
            matched_cov_src, matched_cov_tgt,
            matched_errors, matched_jacobian
        )
        num_matches = len(matched_idx)
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_matches = 0

    return loss, {
        "num_matches": num_matches,
        **loss_metrics,
        "jacobian": matched_jacobian,
        "covariances_src": matched_cov_src,
        "covariances_tgt": matched_cov_tgt,
        "matched_errors": matched_errors,
    }


__all__ = [
    'compute_detector_loss',
    'compute_ranker_loss',
    'compute_covariance_loss',
]
