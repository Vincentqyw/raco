"""
Matching utilities for keypoint correspondence.
Extracted from train.py for reusability across training, evaluation, and inference.
"""

import torch
from typing import Tuple
from .homography import transform_points_with_homography


def get_valid_mask(points: torch.Tensor, H_val: int, W_val: int) -> torch.Tensor:
    """
    Check which points are within image bounds.

    Args:
        points: (..., 2) tensor of (x, y) coordinates
        H_val: Image height
        W_val: Image width

    Returns:
        Boolean mask indicating valid points
    """
    return ((points[..., 0] >= 0) & (points[..., 0] < W_val) &
            (points[..., 1] >= 0) & (points[..., 1] < H_val))


def find_matches(
    kpts_a: torch.Tensor,
    kpts_b: torch.Tensor,
    H: torch.Tensor,
    threshold: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find matches between keypoints using homography.

    Args:
        kpts_a: Keypoints in view A (N0, 2)
        kpts_b: Keypoints in view B (N1, 2)
        H: Homography matrix from A to B (3, 3)
        threshold: Maximum distance for a valid match

    Returns:
        matches_a: Indices of matched keypoints in A (M,)
        matches_b: Indices of matched keypoints in B (M,)
    """
    kpts_a_in_b = transform_points_with_homography(
        kpts_a.unsqueeze(0), H.unsqueeze(0)
    ).squeeze(0)

    distances = torch.cdist(kpts_a_in_b, kpts_b)
    min_dist_a_to_b, closest_b_to_a = distances.min(dim=1)
    _, closest_a_to_b = distances.min(dim=0)

    matches_a, matches_b = [], []
    for i in range(len(kpts_a)):
        j = closest_b_to_a[i]
        if closest_a_to_b[j] == i and min_dist_a_to_b[i] < threshold:
            matches_a.append(i)
            matches_b.append(j)

    if len(matches_a) == 0:
        return torch.tensor([], dtype=torch.long, device=kpts_a.device), \
               torch.tensor([], dtype=torch.long, device=kpts_b.device)

    return torch.tensor(matches_a, device=kpts_a.device), \
           torch.tensor(matches_b, device=kpts_b.device)


def compute_mutual_dist(
    kpts0_trans: torch.Tensor,
    kpts1: torch.Tensor,
    threshold: float = 3.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Mutual Nearest Neighbors (MNN) between two point sets.

    Args:
        kpts0_trans: Keypoints from view 0 projected to view 1 coordinates (B, N0, 2)
        kpts1: Keypoints from view 1 (B, N1, 2)
        threshold: Maximum Euclidean distance to be considered a valid match

    Returns:
        mutual_mask: Boolean mask (B, N0), indicating which kpts0 have valid MNN
        nearest_idx: Corresponding kpts1 indices (B, N0)
        dist01: Nearest neighbor distances (B, N0)
    """
    B, N0, _ = kpts0_trans.shape
    device = kpts0_trans.device

    # 1. Compute pairwise distance matrix (B, N0, N1)
    dist_mat = torch.cdist(kpts0_trans, kpts1)

    # 2. Compute bidirectional nearest neighbors
    # kpts0's nearest neighbor in kpts1
    dist01, idx01 = dist_mat.min(dim=2)  # (B, N0)
    # kpts1's nearest neighbor in kpts0
    dist10, idx10 = dist_mat.min(dim=1)  # (B, N1)

    # 3. Mutual nearest neighbor check
    # For each kpts0[i], check if kpts1[nn(i)]'s nearest neighbor is i
    target = torch.arange(N0, device=device).unsqueeze(0).expand(B, -1)
    rev_idx = idx10.gather(1, idx01)  # (B, N0)

    # 4. Generate mask: (index match) AND (distance < threshold)
    mutual_mask = (rev_idx == target) & (dist01 < threshold)

    return mutual_mask, idx01, dist01
