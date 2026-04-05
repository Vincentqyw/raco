"""
Matching utilities for keypoint correspondence.
Extracted from train.py for reusability across training, evaluation, and inference.
"""

import torch
from typing import Tuple, Optional
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
    threshold: Optional[float] = None,
    kpts0_orig: Optional[torch.Tensor] = None,
    H0: Optional[int] = None,
    W0: Optional[int] = None,
    H1: Optional[int] = None,
    W1: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Mutual Nearest Neighbors (MNN) between two point sets.

    Args:
        kpts0_trans: Keypoints from view 0 projected to view 1 coordinates (B, N0, 2)
        kpts1: Keypoints from view 1 (B, N1, 2)
        threshold: Maximum Euclidean distance to be considered a valid match
        kpts0_orig: Original keypoints in view 0 before projection (B, N0, 2).
                    If provided, will check if projected points are in target view bounds.
        H0: Height of view 0 (used with kpts0_orig for bounds checking)
        W0: Width of view 0 (used with kpts0_orig for bounds checking)
        H1: Height of view 1 (target image height for bounds checking)
        W1: Width of view 1 (target image width for bounds checking)

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
    if threshold is not None:
        mutual_mask = (rev_idx == target) & (dist01 < threshold)
    else:
        mutual_mask = (rev_idx == target)

    # 5. Additional check: verify projected points are within image bounds
    if H1 is not None and W1 is not None:
        # Check if projected points (kpts0_trans in view 1) are within view 1 bounds
        valid_in_view1 = get_valid_mask(kpts0_trans, H1, W1)  # (B, N0)
    else:
        valid_in_view1 = torch.ones_like(mutual_mask)

    # Also check if original points (kpts0_orig in view 0) are within view 0 bounds
    if kpts0_orig is not None and H0 is not None and W0 is not None:
        valid_in_view0 = get_valid_mask(kpts0_orig, H0, W0)  # (B, N0)
        # Both original point in view 0 AND projected point in view 1 must be valid
        mutual_mask = mutual_mask & valid_in_view0 & valid_in_view1
    else:
        mutual_mask = mutual_mask & valid_in_view1

    return mutual_mask, idx01, dist01
