"""
Homography and geometry operations.
Adapted from glue-factory geometry.homography module.
"""

import torch
import numpy as np
from typing import Tuple


def compute_homography_jacobian(
    homography: torch.Tensor,
    points: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Jacobian of homography transformation at given points.

    Args:
        homography: (B, 3, 3) homography matrix
        points: (B, N, 2) points in pixel coordinates (x, y)

    Returns:
        Jacobian matrices (B, N, 2, 2)
    """
    B, N, _ = points.shape
    device = points.device

    # Convert to homogeneous coordinates
    ones = torch.ones(B, N, 1, device=device)
    points_h = torch.cat([points, ones], dim=-1)  # (B, N, 3)

    # Apply homography
    transformed_h = torch.einsum('bij,bnj->bni', homography, points_h)  # (B, N, 3)
    u, v, w = transformed_h[..., 0], transformed_h[..., 1], transformed_h[..., 2]

    # Avoid division by zero
    w_safe = w.clamp(min=epsilon)

    # Derivatives of normalized coordinates [u/w, v/w]
    J_norm = torch.zeros(B, N, 2, 3, device=device)
    J_norm[..., 0, 0] = 1.0 / w_safe      # d(u/w)/du
    J_norm[..., 0, 2] = -u / (w_safe ** 2)  # d(u/w)/dw
    J_norm[..., 1, 1] = 1.0 / w_safe      # d(v/w)/dv
    J_norm[..., 1, 2] = -v / (w_safe ** 2)  # d(v/w)/dw

    # Full Jacobian: J = J_norm @ H[:, :, :2]
    H_first_two = homography[:, :, :2]  # (B, 3, 2)
    H_expanded = H_first_two[:, None, :, :].expand(B, N, 3, 2)
    J = torch.einsum('bnij,bnjk->bnik', J_norm, H_expanded)

    return J


def transform_points_with_homography(
    points: torch.Tensor,
    homography: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Transform points using homography with numerical stability.

    This operation is sensitive to float16 precision issues during AMP training,
    so we force float32 computation and KEEP float32 output to avoid overflow.

    Args:
        points: (B, N, 2) points in pixel coordinates
        homography: (B, 3, 3) homography matrix

    Returns:
        Transformed points (B, N, 2) - always in float32 for numerical stability
    """
    B, N, _ = points.shape
    device = points.device

    # Force float32 for numerical stability (critical for AMP training)
    # Keep output in float32 to prevent overflow when homography has large values
    points = points.float()
    homography = homography.float()

    ones = torch.ones(B, N, 1, device=device, dtype=torch.float32)
    points_h = torch.cat([points, ones], dim=-1)

    transformed_h = torch.einsum('bij,bnj->bni', homography, points_h)

    # Use additive epsilon (following glue-factory) instead of clamp
    # This is the key difference: add eps to denominator, not clamp w
    w = transformed_h[..., 2:3]
    transformed = transformed_h[..., :2] / (w + epsilon)

    # Keep output in float32 to maintain numerical stability
    # Subsequent operations should handle float32 gracefully
    return transformed


def warp_points(points: torch.Tensor, H: torch.Tensor, inverse: bool = True) -> torch.Tensor:
    """Warp points by homography (numpy or torch)."""
    if isinstance(points, torch.Tensor):
        if inverse:
            H = torch.inverse(H)
        return transform_points_with_homography(points, H)
    else:
        # NumPy version
        points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        if inverse:
            H = np.linalg.inv(H)
        warped = (H @ points_h.T).T
        return warped[:, :2] / warped[:, 2:3]


def sample_homography_corners(
    image_size: Tuple[int, int],
    difficulty: float = 0.8,
    translation: float = 1.0,
    max_angle: float = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a random homography by perturbing image corners.
    Adapted from glue-factory.

    Returns:
        H: (3, 3) homography matrix
        corners_src: (4, 2) source corners
        corners_dst: (4, 2) destination corners
        in_bounds: (4,) whether corners stay in bounds
    """
    w, h = image_size
    corners_src = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    # Sample perturbations
    max_shift = min(w, h) * difficulty * 0.2
    corners_dst = corners_src + np.random.uniform(
        -max_shift, max_shift, corners_src.shape
    ).astype(np.float32)

    # Add rotation
    angle = np.random.uniform(-max_angle, max_angle)
    center = np.array([w / 2, h / 2])
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    corners_dst_centered = corners_dst - center
    corners_dst = (R @ corners_dst_centered.T).T + center

    # Add translation
    tx = np.random.uniform(-translation, translation) * w * 0.1
    ty = np.random.uniform(-translation, translation) * h * 0.1
    corners_dst += np.array([tx, ty])

    # Check if corners stay in image bounds
    in_bounds = (
        (corners_dst[:, 0] >= 0) & (corners_dst[:, 0] < w) &
        (corners_dst[:, 1] >= 0) & (corners_dst[:, 1] < h)
    )

    H, _ = cv2.findHomography(corners_src, corners_dst)
    if H is None:
        H = np.eye(3, dtype=np.float32)

    return H.astype(np.float32), corners_src, corners_dst, in_bounds


def compute_reprojection_error_map(
    prob_map_shape: Tuple[int, int, int, int],
    homography: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute reprojection error for each pixel location in the probability map.

    Args:
        prob_map_shape: (B, 1, H, W)
        homography: (B, 3, 3) H from A to B
        device: Device

    Returns:
        errors: (B, H*W) reprojection errors
        valid_mask: (B, H*W) whether each pixel is valid (in bounds after transform)
    """
    B, _, H, W = prob_map_shape

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    pixel_coords = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1)

    coords_in_b = transform_points_with_homography(pixel_coords, homography)

    x_in_b = coords_in_b[..., 0]
    y_in_b = coords_in_b[..., 1]
    in_bounds = (x_in_b >= 0) & (x_in_b < W) & (y_in_b >= 0) & (y_in_b < H)

    dx = coords_in_b[..., 0] - pixel_coords[..., 0]
    dy = coords_in_b[..., 1] - pixel_coords[..., 1]
    displacement = torch.sqrt(dx**2 + dy**2)

    # Clamp max error for numerical stability
    max_error = torch.tensor(100.0, device=device)
    errors = torch.where(in_bounds, displacement, max_error)

    return errors, in_bounds


def symmetric_homography_error(kpts0, kpts1, H_gt):
    """Compute symmetric homography reprojection error.

    This is the standard metric used in SuperPoint, LoFTR, etc.
    Computes the average of forward and backward reprojection errors.

    Args:
        kpts0: (N, 2) keypoints in image 0
        kpts1: (N, 2) keypoints in image 1
        H_gt: (3, 3) ground truth homography from image 0 to 1

    Returns:
        errors: (N,) symmetric reprojection errors in pixels
    """
    if isinstance(kpts0, np.ndarray):
        # NumPy version
        # Forward: project kpts0 to image 1
        kpts0_h = np.concatenate([kpts0, np.ones((len(kpts0), 1))], axis=1)
        kpts0_in_1 = (H_gt @ kpts0_h.T).T
        kpts0_in_1 = kpts0_in_1[:, :2] / (kpts0_in_1[:, 2:3] + 1e-8)
        dist_fwd = np.linalg.norm(kpts0_in_1 - kpts1, axis=1)

        # Backward: project kpts1 to image 0
        H_inv = np.linalg.inv(H_gt)
        kpts1_h = np.concatenate([kpts1, np.ones((len(kpts1), 1))], axis=1)
        kpts1_in_0 = (H_inv @ kpts1_h.T).T
        kpts1_in_0 = kpts1_in_0[:, :2] / (kpts1_in_0[:, 2:3] + 1e-8)
        dist_bwd = np.linalg.norm(kpts1_in_0 - kpts0, axis=1)

        return (dist_fwd + dist_bwd) / 2.0
    else:
        # PyTorch version
        device = kpts0.device
        dtype = kpts0.dtype

        # Forward: project kpts0 to image 1
        ones = torch.ones(kpts0.shape[0], 1, device=device, dtype=dtype)
        kpts0_h = torch.cat([kpts0, ones], dim=-1)
        kpts0_in_1 = (H_gt @ kpts0_h.T).T
        kpts0_in_1 = kpts0_in_1[:, :2] / (kpts0_in_1[:, 2:3] + 1e-8)
        dist_fwd = torch.norm(kpts0_in_1 - kpts1, dim=1)

        # Backward: project kpts1 to image 0
        H_inv = torch.inverse(H_gt)
        kpts1_h = torch.cat([kpts1, ones], dim=-1)
        kpts1_in_0 = (H_inv @ kpts1_h.T).T
        kpts1_in_0 = kpts1_in_0[:, :2] / (kpts1_in_0[:, 2:3] + 1e-8)
        dist_bwd = torch.norm(kpts1_in_0 - kpts0, dim=1)

        return (dist_fwd + dist_bwd) / 2.0


# Import cv2 only when needed
try:
    import cv2
except ImportError:
    cv2 = None
