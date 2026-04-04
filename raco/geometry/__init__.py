"""Geometry module for homography and coordinate operations."""

from .homography import (
    compute_homography_jacobian,
    transform_points_with_homography,
    sample_homography_corners,
    warp_points,
    compute_reprojection_error_map,
)

__all__ = [
    "compute_homography_jacobian",
    "transform_points_with_homography",
    "sample_homography_corners",
    "warp_points",
    "compute_reprojection_error_map",
]
