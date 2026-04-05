"""
Enhanced TensorBoard visualization for RaCo training.
Logs HPatches specific scenes with heatmaps, covariances, and ranker scores.
"""

import io
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from PIL import Image


# HPatches scenes to track during training
# Select representative scenes from both vantage and illumination
DEFAULT_TRACKED_SCENES = [
    "v_boat",  # vantage: viewpoint change
    "v_adam",  # vantage: viewpoint change
    "i_dom",  # illumination: lighting change
    "i_ss2",  # illumination: lighting change
]


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Denormalize ImageNet-normalized image to [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(-1, 1, 1)
    return (image * std + mean).clamp(0, 1)


def figure_to_tensor(fig: plt.Figure) -> torch.Tensor:
    """Convert matplotlib figure to torch tensor for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    buf.close()
    plt.close(fig)
    return img_tensor


def create_heatmap_figname(
    heatmap: np.ndarray,
    title: str = "Heatmap",
    cmap: str = "viridis",
    colorbar_label: str = "",
) -> torch.Tensor:
    """Create a heatmap visualization figure."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(heatmap, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.axis("off")

    # Use make_axes_locatable for proper colorbar alignment
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="3%")
    cbar = plt.colorbar(im, cax=cax)
    if colorbar_label:
        cbar.set_label(colorbar_label, rotation=270, labelpad=15)

    fig.tight_layout()
    return figure_to_tensor(fig)


def create_overlay_figure(
    image: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    title: str = "Overlay",
) -> torch.Tensor:
    """Overlay heatmap on RGB image."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert image to numpy [H, W, 3]
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image

    ax.imshow(img_np)
    im = ax.imshow(heatmap, cmap="jet", alpha=alpha, aspect="auto")
    ax.set_title(title)
    ax.axis("off")

    # Use make_axes_locatable for proper colorbar alignment
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="3%")
    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    return figure_to_tensor(fig)


def create_covariance_figures(
    cov_map: torch.Tensor,
    title_prefix: str = "Covariance",
    cmaps: List[str] = ["viridis", "RdBu_r", "plasma"],
) -> Dict[str, torch.Tensor]:
    """
    Create visualization for covariance as ellipse visualization.

    Color represents the major axis angle of covariance ellipse.
    Intensity is weighted by |Σ| (determinant), with higher uncertainty appearing whiter.

    Args:
        cov_map: Tensor of shape [H, W, 3] containing (L11, L21, L22) Cholesky factors
        title_prefix: Prefix for figure titles
        cmaps: List of colormaps (not used, kept for compatibility)

    Returns:
        Dictionary mapping channel names to figure tensors
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if isinstance(cov_map, torch.Tensor):
        cov_map = cov_map.cpu().numpy()

    # Ensure shape is [H, W, 3]
    if cov_map.shape[0] == 3:
        cov_map = np.transpose(cov_map, (1, 2, 0))

    H, W, C = cov_map.shape
    assert C == 3, f"Expected 3 channels, got {C}"

    # Extract Cholesky factors
    L11 = cov_map[..., 0]  # sqrt(var_x)
    L21 = cov_map[..., 1]  # cov_xy / sqrt(var_x)
    L22 = cov_map[..., 2]  # sqrt(var_y)

    # Reconstruct covariance matrices: Σ = L @ L.T
    # L = [[L11,  0 ],
    #      [L21, L22]]
    # Σ = [[L11^2,        L11*L21],
    #      [L11*L21, L21^2+L22^2]]

    Sigma_00 = L11 ** 2
    Sigma_01 = L11 * L21
    Sigma_11 = L21 ** 2 + L22 ** 2

    # Compute eigenvalues for each pixel
    # λ = (trace ± sqrt(trace^2 - 4*det)) / 2
    trace = Sigma_00 + Sigma_11
    det = Sigma_00 * Sigma_11 - Sigma_01 ** 2

    # Discriminant (ensure non-negative for numerical stability)
    disc = np.maximum(trace ** 2 - 4 * det, 0)
    sqrt_disc = np.sqrt(disc)

    # Eigenvalues: λ1 >= λ2
    lambda_1 = (trace + sqrt_disc) / 2  # Major eigenvalue
    lambda_2 = (trace - sqrt_disc) / 2  # Minor eigenvalue

    # Major axis angle: θ = 0.5 * atan2(2*σ_01, σ_00 - σ_11)
    # Range: [-π/2, π/2]
    angle = 0.5 * np.arctan2(2 * Sigma_01, Sigma_00 - Sigma_11)

    # Normalize angle to [0, 1] for colormap
    # Angle range [-π/2, π/2] -> [0, 1]
    angle_normalized = (angle + np.pi / 2) / np.pi

    # Weight by determinant |Σ|: higher uncertainty = whiter
    # Normalize determinant to [0, 1]
    det_normalized = det / (det.max() + 1e-8)

    # Create HSV image: Hue from angle, Value weighted by det
    # Higher det (more uncertainty) -> lower value (whiter in final image)
    # Use HSV colormap for angles
    from matplotlib.colors import hsv_to_rgb

    # Hue: angle (0-1)
    hue = angle_normalized

    # Saturation: always full
    saturation = np.ones_like(hue)

    # Value: inverse of uncertainty (high det -> low value -> whiter)
    value = 1.0 - det_normalized * 0.7  # Scale to keep some visibility

    # Stack to HSV
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Create figure with consistent size
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(rgb, aspect="auto")
    ax.set_title(f"{title_prefix}\nColor: Major axis angle | Brightness: |Σ| (whiter=higher uncertainty)",
                 fontsize=10)
    ax.axis("off")

    # Use make_axes_locatable for consistent colorbar width (5%)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="3%")

    # Create angle reference gradient
    angle_gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    angle_gradient = np.repeat(angle_gradient, 20, axis=1)

    hue_ref = angle_gradient
    sat_ref = np.ones_like(angle_gradient)
    val_ref = np.ones_like(angle_gradient)
    hsv_ref = np.stack([hue_ref, sat_ref, val_ref], axis=-1)
    rgb_ref = hsv_to_rgb(hsv_ref)

    cax.imshow(rgb_ref, aspect="auto", origin="lower", extent=[0, 1, -90, 90])
    cax.set_ylabel("Major Axis Angle (°)", fontsize=9)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.tick_params(labelsize=8)

    fig.tight_layout()

    tensor = figure_to_tensor(fig)
    results = {"combined": tensor}

    return results


def create_keypoints_overlay_figure(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    title: str = "Keypoints",
    max_kpts: int = 500,
) -> torch.Tensor:
    """Create keypoints overlay on image."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert image
    if isinstance(image, torch.Tensor):
        img_np = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image

    ax.imshow(img_np)

    # Convert keypoints
    if isinstance(keypoints, torch.Tensor):
        kpts_np = keypoints.cpu().numpy()
    else:
        kpts_np = keypoints

    # Subsample if too many
    if len(kpts_np) > max_kpts:
        indices = np.linspace(0, len(kpts_np) - 1, max_kpts, dtype=int)
        kpts_np = kpts_np[indices]
        if scores is not None:
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            scores = scores[indices]

    # Plot keypoints
    if scores is not None:
        scatter = ax.scatter(
            kpts_np[:, 0],
            kpts_np[:, 1],
            c=scores,
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        # Use make_axes_locatable for proper colorbar alignment
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="3%")
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label("Score", rotation=270, labelpad=15)
    else:
        ax.scatter(kpts_np[:, 0], kpts_np[:, 1], c="lime", s=10, alpha=0.7)

    ax.set_title(f"{title} ({len(keypoints)} keypoints)")
    ax.axis("off")
    fig.tight_layout()
    return figure_to_tensor(fig)


class EnhancedTensorBoardLogger:
    """
    Enhanced TensorBoard logger for RaCo training visualization.

    Tracks specific HPatches scenes and logs:
    - RGB images
    - Detection probability heatmaps
    - Covariance Cholesky factor maps
    - Ranker score heatmaps
    - Keypoint overlays
    """

    def __init__(
        self,
        writer,
        tracked_scenes: Optional[List[str]] = None,
        num_scenes_per_eval: int = 4,
    ):
        """
        Args:
            writer: SummaryWriter instance
            tracked_scenes: List of scene names to track (e.g., ["v_boat", "i_dom"])
            num_scenes_per_eval: Number of scenes to visualize per eval
        """
        self.writer = writer
        self.tracked_scenes = tracked_scenes # or DEFAULT_TRACKED_SCENES
        self.num_scenes_per_eval = num_scenes_per_eval

        # Store predictions for tracked scenes to enable step-by-step comparison
        self.scene_history: Dict[str, Dict] = {}

    def should_log_scene(self, seq_name: str) -> bool:
        """Check if a scene should be logged."""
        if self.tracked_scenes is not None:
            return seq_name in self.tracked_scenes
        else:
            return True

    def log_scene_prediction(
        self,
        seq_name: str,
        data: Dict,
        pred: Dict,
        global_step: int,
        img_idx: int = 0,
    ):
        """
        Log all visualizations for a single scene prediction.

        Args:
            seq_name: Scene name (e.g., "v_boat")
            data: Data dictionary containing "image0", "image1", etc.
            pred: Prediction dictionary containing model outputs
            global_step: Training step
            img_idx: Image pair index within scene
        """
        if not self.should_log_scene(seq_name):
            return

        tag_prefix = f"scenes/{seq_name}/{img_idx.item()}"

        # Get reference image for overlays
        img0 = data["image0"]["image"][0]  # [3, H, W]
        img_denorm = denormalize_image(img0)

        # 1. Log prob_map heatmap (detection probability)
        if "prob_map" in pred.get("image0", {}):
            prob_map = pred["image0"]["prob_map"][0, 0].cpu().numpy()  # [H, W]

            # Heatmap only
            prob_fig = create_heatmap_figname(
                prob_map,
                title=f"Detection Probability - {seq_name}",
                cmap="viridis",
                colorbar_label="Probability",
            )
            self.writer.add_image(f"{tag_prefix}/prob_map", prob_fig, global_step)

            # Overlay on image
            overlay_fig = create_overlay_figure(
                img_denorm,
                prob_map,
                alpha=0.5,
                title=f"Probability Overlay - {seq_name}",
            )
            self.writer.add_image(f"{tag_prefix}/prob_overlay", overlay_fig, global_step)

        # 2. Log ranker score map if available
        if "ranker_scores" in pred.get("image0", {}):
            # Ranker scores are per-keypoint, need to scatter to image
            ranker_scores = pred["image0"]["ranker_scores"][0].cpu().numpy()  # [N]
            keypoints = pred["image0"]["keypoints"][0].cpu().numpy()  # [N, 2]

            if len(keypoints) > 0:
                # Create dense ranker map
                H, W = img0.shape[-2:]
                ranker_map = create_dense_map_from_points(
                    keypoints, ranker_scores, (H, W), sigma=3.0
                )

                ranker_fig = create_heatmap_figname(
                    ranker_map,
                    title=f"Ranker Scores - {seq_name}",
                    cmap="plasma",
                    colorbar_label="Rank",
                )
                self.writer.add_image(f"{tag_prefix}/ranker_map", ranker_fig, global_step)

        # 3. Log covariance maps if available
        if "covariances" in pred.get("image0", {}):
            covariances = pred["image0"]["covariances"][0].cpu().numpy()  # [N, 2, 2]
            keypoints = pred["image0"]["keypoints"][0].cpu().numpy()  # [N, 2]

            if len(keypoints) > 0:
                H, W = img0.shape[-2:]

                # Extract Cholesky factors from covariances and create maps
                # cov = L @ L.T where L = [[L11, 0], [L21, L22]]
                L11_vals = np.sqrt(covariances[:, 0, 0])  # sqrt of variance
                L22_vals = np.sqrt(covariances[:, 1, 1])
                L21_vals = covariances[:, 1, 0] / (L11_vals + 1e-8)  # covariance / sqrt(var_x)

                # Create dense maps
                L11_map = create_dense_map_from_points(keypoints, L11_vals, (H, W), sigma=3.0)
                L21_map = create_dense_map_from_points(keypoints, L21_vals, (H, W), sigma=3.0)
                L22_map = create_dense_map_from_points(keypoints, L22_vals, (H, W), sigma=3.0)

                cov_maps = np.stack([L11_map, L21_map, L22_map], axis=-1)
                cov_figs = create_covariance_figures(
                    cov_maps,
                    title_prefix=f"{seq_name}",
                )

                self.writer.add_image(
                    f"{tag_prefix}/covariance_maps", cov_figs["combined"], global_step
                )

        # 4. Log keypoints overlay
        keypoints = pred["keypoints_0"][0]  # [N, 2]
        if "keypoint_scores_0" in pred:
            scores = pred["keypoint_scores_0"][0]  # [N]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu()
        else:
            scores = None

        kpts_fig = create_keypoints_overlay_figure(
            img0,
            keypoints,
            scores=scores,
            title=f"Keypoints - {seq_name}",
        )
        self.writer.add_image(f"{tag_prefix}/keypoints", kpts_fig, global_step)

        # Store in history for potential later comparison
        self.scene_history[f"{seq_name}_{img_idx}_{global_step}"] = {
            "pred": pred,
            "data": data,
        }

    def log_metrics_comparison(self, global_step: int):
        """Log comparison across different steps for tracked scenes."""
        # This can be used to create side-by-side comparisons
        # For now, TensorBoard's built-in slider handles step comparison
        pass


def create_dense_map_from_points(
    points: np.ndarray,
    values: np.ndarray,
    output_size: Tuple[int, int],
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Create a dense heatmap from sparse point values using Gaussian splatting.

    Args:
        points: [N, 2] array of (x, y) coordinates
        values: [N] array of values at each point
        output_size: (H, W) output map size
        sigma: Gaussian kernel sigma

    Returns:
        [H, W] dense map
    """
    H, W = output_size
    output = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    for (x, y), val in zip(points, values):
        if not (0 <= x < W and 0 <= y < H):
            continue

        # Gaussian kernel around point
        dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma**2))

        output += val * gaussian
        weight += gaussian

    # Normalize by weight
    output = np.divide(output, weight, out=np.zeros_like(output), where=weight > 1e-8)

    return output


# Convenience function for integration with train.py
def create_scene_logger(writer, tracked_scenes: Optional[List[str]] = None):
    """Create an EnhancedTensorBoardLogger with specified tracked scenes."""
    return EnhancedTensorBoardLogger(writer, tracked_scenes)
