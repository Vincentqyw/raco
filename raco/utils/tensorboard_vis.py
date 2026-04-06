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
    separate_channels: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Create comprehensive covariance visualizations.

    Provides multiple visualization options:
    - Combined HSV: Angle (hue) + Uncertainty (whiter = more uncertain)
    - Separate Cholesky factors (L11, L21, L22)
    - Determinant map (uncertainty landscape)
    - Anisotropy map (ellipse elongation)

    Paper description: "colored by the angle of the covariance's first eigenvector"
                      "illustrated by opacity" (larger covariance = more transparent/whiter)

    Args:
        cov_map: Tensor of shape [H, W, 3] or [3, H, W] containing (L11, L21, L22) Cholesky factors
        title_prefix: Prefix for figure titles
        cmaps: Colormaps for separate channels [L11, L21, L22]
        separate_channels: If True, create individual channel visualizations

    Returns:
        Dictionary mapping visualization names to figure tensors
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import hsv_to_rgb

    if isinstance(cov_map, torch.Tensor):
        cov_map = cov_map.cpu().numpy()

    # Ensure shape is [H, W, 3]
    if cov_map.shape[0] == 3:
        cov_map = np.transpose(cov_map, (1, 2, 0))

    H, W, C = cov_map.shape
    assert C == 3, f"Expected 3 channels, got {C}"

    # === Extract Cholesky factors ===
    L11 = cov_map[..., 0]  # sqrt(var_x)
    L21 = cov_map[..., 1]  # cov_xy / sqrt(var_x)
    L22 = cov_map[..., 2]  # sqrt(var_y)

    # === Reconstruct covariance matrices: Σ = L @ L.T ===
    # L = [[L11,  0 ],     Σ = [[L11^2,        L11*L21],
    #      [L21, L22]]          [L11*L21, L21^2+L22^2]]

    Sigma_00 = L11 ** 2
    Sigma_01 = L11 * L21
    Sigma_11 = L21 ** 2 + L22 ** 2

    # === Compute eigenvalues and derived quantities ===
    trace = Sigma_00 + Sigma_11
    det = Sigma_00 * Sigma_11 - Sigma_01 ** 2

    # Discriminant (ensure non-negative for numerical stability)
    disc = np.maximum(trace ** 2 - 4 * det, 0)
    sqrt_disc = np.sqrt(disc)

    # Eigenvalues: λ1 >= λ2
    lambda_1 = (trace + sqrt_disc) / 2  # Major eigenvalue
    lambda_2 = (trace - sqrt_disc) / 2  # Minor eigenvalue

    # Anisotropy ratio: measures ellipse elongation (λ1/λ2)
    # Higher = more elongated (anisotropic)
    anisotropy = lambda_1 / (lambda_2 + 1e-8)

    # Major axis angle: θ = 0.5 * atan2(2*σ_01, σ_00 - σ_11)
    angle = 0.5 * np.arctan2(2 * Sigma_01, Sigma_00 - Sigma_11)
    angle_normalized = (angle + np.pi / 2) / np.pi  # [-π/2, π/2] -> [0, 1]

    # === Create visualizations ===
    results = {}

    # Visualization 1: Combined HSV (angle=hue, det=value)
    # Higher uncertainty -> whiter (lower value)
    det_normalized = det / (det.max() + 1e-8)

    hue = angle_normalized
    saturation = np.ones_like(hue)
    value = 1.0 - det_normalized * 0.7  # Whiter = more uncertain

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rgb, aspect="auto")
    ax.set_title(
        f"{title_prefix} - Combined\n"
        f"Color: Major axis angle | Whiter: Higher uncertainty",
        fontsize=10
    )
    ax.axis("off")

    # Colorbar showing angle reference
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="3%")

    angle_gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    angle_gradient = np.repeat(angle_gradient, 20, axis=1)
    hsv_ref = np.stack([angle_gradient, np.ones_like(angle_gradient), np.ones_like(angle_gradient)], axis=-1)
    rgb_ref = hsv_to_rgb(hsv_ref)

    cax.imshow(rgb_ref, aspect="auto", origin="lower", extent=[0, 1, -90, 90])
    cax.set_ylabel("Major Axis Angle (°)", fontsize=9)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.tick_params(labelsize=8)

    fig.tight_layout()
    # results["combined"] = figure_to_tensor(fig)

    # Visualization 2: Separate Cholesky factor channels
    if separate_channels:
        # L11 = sqrt(Var_x)
        results["L11_sqrt_var_x"] = create_heatmap_figname(
            L11, f"{title_prefix} - L11 = √(Var_x)",
            cmap=cmaps[0], colorbar_label="√(Var_x)"
        )

        # L21 = Cov_xy / sqrt(Var_x)
        results["L21_cov_xy_term"] = create_heatmap_figname(
            L21, f"{title_prefix} - L21 = Cov_xy / √(Var_x)",
            cmap=cmaps[1], colorbar_label="Cov_xy / √(Var_x)"
        )

        # L22 = sqrt(Var_y)
        results["L22_sqrt_var_y"] = create_heatmap_figname(
            L22, f"{title_prefix} - L22 = √(Var_y)",
            cmap=cmaps[2], colorbar_label="√(Var_y)"
        )

    # Visualization 3: Determinant map (uncertainty landscape)
    # results["determinant"] = create_heatmap_figname(
    #     det, f"{title_prefix} - Determinant |Σ| (Uncertainty)",
    #     cmap="plasma", colorbar_label="|Σ|"
    # )

    # Visualization 4: Anisotropy map (ellipse elongation)
    # Clip to reasonable range for visualization
    # anisotropy_clipped = np.clip(anisotropy, 1, 10)
    # results["anisotropy"] = create_heatmap_figname(
    #     anisotropy_clipped, f"{title_prefix} - Anisotropy λ₁/λ₂",
    #     cmap="magma", colorbar_label="λ₁/λ₂ (clipped to [1, 10])"
    # )

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


def create_keypoints_with_covariance_ellipses_overlay_figure(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    covariances: torch.Tensor,
    subsample: int = 10,
    sigma: int = 20,
    title: str = "RaCo Keypoint Detection with Uncertainties",
    seed: int = 42,
) -> torch.Tensor:
    """
    Create visualization of keypoints with covariance ellipses overlay on image.

    Uses viz2d utilities for consistent visualization with training examples.

    Args:
        image: Image tensor [3, H, W] (normalized)
        keypoints: Keypoints tensor [N, 2]
        covariances: Covariance matrices tensor [N, 2, 2]
        subsample: Subsample factor for cleaner visualization (default: 10)
        sigma: Number of standard deviations for ellipse size (default: 20)
        title: Plot title
        seed: Random seed for reproducible subsampling

    Returns:
        Figure as torch tensor [3, H, W] for TensorBoard
    """
    from raco.utils import viz2d

    # Denormalize image
    if isinstance(image, torch.Tensor):
        img_np = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image

    # Convert to numpy if needed
    if isinstance(keypoints, torch.Tensor):
        keypoints_np = keypoints.cpu().detach().numpy()
    else:
        keypoints_np = keypoints

    if isinstance(covariances, torch.Tensor):
        covariances_np = covariances.cpu().detach().numpy()
    else:
        covariances_np = covariances

    # Subsample keypoints for cleaner visualization
    n_keypoints = len(keypoints_np)
    if n_keypoints > subsample:
        idxs = np.random.RandomState(seed).permutation(n_keypoints)[::subsample]
        subsampled_keypoints = keypoints_np[idxs]
        subsampled_covariances = covariances_np[idxs]
    else:
        subsampled_keypoints = keypoints_np
        subsampled_covariances = covariances_np

    # Create figure using viz2d
    ax = viz2d.plot_images([img_np])

    # Plot covariance ellipses on subsampled keypoints
    viz2d.plot_covariance_ellipses(
        [subsampled_keypoints],
        [subsampled_covariances],
        axes=ax,
        sigma=sigma,
    )

    # Plot all keypoints
    viz2d.plot_keypoints(
        [keypoints_np],
        axes=ax,
    )

    # Add title
    plt.suptitle(title, fontsize=9, y=0.95)
    plt.tight_layout()

    # Convert to tensor
    return figure_to_tensor(plt.gcf())


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
        seed: int = 42,
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
        if "ranker_map" in pred.get("image0", {}):
            # Ranker scores are per-keypoint, need to scatter to image
            ranker_map = pred["image0"]["ranker_map"][0].cpu().numpy().squeeze()
            ranker_fig = create_heatmap_figname(
                ranker_map,
                title=f"Ranker Scores - {seq_name}",
                cmap="plasma",
                colorbar_label="Rank",
            )
            self.writer.add_image(f"{tag_prefix}/ranker_map", ranker_fig, global_step)

        # 3. Log covariance maps if available
        if "covariances_map" in pred.get("image0", {}):
            covariances_map = pred["image0"]["covariances_map"][0].cpu().numpy()  # [3, H, W]
            covariances_map = covariances_map.transpose(1, 2, 0)  # [H, W, 3]
            cov_figs = create_covariance_figures(
                covariances_map,
                title_prefix=f"{seq_name}",
                separate_channels=True  # Enable all visualizations
            )

            # Log all visualizations to TensorBoard
            for viz_name, viz_tensor in cov_figs.items():
                self.writer.add_image(
                    f"{tag_prefix}/covariance_{viz_name}",
                    viz_tensor,
                    global_step
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

        kpts_cov = create_keypoints_with_covariance_ellipses_overlay_figure(
            img0,         # [3, H, W]
            keypoints,    # [N, 2]
            pred["image0"]["covariances"][0].cpu().numpy(), # [N, 2, 2]
            subsample=10,
            sigma=20,
            seed=seed,
        )
        self.writer.add_image(f"{tag_prefix}/keypoints_with_cov", kpts_cov, global_step)

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
