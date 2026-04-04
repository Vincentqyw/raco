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
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(heatmap, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.axis("off")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label)

    fig.tight_layout()
    return figure_to_tensor(fig)


def create_overlay_figure(
    image: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    title: str = "Overlay",
) -> torch.Tensor:
    """Overlay heatmap on RGB image."""
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

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return figure_to_tensor(fig)


def create_covariance_figures(
    cov_map: torch.Tensor,
    title_prefix: str = "Covariance",
    cmaps: List[str] = ["viridis", "RdBu_r", "plasma"],
) -> Dict[str, torch.Tensor]:
    """
    Create visualization for covariance Cholesky factors.

    Args:
        cov_map: Tensor of shape [H, W, 3] containing (L11, L21, L22)
        title_prefix: Prefix for figure titles
        cmaps: List of colormaps for each channel

    Returns:
        Dictionary mapping channel names to figure tensors
    """
    if isinstance(cov_map, torch.Tensor):
        cov_map = cov_map.cpu().numpy()

    # Ensure shape is [H, W, 3]
    if cov_map.shape[0] == 3:
        cov_map = np.transpose(cov_map, (1, 2, 0))

    H, W, C = cov_map.shape
    assert C == 3, f"Expected 3 channels, got {C}"

    channel_names = ["L11 (var_x)", "L21 (cov_xy)", "L22 (var_y)"]
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (name, cmap) in enumerate(zip(channel_names, cmaps)):
        ax = axes[i]
        data = cov_map[..., i]

        # Use symmetric normalization for L21 (can be negative)
        if i == 1:
            vmax = np.abs(data).max()
            vmin = -vmax
        else:
            vmin, vmax = data.min(), data.max()

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{name}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title_prefix} - Cholesky Factors", fontsize=14)
    fig.tight_layout()

    tensor = figure_to_tensor(fig)
    results["combined"] = tensor

    return results


def create_keypoints_overlay_figure(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    title: str = "Keypoints",
    max_kpts: int = 500,
) -> torch.Tensor:
    """Create keypoints overlay on image."""
    fig, ax = plt.subplots(figsize=(10, 8))

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
        plt.colorbar(scatter, ax=ax, label="Score")
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
        self.tracked_scenes = tracked_scenes or DEFAULT_TRACKED_SCENES
        self.num_scenes_per_eval = num_scenes_per_eval

        # Store predictions for tracked scenes to enable step-by-step comparison
        self.scene_history: Dict[str, Dict] = {}

    def should_log_scene(self, seq_name: str) -> bool:
        """Check if a scene should be logged."""
        return seq_name in self.tracked_scenes

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

        tag_prefix = f"scenes/{seq_name}/{img_idx}"

        # 1. Log RGB image
        img0 = data["image0"]["image"][0]  # [3, H, W]
        img_denorm = denormalize_image(img0)
        self.writer.add_image(f"{tag_prefix}/rgb", img_denorm, global_step)

        # 2. Log prob_map heatmap (detection probability)
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

        # 3. Log ranker score map if available
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

        # 4. Log covariance maps if available
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

        # 5. Log keypoints overlay
        keypoints = pred["keypoints_0"][0]  # [N, 2]
        scores = pred.get("keypoint_scores_0", [None])[0] if "keypoint_scores_0" in pred else None

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
