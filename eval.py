#!/usr/bin/env python
"""
RaCo evaluation script on HPatches.
Following glue-factory eval structure:
- Support for separate vantage/illumination evaluation
- Precision@threshold metrics
- Homography estimation evaluation (DLT + RANSAC)
- Per-scene aggregation
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from raco.datasets import get_dataset
from raco.datasets.hpatches import IGNORED_SCENES
from raco.geometry.homography import symmetric_homography_error
from raco.models import get_model
from raco.utils.tensorboard_vis import create_scene_logger, denormalize_image


def find_matches(kpts0, kpts1, H_gt, threshold=3.0):
    """Find mutual nearest neighbor matches between keypoints."""
    if len(kpts0) == 0 or len(kpts1) == 0:
        return [], []

    kpts0_np = kpts0.cpu().numpy() if torch.is_tensor(kpts0) else kpts0
    kpts1_np = kpts1.cpu().numpy() if torch.is_tensor(kpts1) else kpts1
    H = H_gt.cpu().numpy() if torch.is_tensor(H_gt) else H_gt

    # Project kpts0 to image 1
    ones = np.ones((len(kpts0_np), 1))
    kpts0_h = np.hstack([kpts0_np, ones])
    kpts0_proj = (H @ kpts0_h.T).T
    kpts0_proj = kpts0_proj[:, :2] / kpts0_proj[:, 2:3]

    # Find mutual nearest neighbors
    distances = np.linalg.norm(kpts1_np[None, :, :] - kpts0_proj[:, None, :], axis=-1)

    matches_0to1 = distances.argmin(axis=1)
    min_dist_0to1 = distances.min(axis=1)

    matches_1to0 = distances.argmin(axis=0)

    matches = []
    match_distances = []
    for i in range(len(kpts0_np)):
        j = matches_0to1[i]
        if matches_1to0[j] == i and min_dist_0to1[i] < threshold:
            matches.append((i, j))
            match_distances.append(min_dist_0to1[i])

    return matches, match_distances


def compute_repeatability(kpts0, kpts1, H_gt):
    """Compute repeatability: fraction of keypoints with match < threshold."""
    if len(kpts0) == 0 or len(kpts1) == 0:
        return 0.0, 0.0

    kpts0_np = kpts0.cpu().numpy() if torch.is_tensor(kpts0) else kpts0
    kpts1_np = kpts1.cpu().numpy() if torch.is_tensor(kpts1) else kpts1
    H = H_gt.cpu().numpy() if torch.is_tensor(H_gt) else H_gt

    # Project kpts0 to image 1
    ones = np.ones((len(kpts0_np), 1))
    kpts0_h = np.hstack([kpts0_np, ones])
    kpts0_proj = (H @ kpts0_h.T).T
    kpts0_proj = kpts0_proj[:, :2] / kpts0_proj[:, 2:3]

    # For each projected kpt0, find distance to nearest kpt1
    distances = np.linalg.norm(kpts1_np[None, :, :] - kpts0_proj[:, None, :], axis=-1)
    min_dist = distances.min(axis=1)

    repeatability_3px = (min_dist < 3.0).mean()
    repeatability_1px = (min_dist < 1.0).mean()

    return repeatability_3px, repeatability_1px


def estimate_homography_dlt(kpts0, kpts1, matches):
    """Estimate homography using DLT on matched keypoints."""
    if len(matches) < 4:
        return None, 0

    pts0 = np.float32([kpts0[i] for i, j in matches])
    pts1 = np.float32([kpts1[j] for i, j in matches])

    H_est, inliers = cv2.findHomography(pts0, pts1, method=0)
    num_inliers = np.sum(inliers) if inliers is not None else 0

    return H_est, num_inliers


def estimate_homography_ransac(kpts0, kpts1, matches, ransac_thresh=3.0):
    """Estimate homography using RANSAC on matched keypoints."""
    if len(matches) < 4:
        return None, 0, []

    pts0 = np.float32([kpts0[i] for i, j in matches])
    pts1 = np.float32([kpts1[j] for i, j in matches])

    H_est, mask = cv2.findHomography(
        pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), dtype=bool)
    num_inliers = int(np.sum(inliers))

    return H_est, num_inliers, inliers


def compute_homography_corner_error(H_est, H_gt, img_size):
    """Compute corner error between estimated and GT homography."""
    if H_est is None:
        return float('inf')

    h, w = img_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Transform corners with both homographies
    ones = np.ones((4, 1))
    corners_h = np.hstack([corners, ones])

    corners_est = (H_est @ corners_h.T).T
    corners_est = corners_est[:, :2] / corners_est[:, 2:3]

    corners_gt = (H_gt @ corners_h.T).T
    corners_gt = corners_gt[:, :2] / corners_gt[:, 2:3]

    corner_errors = np.linalg.norm(corners_est - corners_gt, axis=1)
    return np.mean(corner_errors)


def evaluate_pair(pred, data):
    """Evaluate a single image pair."""
    kpts0 = pred['keypoints_0'][0].cpu().numpy()
    kpts1 = pred['keypoints_1'][0].cpu().numpy()
    H_gt = data['H_0to1'][0].cpu().numpy()

    result = {}

    # Find matches
    matches, match_distances = find_matches(kpts0, kpts1, H_gt, threshold=3.0)

    # Basic stats
    result['num_keypoints'] = len(kpts0) + len(kpts1)
    result['num_matches'] = len(matches)

    # Repeatability
    rep_3px, rep_1px = compute_repeatability(kpts0, kpts1, H_gt, threshold=3.0)
    result['repeatability_3px'] = rep_3px
    result['repeatability_1px'] = rep_1px

    # Matching metrics
    if len(matches) > 0:
        result['matching_score'] = len(matches) / (len(kpts0) + len(kpts1))
        result['match_precision_1px'] = np.mean(np.array(match_distances) < 1.0)
        result['match_precision_3px'] = np.mean(np.array(match_distances) < 3.0)
        result['mean_match_distance'] = np.mean(match_distances)

        # Homography estimation via DLT
        H_dlt, inliers_dlt_count = estimate_homography_dlt(kpts0, kpts1, matches)
        result['dlt_inliers'] = inliers_dlt_count
        if H_dlt is not None:
            # Compute corner error
            img_size = data.get('image_size', torch.tensor([[640, 480]]))[0].cpu().numpy()
            if len(img_size) == 2:
                h, w = img_size[1], img_size[0]  # (W, H) format
            else:
                h, w = 480, 640
            corner_error_dlt = compute_homography_corner_error(H_dlt, H_gt, (h, w))
            result['H_error_dlt'] = corner_error_dlt

            # Symmetric reprojection error on all matches
            pts0 = np.float32([kpts0[i] for i, j in matches])
            pts1 = np.float32([kpts1[j] for i, j in matches])
            sym_errors = symmetric_homography_error(pts0, pts1, H_gt)
            result['mean_sym_error'] = np.mean(sym_errors)

        # Homography estimation via RANSAC
        H_ransac, inliers_ransac, _ = estimate_homography_ransac(
            kpts0, kpts1, matches, ransac_thresh=3.0
        )
        result['ransac_inliers'] = inliers_ransac
        if H_ransac is not None:
            img_size = data.get('image_size', torch.tensor([[640, 480]]))[0].cpu().numpy()
            if len(img_size) == 2:
                h, w = img_size[1], img_size[0]
            else:
                h, w = 480, 640
            corner_error_ransac = compute_homography_corner_error(H_ransac, H_gt, (h, w))
            result['H_error_ransac'] = corner_error_ransac
    else:
        result['matching_score'] = 0.0
        result['match_precision_1px'] = 0.0
        result['match_precision_3px'] = 0.0
        result['mean_match_distance'] = float('inf')

    return result, matches


def aggregate_metrics(results_list, scene_types=None):
    """Aggregate metrics across all evaluated pairs."""
    summary = {}

    # Overall metrics
    for key in results_list[0].keys():
        values = [r[key] for r in results_list if key in r and np.isfinite(r[key])]
        if values:
            summary[f'm{key}'] = np.median(values)
            summary[f'mean_{key}'] = np.mean(values)

    # Separate by scene type if provided
    if scene_types is not None:
        illu_results = [r for r, st in zip(results_list, scene_types) if st]
        vant_results = [r for r, st in zip(results_list, scene_types) if not st]

        for key in results_list[0].keys():
            if illu_results:
                values = [r[key] for r in illu_results if key in r and np.isfinite(r[key])]
                if values:
                    summary[f'm{key}_illumination'] = np.median(values)
            if vant_results:
                values = [r[key] for r in vant_results if key in r and np.isfinite(r[key])]
                if values:
                    summary[f'm{key}_vantage'] = np.median(values)

    return summary


def visualize_pair(data, pred, matches, output_path):
    """Create visualization of keypoints and matches."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        img0 = data['image0']['image'][0].cpu()
        img1 = data['image1']['image'][0].cpu()

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img0 = (img0 * std + mean).clamp(0, 1)
        img1 = (img1 * std + mean).clamp(0, 1)

        img0 = img0.permute(1, 2, 0).numpy()
        img1 = img1.permute(1, 2, 0).numpy()

        kpts0 = pred['keypoints_0'][0].cpu().numpy()
        kpts1 = pred['keypoints_1'][0].cpu().numpy()
        scores0 = pred['keypoint_scores_0'][0].cpu().numpy()

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Image 0 with keypoints colored by score
        axes[0, 0].imshow(img0)
        scatter = axes[0, 0].scatter(kpts0[:, 0], kpts0[:, 1], c=scores0, cmap='viridis', s=5)
        axes[0, 0].set_title(f'View 0: {len(kpts0)} keypoints')
        axes[0, 0].axis('off')
        plt.colorbar(scatter, ax=axes[0, 0], label='Score')

        # Image 1 with keypoints
        axes[0, 1].imshow(img1)
        axes[0, 1].scatter(kpts1[:, 0], kpts1[:, 1], c='lime', s=5)
        axes[0, 1].set_title(f'View 1: {len(kpts1)} keypoints')
        axes[0, 1].axis('off')

        # Matches
        axes[1, 0].imshow(np.hstack([img0, img1]))
        for i, j in matches[:50]:  # Show first 50 matches
            pt0 = kpts0[i]
            pt1 = kpts1[j] + [img0.shape[1], 0]
            axes[1, 0].plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'b-', alpha=0.3, linewidth=0.5)
        axes[1, 0].set_title(f'Matches: {len(matches)}')
        axes[1, 0].axis('off')

        # Covariance visualization if available
        if 'covariances_0' in pred and len(kpts0) > 0:
            covs0 = pred['covariances_0'][0].cpu().numpy()
            axes[1, 1].imshow(img0)
            # Draw covariance ellipses for a subset
            n_viz = min(50, len(kpts0))
            for i in range(n_viz):
                # Draw simple ellipse approximation
                cov = covs0[i]
                # Eigen decomposition for ellipse orientation
                eigvals, eigvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                width, height = 2 * np.sqrt(eigvals) * 3  # 3-sigma ellipse
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(kpts0[i], width, height, angle=angle,
                                  fill=False, edgecolor='red', linewidth=0.5)
                axes[1, 1].add_patch(ellipse)
            axes[1, 1].scatter(kpts0[:n_viz, 0], kpts0[:n_viz, 1], c='blue', s=5)
            axes[1, 1].set_title(f'Covariances (3-sigma, {n_viz} shown)')
            axes[1, 1].axis('off')
        else:
            # Show ranker scores if available
            if 'ranker_scores_0' in pred and len(kpts0) > 0:
                ranker_scores = pred['ranker_scores_0'][0].cpu().numpy()
                axes[1, 1].imshow(img0)
                scatter = axes[1, 1].scatter(kpts0[:, 0], kpts0[:, 1], c=ranker_scores, cmap='plasma', s=5)
                axes[1, 1].set_title(f'Ranker Scores: {len(kpts0)} keypoints')
                axes[1, 1].axis('off')
                plt.colorbar(scatter, ax=axes[1, 1], label='Rank')
            else:
                axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scene_type", type=str, default="all",
                        choices=["all", "vantage", "illumination"])
    parser.add_argument("--num_vis", type=int, default=10,
                        help="Number of visualizations to save")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--ignore_large_scenes", type=str, default="true",
                        help="Ignore large scenes for fair comparison")
    args = parser.parse_args()

    ignore_large = args.ignore_large_scenes.lower() == "true"

    # Load config
    conf = OmegaConf.load(args.conf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluation scene type: {args.scene_type}")
    logger.info(f"Ignore large scenes: {ignore_large}")
    if ignore_large:
        logger.info(f"Ignored scenes: {IGNORED_SCENES}")

    # Create output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(output_dir / "tb_logs")

    # Load model
    model = get_model(conf.model.name)(conf.model).to(device)
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Load dataset
    eval_conf = OmegaConf.create({
        "data_dir": "/mnt/e/datasets/hpatches-sequences-release",
        "scene_type": args.scene_type,
        "ignore_large_scenes": ignore_large,
        "batch_size": 1,
        "num_workers": 4,
    })
    dataset = get_dataset("hpatches")(eval_conf)
    loader = dataset.get_data_loader("test", shuffle=False)
    logger.info(f"Evaluating on {len(loader.dataset)} pairs")

    # Evaluation loop
    all_results = []
    scene_types = []
    vis_count = 0

    # Create scene logger for enhanced visualization
    scene_logger = create_scene_logger(writer)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc="Evaluating")):
            # Move to device
            for key in data:
                if isinstance(data[key], dict):
                    for k in data[key]:
                        if torch.is_tensor(data[key][k]):
                            data[key][k] = data[key][k].to(device)
                elif torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            # Forward
            pred = model(data)

            # Evaluate
            result, matches = evaluate_pair(pred, data)
            all_results.append(result)
            scene_types.append(data.get('is_illumination', [False])[0])

            # Get sequence info
            seq_name = data['seq_name'][0] if isinstance(data['seq_name'], list) else data['seq_name']
            img_idx = data.get('img_idx', [1])[0] if isinstance(data.get('img_idx'), list) else data.get('img_idx', 1)

            # Enhanced visualization using scene_logger for tracked scenes
            if scene_logger.should_log_scene(seq_name):
                # Prepare prediction dict
                pred_formatted = {
                    "image0": {
                        "prob_map": pred.get("image0", {}).get("prob_map", None),
                        "ranker_scores": pred.get("ranker_scores_0", None),
                        "covariances": pred.get("covariances_0", None),
                        "keypoints": pred.get("keypoints_0", None),
                    },
                    "keypoints_0": pred.get("keypoints_0"),
                    "keypoint_scores_0": pred.get("keypoint_scores_0"),
                }
                try:
                    scene_logger.log_scene_prediction(
                        seq_name=seq_name,
                        data=data,
                        pred=pred_formatted,
                        global_step=0,  # Single eval, use step 0
                        img_idx=img_idx,
                        seed=conf.dataset.seed,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log scene {seq_name}: {e}")

            # Traditional visualization for other scenes
            if vis_count < args.num_vis and len(matches) > 0:
                vis_path = vis_dir / f"vis_{vis_count:03d}_{seq_name}.png"
                visualize_pair(data, pred, matches, vis_path)
                vis_count += 1

    # Aggregate metrics
    summary = aggregate_metrics(all_results, scene_types)

    # Log to TensorBoard and console
    logger.info("\n=== Evaluation Results ===")
    for key, value in sorted(summary.items()):
        if key.startswith('m'):  # Median metrics
            logger.info(f"{key}: {value:.4f}")
            writer.add_scalar(f"eval/{key}", value, 0)

    # Also log additional metrics
    for key in ['num_keypoints', 'num_matches', 'matching_score', 'repeatability_3px']:
        values = [r[key] for r in all_results if key in r]
        if values:
            mean_val = np.mean(values)
            logger.info(f"mean_{key}: {mean_val:.4f}")
            writer.add_scalar(f"eval/{key}", mean_val, 0)

    writer.close()
    logger.info(f"Results saved to {output_dir}")

    # Print ignored scenes reminder
    if ignore_large:
        logger.info(f"\nNote: Excluded {len(IGNORED_SCENES)} large scenes for fair comparison")
        logger.info(f"Full evaluation should use --ignore_large_scenes=false")


if __name__ == "__main__":
    main()
