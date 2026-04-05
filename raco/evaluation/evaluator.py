"""
Evaluation utilities for RaCo training.
Extracted from train.py for reusability.
"""

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional

from raco.geometry.homography import transform_points_with_homography
from raco.geometry.matching import find_matches


def run_eval(
    model,
    eval_loader,
    device: str,
    writer: SummaryWriter,
    global_step: int,
    num_vis: int = 5,
    scene_logger=None
) -> Dict[str, float]:
    """
    Run evaluation and log metrics to TensorBoard.

    Args:
        model: RaCo model
        eval_loader: Data loader for evaluation
        device: Device to run on
        writer: TensorBoard writer
        global_step: Current training step
        num_vis: Number of scenes to visualize (deprecated, use scene_logger)
        scene_logger: Enhanced scene logger for visualization

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_repeatability = []
    all_matching_scores = []

    logger.info("Running evaluation...")

    # Initialize scene logger if not provided
    if scene_logger is None:
        from raco.utils.visualization import create_scene_logger
        scene_logger = create_scene_logger(writer)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(eval_loader, desc="Eval", leave=False)):
            # Move to device
            for key in data:
                if isinstance(data[key], dict):
                    for k in data[key]:
                        if torch.is_tensor(data[key][k]):
                            data[key][k] = data[key][k].to(device)
                elif torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            pred = model.forward_dual(data)

            # Compute metrics
            kpts0 = pred['keypoints_0'][0]
            kpts1 = pred['keypoints_1'][0]
            H_gt = data['H_0to1'][0]

            matches_a, matches_b = find_matches(kpts0, kpts1, H_gt)
            num_matches = len(matches_a)

            # Repeatability: fraction of keypoints with match < 3px
            if len(kpts0) > 0:
                kpts0_proj = transform_points_with_homography(
                    kpts0.unsqueeze(0), H_gt.unsqueeze(0)
                ).squeeze(0)
                distances = torch.cdist(kpts0_proj, kpts1)
                min_dist = distances.min(dim=1)[0]
                repeatability = (min_dist < 3.0).float().mean().item()
            else:
                repeatability = 0.0

            all_repeatability.append(repeatability)

            if len(kpts0) + len(kpts1) > 0:
                matching_score = num_matches / (len(kpts0) + len(kpts1))
                all_matching_scores.append(matching_score)

            # Log scene-specific visualizations using enhanced logger
            seq_name = data.get('seq_name', ['unknown'])[0] if isinstance(data.get('seq_name'), list) else data.get('seq_name', 'unknown')
            img_idx = data.get('img_idx', [0])[0] if isinstance(data.get('img_idx'), list) else data.get('img_idx', 0)

            # Use enhanced logging for tracked scenes
            if scene_logger.should_log_scene(seq_name):
                # Prepare prediction dict in format expected by logger
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
                        global_step=global_step,
                        img_idx=img_idx,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log scene {seq_name}: {e}")

            # Fallback: Log basic visualizations for non-tracked scenes
            else:
                logger.warning(f"Skip logging to tensorboard")

    # Log metrics
    mean_rep = np.mean(all_repeatability) if all_repeatability else 0.0
    mean_ms = np.mean(all_matching_scores) if all_matching_scores else 0.0

    writer.add_scalar("eval/repeatability", mean_rep, global_step)
    writer.add_scalar("eval/matching_score", mean_ms, global_step)

    logger.info(f"Eval - Repeatability: {mean_rep:.4f}, Matching Score: {mean_ms:.4f}")

    model.train()
    return {"repeatability": mean_rep, "matching_score": mean_ms}
