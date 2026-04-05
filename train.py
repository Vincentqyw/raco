#!/usr/bin/env python
"""
RaCo training script with eval and TensorBoard logging.
Follows glue-factory training pattern.
"""

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from raco.datasets import get_dataset
from raco.models import get_model
from raco.models.losses import DetectorLoss, RankingLoss, CovarianceLoss
from raco.geometry.homography import transform_points_with_homography, compute_homography_jacobian
from raco.utils.tensorboard_vis import create_scene_logger

# Mixed precision training
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False

def get_valid_mask(points, H_val, W_val):
    return ((points[..., 0] >= 0) & (points[..., 0] < W_val) &
            (points[..., 1] >= 0) & (points[..., 1] < H_val))

def find_matches(kpts_a, kpts_b, H, threshold=3.0):
    """Find matches between keypoints using homography."""

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


def compute_mutual_dist(kpts0_trans, kpts1, threshold=3.0):
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
    dist10, idx10 = dist_mat.min(dim=1)  # (B, N1) - unused

    # 3. Mutual nearest neighbor check
    # For each kpts0[i], check if kpts1[nn(i)]'s nearest neighbor is i
    target = torch.arange(N0, device=device).unsqueeze(0).expand(B, -1)
    rev_idx = idx10.gather(1, idx01)  # (B, N0)

    # 4. Generate mask: (index match) AND (distance < threshold)
    mutual_mask = (rev_idx == target) & (dist01 < threshold)

    return mutual_mask, idx01, dist01

def run_eval(model, eval_loader, device, writer, global_step, num_vis=5, scene_logger=None):
    """Run evaluation and log to tensorboard."""
    model.eval()
    all_repeatability = []
    all_matching_scores = []

    logger.info("Running evaluation...")

    # Initialize scene logger if not provided
    if scene_logger is None:
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


def set_stage_require_grad(model, stage):
    """Set requires_grad for parameters based on training stage.

    Detector params: encoder (block1-4, conv1-4, pool2, pool4, gate) + score_head
    Ranker params: ranker_head
    Covariance params: covariance_estimator_head
    """
    detector_names = [
        "block1", "block2", "block3", "block4",
        "conv1", "conv2", "conv3", "conv4",
        "pool2", "pool4", "gate", "normalizer",
        "score_head"
    ]

    for name, param in model.named_parameters():
        if stage == "detector":
            # Train everything
            param.requires_grad = True
        elif stage == "ranker":
            # Only ranker_head
            param.requires_grad = "ranker_head" in name
        elif stage == "covariance":
            # Only covariance_estimator_head
            param.requires_grad = ("covariance_estimator_head" in name) or ("var_activation" in name)
        else:
            param.requires_grad = False

    # Log which parameters are trainable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Stage '{stage}' - Trainable params: {trainable}")


def train_model(model, train_loader, eval_loader, device, conf, writer, stage = "detector", start_iter=0, scene_logger=None):
    """Training loop with eval."""
    max_steps = {
        "detector": conf.train.detector_steps,
        "ranker": len(train_loader) * conf.train.ranker_epochs,
        "covariance": conf.train.covariance_steps,
    }[stage]

    log_interval = conf.train.log_interval
    save_interval = conf.train.save_interval
    eval_interval = conf.train.get("eval_interval", 5000)

    # Mixed precision training setup
    use_amp = conf.train.get("use_amp", True) and AMP_AVAILABLE and device.type == "cuda"
    if use_amp:
        # PyTorch 2.0+ uses device-specific GradScaler
        try:
            scaler = GradScaler(device='cuda')
        except TypeError:
            # Fallback for older PyTorch versions
            scaler = GradScaler()
        logger.info("Using mixed precision training (AMP)")
    else:
        scaler = None

    # Initialize loss functions
    det_loss_fn = None
    rank_loss_fn = None
    cov_loss_fn = None

    if stage == "detector":
        det_loss_fn = DetectorLoss(
            d_max=conf.model.detector.d_max,
            rho_pos=conf.model.detector.rho_pos,
            rho_neg_max=conf.model.detector.rho_neg_max,
        )
    elif stage == "ranker":
        rank_loss_fn = RankingLoss(
            lambda_ranker=conf.model.ranker.get("lambda_ranker", 1.0),
        )
    elif stage == "covariance":
        cov_loss_fn = CovarianceLoss()

    # Freeze parameters based on stage
    set_stage_require_grad(model, stage)

    # Optimizer
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=conf.train.lr, weight_decay=conf.train.weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=1e-6
    )

    model.train()
    iteration = start_iter
    pbar = tqdm(total=max_steps, desc=f"Training {stage}")

    while iteration < max_steps:
        for batch in train_loader:
            # Move to device
            for key in batch:
                if isinstance(batch[key], dict):
                    for k in batch[key]:
                        if torch.is_tensor(batch[key][k]):
                            batch[key][k] = batch[key][k].to(device, non_blocking=True)
                elif torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device, non_blocking=True)

            # Mixed precision forward pass
            with autocast(device_type=device.type, enabled=use_amp):
                pred = model.forward_dual(batch)

                # Pre compute reprojection error
                B, _, H, W = pred["image0"]["raw_scores"].shape
                kpts0 = pred["keypoints_0"]  # (B, N, 2)
                kpts1 = pred["keypoints_1"]
                H_0to1 = batch["H_0to1"]

                # Skip batches with degenerate homographies (before transformation)
                # H_max = H_0to1.abs().max().item()
                # if H_max > 100:  # Reasonable threshold
                #     if iteration % 100 == 0:  # Log occasionally to avoid spam
                #         logger.warning(f"Skipping batch {iteration}: Homography values too large (max={H_max:.1f})")
                #     continue

                # Transform keypoints (outputs float32 for numerical stability)
                kpts0_in_1 = transform_points_with_homography(kpts0, H_0to1)
                kpts1_in_0 = transform_points_with_homography(kpts1, torch.inverse(H_0to1))

                # Check for NaN/Inf immediately after transformation (early detection)
                if not torch.isfinite(kpts0_in_1).all() or not torch.isfinite(kpts1_in_0).all():
                    logger.warning(f"Skipping batch {iteration}: NaN/Inf in homography transformation")
                    logger.warning(f"  H_0to1 range: [{H_0to1.min().item():.4f}, {H_0to1.max().item():.4f}]")
                    logger.warning(f"  kpts0 range: [{kpts0.min().item():.4f}, {kpts0.max().item():.4f}]")
                # breakpoint()
                    continue

                # Compute distance to nearest neighbor (matching the paper's definition)
                # For each kpts0_in_1, find nearest neighbor in kpts1, then compute distance
                # This is: d(xiA) = ||H_A→B(xiA) - NN(H_A→B(xiA))||
                match_threshold = 3.0
                mutual_mask_0_to_1, nearest_idx_0_to_1, distances_0_to_1 = compute_mutual_dist(kpts0_in_1, kpts1, threshold=match_threshold)
                mutual_mask_1_to_0, nearest_idx_1_to_0, distances_1_to_0 = compute_mutual_dist(kpts1_in_0, kpts0, threshold=match_threshold)

                valid_0_to_1 = get_valid_mask(kpts0_in_1, H, W)
                valid_1_to_0 = get_valid_mask(kpts1_in_0, H, W)

                # Check for NaN/Inf in reprojection errors before loss computation
                if not torch.isfinite(distances_0_to_1).all() or not torch.isfinite(distances_1_to_0).all():
                    logger.warning(f"Skipping batch {iteration}: NaN/Inf in reprojection errors")
                    continue
                # breakpoint()


                # Debug: visualize training matches
                if iteration % 100 == 0:
                    from visualize_training_matches import visualize_training_batch, visualize_homography_effect
                    visualize_training_batch(
                        batch, pred,
                        kpts0_in_1, kpts1_in_0,
                        kpts0, kpts1,
                        mutual_mask_0_to_1, mutual_mask_1_to_0,
                        nearest_idx_0_to_1, nearest_idx_1_to_0,
                        valid_0_to_1, valid_1_to_0,
                        iteration=iteration,
                        save_dir=str(Path(conf.output.output_dir) / "debug_training_matches")
                    )
                    visualize_homography_effect(
                        batch, pred, kpts0, H_0to1,
                        iteration=iteration,
                        save_dir=str(Path(conf.output.output_dir) / "debug_training_matches")
                    )
                    # breakpoint()

                # Compute loss based on stage
                if stage == "detector":
                    # Check if keypoints are in valid region after transformation
                    prob_0_sparse = pred["image0"]["keypoint_scores"]
                    prob_1_sparse = pred["image1"]["keypoint_scores"]

                    # Compute loss on sparse samples (pass distances, not error vectors)
                    loss_0, loss_0_details = det_loss_fn.forward_sparse(prob_0_sparse, distances_0_to_1, valid_0_to_1)
                    loss_1, loss_1_details = det_loss_fn.forward_sparse(prob_1_sparse, distances_1_to_0, valid_1_to_0)
                    loss = (loss_0 + loss_1) / 2.0
                    
                    detector_loss_0_rewards = loss_0_details['rewards']
                    detector_loss_1_rewards = loss_1_details['rewards']

                    # Debug: check if loss is 0 and why (now handled by early exit)
                    if loss.item() == 0 and iteration % 10 == 0:
                        distances0 = loss_0_details['distances']
                        distances1 = loss_1_details['distances']
                        logger.info(
                            f"Zero loss at iter {iteration}: valid_ratio={valid_0_to_1.float().mean():.3f}, "
                            f"prob_mean={prob_0_sparse.mean():.6f}, ",
                            f"distances0={distances0.mean():.6f}, ",
                            f"distances1={distances1.mean():.6f}",
                        )
                        # breakpoint()
                        continue

                    det_loss_fn.set_step(iteration)

                elif stage == "ranker":
                    # Ranking loss using soft ranking (reuses mutual_dist computed earlier)
                    ranker_scores_0 = pred["ranker_scores_0"]  # (B, N)
                    ranker_scores_1 = pred["ranker_scores_1"]

                    # Pass mutual_mask directly to loss function
                    loss_0_to_1, _ = rank_loss_fn(ranker_scores_0, ranker_scores_1,mutual_mask_a=mutual_mask_0_to_1,nearest_idx_a=nearest_idx_0_to_1)
                    loss_1_to_0, _ = rank_loss_fn(ranker_scores_1, ranker_scores_0,mutual_mask_a=mutual_mask_1_to_0,nearest_idx_a=nearest_idx_1_to_0)
                    loss = (loss_0_to_1 + loss_1_to_0) / 2.0

                elif stage == "covariance":
                    # Covariance loss using reprojection error (Eq. 6-7 in paper)
                    # Only compute loss on matched keypoints (M)
                    covariances_0 = pred["covariances_0"]  # (B, N, 2, 2)
                    covariances_1 = pred["covariances_1"]

                    # Compute Jacobians
                    jacobian_0_to_1 = compute_homography_jacobian(H_0to1, kpts0)
                    jacobian_1_to_0 = compute_homography_jacobian(torch.inverse(H_0to1), kpts1)

                    # # Compute reprojection error vectors for covariance loss (not for detector loss)
                    batch_idx = torch.arange(B, device=kpts0.device).unsqueeze(1).expand(-1, kpts0.shape[1])
                    nearest_kpts1 = kpts1[batch_idx, nearest_idx_0_to_1]  # (B, N, 2)
                    nearest_kpts0 = kpts0[batch_idx, nearest_idx_1_to_0]  # (B, N, 2)

                    # Error vectors (for covariance loss) - (B, N, 2) for einsum
                    errors_0_to_1 = (kpts0_in_1 - nearest_kpts1)  # (B, N, 2)
                    errors_1_to_0 = (kpts1_in_0 - nearest_kpts0)  # (B, N, 2)

                    # Compute bidirectional loss with separate masks for each direction
                    loss, _ = cov_loss_fn.forward_bidirectional(
                        covariances_0, covariances_1,
                        errors_0_to_1, errors_1_to_0,
                        jacobian_0_to_1, jacobian_1_to_0,
                        valid_mask_a_to_b=mutual_mask_0_to_1,
                        valid_mask_b_to_a=mutual_mask_1_to_0,
                    )

                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Backward with mixed precision
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Step scheduler AFTER optimizer (PyTorch best practice)
            scheduler.step()

            # Logging with more debug info
            if iteration % log_interval == 0:
                writer.add_scalar(f"{stage}/loss", loss.item(), iteration)
                writer.add_scalar(f"{stage}/lr", scheduler.get_last_lr()[0], iteration)

                # Debug: log detector-specific statistics
                if stage == "detector":
                    writer.add_scalar(f"{stage}/reward_0_mean", detector_loss_0_rewards.mean().item(), iteration)
                    writer.add_scalar(f"{stage}/reward_0_std", detector_loss_0_rewards.std().item(), iteration)
                    writer.add_scalar(f"{stage}/reward_1_mean", detector_loss_1_rewards.mean().item(), iteration)
                    writer.add_scalar(f"{stage}/reward_1_std", detector_loss_1_rewards.std().item(), iteration)
                    writer.add_scalar(f"{stage}/mutual_ratio", mutual_mask_0_to_1.float().mean().item(), iteration)

                # Debug: log gradient norms every 500 iterations
                if stage == "detector" and iteration % 500 == 0:
                    total_norm = 0.0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2).item()
                            total_norm += param_norm ** 2
                            # Log score_head gradients specifically
                            if "score_head" in name:
                                writer.add_scalar(f"gradients/score_head_{name}", param_norm, iteration)
                    total_norm = total_norm ** 0.5
                    writer.add_scalar(f"gradients/total_norm", total_norm, iteration)

            iteration += 1
            pbar.update(1)

            # Build postfix based on stage
            postfix = {"loss": f"{loss.item():.4f}"}
            if stage == "detector":
                postfix["N"] = prob_0_sparse.shape[1]
                postfix["sum"] = f"{prob_0_sparse.sum(dim=1).mean():.3f}"
                postfix["max"] = f"{prob_0_sparse.max():.4f}"
                postfix["nz"] = f"{(prob_0_sparse > 1e-6).sum(dim=1).float().mean():.0f}"

            pbar.set_postfix(postfix)

            # Save checkpoint
            if iteration % save_interval == 0:
                ckpt_path = Path(conf.output.output_dir) / f"{stage}_step_{iteration}.pth"
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            # Evaluation
            if eval_loader is not None and iteration % eval_interval == 0:
                run_eval(model, eval_loader, device, writer, iteration, scene_logger=scene_logger)

            if iteration >= max_steps:
                break

    pbar.close()
    return iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, default="configs/default.yaml")
    parser.add_argument("-s", "--stage", type=str, default="detector",
                        choices=["detector", "ranker", "covariance", "all"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", help="Run eval only")
    args = parser.parse_args()

    # Load config
    conf = OmegaConf.load(args.conf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output dir with timestamp subfolder
    base_output_dir = Path(conf.output.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    conf.output.output_dir = output_dir  # update
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(conf, f)

    # TensorBoard
    writer = SummaryWriter(output_dir / "tb_logs")

    # Create scene logger for tracking specific HPatches scenes
    scene_logger = create_scene_logger(writer)

    # Load model
    model = get_model(conf.model.name)(conf.model).to(device)

    # Compile model for faster training (PyTorch 2.0+)
    if conf.train.get("compile", False) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()")
        model = torch.compile(model)

    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # Load datasets
    train_dataset = get_dataset(conf.dataset.name)(conf.dataset)
    train_loader = train_dataset.get_data_loader("train")

    # Eval dataset (hpatches) if configured
    eval_loader = None
    if conf.train.get("eval_during_training", False):
        try:
            # HPatches uses data_dir, separate from oxford_paris data_root
            eval_conf = OmegaConf.create({
                "data_dir": conf.eval.get("data_root", "/mnt/e/datasets/hpatches-sequences-release"),
                "scene_type": "all",
                "batch_size": conf.eval.get("batch_size", 1),
                "num_workers": conf.eval.get("num_workers", 2),
                "max_scenes": conf.eval.get("max_scenes", None),
                "max_pairs_per_scene": conf.eval.get("max_pairs_per_scene", None),
            })
            eval_dataset = get_dataset("hpatches")(eval_conf)
            eval_loader = eval_dataset.get_data_loader("test", shuffle=False)
            logger.info(f"Eval dataset: {len(eval_loader.dataset)} pairs")
        except Exception as e:
            logger.warning(f"Could not load eval dataset: {e}")

    logger.info(f"Train dataset: {len(train_loader.dataset)} samples")
    logger.info(f"Model: {conf.model.name}")

    # Run eval only
    if args.eval_only and eval_loader is not None:
        run_eval(model, eval_loader, device, writer, 0, num_vis=10, scene_logger=scene_logger)
        writer.close()
        return

    # Train
    start_iter = 0
    if conf.train.stage == "all":
        stages = ["detector", "ranker", "covariance"]
        logger.info(f"Training all stages...")
    else:
        stages = [conf.train.stage]

    for stage in stages:
        start_iter = train_model(model, train_loader, eval_loader, device, conf, writer, stage, start_iter, scene_logger)

        ckpt_path = output_dir / f"{stage}_final.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved {stage} checkpoint to {ckpt_path}")

    writer.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
