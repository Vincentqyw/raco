#!/usr/bin/env python
"""
RaCo training script with eval and TensorBoard logging.
Follows glue-factory training pattern.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from raco.datasets import get_dataset
from raco.models import get_model
from raco.models.utils.losses import DetectorLoss
from raco.geometry.homography import compute_reprojection_error_map


def find_matches(kpts_a, kpts_b, H, threshold=3.0):
    """Find matches between keypoints using homography."""
    from raco.geometry.homography import transform_points_with_homography

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


def run_eval(model, eval_loader, device, writer, global_step, num_vis=5):
    """Run evaluation and log to tensorboard."""
    model.eval()
    all_repeatability = []
    all_matching_scores = []
    vis_count = 0

    logger.info("Running evaluation...")

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

            pred = model(data)

            # Compute metrics
            kpts0 = pred['keypoints_0'][0]
            kpts1 = pred['keypoints_1'][0]
            H_gt = data['H_0to1'][0]

            matches_a, matches_b = find_matches(kpts0, kpts1, H_gt)
            num_matches = len(matches_a)

            # Repeatability: fraction of keypoints with match < 3px
            if len(kpts0) > 0:
                from raco.geometry.homography import transform_points_with_homography
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

            # Log visualizations
            if vis_count < num_vis and num_matches > 0:
                try:
                    img0 = data['view0']['image'][0]
                    # Denormalize
                    mean = torch.tensor([0.485, 0.456, 0.406], device=img0.device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=img0.device).view(3, 1, 1)
                    img0_vis = (img0 * std + mean).clamp(0, 1)
                    writer.add_image(f"eval/pair_{vis_count}/image", img0_vis, global_step)

                    # Log heatmap of keypoint scores
                    scores = pred['keypoint_scores_0'][0]
                    writer.add_histogram(f"eval/pair_{vis_count}/scores", scores, global_step)

                    vis_count += 1
                except Exception as e:
                    pass

    # Log metrics
    mean_rep = np.mean(all_repeatability) if all_repeatability else 0.0
    mean_ms = np.mean(all_matching_scores) if all_matching_scores else 0.0

    writer.add_scalar("eval/repeatability", mean_rep, global_step)
    writer.add_scalar("eval/matching_score", mean_ms, global_step)

    logger.info(f"Eval - Repeatability: {mean_rep:.4f}, Matching Score: {mean_ms:.4f}")

    model.train()
    return {"repeatability": mean_rep, "matching_score": mean_ms}


def train_model(model, train_loader, eval_loader, device, conf, writer, start_iter=0):
    """Training loop with eval."""
    stage = conf.train.stage
    max_steps = {
        "detector": conf.train.detector_steps,
        "ranker": len(train_loader) * conf.train.ranker_epochs,
        "covariance": conf.train.covariance_steps,
    }[stage]

    log_interval = conf.train.log_interval
    save_interval = conf.train.save_interval
    eval_interval = conf.train.get("eval_interval", 5000)

    # Initialize loss
    det_loss_fn = None
    if stage == "detector":
        det_loss_fn = DetectorLoss(
            d_max=conf.model.detector.d_max,
            rho_pos=conf.model.detector.rho_pos,
            rho_neg_max=conf.model.detector.rho_neg_max,
        )

    # Freeze parameters
    for name, param in model.named_parameters():
        param.requires_grad = {
            "detector": True,
            "ranker": "ranker" in name,
            "covariance": "covariance" in name,
        }[stage]

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
                            batch[key][k] = batch[key][k].to(device)
                elif torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)

            pred = model(batch)

            # Compute loss
            if stage == "detector":
                B, _, H, W = pred["view0"]["raw_scores"].shape
                H_0to1 = batch["H_0to1"]

                errors_a = compute_reprojection_error_map(
                    (B, 1, H, W), H_0to1, device
                )
                errors_b = compute_reprojection_error_map(
                    (B, 1, H, W), torch.inverse(H_0to1), device
                )

                prob_flat_a = torch.softmax(pred["view0"]["raw_scores"].flatten(1), dim=1)
                prob_flat_b = torch.softmax(pred["view1"]["raw_scores"].flatten(1), dim=1)

                loss_a = det_loss_fn(prob_flat_a, errors_a)
                loss_b = det_loss_fn(prob_flat_b, errors_b)
                loss = (loss_a + loss_b) / 2

                det_loss_fn.set_step(iteration)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            if iteration % log_interval == 0:
                writer.add_scalar(f"{stage}/loss", loss.item(), iteration)
                writer.add_scalar(f"{stage}/lr", scheduler.get_last_lr()[0], iteration)

            iteration += 1
            pbar.update(1)
            pbar.set_postfix({"step": iteration, "loss": f"{loss.item():.4f}"})

            # Save checkpoint
            if iteration % save_interval == 0:
                ckpt_path = Path(conf.output.output_dir) / f"{stage}_step_{iteration}.pth"
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            # Evaluation
            if eval_loader is not None and iteration % eval_interval == 0:
                run_eval(model, eval_loader, device, writer, iteration)

            if iteration >= max_steps:
                break

    pbar.close()
    return iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=str, default="detector",
                        choices=["detector", "ranker", "covariance", "all"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true", help="Run eval only")
    args = parser.parse_args()

    # Load config
    conf = OmegaConf.load(args.conf)
    conf.train.stage = args.stage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output dir
    output_dir = Path(conf.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        OmegaConf.save(conf, f)

    # TensorBoard
    writer = SummaryWriter(output_dir / "tb_logs")

    # Load model
    model = get_model(conf.model.name)(conf.model).to(device)

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
                "data_dir": "/mnt/e/datasets/hpatches-sequences-release",
                "scene_type": "all",
                "batch_size": 1,
                "num_workers": 2,
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
        run_eval(model, eval_loader, device, writer, 0, num_vis=10)
        writer.close()
        return

    # Train
    start_iter = 0
    stages = ["detector", "ranker", "covariance"] if args.stage == "all" else [args.stage]

    for stage in stages:
        conf.train.stage = stage
        start_iter = train_model(model, train_loader, eval_loader, device, conf, writer, start_iter)

        # Save checkpoint
        ckpt_path = output_dir / f"{stage}_final.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved {stage} checkpoint to {ckpt_path}")

    writer.close()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
