"""
Training engine for RaCo.
Implements a StageTrainer class that handles single training stage.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from typing import Optional, Dict, Tuple

from raco.models.losses import DetectorLoss, RankingLoss, CovarianceLoss
from raco.geometry.homography import transform_points_with_homography
from raco.geometry.matching import get_valid_mask, compute_mutual_dist
from raco.trainer.mixed_precision import setup_amp, autocast, GradScaler
from raco.trainer.checkpoint import save_checkpoint
from raco.trainer.model_utils import set_stage_require_grad
from raco.trainer.losses import compute_detector_loss, compute_ranker_loss, compute_covariance_loss
from raco.trainer.metrics import (
    log_detector_metrics, log_ranker_metrics, log_covariance_metrics,
    log_gradients, build_postfix
)
from raco.evaluation import run_eval


class StageTrainer:
    """
    Trainer for a single training stage (detector/ranker/covariance).

    Encapsulates all training logic including:
    - Optimizer and scheduler setup
    - Loss function initialization
    - Mixed precision training
    - Training loop with logging
    - Checkpoint saving
    - Evaluation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stage: str,
        conf,
        device: torch.device,
        writer: SummaryWriter,
        scene_logger=None
    ):
        """
        Initialize StageTrainer.

        Args:
            model: RaCo model
            stage: Training stage (detector/ranker/covariance)
            conf: OmegaConf configuration
            device: Device to train on
            writer: TensorBoard writer
            scene_logger: Enhanced scene logger for visualization
        """
        self.model = model
        self.stage = stage
        self.conf = conf
        self.device = device
        self.writer = writer
        self.scene_logger = scene_logger

        # Setup training components
        self._setup_training_params()
        self._setup_amp()
        self._setup_loss()
        self._setup_optimizer()
        self._freeze_parameters()

        # Log which parameters are trainable
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Stage '{stage}' - Trainable params: {trainable}")

    def _setup_training_params(self):
        """Setup training hyperparameters."""
        # Note: ranker max_steps will be set in train() method when train_loader is available
        self.max_steps_dict = {
            "detector": self.conf.train.detector_steps,
            "covariance": self.conf.train.covariance_steps,
        }

        self.log_interval = self.conf.train.log_interval
        self.save_interval = self.conf.train.save_interval
        self.eval_interval = self.conf.train.get("eval_interval", 5000)

    def _setup_amp(self):
        """Setup automatic mixed precision."""
        use_amp = self.conf.train.get("use_amp", True) and self.device.type == "cuda"
        self.scaler, amp_enabled = setup_amp(self.device.type, use_amp)
        self.use_amp = amp_enabled

        if self.use_amp:
            logger.info("Using mixed precision training (AMP)")

    def _setup_loss(self):
        """Initialize loss function based on stage."""
        self.det_loss_fn = None
        self.rank_loss_fn = None
        self.cov_loss_fn = None

        if self.stage == "detector":
            self.det_loss_fn = DetectorLoss(
                d_max=self.conf.model.detector.d_max,
                rho_pos=self.conf.model.detector.rho_pos,
                rho_neg_max=self.conf.model.detector.rho_neg_max,
            )
        elif self.stage == "ranker":
            self.rank_loss_fn = RankingLoss(
                lambda_ranker=self.conf.model.ranker.get("lambda_ranker", 1.0),
            )
        elif self.stage == "covariance":
            self.cov_loss_fn = CovarianceLoss()

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.conf.train.lr,
            weight_decay=self.conf.train.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_steps, eta_min=1e-6
        )

    def _freeze_parameters(self):
        """Freeze parameters based on training stage."""
        set_stage_require_grad(self.model, self.stage)

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device."""
        for key in batch:
            if isinstance(batch[key], dict):
                for k in batch[key]:
                    if torch.is_tensor(batch[key][k]):
                        batch[key][k] = batch[key][k].to(self.device, non_blocking=True)
            elif torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device, non_blocking=True)
        return batch

    def _compute_reprojection_data(
        self,
        pred: Dict,
        batch: Dict
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute reprojection distances and valid masks.

        Returns:
            Tuple of (mutual_mask_0_to_1, nearest_idx_0_to_1, distances_0_to_1,
                     mutual_mask_1_to_0, nearest_idx_1_to_0, distances_1_to_0,
                     valid_0_to_1, valid_1_to_0, kpts0_in_1, kpts1_in_0, B)
        """
        B, _, H, W = pred["image0"]["raw_scores"].shape
        kpts0 = pred["keypoints_0"]  # (B, N, 2)
        kpts1 = pred["keypoints_1"]
        H_0to1 = batch["H_0to1"]

        # Transform keypoints (float32 for stability)
        kpts0_in_1 = transform_points_with_homography(kpts0, H_0to1)
        kpts1_in_0 = transform_points_with_homography(kpts1, torch.inverse(H_0to1))

        # Check for NaN/Inf
        if not torch.isfinite(kpts0_in_1).all() or not torch.isfinite(kpts1_in_0).all():
            logger.warning(f"Skipping batch: NaN/Inf in homography transformation")
            logger.warning(f"  H_0to1 range: [{H_0to1.min().item():.4f}, {H_0to1.max().item():.4f}]")
            logger.warning(f"  kpts0 range: [{kpts0.min().item():.4f}, {kpts0.max().item():.4f}]")
            return None

        # Compute mutual nearest neighbors
        match_threshold = 3.0
        mutual_mask_0_to_1, nearest_idx_0_to_1, distances_0_to_1 = compute_mutual_dist(
            kpts0_in_1, kpts1, threshold=match_threshold
        )
        mutual_mask_1_to_0, nearest_idx_1_to_0, distances_1_to_0 = compute_mutual_dist(
            kpts1_in_0, kpts0, threshold=match_threshold
        )

        # Valid masks
        valid_0_to_1 = get_valid_mask(kpts0_in_1, H, W)
        valid_1_to_0 = get_valid_mask(kpts1_in_0, H, W)

        # Check for NaN/Inf in distances
        if not torch.isfinite(distances_0_to_1).all() or not torch.isfinite(distances_1_to_0).all():
            logger.warning(f"Skipping batch: NaN/Inf in reprojection errors")
            return None

        return (
            mutual_mask_0_to_1, nearest_idx_0_to_1, distances_0_to_1,
            mutual_mask_1_to_0, nearest_idx_1_to_0, distances_1_to_0,
            valid_0_to_1, valid_1_to_0, kpts0_in_1, kpts1_in_0, B, kpts0, kpts1, H_0to1
        )

    def train_step(
        self,
        batch: Dict,
        iteration: int
    ) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Execute single training step.

        Args:
            batch: Input batch
            iteration: Current iteration

        Returns:
            Tuple of (loss, metrics) or None if batch should be skipped
        """
        # Move to device
        batch = self._move_batch_to_device(batch)

        # Forward pass with AMP
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            pred = self.model.forward_dual(batch)

            # Compute reprojection data
            reprojection_data = self._compute_reprojection_data(pred, batch)
            if reprojection_data is None:
                return None

            (
                mutual_mask_0_to_1, nearest_idx_0_to_1, distances_0_to_1,
                mutual_mask_1_to_0, nearest_idx_1_to_0, distances_1_to_0,
                valid_0_to_1, valid_1_to_0, kpts0_in_1, kpts1_in_0, B, kpts0, kpts1, H_0to1
            ) = reprojection_data

            # Compute loss based on stage
            if self.stage == "detector":
                loss, loss_metrics = compute_detector_loss(
                    pred, self.det_loss_fn,
                    distances_0_to_1, distances_1_to_0,
                    valid_0_to_1, valid_1_to_0,
                    iteration
                )
                if loss.item() == 0:
                    return None

            elif self.stage == "ranker":
                loss, loss_metrics = compute_ranker_loss(
                    pred, self.rank_loss_fn,
                    mutual_mask_0_to_1, mutual_mask_1_to_0,
                    nearest_idx_0_to_1, nearest_idx_1_to_0
                )

            elif self.stage == "covariance":
                loss, loss_metrics = compute_covariance_loss(
                    pred, self.cov_loss_fn,
                    kpts0, kpts1, kpts0_in_1, kpts1_in_0, H_0to1,
                    mutual_mask_0_to_1, mutual_mask_1_to_0,
                    nearest_idx_0_to_1, nearest_idx_1_to_0, B
                )
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                loss_metrics = {}

        # Backward with gradient clipping
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Step scheduler
        self.scheduler.step()

        # Return loss and metrics with extra data for logging
        return loss, loss_metrics, pred, mutual_mask_0_to_1

    def train(
        self,
        train_loader,
        eval_loader=None,
        start_iter=0
    ) -> int:
        """
        Run training loop.

        Args:
            train_loader: Training data loader
            eval_loader: Evaluation data loader (optional)
            start_iter: Starting iteration

        Returns:
            Final iteration number
        """
        # Set max_steps for ranker stage (requires train_loader)
        if self.stage == "ranker":
            self.max_steps = len(train_loader) * self.conf.train.ranker_epochs
        else:
            self.max_steps = self.max_steps_dict[self.stage]

        self.model.train()
        iteration = start_iter
        pbar = tqdm(total=self.max_steps, desc=f"Training {self.stage}")

        while iteration < self.max_steps:
            for batch in train_loader:
                if iteration >= self.max_steps:
                    break

                # Train step
                result = self.train_step(batch, iteration)
                if result is None:
                    continue

                loss, loss_metrics, pred, mutual_mask_0_to_1 = result

                # Logging
                if iteration % self.log_interval == 0:
                    self.writer.add_scalar(f"{self.stage}/loss", loss.item(), iteration)
                    self.writer.add_scalar(f"{self.stage}/lr", self.scheduler.get_last_lr()[0], iteration)

                    # Stage-specific metrics
                    if self.stage == "detector":
                        log_detector_metrics(self.writer, iteration, loss_metrics, mutual_mask_0_to_1)
                    elif self.stage == "ranker":
                        log_ranker_metrics(self.writer, iteration, loss_metrics)
                    elif self.stage == "covariance":
                        log_covariance_metrics(self.writer, iteration, loss_metrics)

                    # Gradient norms
                    if iteration % 500 == 0:
                        log_gradients(self.writer, self.model, iteration, self.stage)

                iteration += 1
                pbar.update(1)

                # Progress bar postfix
                prob_0_sparse = pred["image0"]["keypoint_scores"] if self.stage == "detector" else None
                postfix = build_postfix(self.stage, loss.item(), loss_metrics, prob_0_sparse)
                pbar.set_postfix(postfix)

                # Save checkpoint
                if iteration % self.save_interval == 0:
                    save_checkpoint(self.model, self.conf.output.output_dir, self.stage, step=iteration)

                # Evaluation
                if eval_loader is not None and iteration % self.eval_interval == 0:
                    run_eval(self.model, eval_loader, self.device, self.writer, iteration, scene_logger=self.scene_logger)

        pbar.close()
        return iteration


__all__ = ['StageTrainer']
