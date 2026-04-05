"""
Training metrics logging utilities.
Extracted from train.py for better modularity.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional


def log_detector_metrics(
    writer: SummaryWriter,
    iteration: int,
    loss_metrics: Dict[str, float],
    mutual_mask_0_to_1: torch.Tensor,
    prob_0_sparse: Optional[torch.Tensor] = None
) -> None:
    """
    Log detector-specific metrics to TensorBoard.

    Args:
        writer: TensorBoard writer
        iteration: Current training step
        loss_metrics: Dict with loss metrics from compute_detector_loss
        mutual_mask_0_to_1: Mutual match mask (B, N)
        prob_0_sparse: Sparse probability distribution (B, N), optional for postfix
    """
    # Log rewards statistics
    writer.add_scalar("detector/reward_0_mean", loss_metrics['rewards_0'], iteration)
    writer.add_scalar("detector/reward_1_mean", loss_metrics['rewards_1'], iteration)
    writer.add_scalar("detector/mutual_ratio", mutual_mask_0_to_1.float().mean().item(), iteration)


def log_ranker_metrics(
    writer: SummaryWriter,
    iteration: int,
    loss_metrics: Dict[str, float]
) -> None:
    """
    Log ranker-specific metrics to TensorBoard.

    Args:
        writer: TensorBoard writer
        iteration: Current training step
        loss_metrics: Dict with loss metrics from compute_ranker_loss
    """
    writer.add_scalar("ranker/spearman_loss", loss_metrics['spearman_loss'], iteration)
    writer.add_scalar("ranker/pull_loss", loss_metrics['pull_loss'], iteration)


def log_covariance_metrics(
    writer: SummaryWriter,
    iteration: int,
    loss_metrics: Dict[str, float],
    full: bool = False,
) -> None:
    """
    Log covariance-specific metrics to TensorBoard.

    Args:
        writer: TensorBoard writer
        iteration: Current training step
        loss_metrics: Dict with loss metrics from compute_covariance_loss
    """
    # NLL loss metrics
    writer.add_scalar("covariance/nll_0_to_1", loss_metrics['cov_nll_0_to_1'], iteration)
    writer.add_scalar("covariance/nll_1_to_0", loss_metrics['cov_nll_1_to_0'], iteration)
    writer.add_scalar("covariance/total_loss", loss_metrics.get('total_loss', 0), iteration)

    if full:
        # Covariance statistics for debugging large values
        cov0 = loss_metrics["covariances_0"]  # (B, N, 2, 2)
        cov1 = loss_metrics["covariances_1"]  # (B, N, 2, 2)

        # Covariance element-wise statistics
        writer.add_scalar("covariance/cov0_mean", cov0.mean().item(), iteration)
        writer.add_scalar("covariance/cov0_std", cov0.std().item(), iteration)
        writer.add_scalar("covariance/cov0_max", cov0.max().item(), iteration)
        writer.add_scalar("covariance/cov0_min", cov0.min().item(), iteration)

        writer.add_scalar("covariance/cov1_mean", cov1.mean().item(), iteration)
        writer.add_scalar("covariance/cov1_std", cov1.std().item(), iteration)
        writer.add_scalar("covariance/cov1_max", cov1.max().item(), iteration)
        writer.add_scalar("covariance/cov1_min", cov1.min().item(), iteration)

        # Trace (variance) statistics - average variance across keypoints
        trace0 = torch.diagonal(cov0, dim1=-2, dim2=-1).sum(-1)  # (B, N)
        trace1 = torch.diagonal(cov1, dim1=-2, dim2=-1).sum(-1)  # (B, N)
        writer.add_scalar("covariance/trace0_mean", trace0.mean().item(), iteration)
        writer.add_scalar("covariance/trace0_std", trace0.std().item(), iteration)
        writer.add_scalar("covariance/trace0_max", trace0.max().item(), iteration)
        writer.add_scalar("covariance/trace1_mean", trace1.mean().item(), iteration)
        writer.add_scalar("covariance/trace1_std", trace1.std().item(), iteration)
        writer.add_scalar("covariance/trace1_max", trace1.max().item(), iteration)

        # Determinant statistics - measure of overall uncertainty volume
        try:
            det0 = torch.det(cov0)
            det1 = torch.det(cov1)
            writer.add_scalar("covariance/det0_mean", det0.mean().item(), iteration)
            writer.add_scalar("covariance/det0_std", det0.std().item(), iteration)
            writer.add_scalar("covariance/det1_mean", det1.mean().item(), iteration)
            writer.add_scalar("covariance/det1_std", det1.std().item(), iteration)
        except RuntimeError:
            # Determinant computation might fail for non-positive-definite matrices
            pass

        # Eigenvalue statistics - analyze uncertainty structure
        try:
            eigenvalues0, _ = torch.linalg.eigh(cov0)  # (B, N, 2)
            eigenvalues1, _ = torch.linalg.eigh(cov1)
            writer.add_scalar("covariance/eigenvalues0_max", eigenvalues0.max().item(), iteration)
            writer.add_scalar("covariance/eigenvalues1_max", eigenvalues1.max().item(), iteration)
            # Anisotropy ratio (lambda_max / lambda_min)
            anisotropy0 = eigenvalues0[..., 1] / (eigenvalues0[..., 0] + 1e-8)
            anisotropy1 = eigenvalues1[..., 1] / (eigenvalues1[..., 0] + 1e-8)
            writer.add_scalar("covariance/anisotropy0_mean", anisotropy0.mean().item(), iteration)
            writer.add_scalar("covariance/anisotropy1_mean", anisotropy1.mean().item(), iteration)
        except RuntimeError:
            # Eigenvalue decomposition might fail for invalid matrices
            pass

        # Reprojection error statistics (for context)
        errors_0_to_1 = loss_metrics.get("errors_0_to_1")
        errors_1_to_0 = loss_metrics.get("errors_1_to_0")
        if errors_0_to_1 is not None:
            error_norm_0_to_1 = torch.norm(errors_0_to_1, dim=-1)  # (B, N)
            writer.add_scalar("covariance/reproj_error_0_to_1", error_norm_0_to_1.mean().item(), iteration)
        if errors_1_to_0 is not None:
            error_norm_1_to_0 = torch.norm(errors_1_to_0, dim=-1)  # (B, N)
            writer.add_scalar("covariance/reproj_error_1_to_0", error_norm_1_to_0.mean().item(), iteration)  

def log_ranker_covariance_metrics(
    writer: SummaryWriter,
    iteration: int,
    loss_metrics: Dict[str, float]
) -> None:
    """
    Log ranker_covariance joint training metrics to TensorBoard.

    Args:
        writer: TensorBoard writer
        iteration: Current training step
        loss_metrics: Dict with combined loss metrics from ranker and covariance
    """
    # Log ranker metrics
    writer.add_scalar("ranker_covariance/spearman_loss", loss_metrics['spearman_loss'], iteration)
    writer.add_scalar("ranker_covariance/pull_loss", loss_metrics['pull_loss'], iteration)
    # Log covariance metrics
    writer.add_scalar("ranker_covariance/nll_0_to_1", loss_metrics['cov_nll_0_to_1'], iteration)
    writer.add_scalar("ranker_covariance/nll_1_to_0", loss_metrics['cov_nll_1_to_0'], iteration)

    # Log weighted losses (if present)
    if 'weighted_ranker' in loss_metrics:
        writer.add_scalar("ranker_covariance/weighted_ranker", loss_metrics['weighted_ranker'], iteration)
    if 'weighted_cov' in loss_metrics:
        writer.add_scalar("ranker_covariance/weighted_cov", loss_metrics['weighted_cov'], iteration)


def log_gradients(
    writer: SummaryWriter,
    model: torch.nn.Module,
    iteration: int,
    stage: str
) -> None:
    """
    Log gradient norms for debugging.

    Args:
        writer: TensorBoard writer
        model: Model to log gradients for
        iteration: Current training step
        stage: Training stage name
    """
    if stage != "detector":
        return

    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            # Log score_head gradients specifically
            if "score_head" in name:
                writer.add_scalar(f"gradients/score_head_{name}", param_norm, iteration)

    total_norm = total_norm ** 0.5
    writer.add_scalar("gradients/total_norm", total_norm, iteration)


def build_postfix(
    stage: str,
    loss_value: float,
    loss_metrics: Dict[str, float],
    prob_0_sparse: Optional[torch.Tensor] = None
) -> Dict[str, str]:
    """
    Build postfix dict for progress bar display.

    Args:
        stage: Training stage name
        loss_value: Current loss value
        loss_metrics: Dict with loss metrics
        prob_0_sparse: Sparse probability distribution (B, N), for detector stage

    Returns:
        Dict of display_name: formatted_value
    """
    postfix = {"loss": f"{loss_value:.4f}"}

    if stage == "detector" and prob_0_sparse is not None:
        postfix["N"] = prob_0_sparse.shape[1]
        postfix["sum"] = f"{prob_0_sparse.sum(dim=1).mean():.3f}"
        postfix["max"] = f"{prob_0_sparse.max():.4f}"
        postfix["nz"] = f"{(prob_0_sparse > 1e-6).sum(dim=1).float().mean():.0f}"

    elif stage == "ranker":
        postfix["spearman"] = f"{loss_metrics.get('spearman_loss', 0):.4f}"
        postfix["pull"] = f"{loss_metrics.get('pull_loss', 0):.4f}"

    elif stage == "covariance":
        postfix["nll_0"] = f"{loss_metrics.get('cov_nll_0_to_1', 0):.4f}"
        postfix["nll_1"] = f"{loss_metrics.get('cov_nll_1_to_0', 0):.4f}"

    elif stage == "ranker_covariance":
        # Show both ranker and covariance metrics
        postfix["spearman"] = f"{loss_metrics.get('spearman_loss', 0):.4f}"
        postfix["pull"] = f"{loss_metrics.get('pull_loss', 0):.4f}"
        postfix["nll_0"] = f"{loss_metrics.get('cov_nll_0_to_1', 0):.4f}"
        postfix["nll_1"] = f"{loss_metrics.get('cov_nll_1_to_0', 0):.4f}"

    return postfix


__all__ = [
    'log_detector_metrics',
    'log_ranker_metrics',
    'log_covariance_metrics',
    'log_ranker_covariance_metrics',
    'log_gradients',
    'build_postfix',
]
