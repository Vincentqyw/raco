"""RaCo training module."""

from .mixed_precision import setup_amp, AMP_AVAILABLE, autocast, GradScaler
from .checkpoint import save_checkpoint, load_checkpoint, get_checkpoint_path
from .model_utils import set_stage_require_grad
from .losses import compute_detector_loss, compute_ranker_loss, compute_covariance_loss
from .metrics import (
    log_detector_metrics,
    log_ranker_metrics,
    log_covariance_metrics,
    log_gradients,
    build_postfix,
)
from .engine import StageTrainer

__all__ = [
    'setup_amp',
    'AMP_AVAILABLE',
    'autocast',
    'GradScaler',
    'save_checkpoint',
    'load_checkpoint',
    'get_checkpoint_path',
    'set_stage_require_grad',
    'compute_detector_loss',
    'compute_ranker_loss',
    'compute_covariance_loss',
    'log_detector_metrics',
    'log_ranker_metrics',
    'log_covariance_metrics',
    'log_gradients',
    'build_postfix',
    'StageTrainer',
]
