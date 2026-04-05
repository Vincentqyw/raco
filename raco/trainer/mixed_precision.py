"""
Mixed precision training utilities.
Handles AMP (Automatic Mixed Precision) setup and utilities.
"""

import torch
from torch.cuda.amp import GradScaler
from typing import Tuple, Optional

# Check AMP availability
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False


def setup_amp(device: str, use_amp: bool = True) -> Tuple[Optional[GradScaler], bool]:
    """
    Setup Automatic Mixed Precision (AMP) for trainer.

    Args:
        device: Device to run on ('cuda' or 'cpu')
        use_amp: Whether to enable AMP

    Returns:
        Tuple of (GradScaler or None, amp_enabled flag)
    """
    if not use_amp or not AMP_AVAILABLE:
        return None, False

    if device == 'cpu':
        return None, False

    # AMP is available and requested
    scaler = GradScaler()
    return scaler, True


__all__ = ['setup_amp', 'AMP_AVAILABLE', 'autocast', 'GradScaler']
