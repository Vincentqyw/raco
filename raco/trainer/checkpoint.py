"""
Checkpoint utilities for saving and loading model weights.
"""

import torch
from pathlib import Path
from loguru import logger
from typing import Optional, Union


def get_checkpoint_path(
    output_dir: Union[str, Path],
    stage: str,
    step: Optional[int] = None
) -> Path:
    """
    Generate checkpoint file path.

    Args:
        output_dir: Output directory for checkpoints
        stage: Training stage name (detector/ranker/covariance)
        step: Training step number (None for final checkpoint)

    Returns:
        Path to checkpoint file
    """
    output_dir = Path(output_dir)
    if step is None:
        return output_dir / f"{stage}_final.pth"
    else:
        return output_dir / f"{stage}_step_{step}.pth"


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: Union[str, Path],
    stage: str,
    step: Optional[int] = None
) -> Path:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        output_dir: Output directory for checkpoints
        stage: Training stage name
        step: Training step number (None for final checkpoint)

    Returns:
        Path where checkpoint was saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = get_checkpoint_path(output_dir, stage, step)
    torch.save(model.state_dict(), ckpt_path)

    step_str = f"step {step}" if step is not None else "final"
    logger.info(f"Saved {stage} checkpoint ({step_str}) to {ckpt_path}")

    return ckpt_path


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to

    Returns:
        Model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return model


__all__ = ['save_checkpoint', 'load_checkpoint', 'get_checkpoint_path']
