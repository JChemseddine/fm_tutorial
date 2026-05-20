# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""Checkpoint utilities for saving and loading model state."""

import os
import torch
from typing import Dict, Any, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: 'Config',
    checkpoint_dir: str,
    filename: Optional[str] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        step: Current training step
        config: Configuration object
        checkpoint_dir: Directory to save checkpoint
        filename: Optional filename (default: step_{step}.pt)
        extras: Optional dict of additional state to save (e.g. time_warp)

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"step_{step}.pt"

    checkpoint_path = os.path.join(checkpoint_dir, filename)

    # Save the underlying module if model is torch.compile-wrapped, so the
    # checkpoint keys never carry the `_orig_mod.` prefix and round-trip
    # cleanly into either compiled or uncompiled models.
    save_target = getattr(model, '_orig_mod', model)
    checkpoint = {
        'model_state_dict': save_target.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': config.to_dict(),
    }
    if extras:
        checkpoint.update(extras)

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to

    Returns:
        Dictionary with checkpoint info (step, config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Strip `_orig_mod.` from old checkpoints saved from a compiled model,
    # and load into the underlying module to avoid prefix mismatch when the
    # current model is itself compiled.
    msd = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model_state_dict'].items()}
    target = getattr(model, '_orig_mod', model)
    missing, unexpected = target.load_state_dict(msd, strict=False)
    if missing:
        print(f"  Warning: missing keys in checkpoint: {missing}")
    print(f"Model loaded from {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Optimizer loaded from {checkpoint_path}")

    info = {
        'step': checkpoint.get('step', 0),
        'config': checkpoint.get('config', {}),
    }
    # Pass through any extra state (e.g. time_warp_state)
    for key in checkpoint:
        if key not in ('model_state_dict', 'optimizer_state_dict', 'step', 'config'):
            info[key] = checkpoint[key]
    return info


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get path to latest checkpoint in directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ]

    if not checkpoints:
        return None

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    return os.path.join(checkpoint_dir, checkpoints[-1])


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 3) -> None:
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("step_") and f.endswith(".pt")
    ]

    if len(checkpoints) <= keep_last:
        return

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        os.remove(checkpoint_path)
        print(f"Removed old checkpoint: {checkpoint_path}")