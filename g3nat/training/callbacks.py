"""Training callbacks for checkpointing and progress tracking."""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                   train_losses: List[float], val_losses: List[float], args: Dict,
                   energy_grid: np.ndarray, checkpoint_path: str,
                   metric_history: Optional[List[Dict[str, float]]] = None):
    """Save training checkpoint.

    metric_history is optional and backward compatible: existing callers that
    do not pass it keep working unchanged, and when it is omitted nothing new
    is written to the checkpoint dict (no 'metric_history' key at all, rather
    than a key holding None) -- so a resume read that guards with .get()
    behaves exactly as it did before this parameter existed.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': args,
        'energy_grid': energy_grid,
        'timestamp': time.time()
    }
    if metric_history is not None:
        checkpoint['metric_history'] = metric_history

    # Write-then-rename, because these runs are preemptible. The phase runners
    # use SLURM --requeue on ckpt partitions, so being killed part-way through
    # serialization is expected, not exceptional. Writing straight to
    # checkpoint_path leaves a truncated zip and the requeued attempt dies on
    # resume with "PytorchStreamReader failed reading zip archive" -- which is
    # what happened to job 37966189_0 on 2026-07-31, costing ~3h of training.
    # os.replace is atomic within a filesystem, so an interrupted write leaves
    # the previous checkpoint untouched and the requeue resumes from it.
    tmp_path = f"{checkpoint_path}.tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, checkpoint_path)
    except BaseException:
        # Includes SystemExit/KeyboardInterrupt, since the point is surviving a
        # signal. Never let cleanup mask the original failure.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    print(f"Checkpoint saved: {checkpoint_path}")


def save_progress_file(epoch: int, train_loss: float, val_loss: float,
                      checkpoint_dir: str, args: Dict):
    """Save lightweight progress tracking file."""
    progress_file = os.path.join(checkpoint_dir, 'training_progress.json')
    status_file = os.path.join(checkpoint_dir, 'training_status.txt')

    # Save detailed progress
    progress_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': time.time(),
        'args': args
    }

    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

    # Save simple status file for quick monitoring
    with open(status_file, 'w') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Val Loss: {val_loss:.4f}\n")
        f.write(f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
