"""Training callbacks for checkpointing and progress tracking."""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                   train_losses: List[float], val_losses: List[float], args: Dict,
                   energy_grid: np.ndarray, checkpoint_path: str):
    """Save training checkpoint."""
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
    torch.save(checkpoint, checkpoint_path)
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
