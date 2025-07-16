"""
Utility functions for DNA Transport GNN project.

This module contains helper functions for logging, plotting, data handling,
and other common operations used throughout the project.
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import torch
from datetime import datetime


def setup_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        output_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('dna_transport_gnn')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_training_results(train_losses: List[float], val_losses: List[float], 
                         energy_grid: np.ndarray, filepath: str):
    """
    Save training results to a numpy file.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        energy_grid: Energy grid array
        filepath: Path to save the file
    """
    np.savez(filepath,
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses),
             energy_grid=energy_grid)


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        filepath: str, figsize: Tuple[int, int] = (10, 6)):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        filepath: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dos_comparison(true_dos: np.ndarray, pred_dos: np.ndarray, 
                       energy_grid: np.ndarray, filepath: str,
                       title: str = "DOS Comparison"):
    """
    Plot comparison between true and predicted DOS.
    
    Args:
        true_dos: True DOS values
        pred_dos: Predicted DOS values
        energy_grid: Energy grid
        filepath: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(energy_grid, true_dos, 'r-', label='True DOS', linewidth=2)
    plt.plot(energy_grid, pred_dos, 'b--', label='Predicted DOS', linewidth=2)
    
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_transmission_comparison(true_trans: np.ndarray, pred_trans: np.ndarray,
                               energy_grid: np.ndarray, filepath: str,
                               title: str = "Transmission Comparison"):
    """
    Plot comparison between true and predicted transmission.
    
    Args:
        true_trans: True transmission values
        pred_trans: Predicted transmission values
        energy_grid: Energy grid
        filepath: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(energy_grid, true_trans, 'r-', label='True Transmission', linewidth=2)
    plt.plot(energy_grid, pred_trans, 'b--', label='Predicted Transmission', linewidth=2)
    
    plt.xlabel('Energy (eV)')
    plt.ylabel('Transmission')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> dict:
    """
    Calculate various metrics for model evaluation.
    
    Args:
        true_values: True values
        pred_values: Predicted values
        
    Returns:
        Dictionary containing various metrics
    """
    # Mean Squared Error
    mse = np.mean((true_values - pred_values) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_values - pred_values))
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((true_values - pred_values) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Pearson correlation coefficient
    correlation = np.corrcoef(true_values.flatten(), pred_values.flatten())[0, 1]
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'correlation': correlation
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for the output
    """
    print(f"{prefix}Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.6f}")


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, filepath: str, **kwargs):
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         filepath: str) -> Tuple[int, float]:
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint file
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_directory(base_dir: str, experiment_name: str = None) -> str:
    """
    Create an output directory for experiments.
    
    Args:
        base_dir: Base directory
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'experiment_{timestamp}'
    
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir 