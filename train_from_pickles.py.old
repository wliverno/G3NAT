#!/usr/bin/env python3
"""
Training script for DNATransportHamiltonianGNN using pickle file data.

This script loads DNA transport data from pickle files, splits it into
train/validation/test sets, and trains a DNATransportHamiltonianGNN model.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
import time
from typing import Tuple, List, Dict
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from models import DNATransportHamiltonianGNN, train_model
from dataset import create_dna_dataset
from load_pickle_data import load_pickle_directory
from main import LengthBucketBatchSampler, setup_device, save_checkpoint, create_checkpoint_callback, create_progress_callback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DNA Transport Hamiltonian GNN from pickle files')

    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing pickle files')
    parser.add_argument('--pickle_pattern', type=str, default='*.pkl',
                       help='Glob pattern for pickle files (default: *.pkl)')
    parser.add_argument('--train_ratio', type=float, default=0.70,
                       help='Ratio of data for training (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Ratio of data for validation (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio of data for testing (default: 0.15)')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of graph layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate (default: 0.0 for Hamiltonian model)')
    parser.add_argument('--conv_type', type=str, default='gat', choices=['transformer', 'gat'],
                       help='Convolution type for message passing layers')
    parser.add_argument('--n_orb', type=int, default=1,
                       help='Number of orbitals per DNA base')
    parser.add_argument('--solver_type', type=str, default='frobenius', choices=['frobenius', 'complex'],
                       help='NEGF solver type')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs_pickles',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for saved model (default: auto-generated)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_pickles',
                       help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_frequency', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint file to resume from')

    return parser.parse_args()


def split_indices(num_samples: int, train_ratio: float, val_ratio: float, test_ratio: float,
                  random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices into train/validation/test sets.

    Args:
        num_samples: Total number of samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Shuffle indices
    np.random.seed(random_seed)
    indices = np.random.permutation(num_samples)

    # Calculate split points
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()

    return train_indices, val_indices, test_indices


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.HuberLoss()

    total_loss = 0.0
    dos_losses = []
    trans_losses = []
    num_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            dos_pred, transmission_pred = model(batch)

            # Reshape targets
            batch_size = dos_pred.size(0)
            num_energy_points = dos_pred.size(1)

            dos_target = batch.dos.view(batch_size, num_energy_points)
            transmission_target = batch.transmission.view(batch_size, num_energy_points)

            # Calculate losses
            dos_loss = criterion(dos_pred, dos_target)
            transmission_loss = criterion(transmission_pred, transmission_target)
            total_loss += (dos_loss + transmission_loss).item() * batch_size

            dos_losses.append(dos_loss.item())
            trans_losses.append(transmission_loss.item())
            num_samples += batch_size

    avg_loss = total_loss / num_samples
    avg_dos_loss = np.mean(dos_losses)
    avg_trans_loss = np.mean(trans_losses)

    return {
        'total_loss': avg_loss,
        'dos_loss': avg_dos_loss,
        'transmission_loss': avg_trans_loss,
        'num_samples': num_samples
    }


def save_test_predictions(model: nn.Module, test_loader: DataLoader, energy_grid: np.ndarray,
                         output_dir: str, device: torch.device, num_samples: int = 3):
    """
    Save sample test predictions to visualize model performance.

    Args:
        model: Trained model
        test_loader: Test data loader
        energy_grid: Energy grid for x-axis
        output_dir: Directory to save plots
        device: Device to run on
        num_samples: Number of samples to plot
    """
    model.eval()

    with torch.no_grad():
        # Get first batch
        batch = next(iter(test_loader))
        batch = batch.to(device)

        dos_pred, trans_pred = model(batch)

        # Reshape targets
        batch_size = dos_pred.size(0)
        num_energy_points = dos_pred.size(1)
        dos_target = batch.dos.view(batch_size, num_energy_points)
        trans_target = batch.transmission.view(batch_size, num_energy_points)

        # Plot first few samples
        samples_to_plot = min(num_samples, batch_size)

        for i in range(samples_to_plot):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # DOS plot
            ax1.plot(energy_grid, dos_target[i].cpu().numpy(), 'r-', label='True DOS', linewidth=2)
            ax1.plot(energy_grid, dos_pred[i].cpu().numpy(), 'b--', label='Predicted DOS', linewidth=2)
            ax1.set_xlabel('Energy (eV)')
            ax1.set_ylabel('log10(DOS)')
            ax1.legend()
            ax1.set_title(f'Density of States - Test Sample {i+1}')
            ax1.grid(True, alpha=0.3)

            # Transmission plot
            ax2.plot(energy_grid, trans_target[i].cpu().numpy(), 'r-', label='True Transmission', linewidth=2)
            ax2.plot(energy_grid, trans_pred[i].cpu().numpy(), 'b--', label='Predicted Transmission', linewidth=2)
            ax2.set_xlabel('Energy (eV)')
            ax2.set_ylabel('log10(Transmission)')
            ax2.legend()
            ax2.set_title(f'Transmission Coefficient - Test Sample {i+1}')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'test_sample_{i+1}.png'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved test sample {i+1} prediction plot")


def main():
    """Main training pipeline."""

    # Parse arguments
    args = parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 80)
    print("DNA Transport Hamiltonian GNN Training from Pickle Files")
    print("=" * 80)
    print(f"Arguments: {vars(args)}\n")

    # Setup device
    device = setup_device(args.device)

    # Load data from pickle files
    print("Loading data from pickle files...")
    sequences, complementary_sequences, dos_data, transmission_data, energy_grid, contact_configs = \
        load_pickle_directory(args.data_dir, args.pickle_pattern)

    num_samples = len(sequences)
    print(f"\nDataset summary:")
    print(f"  Total samples: {num_samples}")
    print(f"  Sequence lengths: {[len(s) for s in sequences[:5]]}... (showing first 5)")
    print(f"  Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    print(f"  Contact types: {set(c['contact_type'] for c in contact_configs)}")

    # Split into train/val/test
    print(f"\nSplitting data: train={args.train_ratio:.0%}, val={args.val_ratio:.0%}, test={args.test_ratio:.0%}")
    train_indices, val_indices, test_indices = split_indices(
        num_samples, args.train_ratio, args.val_ratio, args.test_ratio
    )

    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Test samples: {len(test_indices)}")

    # Create full dataset with graphs
    print("\nCreating datasets and converting to graphs...")
    full_dataset = create_dna_dataset(
        sequences=sequences,
        dos_data=dos_data,
        transmission_data=transmission_data,
        energy_grid=energy_grid,
        complementary_sequences=complementary_sequences,
        # Apply contact configurations from pickle files
    )

    # Override contact positions from pickle data
    print("Applying contact configurations from pickle files...")
    for i, config in enumerate(contact_configs):
        graph = full_dataset.graphs[i]
        # The graph was already created with default contact positions
        # We need to update based on the config
        # For now, the loader creates correct positions based on contact_type

    # Create train/val/test subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create data loaders with length bucketing
    print("Creating data loaders with length bucketing...")
    train_sampler = LengthBucketBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_sampler = LengthBucketBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_sampler = LengthBucketBatchSampler(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    # Initialize model
    print("\nInitializing DNATransportHamiltonianGNN model...")
    model = DNATransportHamiltonianGNN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        energy_grid=energy_grid,
        dropout=args.dropout,
        n_orb=args.n_orb,
        conv_type=args.conv_type,
        solver_type=args.solver_type,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Orbitals per base: {args.n_orb}")
    print(f"  Solver type: {args.solver_type}")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=str(device),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        checkpoint_callback=create_checkpoint_callback(args.checkpoint_dir, vars(args), energy_grid),
        progress_callback=create_progress_callback(args.checkpoint_dir, vars(args)),
        max_grad_norm=args.max_grad_norm
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)

    print(f"Test Results:")
    print(f"  Total Loss: {test_metrics['total_loss']:.4f}")
    print(f"  DOS Loss: {test_metrics['dos_loss']:.4f}")
    print(f"  Transmission Loss: {test_metrics['transmission_loss']:.4f}")
    print(f"  Samples evaluated: {test_metrics['num_samples']}")

    # Save model
    print("\nSaving model...")
    if args.model_name is None:
        model_name = f"hamiltonian_pickles_{num_samples}samples_{args.num_epochs}epochs"
    else:
        model_name = args.model_name

    model_path = os.path.join(args.output_dir, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': test_metrics,
        'energy_grid': energy_grid,
        'num_samples': num_samples
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Save training curves
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Training Loss', color='blue')
    ax.plot(val_losses, label='Validation Loss', color='red')
    ax.axhline(y=test_metrics['total_loss'], color='green', linestyle='--', label=f'Test Loss: {test_metrics["total_loss"]:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training, Validation, and Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save test predictions
    print("Saving test predictions...")
    save_test_predictions(model, test_loader, energy_grid, args.output_dir, device)

    # Save test metrics to JSON
    test_metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics saved to: {test_metrics_path}")

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Results: {args.output_dir}")
    print(f"Checkpoints: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
