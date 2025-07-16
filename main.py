#!/usr/bin/env python3
"""
Main training script for DNA Transport GNN.

This script demonstrates the complete training pipeline for predicting
DNA transport properties using Graph Neural Networks.
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Tuple, List

from models import DNATransportGNN, train_model
from dataset import DNATransportDataset
from data_generator import create_sample_data, generate_realistic_dna_sequences
from utils import setup_logging, save_training_results, plot_training_curves


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DNA Transport GNN')
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of training samples')
    parser.add_argument('--seq_length', type=int, default=8,
                       help='Length of DNA sequences')
    parser.add_argument('--num_energy_points', type=int, default=100,
                       help='Number of energy points for DOS/transmission')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='dna_transport_model',
                       help='Name for saved model')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup the device for training."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_data_loaders(sequences: List[str], dos_data: np.ndarray, 
                       transmission_data: np.ndarray, energy_grid: np.ndarray,
                       batch_size: int = 32, train_split: float = 0.8):
    """Create training and validation data loaders."""
    
    # Create dataset
    dataset = DNATransportDataset(sequences, dos_data, transmission_data, energy_grid)
    
    # Train/validation split
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def initialize_model(args) -> DNATransportGNN:
    """Initialize the DNA Transport GNN model."""
    
    model = DNATransportGNN(
        node_features=8,
        edge_features=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        output_dim=args.num_energy_points,
        dropout=args.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Number of heads: {args.num_heads}")
    
    return model


def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging and output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("Starting DNA Transport GNN training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup device
    device = setup_device(args.device)
    
    # Generate sample data
    logger.info("Generating sample data...")
    sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        num_energy_points=args.num_energy_points
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        sequences, dos_data, transmission_data, energy_grid,
        batch_size=args.batch_size
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = initialize_model(args)
    
    # Train model
    logger.info("Starting training...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=str(device)
    )
    
    # Save model and results
    logger.info("Saving results...")
    model_path = os.path.join(args.output_dir, f"{args.model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'energy_grid': energy_grid
    }, model_path)
    
    # Save training results
    save_training_results(
        train_losses, val_losses, energy_grid,
        os.path.join(args.output_dir, 'training_results.npz')
    )
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, plot_path)
    
    # Test prediction on a sample
    logger.info("Generating sample predictions...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        sample_batch = sample_batch.to(device)
        dos_pred, trans_pred = model(sample_batch)
        
        # Plot first sample
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # DOS plot
        idx = 0
        ax1.plot(energy_grid, sample_batch.dos[idx].cpu().numpy(), 'r-', label='True DOS', linewidth=2)
        ax1.plot(energy_grid, dos_pred[idx].cpu().numpy(), 'b--', label='Predicted DOS', linewidth=2)
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('DOS')
        ax1.legend()
        ax1.set_title('Density of States Prediction')
        ax1.grid(True, alpha=0.3)
        
        # Transmission plot
        ax2.plot(energy_grid, sample_batch.transmission[idx].cpu().numpy(), 'r-', label='True Transmission', linewidth=2)
        ax2.plot(energy_grid, trans_pred[idx].cpu().numpy(), 'b--', label='Predicted Transmission', linewidth=2)
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Transmission')
        ax2.legend()
        ax2.set_title('Transmission Coefficient Prediction')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 