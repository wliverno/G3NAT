#!/usr/bin/env python3
"""
Main training script for DNA Transport GNN.

This script demonstrates the complete training pipeline for predicting
DNA transport properties using Graph Neural Networks.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Tuple, List

from models import DNATransportGNN, train_model_with_custom_batching
from data_generator import create_sample_data
from dataset import sequence_to_graph


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


def create_graphs_from_sequences(sequences: List[str], dos_data: np.ndarray, 
                                transmission_data: np.ndarray) -> List:
    """Convert DNA sequences to graph representations."""
    print("Converting sequences to graphs...")
    graphs = []
    failed_conversions = 0
    
    for i, seq in enumerate(sequences):
        # Create graph with contacts at ends
        graph = sequence_to_graph(
            primary_sequence=seq,
            left_contact_positions=0,
            right_contact_positions=len(seq)-1,
            left_contact_coupling=0.1,
            right_contact_coupling=0.2
        )
        
        if graph is not None:
            # Add target data to graph
            graph.dos = torch.tensor(dos_data[i], dtype=torch.float32)
            graph.transmission = torch.tensor(transmission_data[i], dtype=torch.float32)
            graphs.append(graph)
        else:
            failed_conversions += 1
    
    print(f"Successfully created {len(graphs)} graphs")
    print(f"Failed conversions: {failed_conversions}")
    
    return graphs


def split_graphs(graphs: List, train_split: float = 0.8):
    """Split graphs into training and validation sets."""
    train_size = int(train_split * len(graphs))
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    print(f"Training set: {len(train_graphs)} graphs")
    print(f"Validation set: {len(val_graphs)} graphs")
    
    return train_graphs, val_graphs


def initialize_model(args) -> DNATransportGNN:
    """Initialize the DNA Transport GNN model."""
    
    model = DNATransportGNN(
        node_features=8,
        edge_features=4,
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


def save_training_results(train_losses, val_losses, energy_grid, output_dir):
    """Save training results to files."""
    # Save loss curves
    np.savez(os.path.join(output_dir, 'training_results.npz'),
             train_losses=train_losses,
             val_losses=val_losses,
             energy_grid=energy_grid)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training results saved to {output_dir}")


def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting DNA Transport GNN training")
    print(f"Arguments: {vars(args)}")
    
    # Setup device
    device = setup_device(args.device)
    
    # Generate sample data
    print("Generating sample data...")
    sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        num_energy_points=args.num_energy_points
    )
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    
    # Convert sequences to graphs
    graphs = create_graphs_from_sequences(sequences, dos_data, transmission_data)
    
    # Split into train/validation sets
    train_graphs, val_graphs = split_graphs(graphs)
    
    # Initialize model
    print("Initializing model...")
    model = initialize_model(args)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model_with_custom_batching(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=str(device)
    )
    
    # Save model and results
    print("Saving results...")
    model_path = os.path.join(args.output_dir, f"{args.model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'energy_grid': energy_grid
    }, model_path)
    
    # Save training results
    save_training_results(train_losses, val_losses, energy_grid, args.output_dir)
    
    # Test prediction on a sample
    print("Generating sample predictions...")
    model.eval()
    with torch.no_grad():
        # Test on first validation sample
        if len(val_graphs) > 0:
            from torch_geometric.data import Batch
            
            # Create single graph batch
            batch_data = Batch.from_data_list([val_graphs[0]])
            dos_target = torch.stack([val_graphs[0].dos])
            transmission_target = torch.stack([val_graphs[0].transmission])
            
            batch_data = batch_data.to(device)
            dos_target = dos_target.to(device)
            transmission_target = transmission_target.to(device)
            
            dos_pred, trans_pred = model(batch_data)
            
            # Plot first sample
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # DOS plot
            ax1.plot(energy_grid, dos_target[0].cpu().numpy(), 'r-', label='True DOS', linewidth=2)
            ax1.plot(energy_grid, dos_pred[0].cpu().numpy(), 'b--', label='Predicted DOS', linewidth=2)
            ax1.set_xlabel('Energy (eV)')
            ax1.set_ylabel('DOS')
            ax1.legend()
            ax1.set_title('Density of States Prediction')
            ax1.grid(True, alpha=0.3)
            
            # Transmission plot
            ax2.plot(energy_grid, transmission_target[0].cpu().numpy(), 'r-', label='True Transmission', linewidth=2)
            ax2.plot(energy_grid, trans_pred[0].cpu().numpy(), 'b--', label='Predicted Transmission', linewidth=2)
            ax2.set_xlabel('Energy (eV)')
            ax2.set_ylabel('Transmission')
            ax2.legend()
            ax2.set_title('Transmission Coefficient Prediction')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'sample_predictions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate MSE
            dos_mse = torch.nn.functional.mse_loss(dos_pred, dos_target).item()
            transmission_mse = torch.nn.functional.mse_loss(trans_pred, transmission_target).item()
            
            print(f"Sample prediction MSE:")
            print(f"  DOS: {dos_mse:.4f}")
            print(f"  Transmission: {transmission_mse:.4f}")
    
    print(f"Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 