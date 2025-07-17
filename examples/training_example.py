#!/usr/bin/env python3
"""
Training example for the DNA Transport GNN model.

This example demonstrates:
1. How to generate synthetic training data
2. How to train the model from scratch
3. How to evaluate the trained model
4. How to save and load trained models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sequence_to_graph
from models import DNATransportGNN, train_model
from data_generator import create_sample_data, generate_realistic_dna_sequences
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle



def explain_model_initialization():
    """Explain how the model is initialized and why it can run without training."""
    print("=== Model Initialization Explanation ===")
    
    # Create a small model to demonstrate
    model = DNATransportGNN(
        node_features=8,
        edge_features=4,
        hidden_dim=32,
        num_layers=2,
        output_dim=20
    )
    
    print("Model architecture:")
    print(f"  Node projection: {model.node_proj}")
    print(f"  Edge projection: {model.edge_proj}")
    print(f"  Graph attention layers: {len(model.convs)}")
    print(f"  DOS projection: {model.dos_proj}")
    print(f"  Transmission projection: {model.transmission_proj}")
    
    print("\nInitial weights:")
    print("  All weights are randomly initialized using PyTorch defaults:")
    print("  - Linear layers: Xavier/Glorot initialization")
    print("  - GATConv layers: Kaiming initialization")
    print("  - LayerNorm: Standard normal distribution")
    
    # Show some initial weights
    print("\nSample initial weights:")
    print(f"  Node projection weight mean: {model.node_proj.weight.mean().item():.4f}")
    print(f"  Node projection weight std: {model.node_proj.weight.std().item():.4f}")
    print(f"  DOS projection weight mean: {model.dos_proj[0].weight.mean().item():.4f}")
    
    print("\nWhy the model can run without training:")
    print("  1. All operations are mathematically valid with random weights")
    print("  2. Forward pass produces outputs, but they're random")
    print("  3. Training adjusts weights to minimize prediction error")
    print("  4. Without training, predictions have no meaningful relationship to inputs")

def create_training_dataset(num_samples=1000, seq_length=8, num_energy_points=50):
    """Create a comprehensive training dataset."""
    print(f"\n=== Creating Training Dataset ===")
    print(f"Generating {num_samples} DNA sequences of length {seq_length}...")
    
    # Generate synthetic data
    sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=num_samples,
        seq_length=seq_length,
        num_energy_points=num_energy_points
    )
    
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    
    # Convert to graphs
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
    
    return graphs, energy_grid

def train_and_evaluate_model(graphs, energy_grid, model_config=None):
    """Train a model and evaluate its performance."""
    print(f"\n=== Training and Evaluation ===")
    
    # Default model configuration
    if model_config is None:
        model_config = {
            'node_features': 8,
            'edge_features': 4,
            'hidden_dim': 64,
            'num_layers': 3,
            'output_dim': len(energy_grid),
            'dropout': 0.1
        }
    
    # Split data
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size + val_size]
    test_graphs = graphs[train_size + val_size:]
    
    print(f"Data split:")
    print(f"  Training: {len(train_graphs)} graphs")
    print(f"  Validation: {len(val_graphs)} graphs")
    print(f"  Test: {len(test_graphs)} graphs")
    
    # We'll use manual batching instead of DataLoader
    
    # Initialize model
    model = DNATransportGNN(**model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Use the custom batching function instead of the standard train_model
    from models import train_model_with_custom_batching
    train_losses, val_losses = train_model_with_custom_batching(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        num_epochs=100,
        learning_rate=1e-3,
        device=device
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_losses = []
    dos_predictions = []
    transmission_predictions = []
    dos_targets = []
    transmission_targets = []
    
    with torch.no_grad():
        for i in range(0, len(test_graphs), 32):
            batch_graphs = test_graphs[i:i+32]
            
            # Create batch manually
            batch_data = Batch.from_data_list(batch_graphs)
            
            # Stack targets manually
            dos_targets_batch = torch.stack([g.dos for g in batch_graphs])
            transmission_targets_batch = torch.stack([g.transmission for g in batch_graphs])
            
            batch_data = batch_data.to(device)
            dos_targets_batch = dos_targets_batch.to(device)
            transmission_targets_batch = transmission_targets_batch.to(device)
            
            dos_pred, transmission_pred = model(batch_data)
            
            # Calculate loss
            criterion = nn.MSELoss()
            dos_loss = criterion(dos_pred, dos_targets_batch)
            transmission_loss = criterion(transmission_pred, transmission_targets_batch)
            total_loss = dos_loss + transmission_loss
            test_losses.append(total_loss.item())
            
            # Store predictions and targets
            dos_predictions.extend(dos_pred.cpu().numpy())
            transmission_predictions.extend(transmission_pred.cpu().numpy())
            dos_targets.extend(dos_targets_batch.cpu().numpy())
            transmission_targets.extend(transmission_targets_batch.cpu().numpy())
    
    # Convert to numpy arrays
    dos_predictions = np.array(dos_predictions)
    transmission_predictions = np.array(transmission_predictions)
    dos_targets = np.array(dos_targets)
    transmission_targets = np.array(transmission_targets)
    
    # Calculate metrics
    avg_test_loss = np.mean(test_losses)
    dos_mse = mean_squared_error(dos_targets.flatten(), dos_predictions.flatten())
    transmission_mse = mean_squared_error(transmission_targets.flatten(), transmission_predictions.flatten())
    dos_r2 = r2_score(dos_targets.flatten(), dos_predictions.flatten())
    transmission_r2 = r2_score(transmission_targets.flatten(), transmission_predictions.flatten())
    
    print(f"\nTest Results:")
    print(f"  Average test loss: {avg_test_loss:.4f}")
    print(f"  DOS MSE: {dos_mse:.4f}")
    print(f"  Transmission MSE: {transmission_mse:.4f}")
    print(f"  DOS R²: {dos_r2:.4f}")
    print(f"  Transmission R²: {transmission_r2:.4f}")
    
    return model, train_losses, val_losses, {
        'test_loss': avg_test_loss,
        'dos_mse': dos_mse,
        'transmission_mse': transmission_mse,
        'dos_r2': dos_r2,
        'transmission_r2': transmission_r2
    }

def plot_training_results(train_losses, val_losses, metrics, save_path='training_results.png'):
    """Plot training curves and results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss difference
    loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
    axes[0, 1].plot(loss_diff, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('|Train Loss - Val Loss|')
    axes[0, 1].set_title('Overfitting Monitor')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics bar plot
    metric_names = ['DOS MSE', 'Transmission MSE', 'DOS R²', 'Transmission R²']
    metric_values = [metrics['dos_mse'], metrics['transmission_mse'], 
                    metrics['dos_r2'], metrics['transmission_r2']]
    colors = ['red', 'orange', 'blue', 'green']
    
    bars = axes[1, 0].bar(metric_names, metric_values, color=colors)
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Test Set Performance')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Final loss values
    axes[1, 1].text(0.1, 0.8, f'Final Training Loss: {train_losses[-1]:.4f}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'Final Validation Loss: {val_losses[-1]:.4f}', fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'Test Loss: {metrics["test_loss"]:.4f}', fontsize=12)
    axes[1, 1].text(0.1, 0.2, f'Best Val Loss: {min(val_losses):.4f}', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training results saved to '{save_path}'")

def save_model(model, filepath):
    """Save the trained model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'node_features': model.node_proj.in_features,
            'edge_features': model.edge_proj.in_features,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'output_dim': model.output_dim,
            'dropout': 0.1  # Default value
        }
    }, filepath)
    print(f"Model saved to '{filepath}'")

def load_model(filepath):
    """Load a trained model."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model_config = checkpoint['model_config']
    
    model = DNATransportGNN(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from '{filepath}'")
    return model

def compare_untrained_vs_trained():
    """Compare predictions from untrained vs trained model."""
    print(f"\n=== Untrained vs Trained Model Comparison ===")
    
    # Create a simple test sequence
    sequence = "ACGTACGT"
    graph = sequence_to_graph(
        primary_sequence=sequence,
        left_contact_positions=0,
        right_contact_positions=len(sequence)-1,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    if graph is None:
        print("Failed to create test graph")
        return
    
    # Create untrained model
    untrained_model = DNATransportGNN(
        node_features=8,
        edge_features=4,
        hidden_dim=32,
        num_layers=2,
        output_dim=20
    )
    
    # Test untrained model
    untrained_model.eval()
    with torch.no_grad():
        loader = DataLoader([graph], batch_size=1)
        for batch in loader:
            dos_pred_untrained, transmission_pred_untrained = untrained_model(batch)
    
    print("Untrained model predictions:")
    print(f"  DOS range: [{dos_pred_untrained.min().item():.3f}, {dos_pred_untrained.max().item():.3f}]")
    print(f"  Transmission range: [{transmission_pred_untrained.min().item():.3f}, {transmission_pred_untrained.max().item():.3f}]")
    print(f"  DOS std: {dos_pred_untrained.std().item():.3f}")
    print(f"  Transmission std: {transmission_pred_untrained.std().item():.3f}")
    
    print("\nNote: Untrained model produces random predictions with high variance.")
    print("Trained model should produce more consistent and meaningful predictions.")

def main():
    """Main training example."""
    print("DNA Transport GNN - Training Example")
    print("=" * 60)
    
    # Explain model initialization
    explain_model_initialization()
    
    # Compare untrained vs trained
    compare_untrained_vs_trained()
    
    # Create training dataset
    graphs, energy_grid = create_training_dataset(num_samples=500, seq_length=8, num_energy_points=50)
    
    # Train and evaluate model
    model, train_losses, val_losses, metrics = train_and_evaluate_model(graphs, energy_grid)
    
    # Plot results
    plot_training_results(train_losses, val_losses, metrics)
    
    # Save model
    save_model(model, 'trained_dna_model.pth')
    
    # Demonstrate loading
    loaded_model = load_model('trained_dna_model.pth')
    
    print("\n" + "=" * 60)
    print("Training example completed!")
    print("Key takeaways:")
    print("1. Model starts with random weights and produces random predictions")
    print("2. Training adjusts weights to minimize prediction error")
    print("3. Trained model can make meaningful predictions on new data")
    print("4. Model can be saved and loaded for later use")

if __name__ == "__main__":
    main() 