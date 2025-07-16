#!/usr/bin/env python3
"""
Basic usage example for DNA Transport GNN.

This script demonstrates how to use the DNA Transport GNN for predicting
transport properties of DNA sequences.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dna_transport_gnn import DNATransportGNN, DNATransportDataset, DNASequenceToGraph
from data_generator import create_sample_data, generate_realistic_dna_sequences
from utils import plot_dos_comparison, plot_transmission_comparison, calculate_metrics, print_metrics


def basic_example():
    """Basic example showing how to use the DNA Transport GNN."""
    
    print("=== DNA Transport GNN Basic Example ===\n")
    
    # 1. Generate sample data
    print("1. Generating sample data...")
    sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=100, seq_length=8, num_energy_points=50
    )
    
    print(f"   Generated {len(sequences)} sequences")
    print(f"   Sequence length: {len(sequences[0])}")
    print(f"   Energy points: {len(energy_grid)}")
    print(f"   Sample sequence: {sequences[0]}")
    
    # 2. Create dataset
    print("\n2. Creating dataset...")
    dataset = DNATransportDataset(sequences, dos_data, transmission_data, energy_grid)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 3. Initialize model
    print("\n3. Initializing model...")
    model = DNATransportGNN(
        node_features=8,
        edge_features=3,
        hidden_dim=64,  # Smaller for quick demo
        num_layers=2,
        num_heads=2,
        output_dim=50,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # 4. Test forward pass
    print("\n4. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        # Get a batch
        batch = next(iter(dataloader))
        
        # Forward pass
        dos_pred, trans_pred = model(batch)
        
        print(f"   Input batch size: {batch.num_graphs}")
        print(f"   DOS prediction shape: {dos_pred.shape}")
        print(f"   Transmission prediction shape: {trans_pred.shape}")
    
    # 5. Visualize results
    print("\n5. Visualizing results...")
    
    # Plot first sample
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # DOS comparison
    idx = 0
    # Get the first sample from the batch - handle batched data robustly
    # PyG batches custom attributes as a concatenated vector, not (batch, ...)
    # So batch.dos shape is (batch_size * num_energy_points,)
    # Instead, use dos_pred[idx] (which is always [num_energy_points]) and get the true from the dataset
    true_dos = dataset[idx].dos.detach().cpu().numpy()
    pred_dos = dos_pred[idx].detach().cpu().numpy()
    print(f"true_dos shape: {true_dos.shape}, pred_dos shape: {pred_dos.shape}, energy_grid shape: {energy_grid.shape}")
    
    ax1.plot(energy_grid, true_dos, 'r-', label='True DOS', linewidth=2)
    ax1.plot(energy_grid, pred_dos, 'b--', label='Predicted DOS', linewidth=2)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('DOS')
    ax1.set_title('Density of States')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Transmission comparison
    true_trans = dataset[idx].transmission.detach().cpu().numpy()
    pred_trans = trans_pred[idx].detach().cpu().numpy()
    print(f"true_trans shape: {true_trans.shape}, pred_trans shape: {pred_trans.shape}")
    
    ax2.plot(energy_grid, true_trans, 'r-', label='True Transmission', linewidth=2)
    ax2.plot(energy_grid, pred_trans, 'b--', label='Predicted Transmission', linewidth=2)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Transmission')
    ax2.set_title('Transmission Coefficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Sequence visualization
    sequence = sequences[idx]
    ax3.text(0.1, 0.5, f"Sequence: {sequence}", fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.3, f"Length: {len(sequence)}", fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.1, f"GC Content: {(sequence.count('G') + sequence.count('C'))/len(sequence):.2f}", 
             fontsize=12, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Sequence Information')
    ax3.axis('off')
    
    # Metrics
    dos_metrics = calculate_metrics(true_dos, pred_dos)
    trans_metrics = calculate_metrics(true_trans, pred_trans)
    
    metrics_text = f"DOS Metrics:\nR² = {dos_metrics['r2']:.3f}\nRMSE = {dos_metrics['rmse']:.3f}\n\n"
    metrics_text += f"Transmission Metrics:\nR² = {trans_metrics['r2']:.3f}\nRMSE = {trans_metrics['rmse']:.3f}"
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=10, transform=ax4.transAxes, 
             verticalalignment='center')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Prediction Metrics')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('basic_example_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Results saved to 'basic_example_results.png'")
    
    # 6. Print metrics
    print("\n6. Model Performance:")
    print_metrics(dos_metrics, "DOS ")
    print_metrics(trans_metrics, "Transmission ")


def sequence_to_graph_example():
    """Example showing how to convert DNA sequences to graphs."""
    
    print("\n=== Sequence to Graph Conversion Example ===\n")
    
    # Initialize converter
    converter = DNASequenceToGraph()
    
    # Test sequences
    test_sequences = ['ATGC', 'GCTA', 'ATCGATCG']
    
    for i, seq in enumerate(test_sequences):
        print(f"Sequence {i+1}: {seq}")
        
        # Convert to graph
        graph = converter.sequence_to_graph(primary_sequence=seq)
        
        print(f"  Nodes: {graph.x.shape[0]} (sequence: {len(seq)} + 2 contacts)")
        print(f"  Node features: {graph.x.shape[1]}")
        print(f"  Edges: {graph.edge_index.shape[1]}")
        print(f"  Edge features: {graph.edge_attr.shape[1]}")
        print()


def realistic_sequences_example():
    """Example with more realistic DNA sequences."""
    
    print("\n=== Realistic Sequences Example ===\n")
    
    # Generate realistic sequences
    sequences = generate_realistic_dna_sequences(
        num_samples=50, min_length=6, max_length=10
    )
    
    print(f"Generated {len(sequences)} realistic sequences")
    print("Sample sequences:")
    for i in range(min(5, len(sequences))):
        seq = sequences[i]
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        print(f"  {seq} (GC: {gc_content:.2f})")
    
    # Create dataset
    _, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=len(sequences), seq_length=8, num_energy_points=50
    )
    
    dataset = DNATransportDataset(sequences, dos_data, transmission_data, energy_grid)
    
    print(f"\nDataset created with {len(dataset)} samples")


if __name__ == "__main__":
    # Run examples
    basic_example()
    sequence_to_graph_example()
    realistic_sequences_example()
    
    print("\n=== Example completed successfully! ===")
    print("You can now run the full training with: python main.py") 