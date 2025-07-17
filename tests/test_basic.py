#!/usr/bin/env python3
"""
Basic test script for DNA Transport GNN.

This script tests the basic functionality of the DNA Transport GNN
to ensure everything is working correctly.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sequence_to_graph
from models import DNATransportGNN
from dataset import DNATransportDataset
from data_generator import create_sample_data


def test_sequence_to_graph():
    """Test DNA sequence to graph conversion."""
    print("Testing sequence to graph conversion...")
    
    # Test with a simple sequence
    sequence = "ATGC"
    graph = sequence_to_graph(primary_sequence=sequence)
    
    # Check that graph was created successfully
    if graph is None:
        raise AssertionError("Graph should not be None")
    
    # Check graph properties
    assert graph.x is not None and graph.x.shape[0] == len(sequence) + 2, f"Expected {len(sequence) + 2} nodes, got {graph.x.shape[0] if graph.x is not None else 'None'}"
    assert graph.x is not None and graph.x.shape[1] == 8, f"Expected 8 node features, got {graph.x.shape[1] if graph.x is not None else 'None'}"
    assert graph.edge_attr is not None and graph.edge_attr.shape[1] == 4, f"Expected 4 edge features, got {graph.edge_attr.shape[1] if graph.edge_attr is not None else 'None'}"
    
    print("✓ Sequence to graph conversion works correctly")


def test_model_initialization():
    """Test model initialization and forward pass."""
    print("Testing model initialization...")
    
    model = DNATransportGNN(
        node_features=8,
        edge_features=4,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        output_dim=50,
        dropout=0.1
    )
    
    # Test forward pass with dummy data
    graph = sequence_to_graph(primary_sequence="ATGC")
    
    # Add batch dimension
    if graph.x is not None:
        graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    else:
        raise AssertionError("Graph should have valid node features")
    
    model.eval()
    with torch.no_grad():
        dos_pred, trans_pred = model(graph)
        
        assert dos_pred.shape[0] == 1, f"Expected batch size 1, got {dos_pred.shape[0]}"
        assert dos_pred.shape[1] == 50, f"Expected 50 output features, got {dos_pred.shape[1]}"
        assert trans_pred.shape == dos_pred.shape, "DOS and transmission should have same shape"
        assert torch.all(trans_pred >= 0) and torch.all(trans_pred <= 1), "Transmission should be in [0,1]"
    
    print("✓ Model initialization and forward pass work correctly")


def test_dataset():
    """Test dataset creation and loading."""
    print("Testing dataset creation...")
    
    # Generate sample data
    sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=10, seq_length=6, num_energy_points=30
    )
    
    # Create dataset
    dataset = DNATransportDataset(sequences, dos_data, transmission_data, energy_grid)
    
    # Test dataset properties
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
    
    # Test getting an item
    sample = dataset[0]
    assert hasattr(sample, 'x'), "Sample should have node features"
    assert hasattr(sample, 'edge_index'), "Sample should have edge indices"
    assert hasattr(sample, 'edge_attr'), "Sample should have edge attributes"
    assert hasattr(sample, 'dos'), "Sample should have DOS data"
    assert hasattr(sample, 'transmission'), "Sample should have transmission data"
    
    print("✓ Dataset creation and loading work correctly")


def test_data_generation():
    """Test synthetic data generation."""
    print("Testing data generation...")
    
    sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=5, seq_length=4, num_energy_points=20
    )
    
    # Check data properties
    assert len(sequences) == 5, f"Expected 5 sequences, got {len(sequences)}"
    assert dos_data.shape == (5, 20), f"Expected DOS shape (5, 20), got {dos_data.shape}"
    assert transmission_data.shape == (5, 20), f"Expected transmission shape (5, 20), got {transmission_data.shape}"
    assert len(energy_grid) == 20, f"Expected 20 energy points, got {len(energy_grid)}"
    
    # Check sequence properties
    for seq in sequences:
        assert len(seq) == 4, f"Expected sequence length 4, got {len(seq)}"
        assert all(base in 'ATGC' for base in seq), f"Invalid bases in sequence {seq}"
    
    # Check transmission bounds
    assert np.all(transmission_data >= 0) and np.all(transmission_data <= 1), "Transmission should be in [0,1]"
    
    print("✓ Data generation works correctly")


def run_all_tests():
    """Run all tests."""
    print("Running DNA Transport GNN tests...\n")
    
    try:
        test_sequence_to_graph()
        test_model_initialization()
        test_dataset()
        test_data_generation()
        
        print("\n🎉 All tests passed successfully!")
        print("The DNA Transport GNN is ready to use.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 