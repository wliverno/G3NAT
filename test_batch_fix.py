#!/usr/bin/env python3
"""
Test script to verify the custom collate function fixes the batching issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import sequence_to_graph
from data_generator import create_sample_data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch

def custom_collate(batch):
    """Custom collate function to properly handle target data batching."""
    # Separate graphs and targets
    graphs = []
    dos_targets = []
    transmission_targets = []
    
    for data in batch:
        graphs.append(data)
        dos_targets.append(data.dos)
        transmission_targets.append(data.transmission)
    
    # Create batch from graphs
    batch_data = Batch.from_data_list(graphs)
    
    # Stack targets properly
    batch_data.dos = torch.stack(dos_targets)
    batch_data.transmission = torch.stack(transmission_targets)
    
    return batch_data

def test_batching():
    """Test that batching works correctly with custom collate function."""
    print("Testing custom collate function...")
    
    # Generate a small dataset
    sequences, dos_data, transmission_data, energy_grid = create_sample_data(
        num_samples=10, 
        seq_length=6, 
        num_energy_points=20
    )
    
    # Create graphs
    graphs = []
    for i, seq in enumerate(sequences):
        graph = sequence_to_graph(
            primary_sequence=seq,
            left_contact_positions=0,
            right_contact_positions=len(seq)-1,
            left_contact_coupling=0.1,
            right_contact_coupling=0.2
        )
        
        if graph is not None:
            graph.dos = torch.tensor(dos_data[i], dtype=torch.float32)
            graph.transmission = torch.tensor(transmission_data[i], dtype=torch.float32)
            graphs.append(graph)
    
    print(f"Created {len(graphs)} graphs")
    
    # Test batching with custom collate
    loader = DataLoader(graphs, batch_size=4, shuffle=False, collate_fn=custom_collate)
    
    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Batch size: {batch.num_graphs}")
        print(f"  DOS shape: {batch.dos.shape}")
        print(f"  Transmission shape: {batch.transmission.shape}")
        print(f"  Expected DOS shape: [4, 20]")
        print(f"  Expected Transmission shape: [4, 20]")
        
        # Verify shapes are correct
        assert batch.dos.shape == (4, 20), f"DOS shape {batch.dos.shape} != (4, 20)"
        assert batch.transmission.shape == (4, 20), f"Transmission shape {batch.transmission.shape} != (4, 20)"
        
        print("  ✓ Shapes are correct!")
    
    print("\n✓ All batching tests passed!")

if __name__ == "__main__":
    test_batching() 