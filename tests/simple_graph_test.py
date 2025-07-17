#!/usr/bin/env python3
"""
Simple test to demonstrate the new left/right contact interface.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sequence_to_graph
from torch_geometric.data import Data
import torch

def test_simple_contacts():
    """Test basic left/right contact functionality."""
        # Simple case: left contact at position 0, right contact at position 4
    sequence = "ACGTA"
    graph: Data = sequence_to_graph(
        primary_sequence=sequence,
        left_contact_positions=0,
        right_contact_positions=4,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    print(f"Sequence: {sequence}")
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")
    print(f"Edge attributes shape: {graph.edge_attr.shape}")
    print(f"Node features shape: {graph.x.shape}")
    
    # Print edge connections
    print("\nEdge connections:")
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 0].item()
        print(f"  {src} -> {dst} (coupling: {coupling:.3f})")

def test_multiple_contacts():
    """Test multiple contact positions per side."""
        # Multiple contacts: left contacts at positions 0 and 1, right contacts at positions 3 and 4
    sequence = "ACGTA"
    graph: Data = sequence_to_graph(
        primary_sequence=sequence,
        left_left_contact_positions=0, 1[0], right_contact_positions=0, 1[-1],
        right_left_contact_positions=3, 4[0], right_contact_positions=3, 4[-1],
        left_contact_coupling=[0.1, 0.15],
        right_contact_coupling=[0.2, 0.25]
    )
    
    print(f"\nSequence: {sequence}")
    print(f"Left contacts at positions: [0, 1]")
    print(f"Right contacts at positions: [3, 4]")
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")
    
    # Print edge connections
    print("\nEdge connections:")
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 0].item()
        print(f"  {src} -> {dst} (coupling: {coupling:.3f})")

def test_double_stranded():
    """Test with complementary strand."""
        # Double-stranded DNA with contacts
    primary = "ACGTA"
    complementary = "TGCAT"  # Full complementary strand
    graph: Data = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=0,
        right_contact_positions=4,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    print(f"\nPrimary sequence: {primary}")
    print(f"Complementary sequence: {complementary}")
    print(f"Number of nodes: {graph.x.shape[0]}")
    print(f"Number of edges: {graph.edge_index.shape[1]}")
    
    # Print edge connections
    print("\nEdge connections:")
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 0].item()
        print(f"  {src} -> {dst} (coupling: {coupling:.3f})")

if __name__ == "__main__":
    print("Testing new left/right contact interface...")
    test_simple_contacts()
    test_multiple_contacts()
    test_double_stranded()
    print("\nAll tests completed!") 