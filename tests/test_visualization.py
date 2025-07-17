#!/usr/bin/env python3
"""
Test script to verify the visualization works with the new node indexing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sequence_to_graph
from visualize_dna_graph import visualize_dna_graph
import matplotlib.pyplot as plt

def test_simple_visualization():
    """Test visualization with a simple case."""
    print("=== Testing Simple Visualization ===")
    print("Primary: ACGC")
    print("Complementary: GCGT")
    
    graph = sequence_to_graph(
        primary_sequence="ACGC",
        complementary_sequence="GCGT",
        left_contact_positions=('primary', [0]),
        right_contact_positions=('complementary', [0]),
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )
    
    print(f"✓ Graph created successfully!")
    print(f"  - Nodes: {graph.x.shape[0]}")
    print(f"  - Edges: {graph.edge_index.shape[1]}")
    
    # Visualize
    fig, ax = visualize_dna_graph(graph, "ACGC", "GCGT")
    plt.show()
    
    return fig, ax

def test_with_blanks():
    """Test visualization with blank bases."""
    print("\n=== Testing Visualization with Blanks ===")
    print("Primary: ACGC__")
    print("Complementary: TGCG__")
    
    graph = sequence_to_graph(
        primary_sequence="ACGAC__",
        complementary_sequence="TGCG__",
        left_contact_positions=('primary', [0,1]),
        right_contact_positions=('complementary', [0,1]),
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )
    
    print(f"✓ Graph created successfully!")
    print(f"  - Nodes: {graph.x.shape[0]}")
    print(f"  - Edges: {graph.edge_index.shape[1]}")
    
    # Visualize
    fig, ax = visualize_dna_graph(graph, "ACGAC__", "TGCG__")
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    test_simple_visualization()
    test_with_blanks() 