#!/usr/bin/env python3
"""
Basic usage example for the DNA Transport GNN with left/right contacts.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dna_graph import sequence_to_graph
from models import DNATransportGNN
from torch_geometric.data import Data
import torch
import numpy as np

def example_single_contact():
    """Example with single contact on each side."""
    print("=== Single Contact Example ===")
    
    sequence = "ACGTACGT"
    
    # Simple case: left contact at first position, right contact at last position
    graph = sequence_to_graph(
        primary_sequence=sequence,
        left_contact_positions=0,  # Connect to first base (A)
        right_contact_positions=7,  # Connect to last base (T)
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    if graph is None:
        raise RuntimeError("Failed to create graph")
    
    print(f"Sequence: {sequence}")
    print(f"Nodes: {graph.x.shape[0]} (1 left contact + {len(sequence)} bases + 1 right contact)")
    print(f"Edges: {graph.edge_index.shape[1]} (4 contact + {len(sequence)-1} backbone connections)")
    
    # Show the contact connections
    print("\nContact connections:")
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 0].item()
        edge_type = graph.edge_attr[i, 1].item()
        if edge_type == 0:  # Contact connection
            if src == 0:  # Left contact
                print(f"  Left contact -> Base {dst-1} (coupling: {coupling:.3f})")
            elif dst == 0:  # To left contact
                print(f"  Base {src-1} -> Left contact (coupling: {coupling:.3f})")
            elif src == graph.x.shape[0] - 1:  # Right contact
                print(f"  Right contact -> Base {dst-1} (coupling: {coupling:.3f})")
            elif dst == graph.x.shape[0] - 1:  # To right contact
                print(f"  Base {src-1} -> Right contact (coupling: {coupling:.3f})")
    
    # Show backbone connections
    print("\nBackbone connections:")
    backbone_edges = 0
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 0].item()
        edge_type = graph.edge_attr[i, 1].item()
        if edge_type == 1:  # Backbone connection
            if src < dst:  # Only show one direction to avoid duplicates
                print(f"  Base {src-1} <-> Base {dst-1} (coupling: {coupling:.3f})")
                backbone_edges += 1
    print(f"  Total backbone connections: {backbone_edges}")

def example_multiple_contacts():
    """Example with multiple contacts on each side."""
    print("\n=== Multiple Contacts Example ===")
    
    sequence = "ACGTACGT"
    
    # Multiple contacts: left contacts at positions 0 and 2, right contacts at positions 5 and 7
    graph = sequence_to_graph(
        primary_sequence=sequence,
        left_contact_positions=[0, 2],  # Connect to first and third bases
        right_contact_positions=[5, 7],  # Connect to sixth and eighth bases
        left_contact_coupling=[0.1, 0.15],  # Different coupling strengths
        right_contact_coupling=[0.2, 0.25]
    )
    
    if graph is None:
        raise RuntimeError("Failed to create graph")
    
    print(f"Sequence: {sequence}")
    print(f"Left contacts at positions: [0, 2]")
    print(f"Right contacts at positions: [5, 7]")
    print(f"Nodes: {graph.x.shape[0]}")
    print(f"Edges: {graph.edge_index.shape[1]} (8 bidirectional contact connections)")
    
    # Show the contact connections
    print("\nContact connections:")
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 0].item()
        if src == 0:  # Left contact
            print(f"  Left contact -> Base {dst-1} (coupling: {coupling:.3f})")
        elif dst == 0:  # To left contact
            print(f"  Base {src-1} -> Left contact (coupling: {coupling:.3f})")
        elif src == graph.x.shape[0] - 1:  # Right contact
            print(f"  Right contact -> Base {dst-1} (coupling: {coupling:.3f})")
        elif dst == graph.x.shape[0] - 1:  # To right contact
            print(f"  Base {src-1} -> Right contact (coupling: {coupling:.3f})")

def example_double_stranded():
    """Example with double-stranded DNA."""
    print("\n=== Double-stranded DNA Example ===")
    
    primary = "ACGTACGT"
    complementary = "TGCATGCA"  # Full complementary strand
    
    # Double-stranded DNA with contacts on both strands
    graph = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=0,  # Primary strand, position 0
        right_contact_positions=('complementary', 7),  # Complementary strand, position 7
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    if graph is None:
        raise RuntimeError("Failed to create graph")
    
    print(f"Primary sequence: {primary}")
    print(f"Complementary sequence: {complementary}")
    print(f"Left contact: Primary strand, position 0")
    print(f"Right contact: Complementary strand, position 7")
    print(f"Nodes: {graph.x.shape[0]} (1 left contact + {len(primary)} primary + {len(complementary)} complementary + 1 right contact)")
    print(f"Edges: {graph.edge_index.shape[1]} (4 contact + backbone + hydrogen bond connections)")
    
    # Show different types of connections
    contact_edges = 0
    backbone_edges = 0
    hbond_edges = 0
    
    for i in range(graph.edge_index.shape[1]):
        edge_type = graph.edge_attr[i, 1].item()
        if edge_type == 0:  # Contact
            contact_edges += 1
        elif edge_type == 1:  # Backbone
            backbone_edges += 1
        elif edge_type == 2:  # Hydrogen bond
            hbond_edges += 1
    
    print(f"\nConnection types:")
    print(f"  Contact connections: {contact_edges}")
    print(f"  Backbone connections: {backbone_edges}")
    print(f"  Hydrogen bond connections: {hbond_edges}")
    
    # Show contact connections
    print("\nContact connections:")
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 0].item()
        edge_type = graph.edge_attr[i, 1].item()
        if edge_type == 0:  # Contact connection
            if src == 0:  # Left contact
                if dst <= len(primary):
                    print(f"  Left contact -> Primary base {dst-1} (coupling: {coupling:.3f})")
                else:
                    print(f"  Left contact -> Complementary base {dst-len(primary)-1} (coupling: {coupling:.3f})")
            elif src == graph.x.shape[0] - 1:  # Right contact
                if dst <= len(primary):
                    print(f"  Right contact -> Primary base {dst-1} (coupling: {coupling:.3f})")
                else:
                    print(f"  Right contact -> Complementary base {dst-len(primary)-1} (coupling: {coupling:.3f})")

def example_model_usage():
    """Example of using the graph with the GNN model."""
    print("\n=== Model Usage Example ===")
    
    # Create a simple graph
    sequence = "ACGT"
    graph = sequence_to_graph(
        primary_sequence=sequence,
        left_contact_positions=0,
        right_contact_positions=3,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    # Create a batch (single graph)
    from torch_geometric.loader import DataLoader
    loader = DataLoader([graph], batch_size=1)
    
    # Initialize model
    model = DNATransportGNN(
        node_features=8,  # Base features + contact features
        edge_features=3,  # Edge attributes
        hidden_dim=64,
        num_layers=2,
        output_dim=50
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        for batch in loader:
            dos_pred, transmission_pred = model(batch)
            print(f"DOS prediction shape: {dos_pred.shape}")
            print(f"Transmission prediction shape: {transmission_pred.shape}")
            print(f"DOS range: [{dos_pred.min().item():.3f}, {dos_pred.max().item():.3f}]")
            print(f"Transmission range: [{transmission_pred.min().item():.3f}, {transmission_pred.max().item():.3f}]")

if __name__ == "__main__":
    print("DNA Transport GNN - Left/Right Contact Interface Examples")
    print("=" * 60)
    
    example_single_contact()
    example_multiple_contacts()
    example_double_stranded()
    example_model_usage()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!") 