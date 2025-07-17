#!/usr/bin/env python3
"""
Test script to demonstrate the edge case fix for contact positioning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sequence_to_graph

def test_edge_case_fix():
    """Test that the edge case is now properly handled."""
        print("=== Testing Edge Case Fix ===")
    print("Primary sequence: 'ACGC__' (length: 5, positions 0-4)")
    print("Complementary sequence: 'AAGC__' (length: 5, positions 0-4)")
    print("Note: Both sequences have '_' at positions 4, so only positions 0-3 have actual bases")
    
    # Test 1: Valid contact positions (only on actual bases)
    print("\n--- Test 1: Valid contact positions ---")
    try:
        seq = sequence_to_graph(
            primary_sequence='ACGC__', 
            complementary_sequence='AAGC__', 
            left_contact_positions=('primary', [0, 1]), 
            right_contact_positions=('complementary', [2, 3]),   # Only on actual bases
            left_contact_coupling=0.1, 
            right_contact_coupling=0.1
        )
        print("✓ Success!")
        print(f"  - Nodes: {seq.x.shape[0]}")
        print(f"  - Edges: {seq.edge_index.shape[1]}")
        
        # Count different types of edges
        edge_attr = seq.edge_attr.numpy()
        contact_edges = sum(1 for edge in edge_attr if edge[1] == 0 and edge[2] == 0)
        backbone_edges = sum(1 for edge in edge_attr if edge[1] == 1)
        h_bond_edges = sum(1 for edge in edge_attr if edge[2] == 1)
        
        print(f"  - Contact edges: {contact_edges}")
        print(f"  - Backbone edges: {backbone_edges}")
        print(f"  - Hydrogen bond edges: {h_bond_edges}")
        
    except ValueError as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Invalid contact positions (on blank bases)
    print("\n--- Test 2: Invalid contact positions (on blank bases) ---")
    try:
        seq = sequence_to_graph(
            primary_sequence='ACGC__', 
            complementary_sequence='AAGC__', 
            left_contact_positions=('primary', [0, 1]), 
            right_contact_positions=('complementary', [3, 4]),   # Position 4 is '_'
            left_contact_coupling=0.1, 
            right_contact_coupling=0.1
        )
        print("✓ Success! (This shouldn't happen)")
    except ValueError as e:
        print(f"✗ Expected error: {e}")
    
    # Test 3: Invalid contact positions (out of range)
    print("\n--- Test 3: Invalid contact positions (out of range) ---")
    try:
        seq = sequence_to_graph(
            primary_sequence='ACGC__', 
            complementary_sequence='AAGC__', 
            left_contact_positions=('primary', [0, 1]), 
            right_contact_positions=('complementary', [4, 5]),   # Position 5 doesn't exist
            left_contact_coupling=0.1, 
            right_contact_coupling=0.1
        )
        print("✓ Success! (This shouldn't happen)")
    except ValueError as e:
        print(f"✗ Expected error: {e}")

def test_primary_strand_blanks():
    """Test handling of blank bases in primary strand."""
    print("\n=== Testing Primary Strand Blanks ===")
    print("Primary sequence: 'A_CG_' (has blanks at positions 1 and 4)")
    print("Complementary sequence: 'TGCAT' (full sequence)")
    
        try:
        seq = sequence_to_graph(
            primary_sequence='A_CG_', 
            complementary_sequence='TGCAT', 
            left_contact_positions=('primary', [0, 2]), 
            right_contact_positions=('complementary', [2, 4]),   
            left_contact_coupling=0.1, 
            right_contact_coupling=0.1
        )
        print("✓ Success!")
        print(f"  - Nodes: {seq.x.shape[0]}")
        print(f"  - Edges: {seq.edge_index.shape[1]}")
        
        # Count different types of edges
        edge_attr = seq.edge_attr.numpy()
        contact_edges = sum(1 for edge in edge_attr if edge[1] == 0 and edge[2] == 0)
        backbone_edges = sum(1 for edge in edge_attr if edge[1] == 1)
        h_bond_edges = sum(1 for edge in edge_attr if edge[2] == 1)
        
        print(f"  - Contact edges: {contact_edges}")
        print(f"  - Backbone edges: {backbone_edges}")
        print(f"  - Hydrogen bond edges: {h_bond_edges}")
        
    except ValueError as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_edge_case_fix()
    test_primary_strand_blanks() 