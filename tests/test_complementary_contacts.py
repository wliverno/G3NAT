#!/usr/bin/env python3
"""
Test complementary strand contact functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sequence_to_graph
import torch

def test_complementary_contacts():
    """Test contacts on complementary strand."""
        # Double-stranded DNA
    primary = "ACGTACGT"
    complementary = "TGCATGCA"
    
    print("=== Complementary Strand Contact Tests ===")
    
    # Test 1: Left contact on primary, right contact on complementary
    print("\nTest 1: Left contact on primary strand, right contact on complementary strand")
    graph = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=0,  # Primary strand, position 0
        right_contact_positions=('complementary', 7),  # Complementary strand, position 7
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    print(f"Primary: {primary}")
    print(f"Complementary: {complementary}")
    print(f"Nodes: {graph.x.shape[0]}")
    print(f"Edges: {graph.edge_index.shape[1]}")
    
    # Show contact connections
    print("\nContact connections:")
    for i in range(graph.edge_index.shape[1]):
        src, dst = graph.edge_index[:, i].tolist()
        coupling = graph.edge_attr[i, 3].item()  # coupling is now at index 3
        contact_flag = graph.edge_attr[i, 2].item()
        if contact_flag == 1:  # Contact connection
            if src == 0:  # Left contact
                if dst <= len(primary):
                    print(f"  Left contact -> Primary base {dst-1} (coupling: {coupling:.3f})")
                else:
                    print(f"  Left contact -> Complementary base {dst-len(primary)-1} (coupling: {coupling:.3f})")
            elif dst == 0:  # To left contact
                if src <= len(primary):
                    print(f"  Primary base {src-1} -> Left contact (coupling: {coupling:.3f})")
                else:
                    print(f"  Complementary base {src-len(primary)-1} -> Left contact (coupling: {coupling:.3f})")
            elif src == graph.x.shape[0] - 1:  # Right contact
                if dst <= len(primary):
                    print(f"  Right contact -> Primary base {dst-1} (coupling: {coupling:.3f})")
                else:
                    print(f"  Right contact -> Complementary base {dst-len(primary)-1} (coupling: {coupling:.3f})")
            elif dst == graph.x.shape[0] - 1:  # To right contact
                if src <= len(primary):
                    print(f"  Primary base {src-1} -> Right contact (coupling: {coupling:.3f})")
                else:
                    print(f"  Complementary base {src-len(primary)-1} -> Right contact (coupling: {coupling:.3f})")
    
    # Test 2: Both contacts on complementary strand
    print("\n" + "="*60)
    print("Test 2: Both contacts on complementary strand")
    graph2 = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=('complementary', 0),  # Complementary strand, position 0
        right_contact_positions=('complementary', 7),  # Complementary strand, position 7
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    print(f"Primary: {primary}")
    print(f"Complementary: {complementary}")
    print(f"Nodes: {graph2.x.shape[0]}")
    print(f"Edges: {graph2.edge_index.shape[1]}")
    
    # Show contact connections
    print("\nContact connections:")
    for i in range(graph2.edge_index.shape[1]):
        src, dst = graph2.edge_index[:, i].tolist()
                        coupling = graph2.edge_attr[i, 2].item()  # coupling is now at index 3
        edge_type = graph2.edge_attr[i, 1].item()
        if edge_type == 0:  # Contact connection
            if src == 0:  # Left contact
                if dst <= len(primary):
                    print(f"  Left contact -> Primary base {dst-1} (coupling: {coupling:.3f})")
                else:
                    print(f"  Left contact -> Complementary base {dst-len(primary)-1} (coupling: {coupling:.3f})")
            elif dst == 0:  # To left contact
                if src <= len(primary):
                    print(f"  Primary base {src-1} -> Left contact (coupling: {coupling:.3f})")
                else:
                    print(f"  Complementary base {src-len(primary)-1} -> Left contact (coupling: {coupling:.3f})")
            elif src == graph2.x.shape[0] - 1:  # Right contact
                if dst <= len(primary):
                    print(f"  Right contact -> Primary base {dst-1} (coupling: {coupling:.3f})")
                else:
                    print(f"  Right contact -> Complementary base {dst-len(primary)-1} (coupling: {coupling:.3f})")
            elif dst == graph2.x.shape[0] - 1:  # To right contact
                if src <= len(primary):
                    print(f"  Primary base {src-1} -> Right contact (coupling: {coupling:.3f})")
                else:
                    print(f"  Complementary base {src-len(primary)-1} -> Right contact (coupling: {coupling:.3f})")
    
    # Test 3: Multiple contacts on complementary strand
    print("\n" + "="*60)
    print("Test 3: Multiple contacts on complementary strand")
    graph3 = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=('complementary', [0, 2]),  # Complementary strand, positions 0 and 2
        right_contact_positions=('complementary', [5, 7]),  # Complementary strand, positions 5 and 7
        left_contact_coupling=[0.1, 0.15],
        right_contact_coupling=[0.2, 0.25]
    )
    
    print(f"Primary: {primary}")
    print(f"Complementary: {complementary}")
    print(f"Nodes: {graph3.x.shape[0]}")
    print(f"Edges: {graph3.edge_index.shape[1]}")
    
    # Count contact connections
    contact_edges = 0
    for i in range(graph3.edge_index.shape[1]):
        edge_type = graph3.edge_attr[i, 1].item()
        if edge_type == 0:  # Contact connection
            contact_edges += 1
    
    print(f"Contact connections: {contact_edges}")

def test_error_handling():
    """Test error handling for invalid complementary strand contacts."""
        print("\n=== Error Handling Tests ===")
    
    # Test: Contact on blank position in complementary strand
    print("\nTest: Contact on blank position in complementary strand")
    try:
        graph = sequence_to_graph(
            primary_sequence="ACGT",
            complementary_sequence="TG__",  # Last two positions are blank
            left_contact_positions=0,
            right_contact_positions=('complementary', 3),  # Position 3 is blank
            left_contact_coupling=0.1,
            right_contact_coupling=0.2
        )
        print("ERROR: Should have raised an exception!")
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Test: Contact position out of range
    print("\nTest: Contact position out of range")
    try:
        graph = sequence_to_graph(
            primary_sequence="ACGT",
            complementary_sequence="TGCAT",
            left_contact_positions=0,
            right_contact_positions=('complementary', 10),  # Position 10 doesn't exist
            left_contact_coupling=0.1,
            right_contact_coupling=0.2
        )
        print("ERROR: Should have raised an exception!")
    except ValueError as e:
        print(f"Correctly caught error: {e}")

if __name__ == "__main__":
    test_complementary_contacts()
    test_error_handling()
    print("\nAll complementary strand contact tests completed!") 