"""
Test script demonstrating the new simplified interface for double-stranded DNA:
- Second strand with '_' for missing bases
- Contact positions as integers or tuples
- Hydrogen bond strengths as trainable parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dna_graph import sequence_to_graph

def test_examples():
    """Test various examples with the new interface."""
    
    print("=== Testing New Double-Stranded DNA Interface ===\n")
    
    # Example 1: Basic double-stranded DNA
    print("Example 1: Basic double-stranded DNA")
    print("Primary: ACGTA")
    print("Complementary: TGCAT")
    print("Contacts: [0, 4] (first and last of primary strand)")
    
    graph1 = sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="TGCAT",
        left_contact_positions=0,
        right_contact_positions=4
    )
    
    print(f"Nodes: {graph1.x.shape[0]}, Edges: {graph1.edge_index.shape[1]}\n")
    
    # Example 2: Unequal strands with missing bases
    print("Example 2: Unequal strands with missing bases")
    print("Primary: ACGTA")
    print("Complementary: TGC__ (only 3 bases, rest missing)")
    print("Contacts: [0, 4] (first and last of primary strand)")
    
    graph2 = sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="TGC__",
        left_contact_positions=0,
        right_contact_positions=4
    )
    
    print(f"Nodes: {graph2.x.shape[0]}, Edges: {graph2.edge_index.shape[1]}\n")
    
    # Example 3: Contacts on different strands
    print("Example 3: Contacts on different strands")
    print("Primary: ACGTA")
    print("Complementary: TGCAT")
    print("Contacts: [(0, 0), (1, 1)] (first A of primary, second T of complementary)")
    
    graph3 = converter.sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="TGCAT",
        contact_positions=[(0, 0), (1, 1)]  # (strand, position)
    )
    
    print(f"Nodes: {graph3.x.shape[0]}, Edges: {graph3.edge_index.shape[1]}\n")
    
    # Example 4: Single-stranded (original behavior)
    print("Example 4: Single-stranded DNA")
    print("Primary: ACGTA")
    print("Complementary: _____ (all missing)")
    print("Contacts: [0, 4] (first and last of primary strand)")
    
    graph4 = converter.sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="_____",
        contact_positions=[0, 4]
    )
    
    print(f"Nodes: {graph4.x.shape[0]}, Edges: {graph4.edge_index.shape[1]}\n")
    
    # Example 5: Mixed contact placement
    print("Example 5: Mixed contact placement")
    print("Primary: ACGTA")
    print("Complementary: TGCAT")
    print("Contacts: [2, (1, 3)] (G on primary, C on complementary)")
    
    graph5 = converter.sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="TGCAT",
        contact_positions=[2, (1, 3)]  # Mix of int and tuple
    )
    
    print(f"Nodes: {graph5.x.shape[0]}, Edges: {graph5.edge_index.shape[1]}\n")
    
    # Example 6: User's specific scenario
    print("Example 6: User's specific scenario")
    print("Primary: ACGTA")
    print("Complementary: TGCAT")
    print("Left contact on first A, Right contact on T (complementary to last A)")
    print("This would be: [(0, 0), (1, 0)] (first A of primary, first T of complementary)")
    
    graph6 = converter.sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="TGCAT",
        contact_positions=[(0, 0), (1, 0)]  # First A of primary, first T of complementary
    )
    
    print(f"Nodes: {graph6.x.shape[0]}, Edges: {graph6.edge_index.shape[1]}\n")
    
    # Example 7: Partial double-stranded with gaps
    print("Example 7: Partial double-stranded with gaps")
    print("Primary: ACGTA")
    print("Complementary: _GC_T (missing first and fourth bases)")
    print("Contacts: [0, 4] (first and last of primary strand)")
    
    graph7 = converter.sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="_GC_T",
        contact_positions=[0, 4]
    )
    
    print(f"Nodes: {graph7.x.shape[0]}, Edges: {graph7.edge_index.shape[1]}\n")

def analyze_graph_structure(graph, name):
    """Analyze the structure of a graph."""
    print(f"\n--- {name} Analysis ---")
    
    # Count different types of edges
    edge_attr = graph.edge_attr.numpy()
    backbone_edges = sum(1 for edge in edge_attr if edge[1] == 1)
    h_bond_edges = sum(1 for edge in edge_attr if edge[2] == 1)
    contact_edges = sum(1 for edge in edge_attr if edge[1] == 0 and edge[2] == 0)
    
    print(f"Total nodes: {graph.x.shape[0]}")
    print(f"Total edges: {graph.edge_index.shape[1]}")
    print(f"  - Backbone edges: {backbone_edges}")
    print(f"  - Hydrogen bond edges: {h_bond_edges}")
    print(f"  - Contact coupling edges: {contact_edges}")
    
    # Show node types
    contact_nodes = sum(1 for i in range(graph.x.shape[0]) if graph.x[i, -1] == 1)
    base_nodes = graph.x.shape[0] - contact_nodes
    print(f"  - Contact nodes: {contact_nodes}")
    print(f"  - Base nodes: {base_nodes}")

if __name__ == "__main__":
    test_examples()
    
    # Create a simple example for detailed analysis
    converter = DNASequenceToGraph()
    simple_graph = converter.sequence_to_graph(
        primary_sequence="ACGTA",
        complementary_sequence="TGCAT",
        contact_positions=[0, 4]
    )
    
    analyze_graph_structure(simple_graph, "Simple Double-Stranded Example")
    
    print("\n=== Interface Summary ===")
    print("New interface supports:")
    print("1. complementary_sequence with '_' for missing bases")
    print("2. contact_positions as list of integers or tuples")
    print("3. Integer: contact on primary strand at that position")
    print("4. Tuple (strand, pos): contact on strand 0 (primary) or 1 (complementary)")
    print("5. Hydrogen bond strengths are trainable parameters (not hardcoded)") 