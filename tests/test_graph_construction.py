import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import sequence_to_graph
from torch_geometric.data import Data
import torch

def visualize_graph(graph_data, title="DNA Graph"):
    """Visualize the graph using networkx."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(graph_data.x.shape[0]):
        node_type = "contact" if i == 0 or i == graph_data.x.shape[0] - 1 else "base"
        G.add_node(i, type=node_type)
    
    # Add edges
    edge_list = graph_data.edge_index.t().numpy()
    for i in range(edge_list.shape[0]):
        G.add_edge(edge_list[i, 0], edge_list[i, 1])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in G.nodes() if G.nodes[n]['type'] == 'contact'],
                          node_color='red', node_size=300, label='Contacts')
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in G.nodes() if G.nodes[n]['type'] == 'base'],
                          node_color='lightblue', node_size=200, label='Bases')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def test_graph_construction():
    """Test various graph construction scenarios."""
        # Test 1: Simple single-stranded DNA
    print("Test 1: Single-stranded DNA 'ACGTA'")
    graph1 = sequence_to_graph("ACGTA")
    assert graph1 is not None, "Graph 1 should not be None"
    print(f"Nodes: {graph1.x.shape[0]}, Edges: {graph1.edge_index.shape[1]}")
    print(f"Node features shape: {graph1.x.shape}")
    print(f"Edge features shape: {graph1.edge_attr.shape}")
    visualize_graph(graph1, "Single-stranded DNA: ACGTA")
    
    # Test 2: Double-stranded DNA with full complement
    print("\nTest 2: Double-stranded DNA 'ACGTA' with complement 'TGCAT'")
    graph2 = sequence_to_graph("ACGTA", "TGCAT")
    assert graph2 is not None, "Graph 2 should not be None"
    print(f"Nodes: {graph2.x.shape[0]}, Edges: {graph2.edge_index.shape[1]}")
    visualize_graph(graph2, "Double-stranded DNA: ACGTA-TGCAT")
    
    # Test 3: Double-stranded DNA with partial complement
    print("\nTest 3: Double-stranded DNA 'ACGTA' with partial complement 'TA___'")
    graph3 = sequence_to_graph("ACGTA", "TA___")
    assert graph3 is not None, "Graph 3 should not be None"
    print(f"Nodes: {graph3.x.shape[0]}, Edges: {graph3.edge_index.shape[1]}")
    visualize_graph(graph3, "Partial Double-stranded: ACGTA-TG___")
    
    # Test 4: Custom contact positions
    print("\nTest 4: Custom contact positions")
    graph4 = sequence_to_graph("ACGTA", "TGCAT", left_contact_positions=(0, 1), (1, 3)[0], right_contact_positions=(0, 1), (1, 3)[-1])
    assert graph4 is not None, "Graph 4 should not be None"
    print(f"Nodes: {graph4.x.shape[0]}, Edges: {graph4.edge_index.shape[1]}")
    visualize_graph(graph4, "Custom Contacts: Primary pos 1, Complementary pos 3")

if __name__ == "__main__":
    test_graph_construction() 