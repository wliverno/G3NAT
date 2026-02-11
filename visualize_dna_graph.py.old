#!/usr/bin/env python3
"""
Visualization module for DNA transport graphs using NetworkX.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from torch_geometric.utils import to_networkx
from dataset import sequence_to_graph
import numpy as np

def visualize_dna_graph(graph, primary_sequence=None, complementary_sequence=None, 
                       figsize=(8, 12), node_size=1000, font_size=10):
    """
    Visualize a DNA graph with proper labeling and styling.
    
    Args:
        graph: PyTorch Geometric Data object
        primary_sequence: Primary DNA sequence for labeling
        complementary_sequence: Complementary DNA sequence for labeling
        figsize: Figure size (width, height)
        node_size: Size of nodes
        font_size: Font size for labels
    """
    # Convert to NetworkX graph
    nx_graph = to_networkx(graph, to_undirected=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine node types and positions
    node_labels = {}
    node_colors = []
    node_positions = {}
    
    # Get edge attributes
    edge_weights = {}
    edge_styles = {}
    edge_colors = []
    
    # Precompute strand segmentation and position maps based on how sequence_to_graph builds nodes
    if primary_sequence is not None:
        primary_positions_nonblank = [i for i, b in enumerate(primary_sequence) if b != '_']
        primary_count = len(primary_positions_nonblank)
    else:
        primary_positions_nonblank = []
        primary_count = 0

    if complementary_sequence is not None:
        complementary_positions_nonblank = [i for i, b in enumerate(complementary_sequence) if b != '_']
        complementary_count = len(complementary_positions_nonblank)
    else:
        complementary_positions_nonblank = []
        complementary_count = 0

    primary_start_idx = 2
    complementary_start_idx = primary_start_idx + primary_count

    # Process nodes
    for node in nx_graph.nodes():
        if node == 0:  # Left contact
            node_labels[node] = 'L'
            node_colors.append('red')
            node_positions[node] = (-2, 0)
            continue
        if node == 1:  # Right contact
            node_labels[node] = 'R'
            node_colors.append('red')
            node_positions[node] = ((len(primary_sequence) if primary_sequence else 10) + 1, 0)
            continue

        # For base nodes, determine the base type from node features
        node_features = graph.x[node].numpy()
        # Skip safety for malformed data
        if np.all(node_features == 0):
            continue

        # Determine base type from one-hot encoding
        base_features = node_features[:4]
        if base_features[0] == 1:
            base = 'A'
        elif base_features[1] == 1:
            base = 'T'
        elif base_features[2] == 1:
            base = 'G'
        elif base_features[3] == 1:
            base = 'C'
        else:
            base = '?'

        # Positioning: map node index back to original sequence position
        if primary_sequence is not None and primary_start_idx <= node < primary_start_idx + primary_count:
            # Primary strand node
            k = node - primary_start_idx
            seq_pos = primary_positions_nonblank[k]
            x_pos = seq_pos
            y_pos = 0.5
            node_colors.append('lightblue')
        elif complementary_sequence is not None and complementary_start_idx <= node < complementary_start_idx + complementary_count:
            # Complementary strand node
            k = node - complementary_start_idx
            comp_seq_pos = complementary_positions_nonblank[k]
            # Align with its paired primary position: x = len(primary) - 1 - comp_pos
            if primary_sequence is not None:
                x_pos = len(primary_sequence) - 1 - comp_seq_pos
            else:
                x_pos = comp_seq_pos
            y_pos = -0.5
            node_colors.append('lightgreen')
        else:
            # Fallback: place sequentially
            x_pos = node
            y_pos = 0
            node_colors.append('gray')

        node_labels[node] = base
        node_positions[node] = (x_pos, y_pos)
    
    # Process edges
    for edge in nx_graph.edges():
        src, dst = edge
        
        # Find the edge in the original graph to get attributes
        edge_found = False
        for i in range(graph.edge_index.shape[1]):
            if (graph.edge_index[0, i].item() == src and graph.edge_index[1, i].item() == dst) or \
               (graph.edge_index[0, i].item() == dst and graph.edge_index[1, i].item() == src):
                # Edge features: [backbone_onehot, hbond_onehot, contact_onehot, directionality, coupling]
                backbone_flag = graph.edge_attr[i, 0].item()
                h_bond_flag = graph.edge_attr[i, 1].item()
                contact_flag = graph.edge_attr[i, 2].item()
                directionality = graph.edge_attr[i, 3].item()
                coupling = graph.edge_attr[i, 4].item()
                edge_found = True
                break
        
        if edge_found:
            
            if h_bond_flag == 1:  # Hydrogen bond
                edge_weights[edge] = ""  # No label for hydrogen bonds
                edge_styles[edge] = 'dotted'
                edge_colors.append('blue')
            elif contact_flag == 1:  # Contact connection
                edge_weights[edge] = f"{coupling:.2f}"  # Only show coupling for contacts
                edge_styles[edge] = 'solid'
                edge_colors.append('red')
            elif backbone_flag == 1:  # Backbone connection
                edge_weights[edge] = ""  # No label for backbone connections
                edge_styles[edge] = 'solid'
                edge_colors.append('black')
    
    # Draw the graph
    nx.draw(nx_graph, pos=node_positions, 
            node_color=node_colors,
            node_size=node_size,
            font_size=font_size,
            font_weight='bold',
            labels=node_labels,
            edge_color=edge_colors,
            style=[edge_styles.get(edge, 'solid') for edge in nx_graph.edges()],
            width=2,
            ax=ax,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Draw edge labels (coupling strengths)
    edge_labels = nx.draw_networkx_edge_labels(nx_graph, pos=node_positions, 
                                              edge_labels=edge_weights,
                                              font_size=8)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='red', label='Contacts'),
        mpatches.Patch(color='lightblue', label='Primary Strand'),
        mpatches.Patch(color='lightgreen', label='Complementary Strand'),
        Line2D([0], [0], color='red', linestyle='-', linewidth=2, label='Contact Connections'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Backbone Connections'),
        Line2D([0], [0], color='blue', linestyle=':', linewidth=2, label='Hydrogen Bonds')
    ]
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1))
    
    # Set title and labels
    if primary_sequence:
        title = f"DNA Graph: {primary_sequence}"
        if complementary_sequence:
            title += f" / {complementary_sequence}"
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Strand', fontsize=12)
    
    # Set axis limits
    if primary_sequence:
        ax.set_xlim(-3, len(primary_sequence) + 2)
    else:
        ax.set_xlim(-3, 12)
    ax.set_ylim(-1, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def create_sample_visualization():
    """Create a sample visualization to demonstrate the functionality."""
    
    # Create a sample graph
    primary = "ACGCTT"
    complementary = "AAGCGT"
    
    graph = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=('primary', [0, 1]),
        right_contact_positions=('complementary', [0, 1]),
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )
    
    # Visualize
    fig, ax = visualize_dna_graph(graph, primary, complementary)
    plt.show()
    
    return fig, ax

def visualize_multiple_examples():
    """Create multiple visualization examples."""
    
    # Example 1: Single-stranded with multiple contacts
    print("Creating visualization for single-stranded DNA...")
    graph1 = sequence_to_graph(
        primary_sequence="ACGTACGT",
        left_contact_positions=[0, 2],
        right_contact_positions=[5, 7],
        left_contact_coupling=[0.1, 0.15],
        right_contact_coupling=[0.2, 0.25]
    )
    
    fig1, ax1 = visualize_dna_graph(graph1, "ACGTACGT")
    plt.savefig('single_stranded_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 2: Double-stranded with mixed contacts
    print("Creating visualization for double-stranded DNA...")
    graph2 = sequence_to_graph(
        primary_sequence="ACGTACGT",
        complementary_sequence="TGCATGCA",
        left_contact_positions=0,
        right_contact_positions=('complementary', 0),
        left_contact_coupling=0.1,
        right_contact_coupling=0.2
    )
    
    fig2, ax2 = visualize_dna_graph(graph2, "ACGTACGT", "TGCATGCA")
    plt.savefig('double_stranded_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 3: Both contacts on complementary strand
    print("Creating visualization for complementary strand contacts...")
    graph3 = sequence_to_graph(
        primary_sequence="ACGTACGT",
        complementary_sequence="TGCATGCA",
        left_contact_positions=('complementary', [5, 7]),
        right_contact_positions=('complementary', [0, 2]),
        left_contact_coupling=[0.1, 0.15],
        right_contact_coupling=[0.2, 0.25]
    )
    
    fig3, ax3 = visualize_dna_graph(graph3, "ACGTACGT", "TGCATGCA")
    plt.savefig('complementary_contacts_example.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("DNA Graph Visualization Examples")
    print("=" * 40)
    
    # Create the sample visualization from your test
    print("Creating sample visualization...")
    create_sample_visualization()
    
    # Create multiple examples
    print("\nCreating multiple examples...")
    visualize_multiple_examples()
    
    print("\nAll visualizations completed!") 