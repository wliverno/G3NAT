import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Base encoding - one-hot + additional features
BASE_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
BASE_FEATURES = {
    'A': [1, 0, 0, 0, 1, 0, 9.0, 331.0],  # one-hot + purine/pyrimidine + num_atoms + molecular_weight
    'T': [0, 1, 0, 0, 0, 1, 10.0, 322.0],
    'G': [0, 0, 1, 0, 1, 0, 10.0, 347.0],
    'C': [0, 0, 0, 1, 0, 1, 9.0, 307.0]
}

# Contact features (electrodes)
CONTACT_FEATURES = [0, 0, 0, 0, 0, 0, 0, 1]  # Special marker for contacts


def sequence_to_graph(primary_sequence: str, 
                     complementary_sequence: Optional[str] = None,
                     left_contact_positions: Optional[Union[int, List[int], Tuple[str, Union[int, List[int]]]]] = None,
                     right_contact_positions: Optional[Union[int, List[int], Tuple[str, Union[int, List[int]]]]] = None,
                     left_contact_coupling: Union[float, List[float]] = 0.1,
                     right_contact_coupling: Union[float, List[float]] = 0.1) -> Data:
    """
    Convert DNA sequence to PyTorch Geometric graph with left and right contacts.
    
    Args:
        primary_sequence: Main DNA sequence string (e.g., 'ACGTA')
        complementary_sequence: Complementary sequence with '_' for missing bases (e.g., 'TGC__')
        left_contact_positions: Position(s) to connect left contact to. Can be:
            - int: Single position on primary strand (0-indexed)
            - list of ints: Multiple positions on primary strand
            - tuple (strand, pos): ('primary', pos) or ('complementary', pos) for single position
            - tuple (strand, positions): ('primary', [pos1, pos2]) or ('complementary', [pos1, pos2]) for multiple positions
        right_contact_positions: Position(s) to connect right contact to. Same format as left_contact_positions.
        left_contact_coupling: Coupling strength(s) for left contact (eV). 
            - float: Single coupling strength for all left connections
            - list of floats: One coupling strength per left connection
        right_contact_coupling: Coupling strength(s) for right contact (eV). Same format as left_contact_coupling.
        
    Returns:
        PyTorch Geometric Data object
    """
    # Handle default contact positions
    if left_contact_positions is None:
        left_contact_positions = 0  # Default: first position of primary strand
    if right_contact_positions is None:
        right_contact_positions = len(primary_sequence) - 1  # Default: last position of primary strand
        
    # Helper function to normalize contact positions
    def normalize_contact_positions(positions, default_strand='primary'):
        if isinstance(positions, int):
            return ('primary', [positions])
        elif isinstance(positions, list):
            return ('primary', positions)
        elif isinstance(positions, tuple):
            strand, pos_list = positions
            if isinstance(pos_list, int):
                pos_list = [pos_list]
            return (strand, pos_list)
        else:
            raise ValueError(f"Invalid contact positions format: {positions}")
    
    # Normalize contact positions
    left_strand, left_positions = normalize_contact_positions(left_contact_positions)
    right_strand, right_positions = normalize_contact_positions(right_contact_positions)
        
    # Handle coupling strengths
    if isinstance(left_contact_coupling, float):
        left_contact_coupling = [left_contact_coupling] * len(left_positions)
    if isinstance(right_contact_coupling, float):
        right_contact_coupling = [right_contact_coupling] * len(right_positions)
        
    # Type assertion for type checker
    assert isinstance(left_contact_coupling, list)
    assert isinstance(right_contact_coupling, list)
        
    # Generate complementary sequence if not provided
    if complementary_sequence is None:
        complementary_sequence = '_' * len(primary_sequence)
    
    # Node features and mapping
    node_features = []
    node_to_strand = []  # Track which strand each node belongs to
    node_mapping = {}  # Map (strand, position) to node index
    
    # Add left contact (node 0)
    node_features.append(CONTACT_FEATURES)
    node_to_strand.append('left_contact')
    node_mapping['left_contact'] = 0
    
    # Add primary strand features (only for existing bases)
    primary_start_idx = 1
    primary_node_count = 0
    for i, base in enumerate(primary_sequence):
        if base != '_':  # Only create nodes for actual bases
            node_features.append(BASE_FEATURES[base])
            node_to_strand.append('primary')
            node_mapping[('primary', i)] = primary_start_idx + primary_node_count
            primary_node_count += 1
        
    # Add complementary strand features (only for existing bases)
    complementary_start_idx = primary_start_idx + primary_node_count
    complementary_node_count = 0
    for i, base in enumerate(complementary_sequence):
        if base != '_':  # Only create nodes for actual bases
            node_features.append(BASE_FEATURES[base])
            node_to_strand.append('complementary')
            node_mapping[('complementary', i)] = complementary_start_idx + complementary_node_count
            complementary_node_count += 1
    
    # Add right contact (last node)
    right_contact_idx = complementary_start_idx + complementary_node_count
    node_features.append(CONTACT_FEATURES)
    node_to_strand.append('right_contact')
    node_mapping['right_contact'] = right_contact_idx
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge construction - contact connections and backbone connections
    edge_index = []
    edge_attr = []
    
    # Left contact connections
    for i, pos in enumerate(left_positions):
        if left_strand == 'primary':
            if pos >= len(primary_sequence):
                raise ValueError(f"Left contact position {pos} is out of range for sequence length {len(primary_sequence)}")
            base_node_idx = node_mapping[('primary', pos)]
        else:  # complementary strand
            if pos >= len(complementary_sequence):
                raise ValueError(f"Left contact position {pos} is out of range for complementary sequence length {len(complementary_sequence)}")
            if complementary_sequence[pos] == '_':
                raise ValueError(f"Left contact position on complementary strand at position {pos} corresponds to a blank base ('_').")
            base_node_idx = node_mapping[('complementary', pos)]
        
        coupling = left_contact_coupling[i] if i < len(left_contact_coupling) else left_contact_coupling[-1]
        
        # Bidirectional connection between left contact and base
        edge_index.extend([[node_mapping['left_contact'], base_node_idx], 
                         [base_node_idx, node_mapping['left_contact']]])
        edge_attr.extend([[coupling, 0, 0], [coupling, 0, 0]])
    
    # Right contact connections
    for i, pos in enumerate(right_positions):
        if right_strand == 'primary':
            if pos >= len(primary_sequence):
                raise ValueError(f"Right contact position {pos} is out of range for sequence length {len(primary_sequence)}")
            base_node_idx = node_mapping[('primary', pos)]
        else:  # complementary strand
            if pos >= len(complementary_sequence):
                raise ValueError(f"Right contact position {pos} is out of range for complementary sequence length {len(complementary_sequence)}")
            if complementary_sequence[pos] == '_':
                raise ValueError(f"Right contact position on complementary strand at position {pos} corresponds to a blank base ('_').")
            base_node_idx = node_mapping[('complementary', pos)]
        
        coupling = right_contact_coupling[i] if i < len(right_contact_coupling) else right_contact_coupling[-1]
        
        # Bidirectional connection between right contact and base
        edge_index.extend([[node_mapping['right_contact'], base_node_idx], 
                         [base_node_idx, node_mapping['right_contact']]])
        edge_attr.extend([[coupling, 0, 0], [coupling, 0, 0]])
    
    # Primary strand backbone connections (connect adjacent existing bases)
    primary_positions = [pos for pos, base in enumerate(primary_sequence) if base != '_']
    for i in range(len(primary_positions) - 1):
        pos1 = primary_positions[i]
        pos2 = primary_positions[i + 1]
        
        # Only connect if they are adjacent in the original sequence
        if pos2 == pos1 + 1:
            node1_idx = node_mapping[('primary', pos1)]
            node2_idx = node_mapping[('primary', pos2)]
            edge_index.extend([[node1_idx, node2_idx], [node2_idx, node1_idx]])
            edge_attr.extend([[0.8, 1, 0], [0.8, 1, 0]])  # Strong backbone coupling
        
    # Complementary strand backbone connections (only between existing bases)
    complementary_positions = [pos for pos, base in enumerate(complementary_sequence) if base != '_']
    for i in range(len(complementary_positions) - 1):
        pos1 = complementary_positions[i]
        pos2 = complementary_positions[i + 1]
        
        # Only connect if they are adjacent in the original sequence
        if pos2 == pos1 + 1:
            node1_idx = node_mapping[('complementary', pos1)]
            node2_idx = node_mapping[('complementary', pos2)]
            edge_index.extend([[node1_idx, node2_idx], [node2_idx, node1_idx]])
            edge_attr.extend([[0.8, 1, 0], [0.8, 1, 0]])  # Strong backbone coupling
    
    # Hydrogen bonding between complementary strands
    # Only create hydrogen bonds where both bases exist
    for i in range(len(primary_sequence)):
        primary_base = primary_sequence[i]
        complementary_base = complementary_sequence[i] if i < len(complementary_sequence) else '_'
        
        # Only create hydrogen bonds if both bases exist (not '_')
        if primary_base != '_' and complementary_base != '_':
            primary_idx = node_mapping[('primary', i)]
            complementary_idx = node_mapping[('complementary', i)]
            
            # Use a default hydrogen bond strength (will be learned by the model)
            h_bond_strength = 0.4  # Default value, model will learn actual strengths
            
            edge_index.extend([[primary_idx, complementary_idx], [complementary_idx, primary_idx]])
            edge_attr.extend([[h_bond_strength, 0, 1], [h_bond_strength, 0, 1]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def create_contact_connections(node_mapping: Dict, contact_positions: List[int], 
                             contact_coupling: List[float], contact_type: str) -> Tuple[List, List]:
    """
    Create contact connections for a given contact type.
    
    Args:
        node_mapping: Mapping of (strand, position) to node index
        contact_positions: List of positions to connect to
        contact_coupling: List of coupling strengths
        contact_type: 'left' or 'right'
        
    Returns:
        Tuple of (edge_index, edge_attr) lists
    """
    edge_index = []
    edge_attr = []
    
    contact_node = node_mapping[f'{contact_type}_contact']
    
    for i, pos in enumerate(contact_positions):
        coupling = contact_coupling[i] if i < len(contact_coupling) else contact_coupling[-1]
        
        # Bidirectional connection between contact and base
        edge_index.extend([[contact_node, pos], [pos, contact_node]])
        edge_attr.extend([[coupling, 0, 0], [coupling, 0, 0]])
    
    return edge_index, edge_attr


def create_backbone_connections(sequence: str, node_mapping: Dict, strand: str) -> Tuple[List, List]:
    """
    Create backbone connections for a DNA strand.
    
    Args:
        sequence: DNA sequence string
        node_mapping: Mapping of (strand, position) to node index
        strand: 'primary' or 'complementary'
        
    Returns:
        Tuple of (edge_index, edge_attr) lists
    """
    edge_index = []
    edge_attr = []
    
    # Get positions of existing bases
    positions = [pos for pos, base in enumerate(sequence) if base != '_']
    
    for i in range(len(positions) - 1):
        pos1 = positions[i]
        pos2 = positions[i + 1]
        
        # Only connect if they are adjacent in the original sequence
        if pos2 == pos1 + 1:
            node1_idx = node_mapping[(strand, pos1)]
            node2_idx = node_mapping[(strand, pos2)]
            edge_index.extend([[node1_idx, node2_idx], [node2_idx, node1_idx]])
            edge_attr.extend([[0.8, 1, 0], [0.8, 1, 0]])  # Strong backbone coupling
    
    return edge_index, edge_attr


def create_hydrogen_bonds(primary_sequence: str, complementary_sequence: str, 
                         node_mapping: Dict) -> Tuple[List, List]:
    """
    Create hydrogen bond connections between complementary strands.
    
    Args:
        primary_sequence: Primary DNA sequence
        complementary_sequence: Complementary DNA sequence
        node_mapping: Mapping of (strand, position) to node index
        
    Returns:
        Tuple of (edge_index, edge_attr) lists
    """
    edge_index = []
    edge_attr = []
    
    for i in range(len(primary_sequence)):
        primary_base = primary_sequence[i]
        complementary_base = complementary_sequence[i] if i < len(complementary_sequence) else '_'
        
        # Only create hydrogen bonds if both bases exist (not '_')
        if primary_base != '_' and complementary_base != '_':
            primary_idx = node_mapping[('primary', i)]
            complementary_idx = node_mapping[('complementary', i)]
            
            # Use a default hydrogen bond strength
            h_bond_strength = 0.4
            
            edge_index.extend([[primary_idx, complementary_idx], [complementary_idx, primary_idx]])
            edge_attr.extend([[h_bond_strength, 0, 1], [h_bond_strength, 0, 1]])
    
    return edge_index, edge_attr 