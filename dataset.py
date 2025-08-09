import torch
import torch.utils.data
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Base encoding - 4x one-hot for A, T, G, C
BASE_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
BASE_FEATURES = {
    'A': [1, 0, 0, 0], 
    'T': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'C': [0, 0, 0, 1]
}
# Contact features (electrodes)
CONTACT_FEATURES = [0, 0, 0, 0]  # Special marker for contacts

# Edge encoding - 3x one-hot + directionality + contact coupling
def get_edge_features(edge_type: str, directionality: int, coupling: float = 0.0):
    """Return edge features without chemical couplings for non-contact edges"""
    if edge_type == 'backbone':
        return [1, 0, 0, directionality, 0.0]  # Coupling always 0 for backbone
    elif edge_type == 'hydrogen_bond':
        return [0, 1, 0, directionality, 0.0]  # Coupling always 0 for H-bonds
    elif edge_type == 'contact':
        return [0, 0, 1, directionality, coupling]  # User-provided coupling
    else:
        raise ValueError(f"Invalid edge type: {edge_type}")

def sequence_to_graph(primary_sequence: str, 
                     complementary_sequence: Optional[str] = None,
                     left_contact_positions: Optional[Union[int, List[int], Tuple[str, Union[int, List[int]]]]] = None,
                     right_contact_positions: Optional[Union[int, List[int], Tuple[str, Union[int, List[int]]]]] = None,
                     left_contact_coupling: Union[float, List[float]] = 0.1,
                     right_contact_coupling: Union[float, List[float]] = 0.1) -> Data:
    """
    Convert DNA sequence to PyTorch Geometric graph with left and right contacts.
    
    Args:
        primary_sequence: Main DNA sequence string, read 5' to 3'(e.g., 'ACGTA')
        complementary_sequence: Complementary sequence with '_' for missing bases, read 5' to 3'(e.g., 'TACGT')
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
    
    # Add right contact (node 1)
    node_features.append(CONTACT_FEATURES)
    node_to_strand.append('right_contact')
    node_mapping['right_contact'] = 1
    
    # Add primary strand features (only for existing bases)
    primary_start_idx = 2
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
        
        # Contact → base (directionality = 1)
        edge_index.append([node_mapping['left_contact'], base_node_idx])
        edge_attr.append(get_edge_features('contact', 1, coupling))
        
        # Base → contact (directionality = -1)
        edge_index.append([base_node_idx, node_mapping['left_contact']])
        edge_attr.append(get_edge_features('contact', -1, coupling))
    
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
        
        # Contact → base (directionality = 1)
        edge_index.append([node_mapping['right_contact'], base_node_idx])
        edge_attr.append(get_edge_features('contact', 1, coupling))
        
        # Base → contact (directionality = -1)
        edge_index.append([base_node_idx, node_mapping['right_contact']])
        edge_attr.append(get_edge_features('contact', -1, coupling))
    
    # Primary strand backbone connections (connect adjacent existing bases)
    primary_positions = [pos for pos, base in enumerate(primary_sequence) if base != '_']
    for i in range(len(primary_positions) - 1):
        pos1 = primary_positions[i]
        pos2 = primary_positions[i + 1]
        
        # Only connect if they are adjacent in the original sequence
        if pos2 == pos1 + 1:
            node1_idx = node_mapping[('primary', pos1)]
            node2_idx = node_mapping[('primary', pos2)]

            # 5'→3' direction (forward)
            edge_index.append([node1_idx, node2_idx])
            edge_attr.append(get_edge_features('backbone', 1))  # Directionality = 1
            
            # 3'→5' direction (backward)
            edge_index.append([node2_idx, node1_idx])
            edge_attr.append(get_edge_features('backbone', -1))  # Directionality = -1

    # Complementary strand backbone connections (only between existing bases)
    complementary_positions = [pos for pos, base in enumerate(complementary_sequence) if base != '_']
    for i in range(len(complementary_positions) - 1):
        pos1 = complementary_positions[i]
        pos2 = complementary_positions[i + 1]
        
        # Only connect if they are adjacent in the original sequence
        if pos2 == pos1 + 1:
            node1_idx = node_mapping[('complementary', pos1)]
            node2_idx = node_mapping[('complementary', pos2)]

            # 5'→3' direction (forward)
            edge_index.append([node1_idx, node2_idx])
            edge_attr.append(get_edge_features('backbone', 1))  # Directionality = 1
            
            # 3'→5' direction (backward)
            edge_index.append([node2_idx, node1_idx])
            edge_attr.append(get_edge_features('backbone', -1))  # Directionality = -1
    
    # Hydrogen bonding between complementary strands
    # Only create hydrogen bonds where both bases exist
    for i in range(len(primary_sequence)):
        primary_base = primary_sequence[i]
        comp_loc = len(primary_sequence) - i - 1
        complementary_base = complementary_sequence[comp_loc] if comp_loc < len(complementary_sequence) else '_'
        
        # Only create hydrogen bonds if both bases exist (not '_')
        if primary_base != '_' and complementary_base != '_':
            primary_idx = node_mapping[('primary', i)]
            complementary_idx = node_mapping[('complementary', comp_loc)]

            # No Directionality for Hydrogen Bonds            
            edge_index.extend([[primary_idx, complementary_idx], [complementary_idx, primary_idx]])
            edge_attr.extend([get_edge_features('hydrogen_bond', 0), get_edge_features('hydrogen_bond', 0)])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # Annotate with number of DNA nodes (total nodes minus 2 contacts)
    try:
        data.num_dna_nodes = int(x.size(0) - 2)
    except Exception:
        pass
    return data


class DNATransportDataset(torch.utils.data.Dataset):
    """Dataset for DNA transport property prediction."""
    
    def __init__(self, sequences: List[str], dos_data: np.ndarray, 
                 transmission_data: np.ndarray, energy_grid: np.ndarray, 
                 gamma_l: Optional[np.ndarray] = None, gamma_r: Optional[np.ndarray] = None,
                 graphs: Optional[List] = None):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of DNA sequences
            dos_data: Density of states data [num_samples, num_energy_points]
            transmission_data: Transmission data [num_samples, num_energy_points]
            energy_grid: Energy grid [num_energy_points]
            gamma_l: Left contact coupling strengths as vectors [num_samples, seq_length * 2] (optional, for debugging)
            gamma_r: Right contact coupling strengths as vectors [num_samples, seq_length * 2] (optional, for debugging)
            graphs: Pre-converted graphs (optional)
        """
        self.sequences = sequences
        self.dos_data = dos_data
        self.transmission_data = transmission_data
        self.energy_grid = energy_grid
        self.graphs = graphs
        
        # Convert to tensors
        self.dos_tensor = torch.tensor(dos_data, dtype=torch.float)
        self.transmission_tensor = torch.tensor(transmission_data, dtype=torch.float)
        self.energy_tensor = torch.tensor(energy_grid, dtype=torch.float)
        
        # Store gamma values as vectors for debugging
        if gamma_l is not None:
            self.gamma_l = torch.tensor(gamma_l, dtype=torch.float)
        else:
            self.gamma_l = None
            
        if gamma_r is not None:
            self.gamma_r = torch.tensor(gamma_r, dtype=torch.float)
        else:
            self.gamma_r = None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        If graphs are provided, returns the pre-converted graph with targets.
        Otherwise, raises NotImplementedError.
        """
        if self.graphs is not None:
            graph = self.graphs[idx]
            dos = self.dos_tensor[idx]
            transmission = self.transmission_tensor[idx]
            
            # Create the base Data object
            data = Data(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                dos=dos,
                transmission=transmission,
                energy_grid=self.energy_tensor
            )
            
            # Add gamma values if available
            if self.gamma_l is not None:
                data.gamma_l = self.gamma_l[idx]  # Vector of left contact couplings for this sequence
            if self.gamma_r is not None:
                data.gamma_r = self.gamma_r[idx]  # Vector of right contact couplings for this sequence
                
            return data
        else:
            raise NotImplementedError(
                "DNATransportDataset requires pre-converted graphs. "
                "Use create_dna_dataset() instead of instantiating DNATransportDataset directly."
            )


def create_default_gamma_vectors(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create default gamma vectors for a list of sequences.
    
    By default:
    - Left contact at position 0 of primary strand with coupling 0.1
    - Right contact at last position of primary strand with coupling 0.1
    - All other positions have coupling 0.0
    
    Args:
        sequences: List of DNA sequences
        
    Returns:
        Tuple of (gamma_l, gamma_r) arrays, each of shape [num_samples, max_seq_length * 2]
    """
    max_seq_length = max(len(seq) for seq in sequences)
    num_samples = len(sequences)
    
    gamma_l = np.zeros((num_samples, max_seq_length * 2))
    gamma_r = np.zeros((num_samples, max_seq_length * 2))
    
    for i, seq in enumerate(sequences):
        seq_length = len(seq)
        # Left contact at position 0 of primary strand
        gamma_l[i, 0] = 0.1
        # Right contact at last position of primary strand  
        gamma_r[i, seq_length - 1] = 0.1
    
    return gamma_l, gamma_r


def create_dna_dataset(sequences: List[str], dos_data: np.ndarray, 
                      transmission_data: np.ndarray, energy_grid: np.ndarray,
                      complementary_sequences: Optional[List[str]] = None,
                      gamma_l: Optional[np.ndarray] = None, gamma_r: Optional[np.ndarray] = None,
                      graph_converter_func=None,
                      **graph_kwargs) -> DNATransportDataset:
    """
    Create a DNA transport dataset with proper graph conversion.
    
    Args:
        sequences: List of DNA sequences
        dos_data: Density of states data
        transmission_data: Transmission data
        energy_grid: Energy grid
        complementary_sequences: List of complementary DNA sequences (optional)
        gamma_l: Left contact coupling strengths as vectors [num_samples, seq_length * 2] (optional, for debugging)
        gamma_r: Right contact coupling strengths as vectors [num_samples, seq_length * 2] (optional, for debugging)
        graph_converter_func: Function to convert sequences to graphs
        **graph_kwargs: Additional arguments to pass to graph_converter_func
        
    Returns:
        DNATransportDataset object
    """
    if graph_converter_func is None:
        # Use default graph conversion
        graph_converter_func = sequence_to_graph
    
    # Convert sequences to graphs
    graphs = []
    for i, sequence in enumerate(sequences):
        # Create a copy of graph_kwargs for this sequence
        seq_kwargs = graph_kwargs.copy()
        
        # Do not override caller-provided right_contact_positions; leave as-is
        
        # Use provided gamma values if available
        if gamma_l is not None and i < len(gamma_l):
            # Extract non-zero gamma values for left contacts
            seq_gamma_l = gamma_l[i]
            left_contacts = np.where(seq_gamma_l > 0)[0]
            if len(left_contacts) > 0:
                # Use the first non-zero position as left contact
                seq_kwargs['left_contact_positions'] = int(left_contacts[0])  # Convert to Python int
                seq_kwargs['left_contact_coupling'] = float(seq_gamma_l[left_contacts[0]])  # Convert to Python float
        
        if gamma_r is not None and i < len(gamma_r):
            # Extract non-zero gamma values for right contacts
            seq_gamma_r = gamma_r[i]
            right_contacts = np.where(seq_gamma_r > 0)[0]
            if len(right_contacts) > 0:
                # Use the last non-zero position as right contact
                seq_kwargs['right_contact_positions'] = int(right_contacts[-1])  # Convert to Python int
                seq_kwargs['right_contact_coupling'] = float(seq_gamma_r[right_contacts[-1]])  # Convert to Python float
        
        if complementary_sequences is not None and i < len(complementary_sequences):
            # Use both primary and complementary sequences
            graph = graph_converter_func(
                primary_sequence=sequence,
                complementary_sequence=complementary_sequences[i],
                **seq_kwargs
            )
        else:
            # Use only primary sequence
            graph = graph_converter_func(
                primary_sequence=sequence,
                **seq_kwargs
            )
        graphs.append(graph)
    
    # Create and return the dataset with pre-converted graphs
    return DNATransportDataset(sequences, dos_data, transmission_data, energy_grid, gamma_l, gamma_r, graphs) 