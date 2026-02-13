"""
Dataset classes for DNA transport property prediction.
"""
import torch
import torch.utils.data
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Optional

from g3nat.graph import sequence_to_graph


class DNATransportDataset(torch.utils.data.Dataset):
    """Dataset for DNA transport property prediction."""

    def __init__(self, sequences: List[str],
                 dos_data: np.ndarray,
                 transmission_data: np.ndarray,
                 energy_grid: np.ndarray,
                 complementary_sequences: Optional[List[str]] = None,
                 gamma_l: Optional[np.ndarray] = None, gamma_r: Optional[np.ndarray] = None,
                 graphs: Optional[List] = None):
        """
        Initialize the dataset.

        Args:
            sequences: List of DNA sequences
            dos_data: Density of states data [num_samples, num_energy_points]
            transmission_data: Transmission data [num_samples, num_energy_points]
            energy_grid: Energy grid [num_energy_points]
            complementary_sequences: List of complementary DNA sequences (optional)
            gamma_l: Left contact coupling strengths as vectors [num_samples, seq_length * 2] (optional, for debugging)
            gamma_r: Right contact coupling strengths as vectors [num_samples, seq_length * 2] (optional, for debugging)
            graphs: Pre-converted graphs (optional)
        """
        self.sequences = sequences
        self.complementary_sequences = complementary_sequences
        self.dos_data = dos_data
        self.transmission_data = transmission_data
        self.energy_grid = energy_grid
        self.graphs = graphs

        # Convert to tensors (use np.asarray to avoid slow list-of-ndarrays path)
        self.dos_tensor = torch.as_tensor(np.asarray(dos_data), dtype=torch.float)
        self.transmission_tensor = torch.as_tensor(np.asarray(transmission_data), dtype=torch.float)
        self.energy_tensor = torch.as_tensor(np.asarray(energy_grid), dtype=torch.float)

        # Store gamma values as vectors for debugging
        if gamma_l is not None:
            self.gamma_l = torch.as_tensor(np.asarray(gamma_l), dtype=torch.float)
        else:
            self.gamma_l = None

        if gamma_r is not None:
            self.gamma_r = torch.as_tensor(np.asarray(gamma_r), dtype=torch.float)
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
            seq = self.sequences[idx]
            comp_seq = self.complementary_sequences[idx]
            transmission = self.transmission_tensor[idx]

            # Create the base Data object
            data = Data(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                dos=dos,
                transmission=transmission,
                energy_grid=self.energy_tensor,
                seq=seq,
                comp_seq=comp_seq
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
                      left_contact_positions_list: Optional[List] = None,
                      right_contact_positions_list: Optional[List] = None,
                      left_contact_coupling_list: Optional[List[float]] = None,
                      right_contact_coupling_list: Optional[List[float]] = None,
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
        left_contact_positions_list: List of left contact positions for each sequence (optional)
            Each entry should be in format accepted by sequence_to_graph: int, list of ints, or tuple (strand, pos)
        right_contact_positions_list: List of right contact positions for each sequence (optional)
        left_contact_coupling_list: List of left contact coupling values (eV) for each sequence (optional)
        right_contact_coupling_list: List of right contact coupling values (eV) for each sequence (optional)
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

        # Priority order for contact configuration:
        # 1. Per-sequence contact lists (from pickle files)
        # 2. Gamma values (legacy, for debugging)
        # 3. graph_kwargs defaults

        # Use per-sequence contact positions if provided (highest priority)
        if left_contact_positions_list is not None and i < len(left_contact_positions_list):
            seq_kwargs['left_contact_positions'] = left_contact_positions_list[i]
        elif gamma_l is not None and i < len(gamma_l):
            # Fallback to gamma values for backward compatibility
            seq_gamma_l = gamma_l[i]
            left_contacts = np.where(seq_gamma_l > 0)[0]
            if len(left_contacts) > 0:
                seq_kwargs['left_contact_positions'] = int(left_contacts[0])

        if right_contact_positions_list is not None and i < len(right_contact_positions_list):
            seq_kwargs['right_contact_positions'] = right_contact_positions_list[i]
        elif gamma_r is not None and i < len(gamma_r):
            # Fallback to gamma values for backward compatibility
            seq_gamma_r = gamma_r[i]
            right_contacts = np.where(seq_gamma_r > 0)[0]
            if len(right_contacts) > 0:
                seq_kwargs['right_contact_positions'] = int(right_contacts[-1])

        # Use per-sequence coupling values if provided
        if left_contact_coupling_list is not None and i < len(left_contact_coupling_list):
            seq_kwargs['left_contact_coupling'] = left_contact_coupling_list[i]
        elif gamma_l is not None and i < len(gamma_l):
            # Fallback to gamma values for backward compatibility
            seq_gamma_l = gamma_l[i]
            left_contacts = np.where(seq_gamma_l > 0)[0]
            if len(left_contacts) > 0:
                seq_kwargs['left_contact_coupling'] = float(seq_gamma_l[left_contacts[0]])

        if right_contact_coupling_list is not None and i < len(right_contact_coupling_list):
            seq_kwargs['right_contact_coupling'] = right_contact_coupling_list[i]
        elif gamma_r is not None and i < len(gamma_r):
            # Fallback to gamma values for backward compatibility
            seq_gamma_r = gamma_r[i]
            right_contacts = np.where(seq_gamma_r > 0)[0]
            if len(right_contacts) > 0:
                seq_kwargs['right_contact_coupling'] = float(seq_gamma_r[right_contacts[-1]])

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
    return DNATransportDataset(sequences, dos_data, transmission_data, energy_grid, complementary_sequences, gamma_l, gamma_r, graphs)
