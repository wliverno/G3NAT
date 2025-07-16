import torch
import torch.utils.data
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Dict, Optional, Union


class DNATransportDataset(torch.utils.data.Dataset):
    """Dataset for DNA transport property prediction."""
    
    def __init__(self, sequences: List[str], dos_data: np.ndarray, 
                 transmission_data: np.ndarray, energy_grid: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            sequences: List of DNA sequences
            dos_data: Density of states data [num_samples, num_energy_points]
            transmission_data: Transmission data [num_samples, num_energy_points]
            energy_grid: Energy grid [num_energy_points]
        """
        self.sequences = sequences
        self.dos_data = dos_data
        self.transmission_data = transmission_data
        self.energy_grid = energy_grid
        
        # Convert to tensors
        self.dos_tensor = torch.tensor(dos_data, dtype=torch.float)
        self.transmission_tensor = torch.tensor(transmission_data, dtype=torch.float)
        self.energy_tensor = torch.tensor(energy_grid, dtype=torch.float)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            PyTorch Geometric Data object with:
                - x: Node features
                - edge_index: Edge indices
                - edge_attr: Edge features
                - dos: Density of states target
                - transmission: Transmission target
                - energy_grid: Energy grid
        """
        sequence = self.sequences[idx]
        dos = self.dos_tensor[idx]
        transmission = self.transmission_tensor[idx]
        
        # Convert sequence to graph (this will be done by the user with dna_graph.sequence_to_graph)
        # For now, we'll create a placeholder graph structure
        # In practice, the user should call sequence_to_graph() to create the actual graph
        
        # Create a simple placeholder graph (this should be replaced with actual graph conversion)
        num_nodes = len(sequence) + 2  # +2 for left and right contacts
        x = torch.randn(num_nodes, 8)  # Placeholder node features
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()  # Placeholder edges
        edge_attr = torch.randn(2, 3)  # Placeholder edge features
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            dos=dos,
            transmission=transmission,
            energy_grid=self.energy_tensor
        )


def create_dna_dataset(sequences: List[str], dos_data: np.ndarray, 
                      transmission_data: np.ndarray, energy_grid: np.ndarray,
                      graph_converter_func=None) -> DNATransportDataset:
    """
    Create a DNA transport dataset with proper graph conversion.
    
    Args:
        sequences: List of DNA sequences
        dos_data: Density of states data
        transmission_data: Transmission data
        energy_grid: Energy grid
        graph_converter_func: Function to convert sequences to graphs
        
    Returns:
        DNATransportDataset object
    """
    if graph_converter_func is None:
        # Use default graph conversion
        from dna_graph import sequence_to_graph
        graph_converter_func = sequence_to_graph
    
    # Convert sequences to graphs
    graphs = []
    for sequence in sequences:
        graph = graph_converter_func(primary_sequence=sequence)
        graphs.append(graph)
    
    # Create dataset with pre-converted graphs
    dataset = DNATransportDataset(sequences, dos_data, transmission_data, energy_grid)
    
    # Override the __getitem__ method to use pre-converted graphs
    def getitem_with_graphs(idx):
        graph = graphs[idx]
        dos = dataset.dos_tensor[idx]
        transmission = dataset.transmission_tensor[idx]
        
        return Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            dos=dos,
            transmission=transmission,
            energy_grid=dataset.energy_tensor
        )
    
    dataset.__getitem__ = getitem_with_graphs
    return dataset 