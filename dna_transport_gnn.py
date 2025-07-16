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


class DNASequenceToGraph:
    """Convert DNA sequences to graph representations for transport calculations."""
    
    def __init__(self):
        # Base encoding - one-hot + additional features
        self.base_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        self.base_features = {
            'A': [1, 0, 0, 0, 1, 0, 9.0, 331.0],  # one-hot + purine/pyrimidine + num_atoms + molecular_weight
            'T': [0, 1, 0, 0, 0, 1, 10.0, 322.0],
            'G': [0, 0, 1, 0, 1, 0, 10.0, 347.0],
            'C': [0, 0, 0, 1, 0, 1, 9.0, 307.0]
        }
        
        # Contact features (electrodes)
        self.contact_features = [0, 0, 0, 0, 0, 0, 0, 1]  # Special marker for contacts
        
    def sequence_to_graph(self, 
                         primary_sequence: str, 
                         complementary_sequence: Optional[str] = None,
                         contact_positions: Optional[List[Union[int, Tuple[int, int]]]] = None,
                         contact_coupling: float = 0.1) -> Data:
        """
        Convert DNA sequence to PyTorch Geometric graph with flexible double-stranded support.
        
        Args:
            primary_sequence: Main DNA sequence string (e.g., 'ACGTA')
            complementary_sequence: Complementary sequence with '_' for missing bases (e.g., 'TGC__')
            contact_positions: List of contact positions. Each can be:
                - int: Contact on primary strand at that position (0-indexed)
                - tuple (strand, pos): Contact on specified strand (0=primary, 1=complementary) at position
            contact_coupling: Coupling strength to contacts
            
        Returns:
            PyTorch Geometric Data object
        """
        # Handle contact positions
        if contact_positions is None:
            contact_positions = [0, len(primary_sequence) - 1]  # Default: first and last of primary
            
        # Generate complementary sequence if not provided
        if complementary_sequence is None:
            complementary_sequence = '_' * len(primary_sequence)
        
        # Determine the effective length for double-stranded region
        double_stranded_length = min(len(primary_sequence), len(complementary_sequence))
        
        # Calculate total nodes: primary strand + complementary strand + 2 contacts
        total_nodes = len(primary_sequence) + len(complementary_sequence) + 2
        
        # Node features
        node_features = []
        node_to_strand = []  # Track which strand each node belongs to
        
        # Add left contact
        node_features.append(self.contact_features)
        node_to_strand.append('contact')
        
        # Add primary strand features
        for i, base in enumerate(primary_sequence):
            node_features.append(self.base_features[base])
            node_to_strand.append('primary')
            
        # Add complementary strand features
        for i, base in enumerate(complementary_sequence):
            if base == '_':
                # Use a neutral feature vector for missing bases
                node_features.append([0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 9.5, 326.75])  # Average of all bases
            else:
                node_features.append(self.base_features[base])
            node_to_strand.append('complementary')
            
        # Add right contact
        node_features.append(self.contact_features)
        node_to_strand.append('contact')
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge construction
        edge_index = []
        edge_attr = []
        
        # Calculate node indices
        left_contact_idx = 0
        primary_start_idx = 1
        complementary_start_idx = primary_start_idx + len(primary_sequence)
        right_contact_idx = complementary_start_idx + len(complementary_sequence)
        
        # Contact connections based on contact_positions
        for contact_pos in contact_positions:
            if isinstance(contact_pos, int):
                # Contact on primary strand
                strand_idx = 0
                base_pos = contact_pos
            else:
                # Contact on specified strand
                strand_idx, base_pos = contact_pos
                
            if strand_idx == 0:  # Primary strand
                contact_node_idx = primary_start_idx + base_pos
            else:  # Complementary strand
                contact_node_idx = complementary_start_idx + base_pos
                
                # Check if the contact position on the complementary strand corresponds to a blank base ('_')
                if complementary_sequence[base_pos] == '_':
                    raise ValueError(f"Contact position on complementary strand at position {base_pos} corresponds to a blank base ('_').")
                
            # Connect to left contact (first contact)
            edge_index.extend([[left_contact_idx, contact_node_idx], [contact_node_idx, left_contact_idx]])
            edge_attr.extend([[contact_coupling, 0, 0], [contact_coupling, 0, 0]])
            
            # Connect to right contact (second contact) if we have more than one contact
            if len(contact_positions) > 1:
                edge_index.extend([[right_contact_idx, contact_node_idx], [contact_node_idx, right_contact_idx]])
                edge_attr.extend([[contact_coupling, 0, 0], [contact_coupling, 0, 0]])
        
        # Primary strand backbone connections
        for i in range(len(primary_sequence) - 1):
            node1_idx = primary_start_idx + i
            node2_idx = primary_start_idx + i + 1
            edge_index.extend([[node1_idx, node2_idx], [node2_idx, node1_idx]])
            edge_attr.extend([[0.8, 1, 0], [0.8, 1, 0]])  # Strong backbone coupling
            
        # Complementary strand backbone connections (only between non-missing bases)
        for i in range(len(complementary_sequence) - 1):
            if complementary_sequence[i] != '_' and complementary_sequence[i + 1] != '_':
                node1_idx = complementary_start_idx + i
                node2_idx = complementary_start_idx + i + 1
                edge_index.extend([[node1_idx, node2_idx], [node2_idx, node1_idx]])
                edge_attr.extend([[0.8, 1, 0], [0.8, 1, 0]])  # Strong backbone coupling
        
        # Hydrogen bonding between complementary strands
        # Align strands and create hydrogen bonds for complementary base pairs
        for i in range(double_stranded_length):
            primary_idx = primary_start_idx + i
            complementary_idx = complementary_start_idx + i
            
            primary_base = primary_sequence[i]
            complementary_base = complementary_sequence[i]
            
            # Only create hydrogen bonds if both bases exist (not '_')
            if complementary_base != '_':
                # Use a default hydrogen bond strength (will be learned by the model)
                h_bond_strength = 0.4  # Default value, model will learn actual strengths
                
                edge_index.extend([[primary_idx, complementary_idx], [complementary_idx, primary_idx]])
                edge_attr.extend([[h_bond_strength, 0, 1], [h_bond_strength, 0, 1]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class DNATransportGNN(nn.Module):
    """Graph Neural Network for DNA transport property prediction."""
    
    def __init__(self, 
                 node_features: int = 8,
                 edge_features: int = 3,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 output_dim: int = 100,  # Number of energy points
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Input projections
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // num_heads, 
                       heads=num_heads, dropout=dropout, edge_dim=hidden_dim)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output heads
        self.dos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.transmission_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial projections
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Graph convolutions with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x + x_res)  # Residual connection
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x_global = global_mean_pool(x, batch)
        
        # Predictions
        dos_pred = self.dos_head(x_global)
        transmission_pred = torch.sigmoid(self.transmission_head(x_global))  # Transmission [0,1]
        
        return dos_pred, transmission_pred


class DNATransportDataset(torch.utils.data.Dataset):
    """Dataset handler for DNA transport data."""
    
    def __init__(self, sequences: List[str], dos_data: np.ndarray, 
                 transmission_data: np.ndarray, energy_grid: np.ndarray):
        self.sequences = sequences
        self.dos_data = dos_data
        self.transmission_data = transmission_data
        self.energy_grid = energy_grid
        self.graph_converter = DNASequenceToGraph()
        
        # Normalize DOS data
        self.dos_scaler = StandardScaler()
        self.dos_data_normalized = self.dos_scaler.fit_transform(dos_data)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        dos = torch.tensor(self.dos_data_normalized[idx], dtype=torch.float)
        transmission = torch.tensor(self.transmission_data[idx], dtype=torch.float)
        
        # Convert sequence to graph
        graph = self.graph_converter.sequence_to_graph(primary_sequence=sequence)
        graph.dos = dos
        graph.transmission = transmission
        
        return graph


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               num_epochs: int = 100, learning_rate: float = 1e-3, device: str = 'cpu'):
    """Train the DNA transport GNN model."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            dos_pred, trans_pred = model(batch)
            
            # Multi-task loss
            dos_loss = F.mse_loss(dos_pred, batch.dos)
            trans_loss = F.mse_loss(trans_pred, batch.transmission)
            
            loss = dos_loss + trans_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                dos_pred, trans_pred = model(batch)
                
                dos_loss = F.mse_loss(dos_pred, batch.dos)
                trans_loss = F.mse_loss(trans_pred, batch.transmission)
                loss = dos_loss + trans_loss
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses 