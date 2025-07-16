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
            conv = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                          dropout=dropout, add_self_loops=True)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projections for DOS and transmission
        self.dos_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.transmission_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Global pooling
        self.global_pool = global_mean_pool
        
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_features]
                - batch: Batch indices [num_nodes]
                
        Returns:
            Tuple of (dos_pred, transmission_pred) tensors
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Project node and edge features
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global pooling
        x = self.global_pool(x, batch)
        
        # Output projections
        dos_pred = self.dos_proj(x)
        transmission_pred = torch.sigmoid(self.transmission_proj(x))  # Ensure transmission is between 0 and 1
        
        return dos_pred, transmission_pred


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               num_epochs: int = 100, learning_rate: float = 1e-3, device: str = 'cpu'):
    """
    Train the DNA Transport GNN model.
    
    Args:
        model: DNATransportGNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (train_losses, val_losses) lists
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            dos_pred, transmission_pred = model(batch)
            
            # Combined loss for DOS and transmission
            dos_loss = criterion(dos_pred, batch.dos)
            transmission_loss = criterion(transmission_pred, batch.transmission)
            total_loss = dos_loss + transmission_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                dos_pred, transmission_pred = model(batch)
                
                dos_loss = criterion(dos_pred, batch.dos)
                transmission_loss = criterion(transmission_pred, batch.transmission)
                total_loss = dos_loss + transmission_loss
                
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses 