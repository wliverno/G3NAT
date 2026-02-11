from numpy._typing import _128Bit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
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
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 output_dim: int = 100,  # Number of energy points
                 dropout: float = 0.2,
                 conv_type: str = 'transformer'):
        super().__init__()
        # Use features specified in dataset.py
        node_features = 4  # 4 one-hot features (A, T, G, C)
        edge_features = 5  # 3 one-hot + directionality + coupling


        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.conv_type = conv_type.lower()

        # Input projections
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if self.conv_type == 'gat':
                conv = GATConv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, add_self_loops=True, edge_dim=hidden_dim
                )
            else:
                conv = TransformerConv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, edge_dim=hidden_dim
                )
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
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = self.global_pool(x, batch)

        # Output projections
        dos_pred = self.dos_proj(x)
        transmission_pred = self.transmission_proj(x)

        return dos_pred, transmission_pred
