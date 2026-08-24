from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool


class DNATransportGNN(nn.Module):
    """Graph Neural Network for DNA transport property prediction."""

    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 output_dim: int = 100,  # Number of energy points
                 # 0.0 matches scripts/train.py's --dropout default, so a model built
                 # with class defaults matches a trained run (was 0.2 before 2026-08-09;
                 # dropout only acts in train mode, so eval behavior is unchanged).
                 dropout: float = 0.0,
                 conv_type: str = 'transformer',
                 use_geometry: bool = False,
                 geom_dim: int = 7,
                 geom_norm_stats: Optional[Dict] = None):
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

        # Optional SE(3)-invariant geometry channel. Default off = byte-for-byte
        # identical model (no extra params/buffers, existing checkpoints load).
        #
        # DELIBERATELY DUPLICATED from DNATransportHamiltonianGNN rather than
        # factored into a shared module: extracting it would rename state-dict keys
        # and break loading for every geometry-trained checkpoint already on disk.
        # Equivalence is enforced behaviourally instead, by
        # tests/test_models/test_standard_geometry.py::
        # test_fusion_matches_the_hamiltonian_model -- which is a stronger guarantee
        # than shared code, since it checks the two agree rather than assuming it.
        #
        # WHY THE BASELINE NEEDS THIS AT ALL: the paper's comparison is between two
        # models that share an encoder and differ ONLY in readout. Without this
        # channel, a geometry-on Hamiltonian model was being compared against a
        # baseline that could not see geometry -- an input advantage unrelated to
        # the readout, which is the thing under test.
        self.use_geometry = use_geometry
        self.geom_dim = geom_dim
        if use_geometry:
            self.geom_encoder = nn.Sequential(
                nn.Linear(geom_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Near-zero output init: switching geometry on must not perturb a fresh
            # model, or the geometry factor is confounded with initialization.
            nn.init.normal_(self.geom_encoder[-1].weight, std=0.01)
            nn.init.zeros_(self.geom_encoder[-1].bias)
            # per-edge-type z-score buffers: row 0 backbone, row 1 hbond
            mean = torch.zeros(2, geom_dim)
            std = torch.ones(2, geom_dim)
            if geom_norm_stats is not None:
                mean[0] = torch.tensor(geom_norm_stats["backbone"]["mean"], dtype=torch.float)
                std[0] = torch.tensor(geom_norm_stats["backbone"]["std"], dtype=torch.float)
                mean[1] = torch.tensor(geom_norm_stats["hbond"]["mean"], dtype=torch.float)
                std[1] = torch.tensor(geom_norm_stats["hbond"]["std"], dtype=torch.float)
            self.register_buffer("geom_mean", mean)
            self.register_buffer("geom_std", std)

    def _fuse_geometry(self, edge_attr_proj, edge_attr_initial, data):
        """Add per-edge-type-normalized geometry to the projected edge embedding.

        Backbone edges (edge_attr_initial[:,0]==1) use row 0 stats; H-bond edges
        (col 1) use row 1. Masked/contact edges contribute 0 (mask gates them).

        Mirrors DNATransportHamiltonianGNN._fuse_geometry exactly -- see the note in
        __init__ for why it is duplicated rather than shared.
        """
        geom = data.edge_geom
        mask = data.edge_geom_mask
        is_bb = (edge_attr_initial[:, 0] == 1).to(geom.dtype).unsqueeze(1)  # [E,1]
        is_hb = (edge_attr_initial[:, 1] == 1).to(geom.dtype).unsqueeze(1)
        mean = is_bb * self.geom_mean[0] + is_hb * self.geom_mean[1]  # [E,7]
        std = is_bb * self.geom_std[0] + is_hb * self.geom_std[1]
        # A geometry column that is constant across the dataset has std 0; without
        # this guard one degenerate column poisons every edge embedding with nan.
        std = torch.where(std == 0, torch.ones_like(std), std)
        normed = (geom - mean) / std
        return edge_attr_proj + self.geom_encoder(normed) * mask

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

        # Project node and edge features. The RAW edge features are kept: the
        # geometry fusion needs the one-hot edge-type columns to pick a
        # normalization row, and the projection overwrites them.
        x = self.node_proj(x)
        edge_attr_initial = edge_attr
        edge_attr = self.edge_proj(edge_attr)
        if self.use_geometry:
            edge_attr = self._fuse_geometry(edge_attr, edge_attr_initial, data)

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
