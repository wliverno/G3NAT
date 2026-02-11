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


class DNATransportHamiltonianGNN(nn.Module):
    """Graph Neural Network for DNA transport Hamiltonian prediction."""

    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 energy_grid: np.ndarray = np.linspace(-3, 3, 100),
                 dropout: float = 0.1,
                 n_orb: int = 1,
                 enforce_hermiticity: bool = True,
                 solver_type: str = "frobenius",  # "frobenius" | "complex"
                 use_log_outputs: bool = True,
                 log_floor: float = 1e-16,
                 complex_eta: float = 1e-12,
                 conv_type: str = 'gat'):
        super().__init__()
        # Use features specified in dataset.py
        node_features = 4  # 4 one-hot features (A, T, G, C)
        edge_features = 5  # 3 one-hot + directionality + coupling


        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.energy_grid = energy_grid
        self.dropout = dropout
        self.output_dim = len(energy_grid)
        self.n_orb = n_orb  # Number of orbitals per site (n_onsite = n_coupling = n_orb)
        self.enforce_hermiticity = enforce_hermiticity
        self.solver_type = solver_type
        self.use_log_outputs = use_log_outputs
        self.log_floor = float(log_floor)
        self.complex_eta = float(complex_eta)
        self.conv_type = conv_type.lower()

        # Size-agnostic: Hamiltonian constructed from graph structure
        # H_size = num_dna_nodes * n_orb (total number of orbitals)

        # Input projections
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)

        # Graph attention layers
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

        # Graph-based Hamiltonian projections
        # Each node contributes n_orb x n_orb onsite energy block
        self.onsite_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_orb * n_orb)
        )

        # Each edge contributes n_orb x n_orb coupling block
        self.coupling_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_orb * n_orb)
        )

        # Global pooling
        self.global_pool = global_mean_pool

    def construct_hamiltonian_from_graph(self,
                                       node_features: torch.Tensor,
                                       edge_features: torch.Tensor,
                                       edge_index: torch.Tensor,
                                       batch: torch.Tensor,
                                       original_node_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Construct Hamiltonian matrix directly from graph structure.

        Args:
            node_features: Node features after GNN layers [num_nodes, hidden_dim]
            edge_features: Edge features after GNN layers [num_edges, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch indices [num_nodes]
            original_node_features: Original node features for contact detection [num_nodes, node_features]

        Returns:
            H_matrix: Hamiltonian matrix [batch_size, H_size, H_size]
            H_size: Size of Hamiltonian (num_dna_nodes * n_orb)
        """
        device = node_features.device

        # Handle batched processing
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)

        batch_size = int(batch.max().item() + 1)

        # Find DNA nodes (non-contact nodes: not all zeros in original features)
        contact_node_mask = torch.all(original_node_features == 0, dim=1)
        dna_node_mask = ~contact_node_mask
        dna_nodes = torch.where(dna_node_mask)[0]

        # Compute per-graph DNA node counts to support batching safely
        node_counts = torch.bincount(batch, minlength=batch_size)
        dna_counts = []
        start_idx = 0
        for b_idx in range(batch_size):
            count = int(node_counts[b_idx].item())
            end_idx = start_idx + count
            graph_mask = torch.zeros_like(batch, dtype=torch.bool)
            graph_mask[start_idx:end_idx] = True
            dna_counts.append(int((dna_node_mask & graph_mask).sum().item()))
            start_idx = end_idx

        # Ensure all graphs in the batch have the same number of DNA nodes
        if batch_size > 1 and not all(dc == dna_counts[0] for dc in dna_counts):
            raise ValueError(
                f"All graphs in a batch must have the same number of DNA nodes. Got {dna_counts}. "
                "Enable length-bucketing DataLoader for the hamiltonian model."
            )
        num_dna_nodes = dna_counts[0]
        H_size = num_dna_nodes * self.n_orb

        # Get onsite energies for DNA nodes only
        dna_node_features = node_features[dna_node_mask]  # [num_dna_nodes, hidden_dim]
        onsite_blocks = self.onsite_proj(dna_node_features)  # [num_dna_nodes, n_orb²]
        onsite_blocks = onsite_blocks.view(-1, self.n_orb, self.n_orb)  # [num_dna_nodes, n_orb, n_orb]
        if self.enforce_hermiticity:
            # Symmetrize onsite blocks to ensure Hermiticity (real symmetric here)
            onsite_blocks = 0.5 * (onsite_blocks + onsite_blocks.transpose(-1, -2))

        # Get coupling blocks for edges between DNA nodes
        coupling_blocks = self.coupling_proj(edge_features)  # [num_edges, n_orb²]
        coupling_blocks = coupling_blocks.view(-1, self.n_orb, self.n_orb)  # [num_edges, n_orb, n_orb]

        # Initialize Hamiltonian matrix
        H_matrix = torch.zeros(batch_size, H_size, H_size,
                              dtype=torch.float32, device=device)

        # Process each graph in the batch
        for batch_idx in range(batch_size):
            # Get nodes and edges for this graph using contiguous block ranges
            # torch_geometric batches concatenate node arrays in order
            nodes_before = int(torch.sum(torch.bincount(batch, minlength=batch_size)[:batch_idx]).item())
            nodes_in_graph = int(torch.bincount(batch, minlength=batch_size)[batch_idx].item())
            graph_node_indices = torch.arange(nodes_before, nodes_before + nodes_in_graph, device=device)
            node_mask = torch.zeros_like(batch, dtype=torch.bool)
            node_mask[graph_node_indices] = True
            dna_nodes_batch = torch.where(node_mask & dna_node_mask)[0]

            # Create mapping from global DNA node indices to local block indices
            dna_to_local = {global_idx.item(): local_idx
                           for local_idx, global_idx in enumerate(dna_nodes_batch)}

            # Fill diagonal blocks (onsite energies)
            for local_idx, global_idx in enumerate(dna_nodes_batch):
                orb_start = local_idx * self.n_orb
                orb_end = orb_start + self.n_orb
                # Find the position of this global_idx in the dna_nodes array
                global_dna_idx = torch.where(dna_nodes == global_idx)[0][0].item()
                H_matrix[batch_idx, orb_start:orb_end, orb_start:orb_end] = onsite_blocks[global_dna_idx]

            # Fill off-diagonal blocks (couplings) using unique undirected pairs only
            edge_mask = torch.isin(edge_index[0], dna_nodes_batch) & torch.isin(edge_index[1], dna_nodes_batch)
            graph_edge_indices = torch.where(edge_mask)[0]  # Global edge indices
            graph_edge_index = edge_index[:, edge_mask]

            processed_pairs = set()
            for local_edge_idx, (src, dst) in enumerate(graph_edge_index.T):
                src_g = src.item(); dst_g = dst.item()
                if src_g in dna_to_local and dst_g in dna_to_local:
                    src_local = dna_to_local[src_g]
                    dst_local = dna_to_local[dst_g]
                    u, v = (src_local, dst_local) if src_local <= dst_local else (dst_local, src_local)
                    pair = (u, v)
                    if u == v or pair in processed_pairs:
                        continue  # skip self-loops and duplicate reverse edges
                    processed_pairs.add(pair)

                    u_orb_start = u * self.n_orb
                    u_orb_end = u_orb_start + self.n_orb
                    v_orb_start = v * self.n_orb
                    v_orb_end = v_orb_start + self.n_orb

                    # Use the first occurrence's coupling block for this undirected pair
                    global_edge_idx = graph_edge_indices[local_edge_idx]
                    coupling_block = coupling_blocks[global_edge_idx]

                    # Set symmetric coupling blocks
                    H_matrix[batch_idx, u_orb_start:u_orb_end, v_orb_start:v_orb_end] = coupling_block
                    H_matrix[batch_idx, v_orb_start:v_orb_end, u_orb_start:u_orb_end] = coupling_block.conj().T

        # Ensure H is positive definite by adding a diagonal shift
        shift = 1e-6  # Small positive value
        identity = torch.eye(H_matrix.size(-1), device=device)
        H_matrix = H_matrix + shift * identity.unsqueeze(0).expand(batch_size, -1, -1)

        return H_matrix, H_size

    def NEGFProjection(self,
        H_matrix: torch.Tensor,
        GammaL: torch.Tensor,
        GammaR: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the transmission and DOS using NEGF with Frobenius formula (real matrices).

        Args:
            H_matrix: Hamiltonian matrix [batch_size, H_size, H_size]
            GammaL: Left lead coupling [batch_size, H_size]
            GammaR: Right lead coupling [batch_size, H_size]

        Returns:
            Tuple of (transmission, DOS, H) tensors for output
        """

        # Handle both single sample and batch cases
        if H_matrix.dim() == 2:
            H_matrix = H_matrix.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Ensure GammaL and GammaR have batch dimension
        if GammaL.dim() == 1:
            GammaL = GammaL.unsqueeze(0)
        if GammaR.dim() == 1:
            GammaR = GammaR.unsqueeze(0)

        batch_size = H_matrix.size(0)
        H_size = H_matrix.size(1)
        device = H_matrix.device

        # Create Identity Matrix
        I = torch.eye(H_size, dtype=H_matrix.dtype, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, H_size, H_size]
        I = I.expand(batch_size, len(self.energy_grid), H_size, H_size) # [batch, energy, H_size, H_size]

        # Expand Hamiltonian to match energy grid
        H_expanded = H_matrix.unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)  # [batch, energy, H_size, H_size]

        # Create sigTot: gamma vectors should be diagonal matrices
        # GammaL + GammaR has shape [batch, H_size], we need [batch, energy, H_size, H_size] diagonal
        gamma_total = GammaL + GammaR  # [batch, H_size]
        sigTotImag_diag = -0.5 * torch.diag_embed(gamma_total)  # [batch, H_size, H_size]
        sigTotImag = sigTotImag_diag.unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)  # [batch, energy, H_size, H_size]

        # Expand energy grid to match batch size and H_size
        energy_grid_tensor = torch.tensor(self.energy_grid, dtype=H_matrix.dtype, device=device)  # [num_energy]
        energy_grid_expanded = energy_grid_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, num_energy, 1, 1]

        # Calculate A matrix components for Frobenius formula
        delta = I*1e-12  # Small imaginary regularization
        A = energy_grid_expanded * I - H_expanded  # Real part
        B = -sigTotImag + delta  # Imaginary part (purely imaginary)

        # Calculate Green's functions using Frobenius formula for maximum stability
        # inv(A + 1j*B) = inv(A + B@inv(A)@B) - 1j*inv(A)@B@inv(A + B@inv(A)@B)
        # Compute inv(A) first
        try:
            A_inv = torch.linalg.solve(A, I)
        except torch.linalg.LinAlgError:
            print("Warning: Singular matrix encountered, using pseudo-inverse")
            A_inv = torch.linalg.pinv(A)

        # Compute B@inv(A)@B (note: B is purely imaginary, so this is real)
        B_Ainv_B = torch.matmul(torch.matmul(B, A_inv), B)

        # Compute inv(A + B@inv(A)@B)
        A_plus_B_Ainv_B = A + B_Ainv_B
        try:
            Gr_real = torch.linalg.solve(A_plus_B_Ainv_B, I)
        except torch.linalg.LinAlgError:
            print("Warning: Singular matrix encountered, using pseudo-inverse")
            Gr_real = torch.linalg.pinv(A_plus_B_Ainv_B)

        # Compute inv(A)@B@inv(A + B@inv(A)@B)
        Gr_imag = -1*torch.matmul(torch.matmul(A_inv, B), Gr_real)

        # Apply Frobenius formula
        #Gr = Gr_real - 1j * Gr_imag
        #Ga = Gr_real.transpose(-2, -1) - 1j*Gr_imag.transpose(-2, -1)

        # Calculate DOS using correct NEGF formula
        # DOS = -Im(Tr(Gr)) / π, but we omit π factor for consistency
        dos_raw = -1*torch.einsum('benn->be', Gr_imag)/np.pi
        # DOS should be positive by construction in NEGF, but add safety check
        dos_safe = torch.clamp(dos_raw, min=1e-16)  # Ensure positive values without abs()
        DOS = torch.log10(dos_safe)

        # Calculate transmission
        # Convert gamma vectors to diagonal matrices for matrix multiplication
        # [batch, H_size] -> [batch, energy, H_size, 1]
        GammaL_diag = (GammaL).unsqueeze(1).unsqueeze(-1).expand(-1, len(self.energy_grid), -1, -1)
        GammaR_diag = (GammaR).unsqueeze(1).unsqueeze(-1).expand(-1, len(self.energy_grid), -1, -1)

        # Element-wise multiplication (equivalent to diagonal matrix multiplication)
        # Gr: [batch, energy, H_size, H_size]
        # Gamma_diag: [batch, energy, H_size, 1]
        # Result: [batch, energy, H_size, H_size]
        gamma1Gr_real = GammaL_diag * Gr_real
        gamma2Ga_real = GammaR_diag * Gr_real.transpose(-2, -1)
        gamma1Gr_imag = GammaL_diag * Gr_imag
        gamma2Ga_imag = -1 * GammaR_diag * Gr_imag.transpose(-2, -1)

        # Do final matrix multiplication to get transmission
        # Imaginary parts cancel out, only real part is left
        Tcoh = torch.matmul(gamma1Gr_real, gamma2Ga_real) - torch.matmul(gamma1Gr_imag, gamma2Ga_imag)
        T_raw = torch.einsum('benn->be', Tcoh)
        # Transmission should be positive by construction, add safety check
        T_safe = torch.clamp(T_raw, min=1e-16)  # Ensure positive values
        T = torch.log10(T_safe)

        if squeeze_output:
            return T.squeeze(0), DOS.squeeze(0), H_matrix.squeeze(0)
        else:
            return T, DOS, H_matrix

    def NEGFProjectionComplex(self,
        H_matrix: torch.Tensor,
        GammaL: torch.Tensor,
        GammaR: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact complex-valued NEGF with wide-band, purely imaginary self-energies.
        Returns log10 or linear outputs depending on flags.
        """
        # Handle both single sample and batch cases
        if H_matrix.dim() == 2:
            H_matrix = H_matrix.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Ensure GammaL and GammaR have batch dimension
        if GammaL.dim() == 1:
            GammaL = GammaL.unsqueeze(0)
        if GammaR.dim() == 1:
            GammaR = GammaR.unsqueeze(0)

        batch_size = H_matrix.size(0)
        H_size = H_matrix.size(1)
        device = H_matrix.device

        def maybe_log10(x: torch.Tensor) -> torch.Tensor:
            if self.use_log_outputs:
                return torch.log10(torch.clamp(x, min=self.log_floor))
            return x

        dtype_real = H_matrix.dtype
        H_expanded = H_matrix.unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)
        energy = torch.tensor(self.energy_grid, dtype=dtype_real, device=device)
        energy = energy.view(1, -1, 1, 1).expand(batch_size, -1, H_size, H_size)
        I = torch.eye(H_size, dtype=dtype_real, device=device).view(1, 1, H_size, H_size).expand_as(H_expanded)

        gamma_total = (GammaL + GammaR)
        Sigma_im = -0.5 * torch.diag_embed(gamma_total)  # [batch, H_size, H_size]
        Sigma = 1j * Sigma_im.unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)

        # Small positive imaginary part for causality
        eta = self.complex_eta
        A = (energy + 1j * eta) * I - H_expanded - Sigma

        # Solve A @ Gr = I
        I_c = torch.eye(H_size, dtype=torch.complex64 if dtype_real==torch.float32 else torch.complex128, device=device)
        I_c = I_c.view(1, 1, H_size, H_size).expand_as(A)
        try:
            Gr = torch.linalg.solve(A, I_c)
        except torch.linalg.LinAlgError:
            Gr = torch.linalg.pinv(A)
        Ga = Gr.conj().transpose(-2, -1)

        # DOS = -1/pi * Im Tr(Gr)
        DOS_lin = (-1/np.pi) * torch.imag(torch.einsum('benn->be', Gr))
        DOS_lin = torch.clamp(DOS_lin, min=self.log_floor)

        # Transmission: Tr[GammaL Gr GammaR Ga]
        GammaL_mat = torch.diag_embed(GammaL).unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)
        GammaR_mat = torch.diag_embed(GammaR).unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)
        M = torch.matmul(torch.matmul(GammaL_mat, Gr), torch.matmul(GammaR_mat, Ga))
        T_lin = torch.real(torch.einsum('benn->be', M))
        T_lin = torch.clamp(T_lin, min=self.log_floor)

        DOS = maybe_log10(DOS_lin)
        T = maybe_log10(T_lin)

        if squeeze_output:
            return T.squeeze(0), DOS.squeeze(0), H_matrix.squeeze(0)
        else:
            return T, DOS, H_matrix


    def get_contact_vectors(self, x: torch.Tensor,
                        edge_attr: torch.Tensor,
                        edge_index: torch.Tensor,
                        batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract left and right contact data from a PyTorch Geometric Data object.
        Assumes left contact is node 0 and right contact is last node in each graph.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch indices [num_nodes]

        Returns:
            gammaL: Left lead coupling [batch_size, H_size] or [H_size] for single graph
            gammaR: Right lead coupling [batch_size, H_size] or [H_size] for single graph
        """
        device = x.device

        # Handle batched processing
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        batch_size = int(batch.max().item() + 1)

        # Calculate nodes per graph and H_size from batch vector
        node_counts = torch.bincount(batch, minlength=batch_size)
        dna_counts = []
        for b_idx in range(batch_size):
            count = int(node_counts[b_idx].item())
            # Two contacts per graph
            dna_counts.append(max(0, count - 2))

        if batch_size > 1 and not all(dc == dna_counts[0] for dc in dna_counts):
            raise ValueError(
                f"All graphs in a batch must have the same number of DNA nodes (nodes-2). Got {dna_counts}. "
                "Enable length-bucketing DataLoader for the hamiltonian model."
            )
        num_dna_nodes = dna_counts[0]
        H_size = num_dna_nodes * self.n_orb

        # Initialize output tensors
        GammaL_batch = torch.zeros(batch_size, H_size, device=device, dtype=x.dtype)
        GammaR_batch = torch.zeros(batch_size, H_size, device=device, dtype=x.dtype)

        # Process each graph in the batch
        for batch_idx in range(batch_size):
            # Get node range for this graph using counts rather than average
            start_node = int(torch.sum(node_counts[:batch_idx]).item())
            end_node = start_node + int(node_counts[batch_idx].item())
            graph_nodes = torch.arange(start_node, end_node, device=device)

            # Left contact is node 0, right contact is node 1
            left_contact_idx = graph_nodes[0]
            right_contact_idx = graph_nodes[1]

            # Get edges for this graph
            edge_mask = torch.isin(edge_index[0], graph_nodes) & torch.isin(edge_index[1], graph_nodes)
            graph_edge_index = edge_index[:, edge_mask]
            graph_edge_attr = edge_attr[edge_mask]

            # Find contact edges (edge_attr[:, 2] == 1 for contact type)
            contact_mask = (graph_edge_attr[:, 2] == 1)
            if contact_mask.sum() == 0:
                continue  # Skip if no contact edges found

            contact_edges = graph_edge_index[:, contact_mask]
            contact_couplings = graph_edge_attr[contact_mask, 4]  # Coupling strength at index 4

            # Find edges connected to left contact (node 0)
            left_contact_mask = (contact_edges[0] == left_contact_idx)
            left_contact_edges = contact_edges[:, left_contact_mask]
            left_contact_couplings = contact_couplings[left_contact_mask]

            # Find edges connected to right contact (last node)
            right_contact_mask = (contact_edges[0] == right_contact_idx)
            right_contact_edges = contact_edges[:, right_contact_mask]
            right_contact_couplings = contact_couplings[right_contact_mask]

            # Map DNA nodes to Hamiltonian indices (excluding contacts)
            # DNA nodes are indices 2 to nodes_per_graph-1 (after both contacts)
            dna_start = graph_nodes[2]  # First DNA node (after both contacts)

            # Set left contact couplings
            for i, coupling in enumerate(left_contact_couplings):
                if i < len(left_contact_edges[1]):
                    dna_node = left_contact_edges[1, i]
                    # Map to Hamiltonian index (subtract contact offset)
                    dna_idx = dna_node - dna_start  # 0-indexed DNA node
                    if 0 <= dna_idx < num_dna_nodes:
                        # Apply coupling to all orbitals of this DNA node
                        orb_start = dna_idx * self.n_orb
                        orb_end = orb_start + self.n_orb
                        if orb_end <= H_size:
                            GammaL_batch[batch_idx, orb_start:orb_end] = coupling

            # Set right contact couplings
            for i, coupling in enumerate(right_contact_couplings):
                if i < len(right_contact_edges[1]):
                    dna_node = right_contact_edges[1, i]
                    # Map to Hamiltonian index (subtract contact offset)
                    dna_idx = dna_node - dna_start  # 0-indexed DNA node
                    if 0 <= dna_idx < num_dna_nodes:
                        # Apply coupling to all orbitals of this DNA node
                        orb_start = dna_idx * self.n_orb
                        orb_end = orb_start + self.n_orb
                        if orb_end <= H_size:
                            GammaR_batch[batch_idx, orb_start:orb_end] = coupling

        # Return squeezed tensors for single batch case to maintain backward compatibility
        if batch_size == 1:
            return GammaL_batch.squeeze(0), GammaR_batch.squeeze(0)

        return GammaL_batch, GammaR_batch

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

        # Get initial node and edge features for Hamiltonian construction
        x_initial = x.clone()  # Keep original features for contact detection
        edge_attr_initial = edge_attr.clone()

        # Compare model gammas with training data gammas for debugging
        # Uncomment the following block to enable gamma debugging during training
        # if hasattr(data, 'gamma_l') and data.gamma_l is not None and hasattr(data, 'gamma_r') and data.gamma_r is not None:
        #     print("=== GAMMA COMPARISON ===")
        #     print(f"Model GammaL shape: {GammaL.shape}")
        #     print(f"Model GammaR shape: {GammaR.shape}")
        #     print(f"Training gamma_l shape: {data.gamma_l.shape}")
        #     print(f"Training gamma_r shape: {data.gamma_r.shape}")
        #
        #     batch_size = GammaL.size(0)
        #     for i in range(min(batch_size, 3)):  # Print first 3 samples to avoid spam
        #         model_gamma_l = GammaL[i]
        #         model_gamma_r = GammaR[i]
        #         train_gamma_l = data.gamma_l[i] if data.gamma_l.dim() > 1 else data.gamma_l
        #         train_gamma_r = data.gamma_r[i] if data.gamma_r.dim() > 1 else data.gamma_r
        #
        #         # Find non-zero positions
        #         model_active_l = torch.nonzero(model_gamma_l).flatten()
        #         model_active_r = torch.nonzero(model_gamma_r).flatten()
        #         train_active_l = torch.nonzero(train_gamma_l).flatten()
        #         train_active_r = torch.nonzero(train_gamma_r).flatten()
        #
        #         print(f"Sample {i}:")
        #         # Format the gamma values for display
        #         model_l_values = [f"{model_gamma_l[pos].item():.3f}" for pos in model_active_l]
        #         model_r_values = [f"{model_gamma_r[pos].item():.3f}" for pos in model_active_r]
        #         train_l_values = [f"{train_gamma_l[pos].item():.3f}" for pos in train_active_l]
        #         train_r_values = [f"{train_gamma_r[pos].item():.3f}" for pos in train_active_r]
        #
        #         print(f"  Model GammaL active: {model_active_l.tolist()} = {model_l_values}")
        #         print(f"  Model GammaR active: {model_active_r.tolist()} = {model_r_values}")
        #         print(f"  Train gamma_l active: {train_active_l.tolist()} = {train_l_values}")
        #         print(f"  Train gamma_r active: {train_active_r.tolist()} = {train_r_values}")
        #
        #         # Check if they match
        #         l_match = torch.allclose(model_gamma_l, train_gamma_l, atol=1e-6)
        #         r_match = torch.allclose(model_gamma_r, train_gamma_r, atol=1e-6)
        #         print(f"  GammaL match: {l_match}")
        #         print(f"  GammaR match: {r_match}")
        #         if not l_match or not r_match:
        #             print(f"  WARNING: Gamma mismatch detected!")
        #     print("========================")

        # Project node and edge features
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Construct Hamiltonian matrix directly from graph structure
        H_matrix, H_size = self.construct_hamiltonian_from_graph(x, edge_attr, edge_index, batch, x_initial)

        # Get contact vectors using original features for contact detection
        GammaL, GammaR = self.get_contact_vectors(x_initial, edge_attr_initial, edge_index, batch)

        # Calculate transport properties using selected NEGF solver
        if getattr(self, 'solver_type', 'frobenius') == 'complex':
            out1, out2, H = self.NEGFProjectionComplex(H_matrix, GammaL, GammaR)
        else:
            out1, out2, H = self.NEGFProjection(H_matrix, GammaL, GammaR)

        # Preserve existing variable order for backward compatibility
        dos_pred, transmission_pred = out1, out2

        self.H = H

        return dos_pred, transmission_pred
