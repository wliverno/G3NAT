from numpy._typing import _128Bit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
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
                 dropout: float = 0.1):
        super().__init__()
        # Use features specified in dataset.py
        node_features = 6  # 6 one-hot features (A, T, G, C, Purine, Pyrimidine)
        edge_features = 5  # 3 one-hot + directionality + coupling
        
        
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
                          dropout=dropout, add_self_loops=True, edge_dim=hidden_dim)
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
        transmission_pred = self.transmission_proj(x)
        
        return dos_pred, transmission_pred



class DNATransportHamiltonianGNN(nn.Module):
    """Graph Neural Network for DNA transport Hamiltonian prediction."""
    
    def __init__(self, 
                 hidden_dim: int = 128, 
                 num_layers: int = 4,
                 num_heads: int = 4,
                 energy_grid: np.ndarray = np.linspace(-3, 3, 100),
                 max_len_dna: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        # Use features specified in dataset.py
        node_features = 6  # 6 one-hot features (A, T, G, C, Purine, Pyrimidine)
        edge_features = 5  # 3 one-hot + directionality + coupling
        
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.energy_grid = energy_grid
        self.dropout = dropout
        self.output_dim = len(energy_grid)

        # Hamiltonian size specification
        self.H_size = max_len_dna*2
        # number of unique elements in upper triangular + diagonal
        self.num_unique_elements = self.H_size + (self.H_size * (self.H_size - 1)) // 2
        
        # Input projections
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                          dropout=dropout, add_self_loops=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projections for DOS and transmission
        self.H_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_unique_elements)
        )
        
        # Global pooling
        self.global_pool = global_mean_pool
        
    def NEGFProjection(self, 
        H_triangular: torch.Tensor, 
        GammaL: torch.Tensor, 
        GammaR: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the transmission and DOS using the NEGF method.
        
        Args:
            H_triangular: Upper triangular Hamiltonian elements [batch_size, num_unique_elements]
            GammaL: Left lead coupling [batch_size, H_size]
            GammaR: Right lead coupling [batch_size, H_size]

        Returns:
            Tuple of (transmission, DOS, H) tensors for output
        """
        
        # Handle both single sample and batch cases
        if H_triangular.dim() == 1:
            H_triangular = H_triangular.unsqueeze(0)
            GammaL = GammaL.unsqueeze(0)
            GammaR = GammaR.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = H_triangular.size(0)
        device = H_triangular.device
        i_indices, j_indices = torch.triu_indices(self.H_size, self.H_size, device=device)
        H_matrix = torch.zeros(batch_size, self.H_size, self.H_size, 
                        dtype=H_triangular.dtype, device=device)
        H_matrix[:, i_indices, j_indices] = H_triangular
        H_matrix[:, j_indices, i_indices] = H_triangular.conj()
        
        # Create Identity Matrix
        I = torch.eye(self.H_size, dtype=H_triangular.dtype, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, H_size, H_size]
        I = I.expand(batch_size, len(self.energy_grid), self.H_size, self.H_size) # [batch, energy, H_size, H_size]

        # Expand Hamiltonian and sigTot to match energy grid
        H_expanded = H_matrix.unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)  # [batch, energy, H_size, H_size]
        sigTot = (-0.5j * (GammaL + GammaR)).unsqueeze(1).unsqueeze(-1).expand(-1, len(self.energy_grid), -1, -1)

        # Expand energy grid to match batch size and H_size
        energy_grid_tensor = torch.tensor(self.energy_grid, dtype=H_triangular.dtype, device=device)  # [num_energy]
        energy_grid_expanded = energy_grid_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, num_energy, 1, 1]

        # Calculate A matrix
        A = energy_grid_expanded * I - H_expanded - sigTot

        # Calculate Green's functions
        # Use more stable matrix solve instead of direct inversion
        try:
            Gr = torch.linalg.solve(A, I+0j)
        except torch.linalg.LinAlgError:
            # Fallback: add small regularization and use pseudo-inverse
            A_reg = A + 1e-8j * I
            Gr = torch.linalg.pinv(A_reg)
        
        # Calculate DOS using einsum for trace
        DOS = -1*torch.einsum('benn->be', torch.imag(Gr))

        # Calculate transmission
        Ga = Gr.conj().transpose(-2, -1)
        # Convert gamma vectors to diagonal matrices for matrix multiplication
        # [batch, H_size] -> [batch, energy, H_size, 1]
        GammaL_diag = (GammaL + 0j).unsqueeze(1).unsqueeze(-1).expand(-1, len(self.energy_grid), -1, -1)
        GammaR_diag = (GammaR + 0j).unsqueeze(1).unsqueeze(-1).expand(-1, len(self.energy_grid), -1, -1)
 
        # Element-wise multiplication (equivalent to diagonal matrix multiplication)
        # Gr: [batch, energy, H_size, H_size]
        # Gamma_diag: [batch, energy, H_size, 1]
        # Result: [batch, energy, H_size, H_size]
        gamma1Gr = GammaL_diag * Gr
        gamma2Ga = GammaR_diag * Ga

        # Do final matrix multiplication to get transmission
        Tcoh = torch.matmul(gamma1Gr, gamma2Ga)
        T = torch.real(torch.einsum('benn->be', Tcoh))

        
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
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch indices [num_nodes]
                
        Returns:
            gammaL: Left lead coupling [batch_size, H_size]
            gammaR: Right lead coupling [batch_size, H_size]
        """
        device = x.device
        
        # Handle batched processing
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        batch_size = batch.max().item() + 1
        
        # Initialize output tensors
        GammaL_batch = torch.zeros(batch_size, self.H_size, device=device, dtype=x.dtype)
        GammaR_batch = torch.zeros(batch_size, self.H_size, device=device, dtype=x.dtype)
        
        # Process each graph in the batch
        for batch_idx in range(batch_size):
            # Get nodes and edges for this graph
            node_mask = batch == batch_idx
            node_indices = torch.where(node_mask)[0]
            
            # Get edges for this graph
            edge_mask = torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices)
            graph_edge_index = edge_index[:, edge_mask]
            graph_edge_attr = edge_attr[edge_mask]
            graph_x = x[node_mask]
            
            # Create mapping from global to local node indices
            global_to_local = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(node_indices)}
            
            # Find contact nodes (all zero features indicate contact nodes)
            contact_node_mask = torch.all(graph_x == 0, dim=1)
            contact_inds = torch.where(contact_node_mask)[0]
            dna_inds = torch.where(~contact_node_mask)[0]

            if len(contact_inds) < 2:
                continue  # Skip if no proper contacts found
            
            # Map global edge indices to local indices
            local_edge_index = torch.zeros_like(graph_edge_index)
            for i in range(graph_edge_index.size(1)):
                local_edge_index[0, i] = global_to_local[graph_edge_index[0, i].item()]
                local_edge_index[1, i] = global_to_local[graph_edge_index[1, i].item()]

            # Find contact edges (edge_attr[:, 2] == 1 for contact type)
            contact_mask = (graph_edge_attr[:, 2] == 1) 
            contact_edges = local_edge_index[:, contact_mask]
            contact_couplings = graph_edge_attr[contact_mask, 4]  # Coupling strength is at index 4
            
            if len(contact_edges[0]) == 0:
                continue  # Skip if no contact edges found
            
            # Find edges connected to left contact (first contact node)
            left_contact_mask = (contact_edges[0] == contact_inds[0]) 
            left_contact_edges = contact_edges[:, left_contact_mask]
            left_contact_couplings = contact_couplings[left_contact_mask]
            
            # Find edges connected to right contact (last contact node)
            right_contact_mask = (contact_edges[0] == contact_inds[-1]) 
            right_contact_edges = contact_edges[:, right_contact_mask]
            right_contact_couplings = contact_couplings[right_contact_mask]
            
            # Get the DNA base nodes connected to contacts (not the contact nodes themselves)
            left_contact_indices = [edge[1].item() for edge in left_contact_edges.T] if len(left_contact_edges[0]) > 0 else []
            right_contact_indices = [edge[1].item() for edge in right_contact_edges.T] if len(right_contact_edges[0]) > 0 else []

            # Initialize coupling vectors for this graph
            GammaL = torch.zeros(self.H_size, device=device, dtype=x.dtype)
            GammaR = torch.zeros(self.H_size, device=device, dtype=x.dtype)
            
            # Set left contact couplings
            for i, coupling in enumerate(left_contact_couplings):
                if i < len(left_contact_indices):
                    ind = torch.where(dna_inds == left_contact_indices[i])[0]
                    if len(ind) > 0 and ind[0] < self.H_size:
                        GammaL[ind[0]] = coupling
            
            # Set right contact couplings
            for i, coupling in enumerate(right_contact_couplings):
                if i < len(right_contact_indices):
                    ind = torch.where(dna_inds == right_contact_indices[i])[0]
                    if len(ind) > 0 and ind[0] < self.H_size:
                        GammaR[ind[0]] = coupling

            GammaL_batch[batch_idx] = GammaL
            GammaR_batch[batch_idx] = GammaR

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
        
        # First, get the contact vectors (returns batched tensors)
        GammaL, GammaR = self.get_contact_vectors(x, edge_attr, edge_index, batch)
        
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

        # Hidden layer (outputs batch_size x num_unique_elements)
        x = self.H_proj(x)
        
        # Output projections (handles batched inputs)
        dos_pred, transmission_pred, H = self.NEGFProjection(x, GammaL, GammaR)
        
        self.H = H

        return dos_pred, transmission_pred

class DNAHamiltonianGNN(nn.Module):
    """Graph Neural Network for DNA Hamiltonian prediction."""
    
    def __init__(self, 
                 hidden_dim: int = 128, 
                 num_layers: int = 4,
                 num_heads: int = 4,
                 energy_grid: np.ndarray = np.linspace(-3, 3, 100),
                 max_len_dna: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        # Use features specified in dataset.py
        node_features = 6  # 6 one-hot features (A, T, G, C, Purine, Pyrimidine)
        edge_features = 5  # 3 one-hot + directionality + coupling
        
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.energy_grid = energy_grid
        self.dropout = dropout
        self.output_dim = len(energy_grid)

        # Hamiltonian size specification
        self.H_size = max_len_dna*2
        # number of unique elements in upper triangular + diagonal
        self.num_unique_elements = self.H_size + (self.H_size * (self.H_size - 1)) // 2
        
        # Input projections
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)
        
        # Graph attention layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            conv = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                          dropout=dropout, add_self_loops=True, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projections for Hamiltonian elements
        self.H_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_unique_elements)
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
            Hamiltonian elements tensor [batch_size, num_unique_elements]
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

        # Output Hamiltonian elements
        H_elements = self.H_proj(x)
        
        return H_elements
class PhysicsDiscriminator(nn.Module):
    def __init__(self, energy_grid: np.ndarray, max_len_dna: int, n_onsite: int = 1):
        super().__init__()
        self.energy_grid = energy_grid  # Energy points
        self.H_size = max_len_dna*2*n_onsite

    def forward(self, H_triangular: torch.Tensor, GammaL: np.ndarray, GammaR: np.ndarray):
        # Handle both single sample and batch cases
        batch_size = H_triangular.size(0)
        device = H_triangular.device

        # Convert gamma vectors to diagonal matrices for matrix multiplication
        # [batch, H_size] -> [batch, energy, H_size, 1]
        GammaL_diag = torch.tensor(GammaL, dtype=H_triangular.dtype, device=device)
        GammaL_diag = GammaL_diag.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, len(self.energy_grid), -1, -1)
        GammaR_diag = torch.tensor(GammaR, dtype=H_triangular.dtype, device=device)
        GammaR_diag = GammaR_diag.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, len(self.energy_grid), -1, -1)

        # Create Hamiltonian matrix
        i_indices, j_indices = torch.triu_indices(self.H_size, self.H_size, device=device)
        H_matrix = torch.zeros(batch_size, self.H_size, self.H_size, 
                        dtype=H_triangular.dtype, device=device)
        H_matrix[:, i_indices, j_indices] = H_triangular
        H_matrix[:, j_indices, i_indices] = H_triangular.conj()
        
        # Create Identity Matrix
        I = torch.eye(self.H_size, dtype=H_triangular.dtype, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, H_size, H_size]
        I = I.expand(batch_size, len(self.energy_grid), self.H_size, self.H_size) # [batch, energy, H_size, H_size]

        # Expand Hamiltonian and sigTot to match energy grid
        H_expanded = H_matrix.unsqueeze(1).expand(-1, len(self.energy_grid), -1, -1)  # [batch, energy, H_size, H_size]
        sigTot = -0.5j *torch.tensor(np.diag(GammaL + GammaR), dtype=H_triangular.dtype, device=device)
        sigTot = sigTot.unsqueeze(0).unsqueeze(0).expand(batch_size, len(self.energy_grid), -1, -1)

        # Expand energy grid to match batch size and H_size
        energy_grid_tensor = torch.tensor(self.energy_grid, dtype=H_triangular.dtype, device=device)  # [num_energy]
        energy_grid_expanded = energy_grid_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, num_energy, 1, 1]

        # Calculate A matrix
        A = energy_grid_expanded * I - H_expanded - sigTot
        # Calculate Green's functions
        # Use more stable matrix solve instead of direct inversion
        try:
            Gr = torch.linalg.solve(A, I+0j)
        except torch.linalg.LinAlgError:
            # Fallback: add small regularization and use pseudo-inverse
            A_reg = A + 1e-8j * I
            Gr = torch.linalg.pinv(A_reg)
        
        # Calculate DOS using einsum for trace
        DOS = -1*torch.einsum('benn->be', torch.imag(Gr))

        # Calculate transmission
        Ga = Gr.conj().transpose(-2, -1)
        
        # Element-wise multiplication (equivalent to diagonal matrix multiplication)
        # Gr: [batch, energy, H_size, H_size]
        # Gamma_diag: [batch, energy, H_size, 1]
        # Result: [batch, energy, H_size, H_size]
        gamma1Gr = GammaL_diag * Gr
        gamma2Ga = GammaR_diag * Ga

        # Do final matrix multiplication to get transmission
        Tcoh = torch.matmul(gamma1Gr, gamma2Ga)
        T = torch.real(torch.einsum('benn->be', Tcoh))
        return T, DOS

def train_physics_informed(
    train_loader,
    val_loader,
    energy_grid = np.linspace(-3, 3, 100),
    max_len_dna  = 8,
    gammaL = np.array([0.1]+[0.0]*7),
    gammaR = np.array([0.0]+[0.1]+[0.0]*6),
    epochs=100,
    lr=1e-3,
    device='auto'
):
    generator = DNAHamiltonianGNN(energy_grid=energy_grid, max_len_dna=max_len_dna).to(device)
    discriminator = PhysicsDiscriminator(energy_grid, max_len_dna).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    train_losses = []  
    val_losses = []
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        generator.train()
        discriminator.train()
        for batch in train_loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass (generates H and computes physics)
            H_pred = generator(batch)
            T_pred, rho_pred = discriminator(H_pred, gammaL, gammaR)
            
            # Physics-based loss
            # Reshape batched targets to match predictions
            batch_size = T_pred.size(0)
            num_energy_points = T_pred.size(1)
            
            transmission_target = batch.transmission.view(batch_size, num_energy_points)
            dos_target = batch.dos.view(batch_size, num_energy_points)
            
            loss_T = F.mse_loss(T_pred, transmission_target)
            loss_rho = F.mse_loss(rho_pred, dos_target)
            loss = loss_T + loss_rho
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        train_loss = total_loss/len(train_loader)
        train_losses.append(train_loss)
        # Validation loop
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                H_pred = generator(batch)
                T_pred, rho_pred = discriminator(H_pred, gammaL, gammaR)
                
                # Reshape batched targets to match predictions
                batch_size = T_pred.size(0)
                num_energy_points = T_pred.size(1)
                
                transmission_target = batch.transmission.view(batch_size, num_energy_points)
                dos_target = batch.dos.view(batch_size, num_energy_points)
                
                loss_T = F.mse_loss(T_pred, transmission_target)
                loss_rho = F.mse_loss(rho_pred, dos_target)
                loss = loss_T + loss_rho
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return generator, train_losses, val_losses  # Return trained Hamiltonian generator

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               num_epochs: int = 100, learning_rate: float = 1e-3, device: str = 'auto'):
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
            
            # Reshape batched targets to match predictions
            batch_size = dos_pred.size(0)
            num_energy_points = dos_pred.size(1)
            
            dos_target = batch.dos.view(batch_size, num_energy_points)
            transmission_target = batch.transmission.view(batch_size, num_energy_points)
            
            # Combined loss for DOS and transmission
            dos_loss = criterion(dos_pred, dos_target)
            transmission_loss = criterion(transmission_pred, transmission_target)
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
                
                # Reshape batched targets to match predictions
                batch_size = dos_pred.size(0)
                num_energy_points = dos_pred.size(1)
                
                dos_target = batch.dos.view(batch_size, num_energy_points)
                transmission_target = batch.transmission.view(batch_size, num_energy_points)
                
                dos_loss = criterion(dos_pred, dos_target)
                transmission_loss = criterion(transmission_pred, transmission_target)
                total_loss = dos_loss + transmission_loss
                
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses





def load_trained_model(model_path: str, device: str = 'auto'):
    """
    Load a trained DNA Transport GNN model.
    
    Args:
        model_path: Path to the saved model (.pth file)
        device: Device to load model on ('auto', 'cpu', 'cuda')
        
    Returns:
        Loaded model and training arguments
    """
    if device == 'auto':
        device_tensor = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_tensor = torch.device(device)
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device_tensor}")
    
    # Load the saved model (allow numpy arrays for energy grid)
    checkpoint = torch.load(model_path, map_location=device_tensor, weights_only=False)
    
    # Extract model arguments
    args = checkpoint.get('args', {})
    energy_grid = checkpoint.get('energy_grid', np.linspace(-3, 3, 100))
    
    # Initialize model with same architecture
    model = DNATransportGNN(
        hidden_dim=args.get('hidden_dim', 128),
        num_layers=args.get('num_layers', 4),
        num_heads=args.get('num_heads', 4),
        output_dim=args.get('num_energy_points', 100),
        dropout=args.get('dropout', 0.1)
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device_tensor)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Energy grid: {len(energy_grid)} points from {energy_grid[0]:.2f} to {energy_grid[-1]:.2f} eV")
    
    return model, energy_grid, device_tensor


def predict_sequence(model, sequence: str, complementary_sequence: str,
                    left_contact_positions=0, right_contact_positions=None,
                    left_contact_coupling=0.1, right_contact_coupling=0.2):
    """
    Predict DOS and transmission for a DNA sequence.
    
    Args:
        model: Trained DNATransportGNN model
        sequence: DNA sequence string (e.g., "ACGTACGT")
        complementary_sequence: Complementary DNA sequence string (e.g., "GCATGCAT")
        left_contact_positions: Position(s) for left contact
        right_contact_positions: Position(s) for right contact (default: last position)
        left_contact_coupling: Coupling strength for left contact
        right_contact_coupling: Coupling strength for right contact
        
    Returns:
        Tuple of (dos_pred, transmission_pred) arrays
    """
    if right_contact_positions is None:
        right_contact_positions = len(sequence) - 1
    
    print(f"Predicting for sequence: {sequence}")
    print(f"Left contact at position {left_contact_positions}, coupling: {left_contact_coupling}")
    print(f"Right contact at position {right_contact_positions}, coupling: {right_contact_coupling}")
    
    # Import here to avoid circular imports
    from dataset import sequence_to_graph
    
    # Convert sequence to graph
    graph = sequence_to_graph(
        primary_sequence=sequence,
        complementary_sequence=complementary_sequence,
        left_contact_positions=left_contact_positions,
        right_contact_positions=right_contact_positions,
        left_contact_coupling=left_contact_coupling,
        right_contact_coupling=right_contact_coupling
    )
    
    if graph is None:
        raise ValueError(f"Failed to create graph for sequence: {sequence}")
    
    # Create batch (single graph)
    batch_data = Batch.from_data_list([graph])
    batch_data = batch_data.to(next(model.parameters()).device)  # type: ignore
    
    # Make prediction
    with torch.no_grad():
        dos_pred, transmission_pred = model(batch_data)
        
        # Convert to numpy arrays
        dos_pred = dos_pred.cpu().numpy()[0]  # Remove batch dimension
        transmission_pred = transmission_pred.cpu().numpy()[0]
    
    print(f"Prediction completed!")
    print(f"DOS range: [{dos_pred.min():.4f}, {dos_pred.max():.4f}]")
    print(f"Transmission range: [{transmission_pred.min():.4f}, {transmission_pred.max():.4f}]")
    
    return dos_pred, transmission_pred


def predict_multiple_sequences(model, sequences: list, energy_grid: np.ndarray,
                             contact_config=None):
    """
    Predict DOS and transmission for multiple DNA sequences.
    
    Args:
        model: Trained DNATransportGNN model
        sequences: List of DNA sequence strings
        energy_grid: Energy grid for predictions
        contact_config: Dictionary with contact configuration for each sequence
        
    Returns:
        Tuple of (dos_predictions, transmission_predictions) arrays
    """
    if contact_config is None:
        contact_config = {}
    
    dos_predictions = []
    transmission_predictions = []
    
    for i, seq in enumerate(sequences):
        print(f"\n--- Sequence {i+1}/{len(sequences)} ---")
        
        # Get contact configuration for this sequence
        config = contact_config.get(i, {})
        left_pos = config.get('left_contact_positions', 0)
        right_pos = config.get('right_contact_positions', len(seq) - 1)
        left_coupling = config.get('left_contact_coupling', 0.1)
        right_coupling = config.get('right_contact_coupling', 0.2)
        
        try:
            dos_pred, trans_pred = predict_sequence(
                model, seq, seq,  # Use same sequence as complementary for now
                left_contact_positions=left_pos,
                right_contact_positions=right_pos,
                left_contact_coupling=left_coupling,
                right_contact_coupling=right_coupling
            )
            
            dos_predictions.append(dos_pred)
            transmission_predictions.append(trans_pred)
            
        except Exception as e:
            print(f"Error predicting sequence {seq}: {e}")
            # Add zeros as fallback
            dos_predictions.append(np.zeros_like(energy_grid))
            transmission_predictions.append(np.zeros_like(energy_grid))
    
    return np.array(dos_predictions), np.array(transmission_predictions) 
