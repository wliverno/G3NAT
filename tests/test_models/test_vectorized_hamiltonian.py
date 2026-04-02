# tests/test_models/test_vectorized_hamiltonian.py
"""Tests for vectorized Hamiltonian construction and contact vectors."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import pytest
from torch_geometric.data import Data, Batch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph


def _make_model(**kwargs):
    """Create a model with small dims for fast testing."""
    defaults = dict(
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        energy_grid=np.linspace(-1, 1, 10),
        n_orb=1,
        enforce_hermiticity=True,
        solver_type='frobenius',
        conv_type='gat',
    )
    defaults.update(kwargs)
    torch.manual_seed(42)
    return defaults, DNATransportHamiltonianGNN(**defaults)


def _run_gnn_layers(model, data):
    """Run the GNN layers to get processed features (mirrors forward() lines 591-598)."""
    x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    x_initial = x.clone()
    edge_attr_initial = edge_attr.clone()
    x = model.node_proj(x)
    edge_attr_proj = model.edge_proj(edge_attr)
    for i in range(model.num_layers):
        x = model.convs[i](x, edge_index, edge_attr_proj)
        x = model.norms[i](x)
        x = torch.nn.functional.relu(x)
    return x, edge_attr_proj, edge_index, batch, x_initial, edge_attr_initial


# ---- construct_hamiltonian_from_graph tests (n_orb=1) ----

class TestConstructHamiltonianNOrb1:
    """Test vectorized construct_hamiltonian_from_graph with n_orb=1."""

    def test_single_graph_matches_reference(self):
        """Vectorized H matches reference implementation for a single graph."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H_vec, size_vec = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
            H_ref, size_ref = model._construct_hamiltonian_reference(
                x, edge_attr, edge_index, batch, x_init)
        assert size_vec == size_ref
        assert torch.allclose(H_vec, H_ref, atol=1e-6), \
            f"Max diff: {(H_vec - H_ref).abs().max()}"

    def test_batched_graphs_match_reference(self):
        """Vectorized H matches reference for a batch of identical-length graphs."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graphs = [
            sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1),
            sequence_to_graph("TGCA", "TGCA", 0, 3, 0.2, 0.2),
            sequence_to_graph("GGCC", "GGCC", 0, 3, 0.1, 0.3),
        ]
        data = Batch.from_data_list(graphs)
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H_vec, size_vec = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
            H_ref, size_ref = model._construct_hamiltonian_reference(
                x, edge_attr, edge_index, batch, x_init)
        assert size_vec == size_ref
        assert torch.allclose(H_vec, H_ref, atol=1e-6), \
            f"Max diff: {(H_vec - H_ref).abs().max()}"

    def test_hermiticity_single(self):
        """H is symmetric (real Hermitian) for a single graph."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graph = sequence_to_graph("ACGTACGT", "ACGTACGT", 0, 7, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H, _ = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
        assert torch.allclose(H, H.transpose(-1, -2), atol=1e-7), \
            f"Hermiticity violation: max diff = {(H - H.transpose(-1, -2)).abs().max()}"

    def test_hermiticity_batched(self):
        """H is symmetric for every graph in a batch."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graphs = [
            sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1),
            sequence_to_graph("TGCA", "TGCA", 0, 3, 0.2, 0.2),
        ]
        data = Batch.from_data_list(graphs)
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H, _ = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
        for b in range(H.size(0)):
            assert torch.allclose(H[b], H[b].T, atol=1e-7), \
                f"Graph {b}: hermiticity violation, max diff = {(H[b] - H[b].T).abs().max()}"

    def test_single_dna_node(self):
        """Edge case: single base pair (1 DNA node per strand, 2 total)."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graph = sequence_to_graph("A", "T", 0, 0, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H_vec, size_vec = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
            H_ref, size_ref = model._construct_hamiltonian_reference(
                x, edge_attr, edge_index, batch, x_init)
        assert size_vec == size_ref
        assert torch.allclose(H_vec, H_ref, atol=1e-6)
        assert torch.allclose(H_vec, H_vec.transpose(-1, -2), atol=1e-7)


# ---- construct_hamiltonian_from_graph tests (n_orb>1) ----

class TestConstructHamiltonianNOrbMulti:
    """Test vectorized construct_hamiltonian_from_graph with n_orb>1."""

    def test_norb2_single_graph_matches_reference(self):
        """Vectorized H matches reference for n_orb=2, single graph."""
        _, model = _make_model(n_orb=2)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H_vec, size_vec = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
            H_ref, size_ref = model._construct_hamiltonian_reference(
                x, edge_attr, edge_index, batch, x_init)
        assert size_vec == size_ref
        assert H_vec.shape[-1] == 8 * 2  # 8 DNA nodes * 2 orbitals
        assert torch.allclose(H_vec, H_ref, atol=1e-6), \
            f"Max diff: {(H_vec - H_ref).abs().max()}"

    def test_norb2_batched_matches_reference(self):
        """Vectorized H matches reference for n_orb=2, batched."""
        _, model = _make_model(n_orb=2)
        model.eval()
        graphs = [
            sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1),
            sequence_to_graph("TGCA", "TGCA", 0, 3, 0.2, 0.2),
        ]
        data = Batch.from_data_list(graphs)
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H_vec, size_vec = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
            H_ref, size_ref = model._construct_hamiltonian_reference(
                x, edge_attr, edge_index, batch, x_init)
        assert size_vec == size_ref
        assert torch.allclose(H_vec, H_ref, atol=1e-6), \
            f"Max diff: {(H_vec - H_ref).abs().max()}"

    def test_norb2_hermiticity(self):
        """H is symmetric for n_orb=2 (orbital blocks must be symmetric too)."""
        _, model = _make_model(n_orb=2)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H, _ = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
        assert torch.allclose(H, H.transpose(-1, -2), atol=1e-7), \
            f"Hermiticity violation: max diff = {(H - H.transpose(-1, -2)).abs().max()}"

    def test_norb3_hermiticity_batched(self):
        """H is symmetric for n_orb=3 across a batch."""
        _, model = _make_model(n_orb=3)
        model.eval()
        graphs = [
            sequence_to_graph("ACG", "CGT", 0, 2, 0.1, 0.1),
            sequence_to_graph("TGC", "GCA", 0, 2, 0.2, 0.3),
        ]
        data = Batch.from_data_list(graphs)
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H, _ = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
        for b in range(H.size(0)):
            assert torch.allclose(H[b], H[b].T, atol=1e-7), \
                f"Graph {b} n_orb=3: hermiticity violation, max diff = {(H[b] - H[b].T).abs().max()}"

    def test_norb2_diagonal_blocks_symmetric(self):
        """Each n_orb x n_orb diagonal block must itself be symmetric."""
        _, model = _make_model(n_orb=2, enforce_hermiticity=True)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            x, edge_attr, edge_index, batch, x_init, _ = _run_gnn_layers(model, data)
            H, H_size = model.construct_hamiltonian_from_graph(
                x, edge_attr, edge_index, batch, x_init)
        n_orb = 2
        num_sites = H_size // n_orb
        for site in range(num_sites):
            s = site * n_orb
            e = s + n_orb
            block = H[0, s:e, s:e]
            assert torch.allclose(block, block.T, atol=1e-7), \
                f"Site {site} diagonal block not symmetric: {block}"
