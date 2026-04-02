# Vectorize Hamiltonian Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Python loops in `construct_hamiltonian_from_graph` and `get_contact_vectors` with vectorized tensor operations for ~10x speedup.

**Architecture:** Keep the same method signatures and return values. Rename old methods to `_*_reference` for correctness testing. New implementations use `edge_index` directly as indexing tensors with `torch.cumsum`-based per-graph offsets. Hermiticity is guaranteed by building diagonal and off-diagonal parts separately, then symmetrizing only the off-diagonal part.

**Tech Stack:** PyTorch, torch_geometric (existing dependencies only)

---

## File Structure

- **Modify:** `g3nat/models/hamiltonian.py` — replace `construct_hamiltonian_from_graph` and `get_contact_vectors`
- **Create:** `tests/test_models/test_vectorized_hamiltonian.py` — correctness and hermiticity tests

---

### Task 1: Create test file with reference-comparison tests for `construct_hamiltonian_from_graph` (n_orb=1)

**Files:**
- Create: `tests/test_models/test_vectorized_hamiltonian.py`

- [ ] **Step 1: Write failing tests for vectorized Hamiltonian construction**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_models/test_vectorized_hamiltonian.py -v --tb=short 2>&1 | head -40`
Expected: FAIL — `_construct_hamiltonian_reference` does not exist yet.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_models/test_vectorized_hamiltonian.py
git commit -m "test: add vectorized Hamiltonian construction tests (n_orb=1 and n_orb>1)"
```

---

### Task 2: Rename old methods to `_*_reference`

**Files:**
- Modify: `g3nat/models/hamiltonian.py`

- [ ] **Step 1: Rename `construct_hamiltonian_from_graph` → `_construct_hamiltonian_reference`**

In `g3nat/models/hamiltonian.py`, rename the method at line 92:

```python
    def _construct_hamiltonian_reference(self,
```

- [ ] **Step 2: Add a new empty `construct_hamiltonian_from_graph` that calls the reference**

Add immediately after the renamed method (after line 220):

```python
    def construct_hamiltonian_from_graph(self,
                                       node_features: torch.Tensor,
                                       edge_features: torch.Tensor,
                                       edge_index: torch.Tensor,
                                       batch: torch.Tensor,
                                       original_node_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Vectorized Hamiltonian construction. Delegates to reference for now."""
        return self._construct_hamiltonian_reference(
            node_features, edge_features, edge_index, batch, original_node_features)
```

- [ ] **Step 3: Run tests to confirm they pass (reference delegates)**

Run: `python -m pytest tests/test_models/test_vectorized_hamiltonian.py -v --tb=short 2>&1 | head -50`
Expected: All tests PASS (vectorized == reference since it's the same code).

- [ ] **Step 4: Run existing tests to confirm nothing is broken**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py
git commit -m "refactor: rename construct_hamiltonian_from_graph to _construct_hamiltonian_reference"
```

---

### Task 3: Implement vectorized `construct_hamiltonian_from_graph`

**Files:**
- Modify: `g3nat/models/hamiltonian.py`

- [ ] **Step 1: Replace the delegating `construct_hamiltonian_from_graph` with vectorized implementation**

Replace the placeholder `construct_hamiltonian_from_graph` method with:

```python
    def construct_hamiltonian_from_graph(self,
                                       node_features: torch.Tensor,
                                       edge_features: torch.Tensor,
                                       edge_index: torch.Tensor,
                                       batch: torch.Tensor,
                                       original_node_features: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Construct Hamiltonian matrix directly from graph structure (vectorized).

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
        n_orb = self.n_orb

        # Handle batched processing
        if batch is None:
            batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)

        batch_size = int(batch.max().item() + 1)

        # --- Step 1: Identify DNA vs contact nodes ---
        contact_mask = torch.all(original_node_features == 0, dim=1)
        dna_mask = ~contact_mask

        # --- Step 2: Compute per-graph offsets and validate equal DNA counts ---
        node_counts = torch.bincount(batch, minlength=batch_size)
        # Contacts are always the first 2 nodes per graph
        dna_counts = node_counts - 2

        if batch_size > 1:
            if not torch.all(dna_counts == dna_counts[0]):
                raise ValueError(
                    f"All graphs in a batch must have the same number of DNA nodes. "
                    f"Got {dna_counts.tolist()}. "
                    "Enable length-bucketing DataLoader for the hamiltonian model."
                )

        num_dna_nodes = int(dna_counts[0].item())
        H_size = num_dna_nodes * n_orb

        # Per-graph start offsets: graph_starts[b] = sum of node_counts[:b]
        graph_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            graph_starts[1:] = torch.cumsum(node_counts[:-1], dim=0)

        # --- Step 3: Compute local Hamiltonian indices for all DNA nodes ---
        dna_indices = torch.where(dna_mask)[0]           # global indices of DNA nodes
        dna_batch = batch[dna_indices]                    # which graph each belongs to
        dna_local = dna_indices - graph_starts[dna_batch] - 2  # local index (0-based, contacts excluded)

        # --- Step 4: Compute onsite blocks and fill diagonal ---
        dna_features = node_features[dna_mask]
        onsite_raw = self.onsite_proj(dna_features)                  # [total_dna, n_orb²]
        onsite_blocks = onsite_raw.view(-1, n_orb, n_orb)           # [total_dna, n_orb, n_orb]
        if self.enforce_hermiticity:
            onsite_blocks = 0.5 * (onsite_blocks + onsite_blocks.transpose(-1, -2))

        H_diag = torch.zeros(batch_size, H_size, H_size, dtype=torch.float32, device=device)

        if n_orb == 1:
            H_diag[dna_batch, dna_local, dna_local] = onsite_blocks.view(-1)
        else:
            # Expand to orbital indices: for DNA node with local index L,
            # orbital indices are [L*n_orb, L*n_orb+1, ..., L*n_orb+n_orb-1]
            orb_offsets = torch.arange(n_orb, device=device)
            # row_orb[i, o] = dna_local[i] * n_orb + o
            row_orb = dna_local.unsqueeze(1) * n_orb + orb_offsets.unsqueeze(0)  # [total_dna, n_orb]
            # For each DNA node, scatter its n_orb x n_orb block onto the diagonal
            for oi in range(n_orb):
                for oj in range(n_orb):
                    H_diag[dna_batch, row_orb[:, oi], row_orb[:, oj]] = onsite_blocks[:, oi, oj]

        # --- Step 5: Filter edges to DNA-only pairs ---
        src, dst = edge_index
        both_dna = dna_mask[src] & dna_mask[dst]
        dna_src = src[both_dna]
        dna_dst = dst[both_dna]
        dna_edge_features = edge_features[both_dna]

        # Convert to local Hamiltonian indices
        src_batch = batch[dna_src]
        src_local = dna_src - graph_starts[src_batch] - 2
        dst_local = dna_dst - graph_starts[src_batch] - 2

        # --- Step 6: Deduplicate to upper triangle (src < dst), skip self-loops ---
        upper_mask = src_local < dst_local
        src_upper = src_local[upper_mask]
        dst_upper = dst_local[upper_mask]
        batch_upper = src_batch[upper_mask]
        edge_feat_upper = dna_edge_features[upper_mask]

        # Compute coupling blocks for upper-triangle edges only
        coupling_raw = self.coupling_proj(edge_feat_upper)         # [num_upper_edges, n_orb²]
        coupling_blocks = coupling_raw.view(-1, n_orb, n_orb)     # [num_upper_edges, n_orb, n_orb]

        # --- Step 7: Fill off-diagonal (upper triangle only) ---
        H_offdiag = torch.zeros(batch_size, H_size, H_size, dtype=torch.float32, device=device)

        if n_orb == 1:
            H_offdiag[batch_upper, src_upper, dst_upper] = coupling_blocks.view(-1)
        else:
            orb_offsets = torch.arange(n_orb, device=device)
            src_orb = src_upper.unsqueeze(1) * n_orb + orb_offsets.unsqueeze(0)  # [num_edges, n_orb]
            dst_orb = dst_upper.unsqueeze(1) * n_orb + orb_offsets.unsqueeze(0)  # [num_edges, n_orb]
            for oi in range(n_orb):
                for oj in range(n_orb):
                    H_offdiag[batch_upper, src_orb[:, oi], dst_orb[:, oj]] = coupling_blocks[:, oi, oj]

        # --- Step 8: Symmetrize and combine ---
        # H_offdiag has values only in upper triangle, so transpose gives lower triangle
        H_matrix = H_diag + H_offdiag + H_offdiag.transpose(-1, -2)

        # Diagonal shift for positive definiteness
        shift = 1e-6
        identity = torch.eye(H_size, device=device)
        H_matrix = H_matrix + shift * identity.unsqueeze(0).expand(batch_size, -1, -1)

        return H_matrix, H_size
```

- [ ] **Step 2: Run the reference-comparison tests**

Run: `python -m pytest tests/test_models/test_vectorized_hamiltonian.py -v --tb=short 2>&1 | head -50`
Expected: All tests PASS — vectorized output matches reference and hermiticity holds.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add g3nat/models/hamiltonian.py
git commit -m "perf: vectorize construct_hamiltonian_from_graph — eliminate Python loops"
```

---

### Task 4: Add tests for vectorized `get_contact_vectors`

**Files:**
- Modify: `tests/test_models/test_vectorized_hamiltonian.py`

- [ ] **Step 1: Add contact vector tests to the test file**

Append the following to `tests/test_models/test_vectorized_hamiltonian.py`:

```python
# ---- get_contact_vectors tests ----

class TestGetContactVectors:
    """Test vectorized get_contact_vectors."""

    def test_single_graph_matches_reference(self):
        """Vectorized gammas match reference for single graph."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            _, _, edge_index, batch, x_init, edge_attr_init = _run_gnn_layers(model, data)
            GL_vec, GR_vec = model.get_contact_vectors(x_init, edge_attr_init, edge_index, batch)
            GL_ref, GR_ref = model._get_contact_vectors_reference(x_init, edge_attr_init, edge_index, batch)
        assert torch.allclose(GL_vec, GL_ref, atol=1e-6), \
            f"GammaL max diff: {(GL_vec - GL_ref).abs().max()}"
        assert torch.allclose(GR_vec, GR_ref, atol=1e-6), \
            f"GammaR max diff: {(GR_vec - GR_ref).abs().max()}"

    def test_batched_matches_reference(self):
        """Vectorized gammas match reference for a batch."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graphs = [
            sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1),
            sequence_to_graph("TGCA", "TGCA", 0, 3, 0.2, 0.2),
            sequence_to_graph("GGCC", "GGCC", 0, 3, 0.1, 0.3),
        ]
        data = Batch.from_data_list(graphs)
        with torch.no_grad():
            _, _, edge_index, batch, x_init, edge_attr_init = _run_gnn_layers(model, data)
            GL_vec, GR_vec = model.get_contact_vectors(x_init, edge_attr_init, edge_index, batch)
            GL_ref, GR_ref = model._get_contact_vectors_reference(x_init, edge_attr_init, edge_index, batch)
        assert torch.allclose(GL_vec, GL_ref, atol=1e-6)
        assert torch.allclose(GR_vec, GR_ref, atol=1e-6)

    def test_norb2_single_graph(self):
        """Vectorized gammas match reference for n_orb=2."""
        _, model = _make_model(n_orb=2)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            _, _, edge_index, batch, x_init, edge_attr_init = _run_gnn_layers(model, data)
            GL_vec, GR_vec = model.get_contact_vectors(x_init, edge_attr_init, edge_index, batch)
            GL_ref, GR_ref = model._get_contact_vectors_reference(x_init, edge_attr_init, edge_index, batch)
        # With n_orb=2, H_size = num_dna_nodes * 2, coupling fills n_orb consecutive entries
        assert GL_vec.shape == GL_ref.shape
        assert torch.allclose(GL_vec, GL_ref, atol=1e-6)
        assert torch.allclose(GR_vec, GR_ref, atol=1e-6)

    def test_norb2_batched(self):
        """Vectorized gammas match reference for n_orb=2, batched."""
        _, model = _make_model(n_orb=2)
        model.eval()
        graphs = [
            sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1),
            sequence_to_graph("TGCA", "TGCA", 0, 3, 0.2, 0.2),
        ]
        data = Batch.from_data_list(graphs)
        with torch.no_grad():
            _, _, edge_index, batch, x_init, edge_attr_init = _run_gnn_layers(model, data)
            GL_vec, GR_vec = model.get_contact_vectors(x_init, edge_attr_init, edge_index, batch)
            GL_ref, GR_ref = model._get_contact_vectors_reference(x_init, edge_attr_init, edge_index, batch)
        assert torch.allclose(GL_vec, GL_ref, atol=1e-6)
        assert torch.allclose(GR_vec, GR_ref, atol=1e-6)

    def test_coupling_values_nonzero(self):
        """Gamma vectors should have non-zero entries at contact positions."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            _, _, edge_index, batch, x_init, edge_attr_init = _run_gnn_layers(model, data)
            GL, GR = model.get_contact_vectors(x_init, edge_attr_init, edge_index, batch)
        # Left contact connects to position 0 → GammaL[0] should be 0.1
        assert GL[0].item() == pytest.approx(0.1, abs=1e-6)
        # Right contact connects to position 3 → GammaR[3] should be 0.1
        assert GR[3].item() == pytest.approx(0.1, abs=1e-6)
```

- [ ] **Step 2: Run to verify tests fail**

Run: `python -m pytest tests/test_models/test_vectorized_hamiltonian.py::TestGetContactVectors -v --tb=short 2>&1 | head -30`
Expected: FAIL — `_get_contact_vectors_reference` does not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_models/test_vectorized_hamiltonian.py
git commit -m "test: add contact vector reference-comparison tests (n_orb=1 and n_orb>1)"
```

---

### Task 5: Rename old `get_contact_vectors` and implement vectorized version

**Files:**
- Modify: `g3nat/models/hamiltonian.py`

- [ ] **Step 1: Rename `get_contact_vectors` → `_get_contact_vectors_reference`**

In `g3nat/models/hamiltonian.py`, rename the method at line 411:

```python
    def _get_contact_vectors_reference(self, x: torch.Tensor,
```

- [ ] **Step 2: Add vectorized `get_contact_vectors`**

Add the new method after `_get_contact_vectors_reference`:

```python
    def get_contact_vectors(self, x: torch.Tensor,
                        edge_attr: torch.Tensor,
                        edge_index: torch.Tensor,
                        batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract left and right contact coupling vectors (vectorized).

        Args:
            x: Node features [num_nodes, node_features]
            edge_attr: Edge features [num_edges, edge_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]

        Returns:
            gammaL: Left lead coupling [batch_size, H_size] or [H_size] for single graph
            gammaR: Right lead coupling [batch_size, H_size] or [H_size] for single graph
        """
        device = x.device
        n_orb = self.n_orb

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        batch_size = int(batch.max().item() + 1)

        # Per-graph node counts and offsets
        node_counts = torch.bincount(batch, minlength=batch_size)
        num_dna_nodes = int((node_counts[0] - 2).item())
        H_size = num_dna_nodes * n_orb

        graph_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 1:
            graph_starts[1:] = torch.cumsum(node_counts[:-1], dim=0)

        GammaL = torch.zeros(batch_size, H_size, device=device, dtype=x.dtype)
        GammaR = torch.zeros(batch_size, H_size, device=device, dtype=x.dtype)

        # Left contact is graph_starts[b] + 0, right is graph_starts[b] + 1
        left_contact_ids = graph_starts        # [batch_size]
        right_contact_ids = graph_starts + 1   # [batch_size]

        # Find all contact-type edges (edge_attr[:, 2] == 1)
        contact_edge_mask = (edge_attr[:, 2] == 1)
        contact_src = edge_index[0, contact_edge_mask]
        contact_dst = edge_index[1, contact_edge_mask]
        contact_coupling = edge_attr[contact_edge_mask, 4]

        # Which graph does each contact edge belong to?
        contact_src_batch = batch[contact_src]

        # Left contact edges: source is the left contact node of its graph
        is_left = (contact_src == left_contact_ids[contact_src_batch])
        # Right contact edges: source is the right contact node of its graph
        is_right = (contact_src == right_contact_ids[contact_src_batch])

        # For left contact edges, map destination to local DNA index
        left_dst = contact_dst[is_left]
        left_batch = contact_src_batch[is_left]
        left_coupling = contact_coupling[is_left]
        left_dna_local = left_dst - graph_starts[left_batch] - 2

        # For right contact edges
        right_dst = contact_dst[is_right]
        right_batch = contact_src_batch[is_right]
        right_coupling = contact_coupling[is_right]
        right_dna_local = right_dst - graph_starts[right_batch] - 2

        # Scatter coupling values into gamma vectors
        if n_orb == 1:
            GammaL[left_batch, left_dna_local] = left_coupling
            GammaR[right_batch, right_dna_local] = right_coupling
        else:
            orb_offsets = torch.arange(n_orb, device=device)
            # Expand each DNA index into n_orb orbital indices
            left_orb_start = left_dna_local.unsqueeze(1) * n_orb + orb_offsets.unsqueeze(0)  # [num_left, n_orb]
            right_orb_start = right_dna_local.unsqueeze(1) * n_orb + orb_offsets.unsqueeze(0)

            # Expand batch indices to match
            left_batch_exp = left_batch.unsqueeze(1).expand_as(left_orb_start)
            right_batch_exp = right_batch.unsqueeze(1).expand_as(right_orb_start)

            # Expand coupling to fill all orbitals
            left_coupling_exp = left_coupling.unsqueeze(1).expand_as(left_orb_start)
            right_coupling_exp = right_coupling.unsqueeze(1).expand_as(right_orb_start)

            GammaL[left_batch_exp.reshape(-1), left_orb_start.reshape(-1)] = left_coupling_exp.reshape(-1)
            GammaR[right_batch_exp.reshape(-1), right_orb_start.reshape(-1)] = right_coupling_exp.reshape(-1)

        # Squeeze for single-graph backward compatibility
        if batch_size == 1:
            return GammaL.squeeze(0), GammaR.squeeze(0)

        return GammaL, GammaR
```

- [ ] **Step 3: Run contact vector tests**

Run: `python -m pytest tests/test_models/test_vectorized_hamiltonian.py::TestGetContactVectors -v --tb=short 2>&1 | head -30`
Expected: All PASS.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py
git commit -m "perf: vectorize get_contact_vectors — eliminate Python loops"
```

---

### Task 6: End-to-end forward pass verification

**Files:**
- Modify: `tests/test_models/test_vectorized_hamiltonian.py`

- [ ] **Step 1: Add end-to-end forward pass test**

Append to `tests/test_models/test_vectorized_hamiltonian.py`:

```python
# ---- End-to-end forward pass tests ----

class TestForwardPassUnchanged:
    """Verify that model.forward() produces identical results after vectorization."""

    def test_forward_single_graph_norb1(self):
        """Full forward pass output unchanged for n_orb=1."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            dos, trans = model(data)
        assert dos.shape == (1, 10)  # energy_grid has 10 points
        assert trans.shape == (1, 10)
        # Verify H stored on model is hermitian
        assert torch.allclose(model.H, model.H.transpose(-1, -2), atol=1e-7)

    def test_forward_batched_norb1(self):
        """Full forward pass works for a batch."""
        _, model = _make_model(n_orb=1)
        model.eval()
        graphs = [
            sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1),
            sequence_to_graph("TGCA", "TGCA", 0, 3, 0.2, 0.2),
        ]
        data = Batch.from_data_list(graphs)
        with torch.no_grad():
            dos, trans = model(data)
        assert dos.shape == (2, 10)
        assert trans.shape == (2, 10)
        for b in range(2):
            assert torch.allclose(model.H[b], model.H[b].T, atol=1e-7)

    def test_forward_norb2(self):
        """Full forward pass works for n_orb=2."""
        _, model = _make_model(n_orb=2, solver_type='complex')
        model.eval()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        with torch.no_grad():
            dos, trans = model(data)
        assert dos.shape == (1, 10)
        assert trans.shape == (1, 10)
        assert torch.allclose(model.H, model.H.transpose(-1, -2), atol=1e-7)

    def test_gradients_flow(self):
        """Verify gradients flow through the vectorized construction."""
        _, model = _make_model(n_orb=1)
        model.train()
        graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
        data = Batch.from_data_list([graph])
        dos, trans = model(data)
        loss = dos.sum() + trans.sum()
        loss.backward()
        # Check that onsite_proj and coupling_proj have gradients
        for name, param in model.named_parameters():
            if 'onsite_proj' in name or 'coupling_proj' in name:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"
```

- [ ] **Step 2: Run end-to-end tests**

Run: `python -m pytest tests/test_models/test_vectorized_hamiltonian.py::TestForwardPassUnchanged -v --tb=short`
Expected: All PASS.

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_models/test_vectorized_hamiltonian.py
git commit -m "test: add end-to-end forward pass and gradient flow tests"
```

---

### Task 7: Clean up reference methods (optional, after verification)

**Files:**
- Modify: `g3nat/models/hamiltonian.py`

This task should be done only after all tests pass and the user has confirmed the vectorized version is correct on a real training run.

- [ ] **Step 1: Remove `_construct_hamiltonian_reference` and `_get_contact_vectors_reference`**

Delete both `_*_reference` methods from `g3nat/models/hamiltonian.py`.

- [ ] **Step 2: Update tests to remove reference comparisons**

In `tests/test_models/test_vectorized_hamiltonian.py`, remove all `_*_reference` calls and the tests that compare vectorized vs reference. Keep the hermiticity tests, forward pass tests, and gradient flow test.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All remaining tests pass.

- [ ] **Step 4: Commit**

```bash
git add g3nat/models/hamiltonian.py tests/test_models/test_vectorized_hamiltonian.py
git commit -m "cleanup: remove reference implementations after vectorization verified"
```
