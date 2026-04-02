# Vectorize Hamiltonian Construction

**Date:** 2026-04-02
**File:** `g3nat/models/hamiltonian.py`
**Methods:** `construct_hamiltonian_from_graph`, `get_contact_vectors`

## Problem

`construct_hamiltonian_from_graph` uses three nested Python loops (batch × nodes × edges) with per-iteration GPU kernel launches (`torch.where`, `torch.isin`, dict lookups, Python sets). For batch_size=64 and 10 DNA nodes, this means ~2000+ individual GPU ops per forward pass — repeated every epoch for 5000 epochs. This is the dominant training bottleneck.

`get_contact_vectors` has the same pattern (Python loops over batch and contact edges).

## Constraint

- H must be real symmetric (Hermitian). This is the critical correctness criterion.
- No zero-padding beyond what already exists (all graphs in a batch already have equal DNA node counts).
- NEGF solvers (`NEGFProjection`, `NEGFProjectionComplex`) are untouched.
- `forward()` signature and return values are unchanged.

## Design

### Core Insight

`edge_index` already encodes the Hamiltonian sparsity pattern. The current code re-derives this structure every forward pass using Python dicts and sets. The vectorized version uses `edge_index` directly as indexing tensors.

### `construct_hamiltonian_from_graph` — Vectorized

#### Step 1: DNA node identification (unchanged logic, no loops)

```python
contact_mask = torch.all(original_node_features == 0, dim=1)  # [num_nodes]
dna_mask = ~contact_mask
```

#### Step 2: Per-graph node offsets (replaces Python dict `dna_to_local`)

```python
node_counts = torch.bincount(batch, minlength=batch_size)
graph_starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                          torch.cumsum(node_counts, dim=0)[:-1]])
# For each DNA node, its local Hamiltonian index:
#   local_idx = global_idx - graph_start[batch[global_idx]] - 2
# The "- 2" accounts for the two contact nodes at the start of each graph.
```

Compute `dna_local_indices` as a tensor for all DNA nodes at once:

```python
dna_indices = torch.where(dna_mask)[0]                       # global indices of all DNA nodes
dna_batch = batch[dna_indices]                                # which graph each belongs to
dna_local = dna_indices - graph_starts[dna_batch] - 2        # local H index (0-based)
```

Validate that `num_dna_nodes` is equal across graphs (existing check).

#### Step 3: Fill diagonal blocks (onsite energies)

Compute onsite blocks for all DNA nodes in one pass:

```python
dna_features = node_features[dna_mask]                        # [total_dna_nodes, hidden_dim]
onsite_blocks = self.onsite_proj(dna_features)                # [total_dna_nodes, n_orb²]
onsite_blocks = onsite_blocks.view(-1, self.n_orb, self.n_orb)
if self.enforce_hermiticity:
    onsite_blocks = 0.5 * (onsite_blocks + onsite_blocks.transpose(-1, -2))
```

For `n_orb == 1` (common case), scatter directly:

```python
H_matrix[dna_batch, dna_local, dna_local] = onsite_blocks.squeeze(-1).squeeze(-1)
```

For `n_orb > 1`, expand indices over orbital dimensions:

```python
orb = torch.arange(self.n_orb, device=device)
row_orb = (dna_local.unsqueeze(1) * self.n_orb + orb).view(-1)  # [total_dna * n_orb]
# Repeat for col dimension, then scatter onsite_blocks entries
```

#### Step 4: Fill off-diagonal blocks (coupling)

Filter edges to DNA-only pairs:

```python
src, dst = edge_index
both_dna = dna_mask[src] & dna_mask[dst]
dna_src = src[both_dna]
dna_dst = dst[both_dna]
dna_edge_features = edge_features[both_dna]
```

Convert to local Hamiltonian indices:

```python
src_batch = batch[dna_src]
src_local = dna_src - graph_starts[src_batch] - 2
dst_local = dna_dst - graph_starts[src_batch] - 2
```

Deduplicate to upper triangle (replaces Python `processed_pairs` set):

```python
upper_mask = src_local < dst_local
src_upper = src_local[upper_mask]
dst_upper = dst_local[upper_mask]
batch_upper = src_batch[upper_mask]
```

Compute coupling blocks for upper-triangle edges:

```python
coupling_blocks = self.coupling_proj(dna_edge_features[upper_mask])
coupling_blocks = coupling_blocks.view(-1, self.n_orb, self.n_orb)
```

Scatter into H (for `n_orb == 1`):

```python
H_matrix[batch_upper, src_upper, dst_upper] = coupling_blocks.squeeze(-1).squeeze(-1)
```

Symmetrize (one op, guarantees H = H^T):

```python
H_matrix = H_matrix + H_matrix.transpose(-1, -2)
```

**Note on hermiticity with the diagonal:** The diagonal was filled in step 3. The transpose-and-add doubles the diagonal values. Two options:
- **Option A:** Zero the diagonal before symmetrizing, then re-add onsite blocks
- **Option B (preferred):** Build off-diagonal and diagonal parts separately, symmetrize only off-diagonal, then add:
  ```python
  H_diag = torch.zeros(...)    # fill with onsite blocks
  H_offdiag = torch.zeros(...) # fill with coupling blocks (upper triangle only)
  H_matrix = H_diag + H_offdiag + H_offdiag.transpose(-1, -2)
  ```

This guarantees hermiticity by construction without any conditional logic.

#### Step 5: Diagonal shift (unchanged)

```python
H_matrix = H_matrix + 1e-6 * I
```

### `get_contact_vectors` — Vectorized

Same principle. Replace per-batch/per-edge Python loops with:

1. Identify contact-type edges: `contact_mask = edge_attr[:, 2] == 1`
2. Separate left/right by source node: left contact is node `graph_starts[b]`, right is `graph_starts[b] + 1`
3. Map destination nodes to local H indices using the same offset arithmetic
4. Scatter coupling strengths (`edge_attr[:, 4]`) into `GammaL_batch` / `GammaR_batch`

### What Doesn't Change

- `__init__` — all layers, projections, hyperparameters unchanged
- `NEGFProjection` / `NEGFProjectionComplex` — untouched
- `forward()` — same call structure and return signature
- Equal-DNA-nodes-per-batch enforcement — still required
- `construction.py` / dataset code — untouched

## Testing

1. **Reference comparison:** Keep the old implementation as `_construct_hamiltonian_reference`. Run both on identical batches, assert `torch.allclose(H_new, H_ref, atol=1e-6)`.

2. **Hermiticity check (critical):** For every test case, assert:
   ```python
   assert torch.allclose(H, H.transpose(-1, -2), atol=1e-7)
   ```
   This verifies H = H^T directly, independent of whether the old code was correct.

3. **Gamma comparison:** Assert `torch.allclose(GammaL_new, GammaL_ref)` and same for GammaR.

4. **Edge cases:** Single graph (batch_size=1), single DNA node, `n_orb=1` and `n_orb>1`.

5. **Performance:** Time the forward pass before/after on a batch of 64 graphs with 10 DNA nodes. Target: >5x speedup.

6. **Training regression:** Run a short training loop (50 epochs) with both versions on the same data, verify loss curves are numerically identical.
