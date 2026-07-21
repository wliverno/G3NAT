# Base-Dependent Coupling Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tight-binding coupling blocks in `DNATransportHamiltonianGNN` depend on the identities of the two bases each edge connects, instead of being a single base-blind value per edge type.

**Architecture:** The coupling head currently sees only the (once-projected, never message-passed) edge feature, so every backbone edge yields the identical coupling block and every H-bond edge yields another single identical block -- base identity never reaches the off-diagonal Hamiltonian. Fix: feed the two endpoint node embeddings (post-conv, base-aware) into the coupling head alongside the edge feature, ordered low-local-index-first so the model can distinguish ordered dinucleotide steps (AG vs GA) while H stays Hermitian via the existing symmetrization.

**Tech Stack:** PyTorch, PyTorch Geometric, pytest.

## Global Constraints

- H must be real-symmetric (Hermitian). It is enforced by the construction: one block per undirected pair, mirrored (`H_offdiag + H_offdiag.T` vectorized; `coupling_block` + `.conj().T` reference). Do not weaken this.
- `construct_hamiltonian_from_graph` (vectorized) and `_construct_hamiltonian_reference` (loop) MUST produce identical H for the same input -- `tests/test_models/test_vectorized_hamiltonian.py` compares them.
- Do NOT touch: the NEGF solvers (`NEGFProjection`, `NEGFProjectionComplex`), contact/gamma logic (`get_contact_vectors`, `_get_contact_vectors_reference`), the 5-dim edge vector, or the contact indices `edge_attr[:,2]` / `edge_attr[:,4]`.
- `forward()` signature and return values unchanged.
- `coupling_proj` final layer keeps its near-zero init.
- Must work for `n_orb == 1` and `n_orb > 1`.
- `g3nat/models/standard.py` (`DNATransportGNN`) is OUT OF SCOPE -- it does not build a Hamiltonian.
- The coupling head's input dim changes (`hidden_dim` -> `3*hidden_dim`), so this fix is architecturally incompatible with pre-fix checkpoints. The committed `trained_models/hamiltonian_2000x_4to10BP_5000epoch.pth` is a LEGACY-format checkpoint: `load_trained_model` (`g3nat/evaluation/inference.py:86-106`) rebuilds `coupling_proj` at the OLD input width and `load_state_dict` SUCCEEDS -- then the first forward pass shape-mismatches (`3*hidden_dim` vs `hidden_dim`). So it does NOT fail cleanly at load; it crashes at forward. That checkpoint must be retrained on the new architecture; `tests/test_models/test_generator.py::test_optimize_with_trained_model` is skipped until then (Task 1, Step 7). Hardening `load_trained_model` to raise a clear error for this case is a recommended follow-up, out of scope here.

---

## File Structure

- Modify: `g3nat/models/hamiltonian.py`
  - constructor `coupling_proj` definition (currently lines 82-94): input dim `hidden_dim` -> `3 * hidden_dim`.
  - `construct_hamiltonian_from_graph` coupling computation (currently lines 310-344): gather endpoint node embeddings and concat into the coupling input.
  - `_construct_hamiltonian_reference` coupling computation (currently lines 161-163 precompute + 197-220 loop): same change in the loop form.
- Create: `tests/test_models/test_base_dependent_coupling.py` -- base-dependence, Hermiticity, and gradient-flow tests.
- Modify: `tests/test_models/test_generator.py` -- skip `test_optimize_with_trained_model` (its committed checkpoint predates the new coupling head; see Global Constraints).

---

### Task 1: Make the coupling head base-aware

**Files:**
- Test: `tests/test_models/test_base_dependent_coupling.py` (create)
- Modify: `g3nat/models/hamiltonian.py:82-94` (constructor)
- Modify: `g3nat/models/hamiltonian.py:310-344` (`construct_hamiltonian_from_graph`)
- Modify: `g3nat/models/hamiltonian.py:161-163,197-220` (`_construct_hamiltonian_reference`)
- Modify: `tests/test_models/test_generator.py` (skip stale legacy-checkpoint test)

**Interfaces:**
- Consumes: `DNATransportHamiltonianGNN(hidden_dim, num_layers, num_heads, n_orb, energy_grid)`; `sequence_to_graph(primary_sequence, complementary_sequence=None, left_contact_positions, right_contact_positions)`; `model.H` (set by `forward`, shape `[batch, H_size, H_size]`, real).
- Produces: coupling block for undirected DNA pair `(i, j)` with local `i < j` equals `coupling_proj(cat[x_i, x_j, edge_feat_ij])` where `x_*` are post-conv node embeddings `[hidden_dim]` and `edge_feat_ij` is the projected edge feature `[hidden_dim]`; `coupling_proj` input dim is `3 * hidden_dim`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models/test_base_dependent_coupling.py`:

```python
import numpy as np
import torch
from torch_geometric.data import Batch

from g3nat.models import DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph


def _H_for(model, seq, n_orb):
    """Run the model on a single-strand sequence and return its Hamiltonian [H_size, H_size]."""
    data = sequence_to_graph(
        seq, left_contact_positions=0, right_contact_positions=len(seq) - 1
    )
    batch = Batch.from_data_list([data])
    model.eval()
    with torch.no_grad():
        model(batch)
    return model.H[0]  # H_size = len(seq) * n_orb


def _coupling_block(H, i, j, n_orb):
    """Off-diagonal coupling block between DNA nodes i and j."""
    return H[i * n_orb:(i + 1) * n_orb, j * n_orb:(j + 1) * n_orb]


def test_backbone_coupling_depends_on_base_norb1():
    torch.manual_seed(0)
    model = DNATransportHamiltonianGNN(
        hidden_dim=32, num_layers=2, num_heads=2, n_orb=1,
        energy_grid=np.linspace(-1.0, 1.0, 5),
    )
    HA = _H_for(model, "AAAA", n_orb=1)
    HG = _H_for(model, "GAAA", n_orb=1)  # differs only at primary position 0
    cA = _coupling_block(HA, 0, 1, 1)
    cG = _coupling_block(HG, 0, 1, 1)
    assert not torch.allclose(cA, cG, atol=1e-6), (
        "backbone coupling H[0,1] identical despite a different base at position 0 "
        "-> coupling head is base-blind"
    )


def test_backbone_coupling_depends_on_base_norb2():
    torch.manual_seed(0)
    model = DNATransportHamiltonianGNN(
        hidden_dim=32, num_layers=2, num_heads=2, n_orb=2,
        energy_grid=np.linspace(-1.0, 1.0, 5),
    )
    HA = _H_for(model, "AAAA", n_orb=2)
    HG = _H_for(model, "GAAA", n_orb=2)
    cA = _coupling_block(HA, 0, 1, 2)
    cG = _coupling_block(HG, 0, 1, 2)
    assert not torch.allclose(cA, cG, atol=1e-6), (
        "n_orb>1 backbone coupling block identical despite different base -> base-blind"
    )
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run: `cd /mmfs1/gscratch/anantram/willll/G3NAT && python -m pytest tests/test_models/test_base_dependent_coupling.py -v`
Expected: both tests FAIL (`allclose` is True -- couplings are currently identical across bases).

- [ ] **Step 3: Widen `coupling_proj` input in the constructor**

In `g3nat/models/hamiltonian.py`, the coupling projection (currently lines 82-94). Change ONLY the first `Linear`'s input dim:

```python
        # Each edge contributes n_orb x n_orb coupling block.
        # Input = [x_low (hidden), x_high (hidden), edge_feat (hidden)] = 3 * hidden_dim,
        # so the coupling depends on BOTH endpoint bases, not just the edge type.
        self.coupling_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_orb * n_orb)
        )
```

Leave the near-zero init block below it unchanged:

```python
        nn.init.normal_(self.onsite_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.onsite_proj[-1].bias)
        nn.init.normal_(self.coupling_proj[-1].weight, std=0.01)
        nn.init.zeros_(self.coupling_proj[-1].bias)
```

- [ ] **Step 4: Feed endpoint embeddings into the vectorized coupling**

In `construct_hamiltonian_from_graph`, the block currently at lines 322-331:

```python
        # --- Step 6: Deduplicate to upper triangle (src < dst), skip self-loops ---
        upper_mask = src_local < dst_local
        src_upper = src_local[upper_mask]
        dst_upper = dst_local[upper_mask]
        batch_upper = src_batch[upper_mask]
        edge_feat_upper = dna_edge_features[upper_mask]

        # Compute coupling blocks for upper-triangle edges only
        coupling_raw = self.coupling_proj(edge_feat_upper)         # [num_upper_edges, n_orb²]
        coupling_blocks = coupling_raw.view(-1, n_orb, n_orb)     # [num_upper_edges, n_orb, n_orb]
```

Replace with (adds the two endpoint node embeddings, low-local first):

```python
        # --- Step 6: Deduplicate to upper triangle (src < dst), skip self-loops ---
        upper_mask = src_local < dst_local
        src_upper = src_local[upper_mask]
        dst_upper = dst_local[upper_mask]
        batch_upper = src_batch[upper_mask]
        edge_feat_upper = dna_edge_features[upper_mask]
        # Global node indices of each undirected pair's endpoints, low-local first
        # (under upper_mask, dna_src is the smaller-local-index endpoint).
        node_low = dna_src[upper_mask]
        node_high = dna_dst[upper_mask]

        # Base-aware coupling: concat both endpoint embeddings (low, high) with the edge feature.
        x_low = node_features[node_low]                            # [num_upper_edges, hidden_dim]
        x_high = node_features[node_high]                          # [num_upper_edges, hidden_dim]
        coupling_in = torch.cat([x_low, x_high, edge_feat_upper], dim=-1)  # [num_upper_edges, 3*hidden_dim]
        coupling_raw = self.coupling_proj(coupling_in)             # [num_upper_edges, n_orb²]
        coupling_blocks = coupling_raw.view(-1, n_orb, n_orb)     # [num_upper_edges, n_orb, n_orb]
```

- [ ] **Step 5: Feed endpoint embeddings into the reference coupling**

In `_construct_hamiltonian_reference`, delete the precompute (currently lines 161-163):

```python
        # Get coupling blocks for edges between DNA nodes
        coupling_blocks = self.coupling_proj(edge_features)  # [num_edges, n_orb²]
        coupling_blocks = coupling_blocks.view(-1, self.n_orb, self.n_orb)  # [num_edges, n_orb, n_orb]
```

(Delete those three lines -- coupling is now computed per pair inside the loop.)

Then in the loop, the block currently at lines 214-220:

```python
                    # Use the first occurrence's coupling block for this undirected pair
                    global_edge_idx = graph_edge_indices[local_edge_idx]
                    coupling_block = coupling_blocks[global_edge_idx]

                    # Set symmetric coupling blocks
                    H_matrix[batch_idx, u_orb_start:u_orb_end, v_orb_start:v_orb_end] = coupling_block
                    H_matrix[batch_idx, v_orb_start:v_orb_end, u_orb_start:u_orb_end] = coupling_block.conj().T
```

Replace with (compute base-aware coupling, low-local node first to match the vectorized path):

```python
                    # Base-aware coupling: both endpoint embeddings (low-local first) + edge feature.
                    global_edge_idx = graph_edge_indices[local_edge_idx]
                    edge_feat = edge_features[global_edge_idx]
                    if src_local <= dst_local:
                        x_lo, x_hi = node_features[src_g], node_features[dst_g]
                    else:
                        x_lo, x_hi = node_features[dst_g], node_features[src_g]
                    coupling_in = torch.cat([x_lo, x_hi, edge_feat], dim=-1)
                    coupling_block = self.coupling_proj(coupling_in).view(self.n_orb, self.n_orb)

                    # Set symmetric coupling blocks
                    H_matrix[batch_idx, u_orb_start:u_orb_end, v_orb_start:v_orb_end] = coupling_block
                    H_matrix[batch_idx, v_orb_start:v_orb_end, u_orb_start:u_orb_end] = coupling_block.conj().T
```

- [ ] **Step 6: Run the new test to verify it PASSES**

Run: `cd /mmfs1/gscratch/anantram/willll/G3NAT && python -m pytest tests/test_models/test_base_dependent_coupling.py -v`
Expected: both tests PASS.

- [ ] **Step 7: Skip the stale legacy-checkpoint integration test**

The committed `trained_models/hamiltonian_2000x_4to10BP_5000epoch.pth` predates the widened coupling head. `load_trained_model` loads it via a legacy shim (`g3nat/evaluation/inference.py:86-106`) that rebuilds `coupling_proj` at the OLD input width, so `load_state_dict` succeeds but the first forward pass shape-mismatches (`3*hidden_dim` vs `hidden_dim`). Retraining is out of scope, so skip until the checkpoint is regenerated.

In `tests/test_models/test_generator.py`, replace the decorator on `test_optimize_with_trained_model` (currently `@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=...)`) with an unconditional skip:

```python
@pytest.mark.skip(
    reason="trained_models/hamiltonian_*.pth predates base-aware coupling "
           "(coupling_proj input is now 3*hidden_dim). load_trained_model's legacy shim "
           "loads it but forward() then shape-mismatches. Retrain, then remove this skip."
)
def test_optimize_with_trained_model():
```

- [ ] **Step 8: Run the full model + integration suite to verify nothing regressed**

Run: `cd /mmfs1/gscratch/anantram/willll/G3NAT && python -m pytest tests/test_models/ tests/integration/ -v`
Expected: all PASS or SKIP (the Step 7 test shows as skipped). In particular `test_vectorized_hamiltonian.py` (vectorized == reference) must still pass -- it confirms Steps 4 and 5 stayed consistent. If it FAILS, the vectorized and reference paths disagree: re-check the low-local-first ordering and the edge feature used in Step 5; a less likely cause is a floating-point difference exceeding the test's `atol=1e-6` now that the reference calls `coupling_proj` per-edge instead of batched over all edges.

- [ ] **Step 9: Commit**

```bash
cd /mmfs1/gscratch/anantram/willll/G3NAT
git add g3nat/models/hamiltonian.py tests/test_models/test_base_dependent_coupling.py tests/test_models/test_generator.py
git commit -m "fix: make Hamiltonian coupling depend on the two bases it connects

The coupling head only saw the (never-message-passed) edge feature, so every
backbone edge produced one identical coupling block and every H-bond edge
another -- base identity never reached the off-diagonal Hamiltonian. Feed both
endpoint node embeddings (post-conv, low-local first) into coupling_proj so
coupling = f(bases, edge). Ordered so AG != GA; H stays Hermitian via the
existing symmetrization. Updated both vectorized and reference construct paths.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011n22khW2GNEbTGN359sFDz"
```

---

### Task 2: Guard tests -- Hermiticity and gradient flow

> **DROPPED during execution (2026-07-20).** The code review found these duplicate
> existing coverage: `tests/test_models/test_vectorized_hamiltonian.py` already asserts
> Hermiticity (`test_hermiticity_single`/`test_hermiticity_batched`) and coupling-head
> gradient flow (`test_gradients_flow`). Not implemented, to keep the change DRY. The new
> behavior (base-dependent coupling) is covered by Task 1's tests.

**Files:**
- Test: `tests/test_models/test_base_dependent_coupling.py` (append)

**Interfaces:**
- Consumes: everything from Task 1 (`model.H`, `coupling_proj`).
- Produces: nothing new; regression guards.

- [ ] **Step 1: Write the Hermiticity + gradient-flow tests**

Append to `tests/test_models/test_base_dependent_coupling.py`:

```python
def test_hamiltonian_is_symmetric():
    torch.manual_seed(0)
    for n_orb in (1, 2):
        model = DNATransportHamiltonianGNN(
            hidden_dim=32, num_layers=2, num_heads=2, n_orb=n_orb,
            energy_grid=np.linspace(-1.0, 1.0, 5),
        )
        H = _H_for(model, "ACGT", n_orb=n_orb)
        assert torch.allclose(H, H.transpose(-1, -2), atol=1e-6), (
            f"H is not symmetric for n_orb={n_orb}"
        )


def test_gradient_flows_through_coupling_head():
    torch.manual_seed(0)
    model = DNATransportHamiltonianGNN(
        hidden_dim=32, num_layers=2, num_heads=2, n_orb=1,
        energy_grid=np.linspace(-1.0, 1.0, 5),
    )
    data = sequence_to_graph("ACGT", left_contact_positions=0, right_contact_positions=3)
    batch = Batch.from_data_list([data])
    model.train()
    dos_pred, trans_pred = model(batch)
    loss = dos_pred.pow(2).mean() + trans_pred.pow(2).mean()
    loss.backward()
    # The first Linear of the coupling head is the one whose input we widened.
    grad = model.coupling_proj[0].weight.grad
    assert grad is not None and torch.any(grad != 0), (
        "no gradient reached the coupling head -- base-aware path is disconnected"
    )
```

- [ ] **Step 2: Run the guard tests**

Run: `cd /mmfs1/gscratch/anantram/willll/G3NAT && python -m pytest tests/test_models/test_base_dependent_coupling.py -v`
Expected: all PASS (Hermiticity holds by construction; gradient reaches the widened coupling head).

- [ ] **Step 3: Commit**

```bash
cd /mmfs1/gscratch/anantram/willll/G3NAT
git add tests/test_models/test_base_dependent_coupling.py
git commit -m "test: guard Hamiltonian symmetry and coupling-head gradient flow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011n22khW2GNEbTGN359sFDz"
```

---

## Self-Review

**1. Spec coverage:**
- "Coupling depends on both endpoint bases" -> Task 1 (fix + `test_backbone_coupling_depends_on_base_norb1/2`).
- "Ordered, not symmetric (AG != GA)" -> Task 1 concat order `[x_low, x_high, edge]`, low-local first, in both paths.
- "Hermiticity preserved" -> Task 2 `test_hamiltonian_is_symmetric`; construction unchanged.
- "Vectorized == reference" -> Task 1 Step 7 runs `test_vectorized_hamiltonian.py`.
- "Works for n_orb 1 and >1" -> both tested.
- "Near-zero init kept / forward unchanged / NEGF+contacts untouched" -> only `coupling_proj` input dim and the two coupling computations change.
- Checkpoint incompatibility (legacy shim loads, then forward-crashes) -> Global Constraints (corrected) + Task 1 Step 7 skips `test_optimize_with_trained_model`; retraining the checkpoint is noted as required, and hardening `load_trained_model`'s error is a noted out-of-scope follow-up.

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code.

**3. Type consistency:** `coupling_proj` input `3 * hidden_dim` in the constructor matches the `cat[x_low, x_high, edge_feat_upper]` (3 x `hidden_dim`) in the vectorized path and `cat[x_lo, x_hi, edge_feat]` in the reference path. `node_features` is the post-conv `x` passed into both construct methods. `model.H` shape `[batch, H_size, H_size]` used consistently in tests.

## Notes for the geometry merge (later)

When `x3dna-edge-geometry` merges with this branch: geometry is added to `edge_attr` before the conv loop (additive channel), and `edge_attr` flows into the `cat[..., edge_feat]` here unchanged -- so after both land, `coupling = f(bases via x, topology + geometry via edge_attr)` with no further coupling-head change. Only a plain merge (both edit `construct_hamiltonian_from_graph`; conflicts are in adjacent lines and mechanical).
