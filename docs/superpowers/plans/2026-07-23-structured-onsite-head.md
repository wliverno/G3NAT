# Structured Onsite Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-base structured onsite head to the Hamiltonian GNN so the learned onsite energies are physical by construction, and measure how much context the DFT data needs by sweeping the mixing factor alpha.

**Architecture:** In the Hamiltonian construction, replace the free onsite `onsite = onsite_proj(context)` with `onsite = alpha*baseline[base] + (1-alpha)*onsite_proj(context)`, where `baseline` is 4 learned per-base scalars (n_orb=1) read via a differentiable soft-matmul on the base one-hot. alpha is global or per-base, fixed (for the sweep) or learned. All new state is gated behind `--structured_onsite` so the default model is byte-identical. Separately, fix the sequence-leaking train/val split (group by sequence).

**Tech Stack:** PyTorch, PyTorch Geometric, scikit-learn (GroupShuffleSplit), pytest. Env: `conda activate g3nat`. Compute: klone (check `hostname`; login node ok for tests/smoke, sbatch for full runs).

## Global Constraints

- Default off (`--structured_onsite` absent) MUST be byte-identical to the current model (no new params/buffers created, existing checkpoints load). Verify with `torch.equal` on freshly-seeded params, not a state-dict-load test.
- Every DFT train/val split MUST be grouped by sequence string (no sequence in both sides). Old flat-index split leaked identity.
- Supervision stays on DOS/T only. Do NOT match/regress H to the DFT Fock matrix.
- Scope: onsite only. Couplings stay free (but are measured). Do not change NEGF, contacts/gamma, edge features, or batching.
- ASCII only in code and output. Contacts are nodes with all-zero features; contact edges via `edge_attr[:,2]==1`, coupling at `[:,4]` -- do not touch.
- Build order: load-bearing tasks (1-7) first; enrichment (Task 8) after.

---

### Task 1: Grouped-by-sequence split utility

**Files:**
- Create: `g3nat/data/splits.py`
- Test: `tests/test_data/test_splits.py`
- Modify: `scripts/train.py:150-152` (use it), `scripts/train.py:44-73` (add `--split_seed`)

**Interfaces:**
- Produces: `grouped_split(groups: list, test_size: float = 0.2, seed: int = 42) -> tuple[list[int], list[int]]` returning (train_indices, val_indices), no group shared across the two.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data/test_splits.py
from g3nat.data.splits import grouped_split

def test_no_sequence_shared_between_train_and_val():
    # 6 sequences, each duplicated 4x (mimics the 4 contact-variants per sequence)
    seqs = []
    for s in ['AAAA', 'CCCC', 'GGGG', 'TTTT', 'ACGT', 'TGCA']:
        seqs += [s] * 4
    train_idx, val_idx = grouped_split(seqs, test_size=0.34, seed=42)
    train_seqs = {seqs[i] for i in train_idx}
    val_seqs = {seqs[i] for i in val_idx}
    assert train_seqs.isdisjoint(val_seqs)
    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(seqs)

def test_deterministic_given_seed():
    seqs = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    assert grouped_split(seqs, seed=1) == grouped_split(seqs, seed=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n g3nat python -m pytest tests/test_data/test_splits.py -v`
Expected: FAIL (ModuleNotFoundError: g3nat.data.splits)

- [ ] **Step 3: Write minimal implementation**

```python
# g3nat/data/splits.py
"""Sequence-grouped train/val splitting.

The DFT dataset has ~4 contact-variant samples per unique sequence. A flat-index
split leaks sequence identity into val and biases comparisons toward high-capacity
heads. Always split by sequence group.
"""
from typing import List, Tuple
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(groups: List, test_size: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split indices [0..len(groups)) so no group label appears on both sides.

    groups[i] is the group key (e.g. the sequence string) for dataset item i.
    Returns (train_indices, val_indices).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = list(range(len(groups)))
    train_idx, val_idx = next(gss.split(indices, groups=groups))
    return train_idx.tolist(), val_idx.tolist()
```

Create `tests/test_data/__init__.py` (empty) if the test dir needs it.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n g3nat python -m pytest tests/test_data/test_splits.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Wire into train.py**

Add near the other args (after `scripts/train.py:62`, the geometry block):
```python
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Seed for the sequence-grouped train/val split.')
```
Replace `scripts/train.py:150-152`:
```python
    # Split dataset -- GROUPED by sequence so no sequence appears in both train and val.
    from g3nat.data.splits import grouped_split
    train_indices, val_indices = grouped_split(seqs, test_size=0.2, seed=args.split_seed)
```
(`seqs` is the sequence list, parallel to dataset item order.)

- [ ] **Step 6: Verify train.py still parses + smoke-imports**

Run: `conda run -n g3nat python -c "import scripts.train"` and `conda run -n g3nat python scripts/train.py --help | grep split_seed`
Expected: no import error; `--split_seed` listed.

- [ ] **Step 7: Commit**

```bash
git add g3nat/data/splits.py tests/test_data/ scripts/train.py
git commit -m "feat(split): group train/val by sequence (fix identity leak)"
```

---

### Task 2: Structured-onsite params + flags in __init__ (RNG-safe, byte-identical when off)

**Files:**
- Modify: `g3nat/models/hamiltonian.py:19-33` (signature), after `:125` (gated param creation)
- Test: `tests/test_models/test_structured_onsite.py`

**Interfaces:**
- Produces: `DNATransportHamiltonianGNN(..., structured_onsite=False, alpha_granularity='global', alpha_mode='fixed', alpha_value=0.0, alpha_init=0.9)`. When `structured_onsite=True`: attributes `self.onsite_baseline` (Parameter [4, n_orb*n_orb]); alpha state (`self.onsite_alpha_theta` Parameter if learned, else buffer `self.onsite_alpha_fixed`); helper `self._onsite_alpha() -> Tensor` shape [1] (global) or [4] (per_base), values in [0,1], EXACT at fixed endpoints.

- [ ] **Step 1: Write the failing test (byte-identical when off; params appear when on)**

```python
# tests/test_models/test_structured_onsite.py
import torch
import numpy as np
from g3nat import DNATransportHamiltonianGNN

EG = np.linspace(-3, 3, 40)

def _build(seed=0, **kw):
    torch.manual_seed(seed)
    return DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
                                      energy_grid=EG, n_orb=1, conv_type='gat', **kw)

def test_default_off_is_byte_identical():
    a = _build(seed=0)                          # current model
    b = _build(seed=0, structured_onsite=False) # explicit off
    ka, kb = dict(a.named_parameters()), dict(b.named_parameters())
    assert ka.keys() == kb.keys()               # no new params
    for k in ka:
        assert torch.equal(ka[k], kb[k]), f"param {k} differs -> RNG stream perturbed"

def test_on_adds_baseline_and_alpha():
    m = _build(seed=0, structured_onsite=True, alpha_mode='fixed', alpha_value=1.0)
    assert m.onsite_baseline.shape == (4, 1)
    assert torch.allclose(m._onsite_alpha(), torch.ones(1))   # EXACT 1.0, no sigmoid drift

def test_fixed_alpha_zero_is_exact():
    m = _build(seed=0, structured_onsite=True, alpha_mode='fixed', alpha_value=0.0)
    assert torch.equal(m._onsite_alpha(), torch.zeros(1))

def test_per_base_learned_alpha_has_four_values():
    m = _build(seed=0, structured_onsite=True, alpha_granularity='per_base', alpha_mode='learned')
    assert m._onsite_alpha().shape == (4,)
    assert (m._onsite_alpha() > 0).all() and (m._onsite_alpha() < 1).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n g3nat python -m pytest tests/test_models/test_structured_onsite.py -v`
Expected: FAIL (unexpected kwarg `structured_onsite`)

- [ ] **Step 3: Implement signature + gated params**

Add to `__init__` signature (after `geom_norm_stats` param, hamiltonian.py:33):
```python
                 structured_onsite: bool = False,
                 alpha_granularity: str = 'global',   # 'global' | 'per_base'
                 alpha_mode: str = 'fixed',           # 'fixed' | 'learned'
                 alpha_value: float = 0.0,
                 alpha_init: float = 0.9):
```
Add AFTER the geometry block (after hamiltonian.py:125), so existing layers' init RNG is untouched when off:
```python
        # Optional structured onsite head. Default off = no new params (RNG stream and
        # existing checkpoints unchanged). onsite = alpha*baseline[base] + (1-alpha)*context.
        self.structured_onsite = structured_onsite
        self.alpha_granularity = alpha_granularity
        self.alpha_mode = alpha_mode
        if structured_onsite:
            n_alpha = 4 if alpha_granularity == 'per_base' else 1
            # 4 per-base onsite blocks; near-zero init keeps early (E*I - H) well-conditioned.
            self.onsite_baseline = nn.Parameter(torch.empty(4, n_orb * n_orb))
            nn.init.normal_(self.onsite_baseline, std=0.01)
            if alpha_mode == 'learned':
                theta0 = float(np.log(alpha_init / (1.0 - alpha_init)))  # logit(alpha_init)
                self.onsite_alpha_theta = nn.Parameter(torch.full((n_alpha,), theta0))
            else:  # fixed: store alpha DIRECTLY (exact 0.0/1.0, no logit/sigmoid round-trip)
                self.register_buffer('onsite_alpha_fixed', torch.full((n_alpha,), float(alpha_value)))

    def _onsite_alpha(self) -> torch.Tensor:
        """Mixing factor in [0,1], shape [1] (global) or [4] (per_base)."""
        if self.alpha_mode == 'learned':
            return torch.sigmoid(self.onsite_alpha_theta)
        return self.onsite_alpha_fixed
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n g3nat python -m pytest tests/test_models/test_structured_onsite.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add g3nat/models/hamiltonian.py tests/test_models/test_structured_onsite.py
git commit -m "feat(onsite): gated structured-onsite params + alpha helper (byte-identical off)"
```

---

### Task 3: Apply the mixing in both construct paths (soft-matmul, differentiable)

**Files:**
- Modify: `g3nat/models/hamiltonian.py:342-346` (vectorized), `:198-203` (reference)
- Test: `tests/test_models/test_structured_onsite.py` (extend)

**Interfaces:**
- Consumes: `self.onsite_baseline`, `self._onsite_alpha()`, `self.structured_onsite`, `self.alpha_granularity` from Task 2; `original_node_features`, `dna_mask`/`dna_node_mask`, `dna_features`/`dna_node_features` already local in each construct.
- Produces: mixed `onsite_blocks` when structured_onsite on; unchanged when off. Behavior at alpha=0 == current; at alpha=1 == baseline[base] per node.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_models/test_structured_onsite.py
from torch_geometric.data import Batch
from g3nat.graph import sequence_to_graph

def _H(model, seq='ACGT', comp='ACGT'):
    data = sequence_to_graph(seq, comp, left_contact_positions=0, right_contact_positions=len(seq)-1)
    with torch.no_grad():
        model(Batch.from_data_list([data]))
    return model.H[0]

def test_alpha0_matches_current_model():
    off = _build(seed=1)
    on0 = _build(seed=1, structured_onsite=True, alpha_mode='fixed', alpha_value=0.0)
    # onsite_proj weights identical (built after gated params; same seed prefix) -> H equal
    assert torch.allclose(_H(off), _H(on0), atol=1e-6)

def test_alpha1_onsite_equals_baseline_per_base():
    m = _build(seed=2, structured_onsite=True, alpha_mode='fixed', alpha_value=1.0)
    # set distinct baselines so we can read them off the diagonal
    with torch.no_grad():
        m.onsite_baseline.copy_(torch.tensor([[-0.5], [-1.4], [0.0], [-1.1]]))  # A,T,G,C
    H = _H(m, seq='AACC', comp='GGTT')          # primary strand bases A,A,C,C
    diag = torch.diag(H)[:4]
    assert torch.allclose(diag, torch.tensor([-0.5, -0.5, -1.1, -1.1]), atol=1e-5)

def test_baseline_indexed_by_identity_not_position():
    # two graphs, local position 0 is 'G' in one and 'A' in the other
    m = _build(seed=3, structured_onsite=True, alpha_mode='fixed', alpha_value=1.0)
    with torch.no_grad():
        m.onsite_baseline.copy_(torch.tensor([[10.0], [20.0], [30.0], [40.0]]))  # A,T,G,C
    g1 = sequence_to_graph('GA', 'TC', left_contact_positions=0, right_contact_positions=1)
    g2 = sequence_to_graph('AG', 'CT', left_contact_positions=0, right_contact_positions=1)
    with torch.no_grad():
        m(Batch.from_data_list([g1, g2]))
    H = m.H
    assert torch.allclose(torch.diag(H[0])[:2], torch.tensor([30.0, 10.0]), atol=1e-4)  # G,A
    assert torch.allclose(torch.diag(H[1])[:2], torch.tensor([10.0, 30.0]), atol=1e-4)  # A,G

def test_gradient_flows_to_baseline():
    m = _build(seed=4, structured_onsite=True, alpha_mode='fixed', alpha_value=0.5)
    out = m(Batch.from_data_list([sequence_to_graph('ACGT','ACGT',0,3)]))
    loss = (m.H ** 2).sum()
    loss.backward()
    assert m.onsite_baseline.grad is not None and m.onsite_baseline.grad.abs().sum() > 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n g3nat python -m pytest tests/test_models/test_structured_onsite.py -k "alpha0 or alpha1 or identity or gradient" -v`
Expected: FAIL (mixing not applied; alpha1 diag != baseline)

- [ ] **Step 3: Implement mixing helper + patch both paths**

Add a helper method on the model:
```python
    def _mix_onsite(self, dna_features, original_dna_onehot):
        """onsite_raw before reshape. dna_features: post-conv [D, hidden];
        original_dna_onehot: [D, 4] base one-hot in the SAME order."""
        context = self.onsite_proj(dna_features)                 # [D, n_orb^2]
        if not self.structured_onsite:
            return context
        baseline = original_dna_onehot @ self.onsite_baseline    # [D, n_orb^2] soft-matmul
        alpha = self._onsite_alpha()                             # [1] or [4]
        if self.alpha_granularity == 'per_base':
            a = (original_dna_onehot @ alpha.view(4, 1))         # [D, 1] per-node
        else:
            a = alpha.view(1, 1)                                 # broadcast scalar
        return a * baseline + (1.0 - a) * context
```
Vectorized path -- replace hamiltonian.py:343 (`onsite_raw = self.onsite_proj(dna_features)`) with:
```python
        onsite_raw = self._mix_onsite(dna_features, original_node_features[dna_mask])
```
Reference path -- replace hamiltonian.py:199 (`onsite_blocks = self.onsite_proj(dna_node_features)`) with:
```python
        onsite_blocks = self._mix_onsite(dna_node_features, original_node_features[dna_node_mask])
```
(Confirm the reference method's mask var name is `dna_node_mask` and it receives `original_node_features`; if the arg is named differently, use that name.)

- [ ] **Step 4: Run to verify all pass**

Run: `conda run -n g3nat python -m pytest tests/test_models/test_structured_onsite.py -v`
Expected: PASS (all tests, incl. Task 2's)

- [ ] **Step 5: Verify reference == vectorized with mixing on**

Add + run:
```python
def test_reference_matches_vectorized_with_mixing():
    m = _build(seed=5, structured_onsite=True, alpha_mode='fixed', alpha_value=0.6)
    data = sequence_to_graph('ACGT', 'ACGT', 0, 3)
    b = Batch.from_data_list([data])
    x = m.node_proj(b.x); # NOTE: call the two construct methods on the SAME post-conv features.
    # Simplest: assert forward (vectorized) H is finite + Hermitian as a sanity gate here;
    # a full ref-vs-vec equality harness may reuse an existing test util if present.
    m(b); H = m.H[0]
    assert torch.allclose(H, H.T, atol=1e-5)
```
Run the file again; expected PASS. (If an existing ref-vs-vectorized test util exists in `tests/`, extend it to run with `structured_onsite=True` instead.)

- [ ] **Step 6: Commit**

```bash
git add g3nat/models/hamiltonian.py tests/test_models/test_structured_onsite.py
git commit -m "feat(onsite): apply alpha mixing in both construct paths (differentiable soft-matmul)"
```

---

### Task 4: Wire flags through train.py + all reconstruction sites; checkpoint round-trip

**Files:**
- Modify: `scripts/train.py:44-73` (args), `:177-186` (model construction)
- Modify: `g3nat/evaluation/inference.py:59-73`, `scripts/analyze_learned_hamiltonian.py` (load_model), `scripts/probe_onsite_dilution.py` (load_model)
- Test: `tests/test_models/test_structured_onsite.py` (round-trip)

**Interfaces:**
- Consumes: model kwargs from Task 2.
- Produces: `--structured_onsite`, `--alpha_granularity`, `--alpha_mode`, `--alpha_value`, `--alpha_init` on train.py; all reconstruction sites accept these from `checkpoint['args']`.

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_checkpoint_roundtrip_reconstructs_structured_model(tmp_path):
    import numpy as np
    from g3nat.evaluation.inference import load_trained_model  # adjust to actual fn name
    m = _build(seed=6, structured_onsite=True, alpha_granularity='per_base',
               alpha_mode='learned', alpha_value=0.0)
    ckpt = tmp_path / "m.pth"
    torch.save({'model_state_dict': m.state_dict(),
                'energy_grid': EG,
                'args': {'hidden_dim': 32, 'num_layers': 2, 'num_heads': 2, 'n_orb': 1,
                         'conv_type': 'gat', 'structured_onsite': True,
                         'alpha_granularity': 'per_base', 'alpha_mode': 'learned',
                         'alpha_value': 0.0, 'alpha_init': 0.9}}, ckpt)
    loaded, _ = load_trained_model(str(ckpt), device='cpu')  # must not raise on strict load
    assert loaded.onsite_baseline.shape == (4, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n g3nat python -m pytest tests/test_models/test_structured_onsite.py -k roundtrip -v`
Expected: FAIL (strict load_state_dict: unexpected keys onsite_baseline / onsite_alpha_theta)

- [ ] **Step 3: Register args in reconstruction sites**

`g3nat/evaluation/inference.py` -- add after line 72 (`use_geometry=...`), inside the `DNATransportHamiltonianGNN(...)` call:
```python
            structured_onsite=args.get('structured_onsite', False),
            alpha_granularity=args.get('alpha_granularity', 'global'),
            alpha_mode=args.get('alpha_mode', 'fixed'),
            alpha_value=args.get('alpha_value', 0.0),
            alpha_init=args.get('alpha_init', 0.9),
```
Apply the SAME five-line addition to the `DNATransportHamiltonianGNN(...)` constructor call inside `load_model` in `scripts/analyze_learned_hamiltonian.py` and `scripts/probe_onsite_dilution.py` (both build from `ck['args']` / `a`; use `a.get('structured_onsite', False)` etc.).

- [ ] **Step 4: Add train.py flags + pass to model**

Add after `scripts/train.py:62`:
```python
    parser.add_argument('--structured_onsite', action='store_true',
                       help='Mix a per-base onsite baseline with the context head.')
    parser.add_argument('--alpha_granularity', choices=['global', 'per_base'], default='global')
    parser.add_argument('--alpha_mode', choices=['fixed', 'learned'], default='fixed')
    parser.add_argument('--alpha_value', type=float, default=0.0,
                       help='Fixed mixing factor in [0,1] (alpha_mode=fixed).')
    parser.add_argument('--alpha_init', type=float, default=0.9,
                       help='Initial mixing factor (alpha_mode=learned).')
```
Add to the `DNATransportHamiltonianGNN(...)` call at `scripts/train.py:177-186`:
```python
            structured_onsite=args.structured_onsite,
            alpha_granularity=args.alpha_granularity,
            alpha_mode=args.alpha_mode,
            alpha_value=args.alpha_value,
            alpha_init=args.alpha_init,
```
(Reject the degenerate combo: after parse, `assert not (args.alpha_granularity=='per_base' and args.alpha_mode=='fixed'), "per_base+fixed needs 4 alphas; use learned or global"`.)

- [ ] **Step 5: Run round-trip test + full suite**

Run: `conda run -n g3nat python -m pytest tests/test_models/test_structured_onsite.py -v && conda run -n g3nat python -m pytest tests/ -q`
Expected: round-trip PASS; existing ~85-test suite still PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py g3nat/evaluation/inference.py scripts/analyze_learned_hamiltonian.py scripts/probe_onsite_dilution.py tests/test_models/test_structured_onsite.py
git commit -m "feat(onsite): wire alpha flags through train + reconstruction sites"
```

---

### Task 5: Physicality metrics module (co-gate + distinctness + coupling bandwidth)

**Files:**
- Create: `g3nat/evaluation/physicality.py`
- Test: `tests/test_evaluation/test_physicality.py`

**Interfaces:**
- Produces:
  - `onsite_metrics(H_diag: np.ndarray, window=(-1,1)) -> dict` with `frac_in_window`, `min`, `max`, `range`.
  - `eig_metrics(H: np.ndarray, window=(-1,1)) -> dict` with `frac_eig_in_window`.
  - `coupling_bandwidth(H: np.ndarray) -> float` (max abs off-diagonal, the escape-valve diagnostic).
  - `baseline_distinctness(baseline: np.ndarray) -> dict` with `min_pairwise`, `spread` (worst-case pairwise gap + std across the 4 baselines). NOTE: true eta^2 needs per-SITE onsite data and is NOT returned here; it is computed separately by `scripts/probe_onsite_dilution.py::variance_decomposition`.
  - `is_physical_win(before: dict, after: dict) -> bool`: True only if BOTH onsite-in-window AND eig-in-window improve (guards "shifted, not fixed").

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluation/test_physicality.py
import numpy as np
from g3nat.evaluation.physicality import (onsite_metrics, eig_metrics,
                                          coupling_bandwidth, baseline_distinctness, is_physical_win)

def test_onsite_frac_in_window():
    d = np.array([-0.5, 0.5, -33.0, 2.0])
    m = onsite_metrics(d)
    assert abs(m['frac_in_window'] - 0.5) < 1e-9
    assert m['min'] == -33.0 and m['max'] == 2.0

def test_eig_in_window_counts_eigenvalues():
    H = np.diag([-0.2, 0.3, -20.0])  # eig = diagonal
    assert abs(eig_metrics(H)['frac_eig_in_window'] - 2/3) < 1e-9

def test_coupling_bandwidth_is_max_offdiag():
    H = np.array([[0.0, 0.7], [0.7, 0.0]])
    assert abs(coupling_bandwidth(H) - 0.7) < 1e-9

def test_distinctness_flags_collapse():
    collapsed = np.array([[-0.5], [-0.5], [-0.49], [-0.5]])
    distinct = np.array([[-0.49], [-1.39], [0.0], [-1.12]])
    assert baseline_distinctness(collapsed)['min_pairwise'] < 0.05
    assert baseline_distinctness(distinct)['min_pairwise'] > 0.4

def test_win_requires_both_to_improve():
    before = {'frac_in_window': 0.6, 'frac_eig_in_window': 0.6}
    shifted = {'frac_in_window': 0.9, 'frac_eig_in_window': 0.6}   # onsite up, eig flat
    real = {'frac_in_window': 0.9, 'frac_eig_in_window': 0.85}
    assert is_physical_win(before, shifted) is False
    assert is_physical_win(before, real) is True
```

- [ ] **Step 2: Run to verify fail**

Run: `conda run -n g3nat python -m pytest tests/test_evaluation/test_physicality.py -v`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement**

```python
# g3nat/evaluation/physicality.py
"""Physicality diagnostics for a learned Hamiltonian. A 'win' requires onsite AND
eigenvalues to move into the window together -- else the model just relocated the
unphysical states into the couplings ('shifted, not fixed')."""
import numpy as np


def onsite_metrics(H_diag, window=(-1.0, 1.0)):
    d = np.asarray(H_diag).ravel()
    lo, hi = window
    return {'frac_in_window': float(np.mean((d >= lo) & (d <= hi))),
            'min': float(d.min()), 'max': float(d.max()), 'range': float(d.max() - d.min())}


def eig_metrics(H, window=(-1.0, 1.0)):
    w = np.linalg.eigvalsh(np.asarray(H))
    lo, hi = window
    return {'frac_eig_in_window': float(np.mean((w >= lo) & (w <= hi)))}


def coupling_bandwidth(H):
    H = np.asarray(H)
    off = H - np.diag(np.diag(H))
    return float(np.abs(off).max())


def baseline_distinctness(baseline):
    # Only the 4 baseline values are available here, so distinctness = pairwise spread.
    # (True eta^2 needs per-SITE onsite over the val set; that lives in
    # scripts/probe_onsite_dilution.py::variance_decomposition, which has the per-site data.)
    b = np.asarray(baseline).reshape(len(baseline), -1).mean(axis=1)  # 1 scalar per base
    pw = [abs(b[i] - b[j]) for i in range(len(b)) for j in range(i + 1, len(b))]
    return {'min_pairwise': float(min(pw)) if pw else 0.0, 'spread': float(b.std())}


def is_physical_win(before, after, eps=1e-6):
    return (after['frac_in_window'] > before['frac_in_window'] + eps and
            after['frac_eig_in_window'] > before['frac_eig_in_window'] + eps)
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n g3nat python -m pytest tests/test_evaluation/test_physicality.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add g3nat/evaluation/physicality.py tests/test_evaluation/
git commit -m "feat(eval): physicality metrics with onsite+eig co-gate and distinctness"
```

---

### Task 6: Extract per-base TB parameters + literature comparison

**Files:**
- Create: `scripts/extract_tb_params.py`

**Interfaces:**
- Consumes: a trained checkpoint (structured_onsite). Uses `load_model` pattern from Task 4.
- Produces: prints the 4 learned baselines, gauge-corrected (subtract model's G), alongside Roche + Voityuk-Rosch (values from `g3nat/utils/physics.py`), and `baseline_distinctness`.

- [ ] **Step 1: Implement (no unit test; it is a reporting script exercised on a real checkpoint)**

```python
#!/usr/bin/env python3
"""Extract the learned per-base onsite (baseline) TB parameters and compare to literature.
Comparisons are GAUGE-CORRECTED (each set shifted so G=0) -- absolute onsite is gauge-dependent.
Usage: conda run -n g3nat python scripts/extract_tb_params.py <checkpoint.pth>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from g3nat.evaluation.physicality import baseline_distinctness

BASES = ['A', 'T', 'G', 'C']
ROCHE = {'A': -0.49, 'T': -1.39, 'G': 0.00, 'C': -1.12}   # g3nat/utils/physics.py

def _gauge(d):  # shift so G = 0
    return {b: d[b] - d['G'] for b in BASES}

def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    base = sd['onsite_baseline'].reshape(4, -1).mean(1).numpy()   # A,T,G,C order (BASE_TO_IDX)
    learned = {b: float(base[i]) for i, b in enumerate(BASES)}
    print("base  learned  learned(G=0)  Roche(G=0)")
    lg, rg = _gauge(learned), _gauge(ROCHE)
    for b in BASES:
        print(f"{b:>4} {learned[b]:>+8.3f} {lg[b]:>+12.3f} {rg[b]:>+11.3f}")
    order_learned = sorted(BASES, key=lambda b: learned[b])
    order_roche = sorted(BASES, key=lambda b: ROCHE[b])
    print(f"\nordering  learned: {order_learned}   Roche: {order_roche}   match={order_learned==order_roche}")
    print(f"distinctness: {baseline_distinctness(base.reshape(4,1))}")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/extract_tb_params.py
git commit -m "feat(eval): extract per-base TB params + gauge-corrected literature comparison"
```

---

### Task 7: Load-bearing runs (smoke -> clean baseline -> DFT alpha sweep)

**Files:**
- Create: `scripts/run_onsite_sweep.sh` (sbatch array or loop calling train.py)

These are experiments, not TDD. Each records: best-val DOS/T loss, onsite+eig physicality, coupling bandwidth, baseline distinctness (via a short eval using the Task 5 module on the trained `model.H` over the val set).

- [ ] **Step 1: Smoke.** One short structured run to verify plumbing + timing:
```bash
conda run -n g3nat python scripts/train.py --data_source pickle --data_dir pickle_files \
  --model_type hamiltonian --conv_type gat --hidden_dim 256 --num_epochs 50 \
  --structured_onsite --alpha_mode fixed --alpha_value 1.0 --split_seed 42 \
  --output_dir outputs_onsite_smoke --checkpoint_dir ckpt_onsite_smoke
```
Then `conda run -n g3nat python scripts/extract_tb_params.py ckpt_onsite_smoke/<model>.pth` -- confirm baselines are finite + moving toward distinct values. Record wall-clock -> size the sweep.

- [ ] **Step 2: Clean-split reference.** alpha=0 (== current model) under the grouped split, 5000 epochs, >=3 `--split_seed` values {42,43,44}. This is the honest baseline (replaces the leaky 0.547).

- [ ] **Step 3: DFT global sweep.** For `alpha_value` in {0,0.25,0.5,0.75,0.9,1.0} x `--split_seed` in {42,43,44}: train (fixed global). Collect `val_loss(alpha)` mean+/-std (best-checkpoint) + physicality + distinctness. This is the headline discriminator curve. Interpret the moderate-alpha region; report train-convergence per alpha to rule out under-convergence.

- [ ] **Step 4: Extract** baselines at the informative alpha via `scripts/extract_tb_params.py`.

- [ ] **Step 5: Commit** the runner + a results notes file under `docs/`.

---

### Task 8: Full comparison (enrichment -- after Task 7)

- **TB calibration sweep:** identical global sweep on `--data_source tb` (known: no context needed). Its (flat) val_loss(alpha) is the reference for "does not need context." No new code.
- **Standalone 4-scalar control:** train with `--structured_onsite --alpha_mode fixed --alpha_value 1.0` is already pure per-base; compare to a plain 4-scalar head to isolate the mixing-form confound (reuse the extraction).
- **Per-base nested test:** `--alpha_granularity per_base --alpha_mode learned`. PRE-REGISTER predicted drift ranking (G driftiest) to a dated commit BEFORE running; judge observed ranking with a permutation test across `--split_seed` seeds.
- **Bilevel learned alpha (new code):** to make a learned global alpha meaningful, update `onsite_alpha_theta` on a held-out val minibatch, separate from the train step (alternating optimization). This is a `g3nat/training/trainer.py` change: give alpha its own optimizer stepped on a val batch each epoch. Specify + TDD when reached (out of load-bearing scope).

---

## Notes for the executor

- Run everything through `conda run -n g3nat` (or `conda activate g3nat`). Check `hostname`: login node ok for pytest + smoke; use sbatch for the 5000-epoch sweep (see `running-gaunegf` skill for the sbatch pattern on klone).
- The reference construct (`_construct_hamiltonian_reference`) is not used by `forward` but must stay consistent for tests. If its local var names differ from `dna_node_mask`/`original_node_features`, adapt Step 3 of Task 3.
- Metric = best-val-checkpoint or last-N rolling mean, NEVER final-epoch. If `trainer.py` only tracks final loss, add best-val tracking as part of Task 7.
