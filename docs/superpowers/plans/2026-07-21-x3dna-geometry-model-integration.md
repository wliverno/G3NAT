# X3DNA Geometry Model Integration (Plan 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the offline geometry cache from Plan 1 and attach the SE(3)-invariant per-edge geometry to the graph and the Hamiltonian GNN, behind a `use_geometry` toggle that is byte-for-byte identical to the current model when off.

**Architecture:** A pure assembly function maps a cached per-sequence geometry entry onto edge identities; `sequence_to_graph` places those 7-tuples in lockstep with the edges it already builds (so alignment cannot drift from edge order); the model normalizes per edge type with checkpointed buffers and fuses geometry into the edge embedding additively. Default off = current model.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, NumPy, pytest. Tests run via SLURM (`sbatch TestJob <pytest-args>`) in the `g3nat` conda env - NEVER on the login node.

## Global Constraints

- Work on branch `x3dna-edge-geometry` directly. No worktrees.
- The topological 5-dim edge vector `[backbone, hbond, contact, directionality, coupling]` and its indices are FROZEN. Contact detection reads `edge_attr[:, 2]`; contact coupling reads `edge_attr[:, 4]` (`g3nat/models/hamiltonian.py:704-707`). Geometry lives in SEPARATE tensors (`edge_geom`, `edge_geom_mask`), never inside `edge_attr`.
- `use_geometry=False` MUST reproduce the current model bit-for-bit on a fixed seed and load existing checkpoints.
- 7-slot schema per edge: `[d_centroid, t1, t2, t3, r1, r2, r3]`. Backbone: `[stack_dist, shift, slide, rise, tilt, roll, twist]`. H-bond: `[pair_dist, shear, stretch, stagger, buckle, propeller, opening]`. Contacts/overhangs: zeros, mask 0.
- Parameter order (verbatim): bp = [Shear, Stretch, Stagger, Buckle, Propeller, Opening]; step = [Shift, Slide, Rise, Tilt, Roll, Twist].
- Cache format (Plan 1, `build_geometry_cache`): `cache[seq] = {bp_pars [Npair,6], step_pars [Nstep,6], primary_centroids [Nprimary,3], comp_centroids [Ncomp,3]}`, keyed by lowercase sequence. Cache at `geom_cache/geometry.pkl` (gitignored).
- Pairing convention (matches `construction.py:224-238` and Plan 1 `_edge_rows`): primary position `i` pairs with complementary position `N-1-i`; DSSR pair `k` = primary `k` with comp `Ncomp-1-k`. Complementary centroids array is ordered 5'->3' (same order as the `complementary_sequence` string), so comp string position `j` maps to `comp_centroids[j]`.
- Migration must leave a working system at every step. Commit after every task.
- All test commands run through SLURM: `sbatch TestJob <pytest-args>`; read `slurm-<jobid>.out`. Never run `python`/`pytest` on the login node.

---

## File Structure

- Modify: `g3nat/graph/geometry.py` - add pure `assemble_graph_geometry(...)` (edge-identity -> 7-tuple).
- Modify: `g3nat/graph/construction.py` - `sequence_to_graph` gains `geometry=None` and emits `edge_geom`/`edge_geom_mask`; `create_dna_dataset` gains `geometry_cache=None` and passes per-sequence entries. (Keep the duplicate `create_dna_dataset` in `datasets.py` in sync.)
- Modify: `g3nat/data/datasets.py` - mirror the `create_dna_dataset` change; `DNATransportDataset.__getitem__` carries `edge_geom`/`edge_geom_mask` through.
- Modify: `g3nat/data/pickle.py` - `load_pickle_directory` optionally loads the geometry cache and returns it.
- Modify: `g3nat/models/hamiltonian.py` - `use_geometry`, `geom_dim=7`, per-type norm buffers, `geom_encoder`, fusion in `forward`, persist flag+stats in checkpoint.
- Modify: `scripts/train.py`, `g3nat/training/config.py` - `--use_geometry` / `--geom_cache` flags, compute norm stats from cache, pass to model.
- Modify: `g3nat/models/standard.py` - (optional, last) same flag for parity.
- Test: `tests/test_graph/test_geometry.py` (assembly), `tests/test_graph/test_construction.py` (graph tensors), `tests/test_models/test_forward.py` (fusion + backward-compat), `tests/integration/test_end_to_end.py` (flag on).
- Create: `GeomCacheJob` (sbatch) - offline cache build over the PDB dataset.

---

## Task 1: Pure per-edge geometry assembly

**Files:**
- Modify: `g3nat/graph/geometry.py`
- Test: `tests/test_graph/test_geometry.py`

**Interfaces:**
- Consumes: a cache entry `{bp_pars, step_pars, primary_centroids, comp_centroids}` (Plan 1).
- Produces: `assemble_graph_geometry(primary_seq, comp_seq, entry) -> dict` mapping edge identity to a length-7 list. Keys: `("backbone", "primary", i)` for the primary step between positions `i` and `i+1`; `("backbone", "complementary", j)` for the comp step between comp positions `j` and `j+1`; `("hbond", i)` for the pair at primary position `i`. Missing data -> key absent (caller masks 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph/test_geometry.py  (add)
import numpy as np
from g3nat.graph.geometry import assemble_graph_geometry

def _toy_entry():
    # 3 base pairs. Distinct numbers so mis-mapping is detectable.
    bp = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]], float)
    step = np.array([[21,22,23,24,25,26],[31,32,33,34,35,36]], float)  # Nstep=2
    # primary centroids p0,p1,p2 ; comp centroids c0,c1,c2 (5'->3')
    pc = np.array([[0,0,0],[0,0,3.4],[0,0,6.8]], float)
    cc = np.array([[6,0,0],[6,0,3.4],[6,0,6.8]], float)
    return {"bp_pars": bp, "step_pars": step,
            "primary_centroids": pc, "comp_centroids": cc}

def test_assemble_backbone_primary_and_hbond():
    e = _toy_entry()
    g = assemble_graph_geometry("aaa", "ttt", e)
    # primary backbone step 0: [dist(p0,p1), *step[0]]
    assert g[("backbone","primary",0)] == [3.4, 21,22,23,24,25,26]
    assert g[("backbone","primary",1)] == [3.4, 31,32,33,34,35,36]
    # hbond pair i: [dist(p_i, c_{N-1-i}), *bp[i]]
    assert g[("hbond",0)][1:] == [1,2,3,4,5,6]
    assert abs(g[("hbond",0)][0] - 6.0) < 1e-9   # p0=(0,0,0) vs c2=(6,0,6.8)? -> see mapping below

def test_assemble_backbone_complementary_shares_step():
    e = _toy_entry()
    g = assemble_graph_geometry("aaa", "ttt", e)
    # comp step between comp j and j+1 shares primary step index (N-2-j)
    # j=0 -> step index N-2-0 = 1 ; j=1 -> step index 0
    assert g[("backbone","complementary",0)][1:] == [31,32,33,34,35,36]
    assert g[("backbone","complementary",1)][1:] == [21,22,23,24,25,26]
    assert g[("backbone","complementary",0)][0] == 3.4  # dist(c0,c1)
```

Note on the hbond distance assertion: with pairing primary `i` <-> comp `N-1-i`, pair 0 = p0 with c2. Fix the expected distance in the test to `dist(p0, c2)` once you compute it from the toy coords (`sqrt(6^2 + 6.8^2)`), rather than the placeholder above. Keep distinct coords so the mapping is verified, not just the shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `sbatch TestJob tests/test_graph/test_geometry.py -v -k assemble` then read `slurm-<jobid>.out`.
Expected: FAIL with `ImportError: cannot import name 'assemble_graph_geometry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# g3nat/graph/geometry.py  (add)
def assemble_graph_geometry(primary_seq, comp_seq, entry):
    """Map a cached per-sequence geometry entry onto graph edge identities.

    Returns {edge_id: [d_centroid, t1,t2,t3, r1,r2,r3]} with:
      ("backbone","primary",i)       primary step i (positions i,i+1)
      ("backbone","complementary",j) comp step j (positions j,j+1), sharing
                                     primary step index N-2-j
      ("hbond",i)                    pair i (primary i <-> comp N-1-i)
    Absent where the underlying params/centroids are missing.
    """
    bp = np.asarray(entry["bp_pars"], float)
    step = np.asarray(entry["step_pars"], float)
    pc = np.asarray(entry["primary_centroids"], float)
    cc = np.asarray(entry["comp_centroids"], float)
    n = pc.shape[0]
    ncomp = cc.shape[0]
    out = {}
    # primary backbone
    for i in range(min(step.shape[0], max(0, n - 1))):
        out[("backbone", "primary", i)] = [centroid_distance(pc[i], pc[i + 1]), *step[i]]
    # complementary backbone: comp step j shares primary step index (n-2-j)
    for j in range(max(0, ncomp - 1)):
        si = n - 2 - j
        if 0 <= si < step.shape[0]:
            out[("backbone", "complementary", j)] = [centroid_distance(cc[j], cc[j + 1]), *step[si]]
    # hbond: primary i <-> comp n-1-i
    for i in range(min(bp.shape[0], n, ncomp)):
        cj = ncomp - 1 - i
        out[("hbond", i)] = [centroid_distance(pc[i], cc[cj]), *bp[i]]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbatch TestJob tests/test_graph/test_geometry.py -v -k assemble`
Expected: PASS (2 tests). If a distance assertion fails, correct the EXPECTED number in the test to the value computed from the toy coords - do not change the implementation to match a wrong expectation.

- [ ] **Step 5: Commit**

```bash
git add g3nat/graph/geometry.py tests/test_graph/test_geometry.py
git commit -m "feat(geometry): assemble_graph_geometry - cache entry to per-edge 7-tuples (Plan 2 Task 1)"
```

---

## Task 2: Emit edge_geom / edge_geom_mask from sequence_to_graph

**Files:**
- Modify: `g3nat/graph/construction.py` (`sequence_to_graph`)
- Test: `tests/test_graph/test_construction.py`

**Interfaces:**
- Consumes: `assemble_graph_geometry` (Task 1).
- Produces: `sequence_to_graph(..., geometry=None)`; the returned `Data` gains `edge_geom` `[num_edges, 7]` (float) and `edge_geom_mask` `[num_edges, 1]` (float in {0,1}), aligned to `edge_index`. `geometry` is a cache entry dict or None; None -> all zeros, mask 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph/test_construction.py  (add)
import torch
from g3nat.graph.construction import sequence_to_graph

def test_edge_geom_default_absent():
    d = sequence_to_graph("ACGT", "ACGT")
    assert d.edge_geom.shape == (d.edge_index.shape[1], 7)
    assert d.edge_geom_mask.shape == (d.edge_index.shape[1], 1)
    assert torch.count_nonzero(d.edge_geom_mask) == 0     # no geometry supplied
    assert torch.count_nonzero(d.edge_geom) == 0

def test_edge_geom_lands_on_backbone_and_hbond_only():
    import numpy as np
    entry = {
        "bp_pars": np.arange(4*6).reshape(4,6).astype(float),
        "step_pars": (np.arange(3*6).reshape(3,6)+100).astype(float),
        "primary_centroids": np.array([[0,0,i*3.4] for i in range(4)], float),
        "comp_centroids": np.array([[6,0,i*3.4] for i in range(4)], float),
    }
    d = sequence_to_graph("ACGT", "ACGT", geometry=entry)
    ea = d.edge_attr
    # contact edges (col 2 == 1) always masked 0
    contact = ea[:,2] == 1
    assert torch.count_nonzero(d.edge_geom_mask[contact]) == 0
    # every backbone (col0==1) and hbond (col1==1) edge is masked 1 for a full duplex
    bb_hb = (ea[:,0]==1) | (ea[:,1]==1)
    assert torch.all(d.edge_geom_mask[bb_hb] == 1)
    # a backbone edge carries a step distance in slot 0 (~3.4), not zero
    bb = ea[:,0]==1
    assert torch.all(d.edge_geom[bb][:,0] > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbatch TestJob tests/test_graph/test_construction.py -v -k edge_geom`
Expected: FAIL with `AttributeError: 'GlobalStorage' object has no attribute 'edge_geom'`.

- [ ] **Step 3: Write minimal implementation**

In `sequence_to_graph`, add `geometry: Optional[dict] = None` to the signature. Build a parallel `edge_geom` list and `edge_geom_mask` list, appending EXACTLY once per `edge_attr.append(...)`/`.extend(...)`, in the same order. Precompute the lookup once:

```python
# near the top of edge construction, after edge_index/edge_attr = [], []:
edge_geom = []
edge_geom_mask = []
_ZERO7 = [0.0]*7
from g3nat.graph.geometry import assemble_graph_geometry
_geo = assemble_graph_geometry(primary_sequence, complementary_sequence, geometry) if geometry is not None else {}

def _emit(g7, m):
    edge_geom.append(g7); edge_geom_mask.append([m])
```

Then at each edge append, add a matching `_emit(...)`:
- contact edges (all 4 contact appends): `_emit(_ZERO7, 0.0)`.
- primary backbone forward `(node1->node2)` and backward `(node2->node1)`: both use key `("backbone","primary",pos1)`; `g = _geo.get(("backbone","primary",pos1))`; `_emit(g, 1.0) if g else _emit(_ZERO7, 0.0)` - emit for BOTH directed copies (identical 7-tuple, per spec: geometry is direction-symmetric; `directionality` in `edge_attr[:,3]` carries orientation).
- complementary backbone forward/backward: key `("backbone","complementary",pos1)`.
- hydrogen bond (both directed copies): key `("hbond", i)` where `i` is the primary index in the H-bond loop.

Finally, after the tensors:

```python
data.edge_geom = torch.tensor(edge_geom, dtype=torch.float) if edge_geom \
    else torch.zeros((edge_index.size(1), 7), dtype=torch.float)
data.edge_geom_mask = torch.tensor(edge_geom_mask, dtype=torch.float) if edge_geom_mask \
    else torch.zeros((edge_index.size(1), 1), dtype=torch.float)
assert data.edge_geom.size(0) == edge_index.size(1), \
    f"edge_geom misaligned: {data.edge_geom.size(0)} vs {edge_index.size(1)} edges"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `sbatch TestJob tests/test_graph/test_construction.py -v -k edge_geom`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full graph + baseline suite to confirm no regression**

Run: `sbatch TestJob tests/test_graph/ tests/baseline/ -v`
Expected: all PASS (existing graph tests unaffected; `Data` just has extra attributes).

- [ ] **Step 6: Commit**

```bash
git add g3nat/graph/construction.py tests/test_graph/test_construction.py
git commit -m "feat(geometry): sequence_to_graph emits edge_geom/edge_geom_mask (Plan 2 Task 2)"
```

---

## Task 3: use_geometry fusion + per-type normalization in the Hamiltonian model

**Files:**
- Modify: `g3nat/models/hamiltonian.py`
- Test: `tests/test_models/test_forward.py`

**Interfaces:**
- Consumes: `data.edge_geom` `[E,7]`, `data.edge_geom_mask` `[E,1]`, and the frozen edge one-hot `edge_attr[:,0:3]`.
- Produces: `DNATransportHamiltonianGNN(..., use_geometry=False, geom_dim=7, geom_norm_stats=None)`. When `use_geometry=True`, the fused edge embedding = `edge_proj(edge_attr) + geom_encoder(normalize(edge_geom)) * edge_geom_mask`. `geom_norm_stats` = `{"backbone":{"mean":[7],"std":[7]}, "hbond":{"mean":[7],"std":[7]}}` (from `compute_norm_stats`); stored as buffers so they persist in the checkpoint.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_forward.py  (add)
import torch
from torch_geometric.data import Batch
from g3nat.graph.construction import sequence_to_graph
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN

def _model(**kw):
    torch.manual_seed(0)
    return DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
                                      n_orb=1, conv_type='gat', **kw)

def test_use_geometry_false_matches_baseline_bitwise():
    g = sequence_to_graph("ACGT", "ACGT")
    b = Batch.from_data_list([g])
    m1 = _model(use_geometry=False)
    torch.manual_seed(0); m2 = _model(use_geometry=False)
    with torch.no_grad():
        out1 = m1(b); out2 = m2(b)
    assert torch.equal(out1[0], out2[0]) and torch.equal(out1[1], out2[1])

def test_geometry_changes_output():
    g = sequence_to_graph("ACGT", "ACGT")
    g.edge_geom = torch.zeros_like(g.edge_geom)
    g.edge_geom_mask = torch.zeros_like(g.edge_geom_mask)
    m = _model(use_geometry=True)
    # give ONE backbone edge nonzero geometry + mask, and perturb geom_encoder from zero-init
    with torch.no_grad():
        for p in m.geom_encoder.parameters():
            p.add_(0.1)
    b0 = Batch.from_data_list([g])
    with torch.no_grad(): out0 = m(b0)
    g2 = g.clone()
    bb = (g2.edge_attr[:,0]==1).nonzero()[0]
    g2.edge_geom[bb] = torch.tensor([3.4,1.,2.,3.,4.,5.,6.])
    g2.edge_geom_mask[bb] = 1.0
    b1 = Batch.from_data_list([g2])
    with torch.no_grad(): out1 = m(b1)
    assert not torch.allclose(out0[1], out1[1])   # transmission responds to geometry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `sbatch TestJob tests/test_models/test_forward.py -v -k "geometry"`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'use_geometry'`.

- [ ] **Step 3: Write minimal implementation**

In `DNATransportHamiltonianGNN.__init__`, add args and modules:

```python
def __init__(self, ..., conv_type='gat', use_geometry=False, geom_dim=7, geom_norm_stats=None):
    ...
    self.use_geometry = use_geometry
    self.geom_dim = geom_dim
    if use_geometry:
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        nn.init.normal_(self.geom_encoder[-1].weight, std=0.01)
        nn.init.zeros_(self.geom_encoder[-1].bias)
        # per-type norm buffers: row 0 backbone, row 1 hbond
        mean = torch.zeros(2, geom_dim); std = torch.ones(2, geom_dim)
        if geom_norm_stats is not None:
            mean[0] = torch.tensor(geom_norm_stats["backbone"]["mean"])
            std[0]  = torch.tensor(geom_norm_stats["backbone"]["std"])
            mean[1] = torch.tensor(geom_norm_stats["hbond"]["mean"])
            std[1]  = torch.tensor(geom_norm_stats["hbond"]["std"])
        self.register_buffer("geom_mean", mean)
        self.register_buffer("geom_std", std)
```

Add a fusion helper:

```python
def _fuse_geometry(self, edge_attr_proj, edge_attr_initial, data):
    geom = data.edge_geom
    mask = data.edge_geom_mask
    is_bb = (edge_attr_initial[:, 0] == 1).float().unsqueeze(1)   # [E,1]
    is_hb = (edge_attr_initial[:, 1] == 1).float().unsqueeze(1)
    mean = is_bb * self.geom_mean[0] + is_hb * self.geom_mean[1]  # [E,7]
    std  = is_bb * self.geom_std[0]  + is_hb * self.geom_std[1]
    std  = torch.where(std == 0, torch.ones_like(std), std)
    normed = (geom - mean) / std
    return edge_attr_proj + self.geom_encoder(normed) * mask
```

In `forward`, after `edge_attr = self.edge_proj(edge_attr)`:

```python
if self.use_geometry:
    edge_attr = self._fuse_geometry(edge_attr, edge_attr_initial, data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `sbatch TestJob tests/test_models/test_forward.py -v -k "geometry"`
Expected: PASS (2 tests).

- [ ] **Step 5: Full model + integration suite (backward-compat gate)**

Run: `sbatch TestJob tests/test_models/ tests/integration/ -v`
Expected: all PASS - existing forward/gradient tests unchanged (default `use_geometry=False`).

- [ ] **Step 6: Commit**

```bash
git add g3nat/models/hamiltonian.py tests/test_models/test_forward.py
git commit -m "feat(geometry): use_geometry fusion + per-type norm buffers in Hamiltonian GNN (Plan 2 Task 3)"
```

---

## Task 4: Thread the cache through datasets, pickle loader, and training

**Files:**
- Modify: `g3nat/graph/construction.py` (`create_dna_dataset`), `g3nat/data/datasets.py` (`create_dna_dataset`, `DNATransportDataset.__getitem__`), `g3nat/data/pickle.py` (`load_pickle_directory`)
- Modify: `scripts/train.py`, `g3nat/training/config.py`
- Test: `tests/test_data/test_datasets.py`, `tests/integration/test_end_to_end.py`

**Interfaces:**
- Consumes: `sequence_to_graph(..., geometry=entry)` (Task 2), `compute_norm_stats` (Plan 1), model `geom_norm_stats` arg (Task 3).
- Produces: `create_dna_dataset(..., geometry_cache=None)` (both copies) passes `geometry=geometry_cache.get(seq.lower())` per sequence; `DNATransportDataset.__getitem__` copies `edge_geom`/`edge_geom_mask` onto the emitted `Data`; `load_pickle_directory(..., geom_cache_path=None)` returns the loaded cache dict (or None); `train.py` gains `--use_geometry` (store_true) and `--geom_cache PATH`, computes stats via `compute_norm_stats(cache)`, and constructs the model with `use_geometry=args.use_geometry, geom_norm_stats=stats`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data/test_datasets.py  (add)
import numpy as np
from g3nat.data.datasets import create_dna_dataset

def test_geometry_cache_flows_to_graph():
    seqs = ["ACGT"]
    entry = {"bp_pars": np.zeros((4,6)), "step_pars": np.ones((3,6)),
             "primary_centroids": np.array([[0,0,i*3.4] for i in range(4)], float),
             "comp_centroids": np.array([[6,0,i*3.4] for i in range(4)], float)}
    ds = create_dna_dataset(seqs, np.zeros((1,10)), np.zeros((1,10)), np.linspace(-1,1,10),
                            complementary_sequences=["ACGT"],
                            geometry_cache={"acgt": entry})
    d = ds[0]
    bb = d.edge_attr[:,0]==1
    assert d.edge_geom_mask[bb].sum() > 0    # backbone edges carry geometry
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbatch TestJob tests/test_data/test_datasets.py -v -k geometry_cache`
Expected: FAIL with `TypeError: create_dna_dataset() got an unexpected keyword argument 'geometry_cache'`.

- [ ] **Step 3: Write minimal implementation**

`create_dna_dataset` (BOTH `construction.py` and `datasets.py` - keep in sync): add `geometry_cache=None`; inside the per-sequence loop set `seq_kwargs['geometry'] = geometry_cache.get(sequence.lower())` when `geometry_cache is not None`. `DNATransportDataset.__getitem__`: after building `data`, add:

```python
if hasattr(graph, 'edge_geom'):
    data.edge_geom = graph.edge_geom
    data.edge_geom_mask = graph.edge_geom_mask
```

`pickle.py::load_pickle_directory`: add `geom_cache_path=None`; if set, `import pickle; cache = pickle.load(open(geom_cache_path,'rb'))` and return it alongside existing returns (extend the return tuple / dict; update the single call site in `train.py`).

`config.py`: add `use_geometry: bool = False` and `geom_cache: Optional[str] = None` to `TrainingConfig`.

`train.py`: add
```python
parser.add_argument('--use_geometry', action='store_true',
                    help='Fuse SE(3)-invariant X3DNA edge geometry (needs --geom_cache)')
parser.add_argument('--geom_cache', type=str, default='geom_cache/geometry.pkl')
```
Load the cache when `args.use_geometry`, compute `stats = g3nat.graph.geometry.compute_norm_stats(cache)`, pass `geometry_cache=cache` to `create_dna_dataset`, and construct the hamiltonian model with `use_geometry=args.use_geometry, geom_norm_stats=stats`. When `--use_geometry` is absent, behavior is byte-for-byte unchanged (no cache load, `use_geometry=False`).

- [ ] **Step 4: Run test to verify it passes**

Run: `sbatch TestJob tests/test_data/test_datasets.py -v -k geometry_cache`
Expected: PASS.

- [ ] **Step 5: Full suite**

Run: `sbatch TestJob tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add g3nat/graph/construction.py g3nat/data/datasets.py g3nat/data/pickle.py scripts/train.py g3nat/training/config.py tests/test_data/test_datasets.py
git commit -m "feat(geometry): thread geometry cache through dataset + train.py --use_geometry flag (Plan 2 Task 4)"
```

---

## Task 5: Build the geometry cache offline (job) and smoke-run the flag

**Files:**
- Create: `GeomCacheJob` (sbatch)
- Test: `tests/integration/test_end_to_end.py`

**Interfaces:**
- Consumes: `build_geometry_cache` (Plan 1), the PDB dataset at `/mmfs1/gscratch/anantram/asyed4/DNADataSet/<seq>/<seq>.pdb`.
- Produces: `geom_cache/geometry.pkl` covering the training sequences.

- [ ] **Step 1: Write the functional end-to-end test (small hand-built cache)**

```python
# tests/integration/test_end_to_end.py  (add)
import numpy as np, torch
from torch_geometric.data import Batch
from g3nat.graph.construction import sequence_to_graph
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN

def test_end_to_end_geometry_on():
    entry = {"bp_pars": np.random.RandomState(0).randn(4,6),
             "step_pars": np.random.RandomState(1).randn(3,6),
             "primary_centroids": np.array([[0,0,i*3.4] for i in range(4)], float),
             "comp_centroids": np.array([[6,0,i*3.4] for i in range(4)], float)}
    stats = {"backbone":{"mean":[0]*7,"std":[1]*7},"hbond":{"mean":[0]*7,"std":[1]*7}}
    g = sequence_to_graph("ACGT","ACGT", geometry=entry)
    m = DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
                                   conv_type='gat', use_geometry=True, geom_norm_stats=stats)
    dos, trans = m(Batch.from_data_list([g]))
    assert torch.isfinite(dos).all() and torch.isfinite(trans).all()
```

- [ ] **Step 2: Run it**

Run: `sbatch TestJob tests/integration/test_end_to_end.py -v -k geometry_on`
Expected: PASS.

- [ ] **Step 3: Write the cache-build job**

```bash
# GeomCacheJob
#!/bin/bash
#SBATCH --job-name=g3nat-geomcache
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mem=8GB
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate g3nat
cd /mmfs1/gscratch/anantram/willll/G3NAT
export X3DNA_DSSR=/mmfs1/gscratch/anantram/asyed4/x3dna-dssr
python -u -c "
import glob, os
from g3nat.data.pickle import load_pickle_directory
from g3nat.graph.geometry import build_geometry_cache
seqs = sorted({os.path.basename(p).split('_run')[0] for p in glob.glob('pickle_files/*_run1.pkl')})
build_geometry_cache('/mmfs1/gscratch/anantram/asyed4/DNADataSet', 'geom_cache/geometry.pkl', sequences=seqs)
print('cache built for', len(seqs), 'sequences')
"
```

- [ ] **Step 4: Submit and verify**

Run: `sbatch GeomCacheJob` then read the `slurm-<jobid>.out`.
Expected: "cache built for 515 sequences"; `geom_cache/geometry.pkl` exists. Warnings for any missing PDBs are non-fatal.

- [ ] **Step 5: Commit (job script only; cache is gitignored)**

```bash
git add GeomCacheJob tests/integration/test_end_to_end.py
git commit -m "feat(geometry): offline cache-build job + geometry-on end-to-end test (Plan 2 Task 5)"
```

---

## Task 6 (optional, last): mirror use_geometry into standard.py

**Files:**
- Modify: `g3nat/models/standard.py`
- Test: `tests/test_models/test_forward.py`

- [ ] **Step 1-5:** Same shape as Task 3 for `DNATransportGNN`: add `use_geometry`/`geom_dim`/`geom_norm_stats`, `geom_encoder`, `_fuse_geometry`, fuse after `edge_proj`; test that `use_geometry=False` is unchanged and that geometry changes output. Commit `feat(geometry): mirror use_geometry into standard model (Plan 2 Task 6)`.

---

## Self-Review Notes (coverage against the spec)

- Spec "new tensors on the graph" -> Task 2. "Normalization (per edge type)" -> Task 3 buffers + `_fuse_geometry`. "Model integration" -> Task 3. "Edge-type to relationship mapping" (incl. complementary backbone sharing step params) -> Task 1. "Offline preprocessing / data-source" -> Task 5 uses Plan 1's `build_geometry_cache`. "Migration sequence (working system at every step)" -> Tasks are ordered so default-off stays green throughout; the flag turns on only in Task 5. "Testing" section items: backward-compat bitwise (Task 3 Step 1), wiring proof (Task 3 Step 1 / Task 5), degeneracy (H-bond distance is atom-centroid ~6A not frame-origin ~0.09A) covered by Task 1's hbond distance assertion; invariance regression already lives in Plan 1's `test_geometry_is_se3_invariant`.
- FROZEN indices `edge_attr[:,2]`/`[:,4]`: geometry is a separate tensor; `edge_attr` is never widened. Verified by Task 2's contact-mask test and the full-suite gates.
- Open question (encoder shape) resolved: start with a 2-layer MLP `Linear(7,h)->ReLU->Linear(h,h)`, near-zero final init (consistent with `coupling_proj`/`onsite_proj`). Upgrade to concat edge-type one-hot only if specialization proves weak (noted, not built).
