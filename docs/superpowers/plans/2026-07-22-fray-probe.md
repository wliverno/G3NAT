# Fray Probe (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Perturb the terminal primary stacking edge's geometry on the geometry-ON model and map how the ENTIRE predicted Hamiltonian `model.H` responds (heatmaps, argmax-change tracking, region decomposition), as a read-only inference probe.

**Architecture:** Pure, unit-testable helpers in a new package module `g3nat/evaluation/fray.py` (edge location, region masks, sweep, metrics); a thin runner `scripts/fray_probe.py` (load model + cache, pick sequences, orchestrate, write `outputs_fray/`, plot); a `FrayProbeJob` sbatch. No change to the model, graph builder, or any Plan 2 code.

**Tech Stack:** Python, PyTorch, PyTorch Geometric, NumPy, matplotlib, pytest. Tests run via SLURM (`sbatch TestJob <pytest-args>`) in the `g3nat` conda env -- NEVER on the login node.

## Global Constraints

- Work on branch `x3dna-edge-geometry`. No worktrees. Read-only probe: do not modify `g3nat/models/`, `g3nat/graph/`, or `g3nat/data/`.
- The trained model is `outputs_pickle_gat_geom/hamiltonian_pickle_model.pth` (geometry-ON, `use_geometry=True` in its saved args, `n_orb=1`, hidden=256). Load with `g3nat.evaluation.load_trained_model`.
- Node layout from `sequence_to_graph` (FROZEN): node 0 = left contact, 1 = right contact, primary bases at nodes `2 .. 2+Np-1`, complementary bases after. So primary position `k` is node `2+k` and Hamiltonian local index `k` (`n_orb=1`). The two terminal primary bases (positions `N-2`, `N-1`) are nodes `N`, `N+1`; their H indices are `N-2`, `N-1`.
- Edge geometry schema (FROZEN): `edge_geom[:, 0]` = centroid distance, `edge_geom[:, 3]` = rise.
- `model.H` is `[batch, M, M]`; for a single-graph batch use `model.H[0]`, `M = 2*Np`.
- Morph only slots 0 and 3, together, on both directed copies of the terminal primary backbone edge; hold all else fixed. Sweep is intentionally wide/out-of-distribution (no physical bound).
- All test commands run through SLURM: `sbatch TestJob <pytest-args>`; read `slurm-<jobid>.out`. Never `python`/`pytest` on the login node.
- Commit after each task.

---

## File Structure

- Create: `g3nat/evaluation/fray.py` -- pure helpers + sweep + metrics.
- Create: `tests/test_models/test_fray.py` -- unit + small-model integration tests.
- Create: `scripts/fray_probe.py` -- runner (load, orchestrate, write `outputs_fray/`, plot).
- Create: `FrayProbeJob` -- sbatch runner.
- Modify: `.gitignore` -- add `outputs_fray/`.

---

## Task 1: Edge-location + region-mask helpers (pure)

**Files:**
- Create: `g3nat/evaluation/fray.py`
- Test: `tests/test_models/test_fray.py`

**Interfaces:**
- Produces:
  - `terminal_backbone_rows(graph, n_primary) -> list[int]`: rows in `graph.edge_geom` for the terminal primary backbone edge (both directed copies), i.e. the backbone edges (`edge_attr[:,0]==1`) whose endpoints are nodes `{n_primary, n_primary+1}`.
  - `terminal_h_indices(n_primary) -> tuple[int,int]`: `(n_primary-1, n_primary-2)` (terminal base, its stacked neighbor; `n_orb=1`).
  - `region_masks(n_dna, n_primary) -> dict[str, np.ndarray]`: boolean `n_dna x n_dna` masks with keys `diag`, `offdiag`, `terminal_local` (any element touching H index `n_primary-1` or `n_primary-2`), `distal` (off-diagonal and not terminal_local), `primary` (both indices `< n_primary`), `comp` (both `>= n_primary`), `cross` (one of each).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_fray.py
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from g3nat.graph.construction import sequence_to_graph
from g3nat.evaluation.fray import terminal_backbone_rows, terminal_h_indices, region_masks

def _entry(n=4):
    return {"bp_pars": np.zeros((n,6)), "step_pars": np.ones((n-1,6)),
            "primary_centroids": np.array([[0,0,i*3.4] for i in range(n)], float),
            "comp_centroids": np.array([[6,0,(n-1-i)*3.4] for i in range(n)], float)}

def test_terminal_backbone_rows_point_at_last_primary_step():
    g = sequence_to_graph("ACGT", "ACGT", geometry=_entry(4))
    rows = terminal_backbone_rows(g, 4)
    assert len(rows) == 2                     # both directed copies
    ei = g.edge_index
    for r in rows:
        assert g.edge_attr[r, 0] == 1         # backbone
        assert {int(ei[0, r]), int(ei[1, r])} == {4, 5}   # nodes N, N+1 (primary pos 2,3)

def test_terminal_h_indices():
    assert terminal_h_indices(4) == (3, 2)

def test_region_masks_partition_and_terminal():
    m = region_masks(8, 4)                    # duplex ACGT: M=8, primary 0..3, comp 4..7
    assert m["diag"].sum() == 8
    assert np.array_equal(m["diag"] | m["offdiag"], np.ones((8,8), bool))
    assert not np.any(m["diag"] & m["offdiag"])
    # terminal_local = touches index 2 or 3
    assert m["terminal_local"][3, 7] and m["terminal_local"][2, 0]
    assert not m["terminal_local"][4, 5]
    # distal is off-diagonal and not terminal_local
    assert not np.any(m["distal"] & m["terminal_local"])
    assert not np.any(m["distal"] & m["diag"])
    # block masks partition the off-diagonal-agnostic index space
    assert np.array_equal(m["primary"] | m["comp"] | m["cross"], np.ones((8,8), bool))
```

- [ ] **Step 2: Run to verify failure**

Run: `sbatch TestJob tests/test_models/test_fray.py -v` ; read `slurm-<jobid>.out`.
Expected: FAIL, `ModuleNotFoundError: No module named 'g3nat.evaluation.fray'`.

- [ ] **Step 3: Implement the helpers**

```python
# g3nat/evaluation/fray.py
"""Read-only probe: how the predicted Hamiltonian responds to terminal destacking."""
import numpy as np
import torch


def terminal_backbone_rows(graph, n_primary):
    """edge_geom rows of the terminal primary backbone edge (both directed copies).

    Primary position k is node 2+k; the last two primary bases are nodes
    n_primary and n_primary+1. The terminal stacking edge is the backbone edge
    (edge_attr[:,0]==1) between them.
    """
    ei = graph.edge_index
    ea = graph.edge_attr
    tgt = {n_primary, n_primary + 1}
    rows = []
    for r in range(ei.shape[1]):
        if ea[r, 0] == 1 and {int(ei[0, r]), int(ei[1, r])} == tgt:
            rows.append(r)
    return rows


def terminal_h_indices(n_primary):
    """Hamiltonian indices (terminal base, stacked neighbor) for n_orb=1."""
    return (n_primary - 1, n_primary - 2)


def region_masks(n_dna, n_primary):
    """Boolean n_dna x n_dna masks decomposing the Hamiltonian into regions."""
    M = n_dna
    idx = np.arange(M)
    diag = np.eye(M, dtype=bool)
    offdiag = ~diag
    term = {n_primary - 1, n_primary - 2}
    ti = np.array([i in term for i in idx])
    terminal_local = ti[:, None] | ti[None, :]
    distal = offdiag & ~terminal_local
    is_primary = idx < n_primary
    primary = is_primary[:, None] & is_primary[None, :]
    comp = (~is_primary)[:, None] & (~is_primary)[None, :]
    cross = ~(primary | comp)
    return {"diag": diag, "offdiag": offdiag, "terminal_local": terminal_local,
            "distal": distal, "primary": primary, "comp": comp, "cross": cross}
```

- [ ] **Step 4: Run to verify pass**

Run: `sbatch TestJob tests/test_models/test_fray.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add g3nat/evaluation/fray.py tests/test_models/test_fray.py
git commit -m "feat(fray): terminal-edge location + region-mask helpers (Fray Task 1)"
```

---

## Task 2: Sweep + whole-Hamiltonian metrics

**Files:**
- Modify: `g3nat/evaluation/fray.py`
- Test: `tests/test_models/test_fray.py`

**Interfaces:**
- Consumes: `terminal_backbone_rows` (Task 1), a `use_geometry=True` `DNATransportHamiltonianGNN`, a geometry-ON graph.
- Produces:
  - `run_fray_sweep(model, graph, rows, deltas) -> np.ndarray [n_delta, M, M]`: stacks `model.H[0]` (numpy, real) over the sweep. At each delta sets `edge_geom[rows,0]=d0+delta` and `edge_geom[rows,3]=r0+delta` (d0/r0 read from the unmorphed graph), forwards a 1-graph batch, restores the graph after. `deltas[0]` must be 0.
  - `sweep_metrics(Hstack, deltas, n_dna, n_primary) -> dict`: with `D=Hstack[k]-Hstack[0]` per k, returns `{"term_coupling":[n_delta], "argmax_ij":[n_delta,2] int, "fro":[n_delta], "region":{name:[n_delta]}}` where `term_coupling[k]=|Hstack[k][t0,t1]|` for `(t0,t1)=terminal_h_indices`, `argmax_ij` = unravel of `argmax|D|`, `fro`=Frobenius norm of D, `region[name]`=sum of `|D|` over that region mask.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models/test_fray.py  (add)
from torch_geometric.data import Batch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.evaluation.fray import run_fray_sweep, sweep_metrics

def _geo_model():
    torch.manual_seed(0)
    stats = {"backbone":{"mean":[0]*7,"std":[1]*7}, "hbond":{"mean":[0]*7,"std":[1]*7}}
    return DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
        energy_grid=np.linspace(-3,3,20), conv_type='gat',
        use_geometry=True, geom_norm_stats=stats)

def test_sweep_starts_at_unmorphed_and_moves():
    g = sequence_to_graph("ACGT","ACGT", geometry=_entry(4))
    m = _geo_model()
    with torch.no_grad():
        for p in m.geom_encoder.parameters(): p.add_(0.1)   # make geometry matter
    rows = terminal_backbone_rows(g, 4)
    H = run_fray_sweep(m, g, rows, np.linspace(0, 3, 5))
    assert H.shape[0] == 5 and H.shape[1] == H.shape[2] == 8
    mets = sweep_metrics(H, np.linspace(0,3,5), 8, 4)
    assert np.allclose(H[0] - H[0], 0)              # trivially
    assert mets["fro"][0] == 0.0                    # delta=0 -> no change
    assert mets["fro"][-1] > 0                       # morph changes H somewhere
    assert mets["argmax_ij"].shape == (5, 2)
    # regions present and non-negative
    for k in ("diag","distal","terminal_local","primary","comp","cross"):
        assert (mets["region"][k] >= 0).all()

def test_sweep_does_not_mutate_caller_graph():
    g = sequence_to_graph("ACGT","ACGT", geometry=_entry(4))
    rows = terminal_backbone_rows(g, 4)
    before = g.edge_geom.clone()
    run_fray_sweep(_geo_model(), g, rows, np.linspace(0, 2, 3))
    assert torch.equal(g.edge_geom, before)          # restored
```

- [ ] **Step 2: Run to verify failure**

Run: `sbatch TestJob tests/test_models/test_fray.py -v -k sweep`
Expected: FAIL, `ImportError: cannot import name 'run_fray_sweep'`.

- [ ] **Step 3: Implement**

```python
# g3nat/evaluation/fray.py  (add)
def run_fray_sweep(model, graph, rows, deltas):
    from torch_geometric.data import Batch
    assert float(deltas[0]) == 0.0, "deltas[0] must be 0 (unmorphed reference)"
    base = graph.edge_geom.clone()
    d0 = base[rows[0], 0].item()
    r0 = base[rows[0], 3].item()
    out = []
    model.eval()
    try:
        for delta in deltas:
            eg = base.clone()
            for r in rows:
                eg[r, 0] = d0 + float(delta)
                eg[r, 3] = r0 + float(delta)
            graph.edge_geom = eg
            with torch.no_grad():
                model(Batch.from_data_list([graph]))
            out.append(model.H[0].detach().cpu().numpy().copy())
    finally:
        graph.edge_geom = base
    return np.stack(out)


def sweep_metrics(Hstack, deltas, n_dna, n_primary):
    masks = region_masks(n_dna, n_primary)
    t0, t1 = terminal_h_indices(n_primary)
    H0 = Hstack[0]
    n = Hstack.shape[0]
    term = np.empty(n); fro = np.empty(n); amax = np.empty((n, 2), int)
    region = {k: np.empty(n) for k in masks}
    for k in range(n):
        D = np.abs(Hstack[k] - H0)
        term[k] = abs(Hstack[k][t0, t1])
        fro[k] = float(np.sqrt((D ** 2).sum()))
        amax[k] = np.unravel_index(int(np.argmax(D)), D.shape)
        for name, msk in masks.items():
            region[name][k] = float(D[msk].sum())
    return {"term_coupling": term, "argmax_ij": amax, "fro": fro, "region": region}
```

- [ ] **Step 4: Run to verify pass**

Run: `sbatch TestJob tests/test_models/test_fray.py -v`
Expected: PASS (all fray tests).

- [ ] **Step 5: Commit**

```bash
git add g3nat/evaluation/fray.py tests/test_models/test_fray.py
git commit -m "feat(fray): fray sweep + whole-Hamiltonian response metrics (Fray Task 2)"
```

---

## Task 3: Runner script + job (produces outputs_fray/)

**Files:**
- Create: `scripts/fray_probe.py`
- Create: `FrayProbeJob`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `load_trained_model`, `sequence_to_graph`, `compute_norm_stats`, all `fray.py` helpers.
- Produces: `outputs_fray/sweep_metrics.csv`, `outputs_fray/Hmats.npz`, `outputs_fray/norm_band.json`.

- [ ] **Step 1: Write the runner**

```python
# scripts/fray_probe.py
import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from g3nat.evaluation import load_trained_model
from g3nat.graph.construction import sequence_to_graph
from g3nat.graph.geometry import compute_norm_stats
from g3nat.evaluation.fray import terminal_backbone_rows, terminal_h_indices, sweep_metrics, run_fray_sweep

MODEL = "outputs_pickle_gat_geom/hamiltonian_pickle_model.pth"
CACHE = "geom_cache/geometry.pkl"
OUT = "outputs_fray"
# 3-4 training sequences spanning terminal steps; validated to exist in the cache at runtime.
CANDIDATES = ["aaac", "ggtc", "gtcc", "cagc", "tttc", "caaa"]
DELTAS = np.linspace(0.0, 5.0, 35)

def complement(seq):
    c = {'A':'T','T':'A','G':'C','C':'G'}
    return ''.join(c[b] for b in seq.upper())[::-1]

def main():
    os.makedirs(OUT, exist_ok=True)
    model, egrid, dev = load_trained_model(MODEL, device='cpu')
    assert getattr(model, 'n_orb', 1) == 1, "probe assumes n_orb=1"
    cache = pickle.load(open(CACHE, "rb"))
    stats = compute_norm_stats(cache)
    band = {"d": {"mean": stats["backbone"]["mean"][0], "std": stats["backbone"]["std"][0]},
            "rise": {"mean": stats["backbone"]["mean"][3], "std": stats["backbone"]["std"][3]}}
    json.dump(band, open(os.path.join(OUT, "norm_band.json"), "w"), indent=2)

    seqs = [s for s in CANDIDATES if s in cache][:4]
    rows_out = []
    Hsave = {}
    for seq in seqs:
        N = len(seq)
        g = sequence_to_graph(seq.upper(), complement(seq), geometry=cache[seq])
        rows = terminal_backbone_rows(g, N)
        if not rows:
            print(f"WARN {seq}: no terminal backbone geometry, skipping"); continue
        H = run_fray_sweep(model, g, rows, DELTAS)
        m = sweep_metrics(H, DELTAS, 2 * N, N)
        Hsave[f"{seq}_H"] = H
        d0 = g.edge_geom[rows[0], 0].item()
        for k, delta in enumerate(DELTAS):
            rows_out.append([seq, float(delta), d0 + float(delta), m["term_coupling"][k],
                             int(m["argmax_ij"][k, 0]), int(m["argmax_ij"][k, 1]),
                             m["fro"][k], m["region"]["terminal_local"][k],
                             m["region"]["distal"][k], m["region"]["diag"][k],
                             m["region"]["cross"][k]])
        print(f"{seq}: swept {len(DELTAS)} points, max||D||_F={m['fro'].max():.4g}, "
              f"final argmax={tuple(m['argmax_ij'][-1])} (terminal idx {terminal_h_indices(N)})")

    import csv
    with open(os.path.join(OUT, "sweep_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq","delta","d","term_coupling","argmax_i","argmax_j","fro",
                    "terminal_local","distal","diag","cross"])
        w.writerows(rows_out)
    np.savez_compressed(os.path.join(OUT, "Hmats.npz"), **Hsave)
    print(f"wrote {OUT}/sweep_metrics.csv, Hmats.npz, norm_band.json for {len(seqs)} sequences")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the job and gitignore entry**

```bash
# FrayProbeJob
#!/bin/bash
#SBATCH --job-name=g3nat-fray
#SBATCH --account=anantram-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --mem=8GB
##
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate g3nat
cd /mmfs1/gscratch/anantram/willll/G3NAT
python -u scripts/fray_probe.py
echo "=== fray exit code: $? ==="
```

Add `outputs_fray/` to `.gitignore`.

- [ ] **Step 3: Run the job**

Run: `sbatch FrayProbeJob` ; read `slurm-<jobid>.out`.
Expected: per-sequence summary lines; `outputs_fray/sweep_metrics.csv`, `Hmats.npz`, `norm_band.json` exist. Sanity: the `fro` column is 0 at `delta=0` for every sequence.

- [ ] **Step 4: Commit (script + job; outputs gitignored)**

```bash
git add scripts/fray_probe.py FrayProbeJob .gitignore
git commit -m "feat(fray): runner + job producing whole-Hamiltonian sweep data (Fray Task 3)"
```

---

## Task 4: Plots (dataviz skill)

**Files:**
- Modify: `scripts/fray_probe.py` (add a `plots()` step reading the saved data), or a `scripts/fray_plots.py`.

**Interfaces:**
- Consumes: `outputs_fray/sweep_metrics.csv`, `Hmats.npz`, `norm_band.json`.
- Produces: `outputs_fray/response_heatmaps.png`, `region_curves.png`, `terminal_coupling.png`.

- [ ] **Step 1: Invoke the dataviz skill** before writing any plotting code (palette, mark specs, light/dark, labeling).

- [ ] **Step 2: Implement plotting** (matplotlib, styled per dataviz):
  - `response_heatmaps.png`: per sequence, `|H(delta)-H(0)|` heatmap at a small-in-distribution delta and 2 large deltas; axes labeled by base index with primary/comp split marked; terminal index annotated.
  - `region_curves.png`: `terminal_local`, `distal`, `diag`, `cross`, and `fro` vs `d`, one panel per sequence (or overlaid), with the in-distribution `d` band (mean +/- 3*std from `norm_band.json`) shaded.
  - `terminal_coupling.png`: `term_coupling` vs `d` per sequence, in-distribution band shaded, physical-decay expectation annotated.

- [ ] **Step 3: Run and eyeball**

Run: `sbatch FrayProbeJob` (regenerates data + plots) or a plots-only job; confirm the three PNGs exist and render. Send them to the user for the read-out.

- [ ] **Step 4: Commit**

```bash
git add scripts/fray_probe.py   # (+ scripts/fray_plots.py if split)
git commit -m "feat(fray): response heatmaps + region + terminal-coupling plots (Fray Task 4)"
```

---

## Self-Review Notes (coverage against the spec)

- Spec readouts -> Task 2 `sweep_metrics` (heatmap data via `Hmats.npz`, argmax, region decomposition, fro, terminal coupling) + Task 4 plots.
- "Which part changes most / other side of the bases" -> `argmax_ij` tracking + `terminal_local` vs `distal` + `primary`/`comp`/`cross` region sums + the heatmap. Covered.
- Topology never changed; only slots 0 and 3 on the terminal backbone edge move -> `run_fray_sweep` (Task 2) restores the graph and touches only `rows`; Task 1 test asserts the rows are the correct terminal backbone edge.
- No-DFT, no model change, read-only -> Tasks import from `g3nat` and add only `g3nat/evaluation/fray.py` + `scripts/` + a job.
- `delta=0 -> D=0` sanity -> Task 2 test + Task 3 job-log check.
- `n_orb=1` assumption -> asserted in the runner; index mapping documented in Global Constraints.
