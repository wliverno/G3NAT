# X3DNA Geometry Extraction (offline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `g3nat/graph/geometry.py` — the offline module that turns a DNA PDB into the SE(3)-invariant per-edge geometry the GNN will consume: parse X3DNA/DSSR bp/step parameters, compute base-centroid distances, and cache them per sequence with per-edge-type normalization stats.

**Architecture:** Pure, independently-testable functions. `parse_dssr_out` and `base_centroids` are pure (fixture-tested, no DSSR needed). `run_dssr` shells out to the DSSR binary; `build_geometry_cache` ties them together over the PDB dataset and writes a cache; `compute_norm_stats` produces per-edge-type z-score stats. This plan produces the cache + stats — a self-contained subsystem. A separate Plan 2 consumes the cache to attach geometry to graph edges and the model.

**Tech Stack:** Python, NumPy, DSSR v2.8 (`/mmfs1/gscratch/anantram/asyed4/x3dna-dssr`), pytest. Tests run via SLURM (`sbatch TestJob <args>`) in the `g3nat` conda env — never on the login node.

## Global Constraints

- This plan touches ONLY `g3nat/graph/geometry.py` (new) and its tests + fixtures. It does NOT modify `construction.py`, `hamiltonian.py`, or any pipeline code — that is Plan 2.
- Distances and X3DNA params are SE(3)-invariant; `base_centroids` uses only intra-structure geometry.
- The DSSR binary path resolves from arg, then `$X3DNA_DSSR`, then default `/mmfs1/gscratch/anantram/asyed4/x3dna-dssr`.
- Cache is keyed by lowercase sequence string; lives at `geom_cache/geometry.pkl` (repo root; `geom_cache/` is gitignored — add it).
- Parameter order (verbatim from DSSR / spec): bp = [Shear, Stretch, Stagger, Buckle, Propeller, Opening]; step = [Shift, Slide, Rise, Tilt, Roll, Twist].
- DSSR prints the step-pars block twice; the parser returns exactly `Npair-1` step rows.
- All test commands run through SLURM: `sbatch TestJob <pytest-args>`; read `slurm-<jobid>.out`. Never `python` on the login node.

---

## File Structure

- Create: `g3nat/graph/geometry.py` — the extraction module.
- Create: `tests/test_graph/fixtures/aaac.out`, `tests/test_graph/fixtures/aaac.pdb` — static DSSR + PDB fixtures (copied from `/mmfs1/gscratch/anantram/asyed4/DNADataSet/aaac/`) so parser/centroid tests need no DSSR run.
- Create: `tests/test_graph/test_geometry.py` — unit + integration tests.
- Modify: `.gitignore` — add `geom_cache/`.

---

### Task 1: DSSR parameter parser

**Files:**
- Create: `g3nat/graph/geometry.py`
- Create: `tests/test_graph/fixtures/aaac.out` (copy)
- Create: `tests/test_graph/test_geometry.py`

**Interfaces:**
- Produces: `parse_dssr_out(out_text: str) -> dict` returning `{"bp_pars": np.ndarray[Npair,6], "step_pars": np.ndarray[Nstep,6]}` with `Nstep == Npair-1`, dtype float.

- [ ] **Step 1: Copy the DSSR fixture**

```bash
mkdir -p tests/test_graph/fixtures
cp /mmfs1/gscratch/anantram/asyed4/DNADataSet/aaac/aaac.out tests/test_graph/fixtures/aaac.out
```

- [ ] **Step 2: Write the failing parser test**

Create `tests/test_graph/test_geometry.py`:

```python
import os
import numpy as np
from g3nat.graph import geometry

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

def test_parse_dssr_out_aaac():
    out = open(os.path.join(FIX, "aaac.out")).read()
    r = geometry.parse_dssr_out(out)
    bp, step = r["bp_pars"], r["step_pars"]
    assert bp.shape == (4, 6), bp.shape          # aaac = 4 base pairs
    assert step.shape == (3, 6), step.shape      # 3 steps, deduped from the doubled block
    # first bp row: Shear,Stretch,Stagger,Buckle,Propeller,Opening
    np.testing.assert_allclose(bp[0], [0.00, -0.09, -0.00, 0.01, -1.23, -2.68], atol=1e-6)
    # first step row: Shift,Slide,Rise,Tilt,Roll,Twist
    np.testing.assert_allclose(step[0], [0.00, -0.20, 3.37, 0.01, -2.81, 35.90], atol=1e-6)
```

- [ ] **Step 3: Run it, expect FAIL** (module/function missing)

Run: `sbatch TestJob tests/test_graph/test_geometry.py::test_parse_dssr_out_aaac -v` ; read `slurm-<id>.out`.
Expected: FAIL (`ModuleNotFoundError`/`AttributeError`).

- [ ] **Step 4: Implement the parser**

Create `g3nat/graph/geometry.py`:

```python
"""Offline extraction of SE(3)-invariant edge geometry from DNA PDB structures."""
import os
import re
import subprocess
import numpy as np

_BRACKET = re.compile(r"\[([^\]]*)\]")

def _rows(out_text, tag):
    rows = []
    for line in out_text.splitlines():
        if tag in line:
            m = _BRACKET.search(line)
            if m:
                rows.append([float(x) for x in m.group(1).split()])
    return np.array(rows, dtype=float) if rows else np.zeros((0, 6))

def parse_dssr_out(out_text):
    """Parse bp-pars and step-pars from a DSSR --more .out file.

    Returns {"bp_pars": [Npair,6], "step_pars": [Nstep,6]} with Nstep == Npair-1.
    DSSR prints the step-pars block twice; we keep the first Npair-1 rows.
    """
    bp = _rows(out_text, "bp-pars:")
    step_all = _rows(out_text, "step-pars:")
    n_step = max(0, bp.shape[0] - 1)
    step = step_all[:n_step]
    return {"bp_pars": bp, "step_pars": step}
```

- [ ] **Step 5: Run it, expect PASS**

Run: `sbatch TestJob tests/test_graph/test_geometry.py::test_parse_dssr_out_aaac -v` ; read `slurm-<id>.out`.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add g3nat/graph/geometry.py tests/test_graph/test_geometry.py tests/test_graph/fixtures/aaac.out
git commit -m "feat(geometry): DSSR bp/step-pars parser + fixture

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011n22khW2GNEbTGN359sFDz"
```

---

### Task 2: base centroids + centroid distance

**Files:**
- Modify: `g3nat/graph/geometry.py`
- Create: `tests/test_graph/fixtures/aaac.pdb` (copy)
- Modify: `tests/test_graph/test_geometry.py`

**Interfaces:**
- Produces:
  - `base_centroids(pdb_path: str) -> dict[(chain:int, resseq:int) -> np.ndarray[3]]` — centroid of each residue's base (non-backbone, non-hydrogen) heavy atoms. `chain` is 0-based strand index (TER-delimited).
  - `centroid_distance(a: np.ndarray, b: np.ndarray) -> float`.

- [ ] **Step 1: Copy the PDB fixture**

```bash
cp /mmfs1/gscratch/anantram/asyed4/DNADataSet/aaac/aaac.pdb tests/test_graph/fixtures/aaac.pdb
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_graph/test_geometry.py`:

```python
def test_base_centroids_and_distances():
    pdb = os.path.join(FIX, "aaac.pdb")
    cent = geometry.base_centroids(pdb)
    # aaac: strand 0 = DA DA DA DC (resseq 1..4), strand 1 = DG DT DT DT (resseq 1..4)
    assert (0, 1) in cent and (1, 1) in cent
    # complementary (Watson-Crick) distance: primary res i pairs with strand1 res (N+1-i)
    d_pair = geometry.centroid_distance(cent[(0, 1)], cent[(1, 4)])   # DA1 : DT8
    assert 4.5 < d_pair < 7.5, d_pair          # ~6.0 A, NOT the ~0.09 A frame-origin degeneracy
    # stacking (neighbor) distance on strand 0
    d_stack = geometry.centroid_distance(cent[(0, 1)], cent[(0, 2)])  # DA1 : DA2
    assert 3.0 < d_stack < 4.5, d_stack        # ~3.7 A
```

- [ ] **Step 3: Run it, expect FAIL**

Run: `sbatch TestJob tests/test_graph/test_geometry.py::test_base_centroids_and_distances -v`
Expected: FAIL (`AttributeError: base_centroids`).

- [ ] **Step 4: Implement**

Append to `g3nat/graph/geometry.py`:

```python
# sugar-phosphate backbone atom names to exclude when taking the base centroid
_BACKBONE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
             "C2'", "C1'", "HO5'", "HO3'", "H5'", "H5''", "H4'", "H3'",
             "H2'", "H2''", "H1'"}

def base_centroids(pdb_path):
    """Centroid of each residue's base (non-backbone, non-hydrogen) heavy atoms.
    Keyed by (chain_index, resseq); chain_index increments at each TER record."""
    coords = {}
    chain = 0
    for ln in open(pdb_path):
        if ln.startswith("TER"):
            chain += 1
            continue
        if ln.startswith(("ATOM", "HETATM")):
            name = ln[12:16].strip()
            if name.startswith("H") or "'" in name or name in _BACKBONE:
                continue
            resseq = int(ln[22:26])
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            coords.setdefault((chain, resseq), []).append(xyz)
    return {k: np.mean(v, axis=0) for k, v in coords.items()}

def centroid_distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
```

- [ ] **Step 5: Run it, expect PASS**

Run: `sbatch TestJob tests/test_graph/test_geometry.py::test_base_centroids_and_distances -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add g3nat/graph/geometry.py tests/test_graph/test_geometry.py tests/test_graph/fixtures/aaac.pdb
git commit -m "feat(geometry): base-centroid extraction + distance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011n22khW2GNEbTGN359sFDz"
```

---

### Task 3: DSSR runner + cache builder + norm stats

**Files:**
- Modify: `g3nat/graph/geometry.py`
- Modify: `.gitignore` (add `geom_cache/`)
- Modify: `tests/test_graph/test_geometry.py`

**Interfaces:**
- Consumes: `parse_dssr_out`, `base_centroids` (Tasks 1-2).
- Produces:
  - `run_dssr(pdb_path, dssr_bin=None, workdir=None) -> str` — runs `dssr -i=<pdb> --more -o=<out>` and returns the `.out` text.
  - `build_geometry_cache(dataset_dir, out_path, sequences=None) -> dict` — for each `<seq>/<seq>.pdb`, store `{seq: {"bp_pars":[Npair,6], "step_pars":[Nstep,6], "primary_centroids":[Nprimary,3], "comp_centroids":[Ncomp,3]}}`. Primary = strand 0 by resseq; comp = strand 1 by resseq. Missing/failed sequences are recorded absent (warned), not fatal. Writes a pickle to `out_path`.
  - `compute_norm_stats(cache) -> {"backbone": {"mean":[7],"std":[7]}, "hbond": {"mean":[7],"std":[7]}}` — assembles the 7-tuples per edge type (backbone = `[stack_dist, shift, slide, rise, tilt, roll, twist]`, hbond = `[pair_dist, shear, stretch, stagger, buckle, propeller, opening]`) over all cached sequences and returns per-type z-score stats (std floored at 1e-6).

- [ ] **Step 1: Gitignore the cache dir**

Add `geom_cache/` to `.gitignore` (new line). Commit is folded into Step 6.

- [ ] **Step 2: Write the failing integration test**

Append to `tests/test_graph/test_geometry.py`:

```python
import shutil
import pytest

DATASET = "/mmfs1/gscratch/anantram/asyed4/DNADataSet"

@pytest.mark.skipif(not os.path.isdir(DATASET), reason="DSSR dataset not present")
def test_build_cache_and_norm_stats(tmp_path):
    out = str(tmp_path / "geom.pkl")
    cache = geometry.build_geometry_cache(DATASET, out, sequences=["aaac", "aaat"])
    assert "aaac" in cache
    e = cache["aaac"]
    assert e["bp_pars"].shape == (4, 6) and e["step_pars"].shape == (3, 6)
    assert e["primary_centroids"].shape == (4, 3) and e["comp_centroids"].shape == (4, 3)
    assert os.path.exists(out)
    stats = geometry.compute_norm_stats(cache)
    for t in ("backbone", "hbond"):
        assert np.asarray(stats[t]["mean"]).shape == (7,)
        assert np.all(np.asarray(stats[t]["std"]) >= 1e-6)
    # hbond distance channel (slot 0) is the atom distance ~6, not the degenerate ~0.09
    assert stats["hbond"]["mean"][0] > 4.0
```

- [ ] **Step 3: Run it, expect FAIL**

Run: `sbatch TestJob tests/test_graph/test_geometry.py::test_build_cache_and_norm_stats -v`
Expected: FAIL (functions missing).

- [ ] **Step 4: Implement**

Append to `g3nat/graph/geometry.py`:

```python
import pickle
import warnings

_DSSR_DEFAULT = "/mmfs1/gscratch/anantram/asyed4/x3dna-dssr"

def _dssr_bin(dssr_bin=None):
    return dssr_bin or os.environ.get("X3DNA_DSSR", _DSSR_DEFAULT)

def run_dssr(pdb_path, dssr_bin=None, workdir=None):
    workdir = workdir or os.path.dirname(os.path.abspath(pdb_path))
    out = os.path.join(workdir, "_dssr_tmp.out")
    subprocess.run([_dssr_bin(dssr_bin), f"-i={pdb_path}", "--more", f"-o={out}"],
                   check=True, capture_output=True, cwd=workdir)
    text = open(out).read()
    try:
        os.remove(out)
    except OSError:
        pass
    return text

def _centroids_by_strand(pdb_path):
    cent = base_centroids(pdb_path)
    strands = {}
    for (chain, resseq), xyz in cent.items():
        strands.setdefault(chain, []).append((resseq, xyz))
    ordered = {}
    for chain, items in strands.items():
        items.sort(key=lambda t: t[0])
        ordered[chain] = np.array([xyz for _, xyz in items])
    return ordered

def build_geometry_cache(dataset_dir, out_path, sequences=None):
    if sequences is None:
        sequences = sorted(d for d in os.listdir(dataset_dir)
                           if os.path.isdir(os.path.join(dataset_dir, d)))
    cache = {}
    for seq in sequences:
        pdb = os.path.join(dataset_dir, seq, f"{seq}.pdb")
        if not os.path.exists(pdb):
            warnings.warn(f"missing pdb for {seq}"); continue
        try:
            pars = parse_dssr_out(run_dssr(pdb))
            strands = _centroids_by_strand(pdb)
            cache[seq] = {
                "bp_pars": pars["bp_pars"], "step_pars": pars["step_pars"],
                "primary_centroids": strands.get(0, np.zeros((0, 3))),
                "comp_centroids": strands.get(1, np.zeros((0, 3))),
            }
        except Exception as ex:  # noqa: BLE001
            warnings.warn(f"geometry failed for {seq}: {ex}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    return cache

def _edge_rows(cache):
    """Assemble backbone and hbond 7-tuples across all sequences (for norm stats)."""
    back, hb = [], []
    for e in cache.values():
        pc, cc = e["primary_centroids"], e["comp_centroids"]
        step, bp = e["step_pars"], e["bp_pars"]
        n = pc.shape[0]
        # backbone (primary strand): step k between primary k and k+1
        for k in range(min(step.shape[0], n - 1)):
            d = centroid_distance(pc[k], pc[k + 1])
            back.append([d, *step[k]])
        # hbond: pair k = primary k with comp (N-1-k)
        for k in range(min(bp.shape[0], n, cc.shape[0])):
            d = centroid_distance(pc[k], cc[cc.shape[0] - 1 - k])
            hb.append([d, *bp[k]])
    return np.array(back), np.array(hb)

def compute_norm_stats(cache):
    back, hb = _edge_rows(cache)
    def st(a):
        if a.size == 0:
            return {"mean": [0.0] * 7, "std": [1.0] * 7}
        return {"mean": a.mean(0).tolist(),
                "std": np.maximum(a.std(0), 1e-6).tolist()}
    return {"backbone": st(back), "hbond": st(hb)}
```

- [ ] **Step 5: Run it, expect PASS**

Run: `sbatch TestJob tests/test_graph/test_geometry.py -v` (whole file — Tasks 1-3).
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add g3nat/graph/geometry.py tests/test_graph/test_geometry.py .gitignore
git commit -m "feat(geometry): DSSR runner, per-sequence cache, per-edge-type norm stats

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011n22khW2GNEbTGN359sFDz"
```

---

### Task 4: SE(3)-invariance regression test

**Files:**
- Modify: `tests/test_graph/test_geometry.py`

**Interfaces:** consumes `run_dssr`, `parse_dssr_out`, `base_centroids`, `centroid_distance`.

- [ ] **Step 1: Write the invariance test**

Append to `tests/test_graph/test_geometry.py`:

```python
@pytest.mark.skipif(not os.path.isdir(DATASET), reason="DSSR dataset not present")
def test_geometry_is_se3_invariant(tmp_path):
    src = os.path.join(DATASET, "aaac", "aaac.pdb")
    lines = open(src).read().splitlines()
    rng = np.random.RandomState(3)
    A = rng.randn(3, 3); Q, R = np.linalg.qr(A); Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0: Q[:, 0] = -Q[:, 0]
    t = np.array([11.0, -22.0, 33.0])
    rot = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            v = Q @ xyz + t
            ln = ln[:30] + f"{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}" + ln[54:]
        rot.append(ln)
    rp = str(tmp_path / "aaac_rot.pdb"); open(rp, "w").write("\n".join(rot) + "\n")

    p0 = geometry.parse_dssr_out(geometry.run_dssr(src, workdir=str(tmp_path)))
    p1 = geometry.parse_dssr_out(geometry.run_dssr(rp, workdir=str(tmp_path)))
    np.testing.assert_allclose(p0["step_pars"], p1["step_pars"], atol=0.05)
    np.testing.assert_allclose(p0["bp_pars"], p1["bp_pars"], atol=0.05)
    c0, c1 = geometry.base_centroids(src), geometry.base_centroids(rp)
    d0 = geometry.centroid_distance(c0[(0, 1)], c0[(0, 2)])
    d1 = geometry.centroid_distance(c1[(0, 1)], c1[(0, 2)])
    assert abs(d0 - d1) < 1e-3
```

- [ ] **Step 2: Run it, expect PASS**

Run: `sbatch TestJob tests/test_graph/test_geometry.py::test_geometry_is_se3_invariant -v`
Expected: PASS (DSSR params unchanged within rounding; centroid distance exactly unchanged).

- [ ] **Step 3: Commit**

```bash
git add tests/test_graph/test_geometry.py
git commit -m "test(geometry): SE(3)-invariance regression (rotate+translate)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011n22khW2GNEbTGN359sFDz"
```

---

## Self-Review

**1. Spec coverage:** parser (Task 1), centroid distances (Task 2), DSSR runner + cache + per-edge-type norm stats (Task 3), invariance regression (Task 4) — all map to the spec's "Offline preprocessing" + "Normalization" + "Invariance" sections. Deferred to Plan 2 (integration): `edge_geom`/`edge_geom_mask` tensors in `construction.py`, the geom encoder + `use_geometry` in the model, data/train threading, functional + backward-compat tests.

**2. Placeholder scan:** every code step has complete code; no TBD/TODO.

**3. Type consistency:** cache entry keys (`bp_pars`, `step_pars`, `primary_centroids`, `comp_centroids`) are used identically in `build_geometry_cache`, `_edge_rows`, and Task 3's test. `base_centroids` returns `(chain, resseq)` keys used consistently in Task 2 and Task 4. 7-tuple order (dist, then 6 X3DNA params) is consistent between `_edge_rows` and the spec.

## Open questions / notes for Plan 2

- **Residue-order mapping is the crux of Plan 2:** the cache stores centroids by PDB `(chain, resseq)`; Plan 2 must map primary node `i` -> `primary_centroids[i]`, complementary node `j` -> the right `comp_centroids` entry, and the H-bond pairing (primary `i` <-> comp `N-1-i`) exactly as `sequence_to_graph` builds them. Validate against a known duplex before wiring.
- Backbone step params are per base-pair-step (shared by both strands' backbone edges at a level); only the `d_centroid` slot is per-strand. Plan 2 assembles the final per-edge 7-tuples.
