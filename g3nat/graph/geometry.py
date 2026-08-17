"""Offline extraction of SE(3)-invariant edge geometry from DNA PDB structures."""
import os
import re
import pickle
import shutil
import tempfile
import warnings
import subprocess
import numpy as np

_BRACKET = re.compile(r"\[([^\]]*)\]")
_DSSR_DEFAULT = "/mmfs1/gscratch/anantram/asyed4/x3dna-dssr"


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


# sugar-phosphate backbone atom names to exclude when taking the base centroid
_BACKBONE = {"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'",
             "C2'", "C1'", "HO5'", "HO3'", "H5'", "H5''", "H4'", "H3'",
             "H2'", "H2''", "H1'"}


def base_centroids(pdb_path):
    """Centroid of each residue's base (non-backbone, non-hydrogen) heavy atoms.

    Keyed by (chain_index, resseq); chain_index increments at each TER record.
    """
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


def _dssr_bin(dssr_bin=None):
    return dssr_bin or os.environ.get("X3DNA_DSSR", _DSSR_DEFAULT)


def run_dssr(pdb_path, dssr_bin=None, workdir=None):
    """Run DSSR on a PDB and return the .out text.

    DSSR writes its output + auxiliary files into the working directory, so we run
    it in a writable temp dir by default (the source PDB dir may be read-only).
    """
    pdb_path = os.path.abspath(pdb_path)
    owns_dir = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="dssr_")
    out = os.path.join(workdir, "dssr.out")
    try:
        subprocess.run([_dssr_bin(dssr_bin), f"-i={pdb_path}", "--more", f"-o={out}"],
                       check=True, capture_output=True, cwd=workdir)
        text = open(out).read()
    finally:
        if owns_dir:
            shutil.rmtree(workdir, ignore_errors=True)
    return text


def _centroids_by_strand(pdb_path):
    """Per-strand base centroids as arrays ordered by residue number.

    Returns {chain_index: np.ndarray[N_res, 3]}.
    """
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
    """Run DSSR + centroids over <seq>/<seq>.pdb and cache per-sequence geometry.

    cache[seq] = {bp_pars [Npair,6], step_pars [Nstep,6],
                  primary_centroids [Nprimary,3], comp_centroids [Ncomp,3]}.
    Missing/failed sequences are warned and skipped, not fatal. Writes a pickle.
    """
    if sequences is None:
        sequences = sorted(d for d in os.listdir(dataset_dir)
                           if os.path.isdir(os.path.join(dataset_dir, d)))
    cache = {}
    for seq in sequences:
        pdb = os.path.join(dataset_dir, seq, f"{seq}.pdb")
        if not os.path.exists(pdb):
            warnings.warn(f"missing pdb for {seq}")
            continue
        try:
            pars = parse_dssr_out(run_dssr(pdb))
            strands = _centroids_by_strand(pdb)
            cache[seq] = {
                "bp_pars": pars["bp_pars"],
                "step_pars": pars["step_pars"],
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
    """Assemble backbone and hbond 7-tuples across all sequences (for norm stats).

    backbone = [stack_dist, shift, slide, rise, tilt, roll, twist]
    hbond    = [pair_dist,  shear, stretch, stagger, buckle, propeller, opening]
    """
    back, hb = [], []
    for e in cache.values():
        pc, cc = e["primary_centroids"], e["comp_centroids"]
        step, bp = e["step_pars"], e["bp_pars"]
        n = pc.shape[0]
        # backbone (primary strand): step k between primary k and k+1
        for k in range(min(step.shape[0], max(0, n - 1))):
            back.append([centroid_distance(pc[k], pc[k + 1]), *step[k]])
        # hbond: pair k = primary k with comp (Ncomp-1-k)
        for k in range(min(bp.shape[0], n, cc.shape[0])):
            hb.append([centroid_distance(pc[k], cc[cc.shape[0] - 1 - k]), *bp[k]])
    return np.array(back), np.array(hb)


def assemble_graph_geometry(primary_seq, comp_seq, entry):
    """Map a cached per-sequence geometry entry onto graph edge identities.

    Returns {edge_id: [d_centroid, t1, t2, t3, r1, r2, r3]} with edge ids:
      ("backbone", "primary", i)        primary step between positions i and i+1
      ("backbone", "complementary", j)  comp step between comp positions j and
                                        j+1, sharing primary step index n-2-j
      ("hbond", i)                      pair i (primary i <-> comp n-1-i)
    Keys are absent where the underlying params/centroids are missing; callers
    mask those edges to 0. Backbone slot 0 is the per-strand stacking centroid
    distance; H-bond slot 0 is the atom-centroid pairing distance (~6 A), never
    the degenerate frame-origin distance.
    """
    bp = np.asarray(entry["bp_pars"], float)
    step = np.asarray(entry["step_pars"], float)
    pc = np.asarray(entry["primary_centroids"], float)
    cc = np.asarray(entry["comp_centroids"], float)
    n = pc.shape[0]
    ncomp = cc.shape[0]
    out = {}
    # primary backbone: step i between primary i and i+1
    for i in range(min(step.shape[0], max(0, n - 1))):
        out[("backbone", "primary", i)] = [centroid_distance(pc[i], pc[i + 1]), *step[i]]
    # complementary backbone: comp step j shares primary step index n-2-j
    for j in range(max(0, ncomp - 1)):
        si = n - 2 - j
        if 0 <= si < step.shape[0]:
            out[("backbone", "complementary", j)] = [centroid_distance(cc[j], cc[j + 1]), *step[si]]
    # hbond: primary i pairs comp n-1-i
    for i in range(min(bp.shape[0], n, ncomp)):
        out[("hbond", i)] = [centroid_distance(pc[i], cc[ncomp - 1 - i]), *bp[i]]
    return out


def compute_norm_stats(cache, sequences=None):
    """Per-edge-type z-score stats over the assembled 7-tuples (std floored at 1e-6).

    If `sequences` is given, stats are computed only over those keys (lowercased),
    restricted to entries present in `cache` -- e.g. the training split only, so
    val/test sequences do not leak into the normalization stats.
    """
    if sequences is not None:
        keys = {s.lower() for s in sequences}
        cache = {k: v for k, v in cache.items() if k in keys}
    back, hb = _edge_rows(cache)

    def st(a):
        if a.size == 0:
            return {"mean": [0.0] * 7, "std": [1.0] * 7}
        return {"mean": a.mean(0).tolist(),
                "std": np.maximum(a.std(0), 1e-6).tolist()}

    return {"backbone": st(back), "hbond": st(hb)}
