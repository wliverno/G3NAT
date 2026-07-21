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
