#!/usr/bin/env python3
"""Write Fock/overlap matrices and the per-row orbital map to matrices.h5 (companion to transport.h5)."""
import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaussian_basis import ELEMENT_OF_Z, derive_basis, parse_dump_header  # noqa: E402

ROOT_ATTRS = {
    "units_fock": "Hartree",
    "units_overlap": "dimensionless",
    "basis": "6-31G(d,p), Cartesian d (6 functions), Gaussian 16 function order",
    "component_order_d": "DXX DYY DZZ DXY DXZ DYZ (Gaussian Cartesian order; type_code 2001-2006)",
    "row_to_atom": ("basis/atom_index is 0-based into the atoms table of transport.h5 "
                    "(/<sequence>/<run>/atoms/*)"),
    "license": "CC-BY-4.0",
    "companion": "transport.h5",
}
VERIFIED = "derived; verified against this sequence's Gaussian matrix-element file"


def _load(path, var):
    return np.asarray(loadmat(path, variable_names=[var])[var], dtype=np.float64)


def export_matrices(jobs, out_path, attrs):
    n = 0
    with h5py.File(out_path, "w") as h:
        for k, v in attrs.items():
            h.attrs[k] = v
        for seq, elements, fock_mat, overlap_mat, dump_header in jobs:
            b = derive_basis(elements)
            F = _load(fock_mat, f"{seq}_Fock")
            S = _load(overlap_mat, f"{seq}_Overlap")
            nb = len(b["atom_index"])
            for name, M in (("fock", F), ("overlap", S)):
                if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != nb:
                    raise ValueError(f"{seq}: {name} is {M.shape} but the basis has {nb} functions")
                if not np.array_equal(M, M.T):
                    raise ValueError(f"{seq}: {name} is not exactly symmetric")
            source = "derived"
            if dump_header is not None:
                d = parse_dump_header(dump_header)
                if [ELEMENT_OF_Z[z] for z in d["atomic_numbers"]] != list(elements):
                    raise ValueError(f"{seq}: dump atom order differs from the atoms table")
                if b["type_code"].tolist() != d["ibftyp"]:
                    raise ValueError(f"{seq}: derived type codes differ from the dump's IBfTyp")
                source = VERIFIED
            g = h.create_group(f"/{seq}")
            g.create_dataset("fock", data=F, compression="gzip", compression_opts=4)
            g.create_dataset("overlap", data=S, compression="gzip", compression_opts=4)
            bg = g.create_group("basis")
            for k, v in b.items():
                bg.create_dataset(k, data=v)
            g.attrs["label_source"] = source
            n += 1
    return n


def _jobs(pickle_dirs, train_mats, heldout_mats):
    recs = {}
    for d in pickle_dirs:
        for p in sorted(Path(d).glob("*_run1.pkl")):
            with open(p, "rb") as f:
                r = pickle.load(f)
            recs[r["sequence"]] = list(r["atoms"]["element"])
    roots = [(Path(train_mats), False)] + [(Path(x), True) for x in heldout_mats]
    jobs, missing = [], []
    for seq, els in sorted(recs.items()):
        for root, held in roots:
            fm, om = root / seq / f"{seq}_Fock.mat", root / seq / f"{seq}_Overlap.mat"
            if fm.exists() and om.exists():
                jobs.append((seq, els, fm, om, root / seq / f"{seq}.txt" if held else None))
                break
        else:
            missing.append(seq)
    if missing:
        raise SystemExit(f"{len(missing)} sequences with records have no Fock/Overlap: {missing}")
    have = {p.parent.name for root, _ in roots for p in root.glob("*/*_Fock.mat")}
    print("matrices without transport records (excluded):", sorted(have - set(recs)))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_path")
    ap.add_argument("--pickles", nargs="+", required=True)
    ap.add_argument("--train-mats", required=True)
    ap.add_argument("--heldout-mats", nargs="+", required=True)
    a = ap.parse_args()
    n = export_matrices(_jobs(a.pickles, a.train_mats, a.heldout_mats), a.out_path, ROOT_ATTRS)
    print(f"wrote {n} sequences to {a.out_path}")
