#!/usr/bin/env python3
"""Convert the published transport.h5 back into the pickle records the training code reads.

    python DNADataset/import_hdf5.py transport.h5 out_dir [--split train|heldout]

Exact inverse of export_hdf5.py: every field, type and value of the original pickle is
restored (atoms.chain, which is blank in every source record, is restored as blank).
Refuses to overwrite existing files.
"""
import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def read_record(g, sequence):
    a = g["atoms"]
    n = a["resseq"].shape[0]
    return {
        "sequence": sequence,
        "complementary_sequence": _s(g["complementary_sequence"][()]),
        "gjf_text": _s(g["gjf_text"][()]),
        "contacts": {
            "left_atoms": [int(x) for x in g["left_atoms"][:]],
            "right_atoms": [int(x) for x in g["right_atoms"][:]],
            "coupling_eV": float(g.attrs["coupling_eV"]),
            "contact_type": _s(g.attrs["contact_type"]),
        },
        "Egrid": g["Egrid"][:].astype(np.float64),
        "T": g["T"][:].astype(np.float64),
        "DOS": g["DOS"][:].astype(np.float64),
        "DOSAtom": g["DOSAtom"][:].astype(np.float64),
        "atoms": {
            "element": [_s(x) for x in a["element"][:]],
            "name": [_s(x) for x in a["name"][:]],
            "resseq": [int(x) for x in a["resseq"][:]],
            "resname": [_s(x) for x in a["resname"][:]],
            "chain": [" "] * n,
            "xyz": a["xyz"][:].astype(np.float64),
        },
        "energy_reference_eV": float(g.attrs["energy_reference_eV"]),
    }


def import_archive(h5_path, out_dir, split=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h:
        todo = [(s, r) for s in h for r in h[s]
                if split is None or _s(h[s][r].attrs["split"]) == split]
        clash = [f"{s}_{r}.pkl" for s, r in todo if (out_dir / f"{s}_{r}.pkl").exists()]
        if clash:
            raise FileExistsError(f"{len(clash)} target files exist in {out_dir}, e.g. {clash[0]}")
        for s, r in todo:
            with open(out_dir / f"{s}_{r}.pkl", "wb") as f:
                pickle.dump(read_record(h[s][r], s), f)
    return len(todo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path")
    ap.add_argument("out_dir")
    ap.add_argument("--split", choices=("train", "heldout"))
    a = ap.parse_args()
    print(f"wrote {import_archive(a.h5_path, a.out_dir, a.split)} records to {a.out_dir}")
