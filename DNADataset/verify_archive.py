#!/usr/bin/env python3
"""Exact comparison of two records, and a whole-archive round-trip check.

    python DNADataset/verify_archive.py transport.h5 pickle_files_v2 [heldout_dir ...]

Imports every run group of the archive and compares it field by field with its source
pickle. Exit status 1 and a list of differences if anything differs.
"""
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np


def records_equal(a, b, path=""):
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        if list(a) != list(b):
            return [f"{path}: keys {list(a)} != {list(b)}"]
        out = []
        for k in a:
            out += records_equal(a[k], b[k], f"{path}.{k}" if path else k)
        return out
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{path}: dtype/shape {a.dtype}{a.shape} != {b.dtype}{b.shape}"]
        return [] if np.array_equal(a, b) else [f"{path}: values differ"]
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += records_equal(x, y, f"{path}[{i}]")
            if len(out) > 5:
                break
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main(h5_path, *pickle_dirs):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from import_hdf5 import read_record
    src = {}
    for d in pickle_dirs:
        for p in Path(d).glob("*_run*.pkl"):
            src[p.stem] = p
    bad, n = [], 0
    with h5py.File(h5_path, "r") as h:
        for seq in h:
            for run in h[seq]:
                n += 1
                key = f"{seq}_{run}"
                if key not in src:
                    bad.append(f"{key}: no source pickle")
                    continue
                with open(src[key], "rb") as f:
                    diffs = records_equal(pickle.load(f), read_record(h[seq][run], seq))
                bad += [f"{key}: {d}" for d in diffs]
    print(f"checked {n} records against {len(src)} source pickles; {len(bad)} differences")
    if n != len(src):
        bad.append(f"archive has {n} records, sources have {len(src)}")
    for b in bad[:50]:
        print("  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
