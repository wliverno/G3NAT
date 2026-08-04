#!/usr/bin/env python3
"""Python port of DOS_calc.m.

Computes total DOS(E) = -sum(imag(diag(Gr)))/pi and per-atom DOSAtom via
orbital-block sums, and writes DOS_<strand>_gammaL_<gL>_gammaR_<gR>.mat with
`Energy`, `DOS`, `DOSAtom` -- same variable names/shapes as the MATLAB
script (`Energy`/`DOS` are (1, NE) row vectors, `DOSAtom` is
(NAtoms, NE), to stay readable by DNADataset/convert_to_pickle.py
unchanged).

eta is read from Parameters.txt (DOS_calc.m does NOT override it to 0 the
way DNATransmission_Ballistic.m does), but in this dataset it is always
0.000000 ("Broadening (for DOS)" section), so eta = 0 in practice here too
-- settled project policy per the task brief. Not hardcoded in this file
since DOS_calc.m itself reads it; if Parameters.txt ever has a nonzero
Broadening value that is a genuine input change, not something this port
should silently override.

MATLAB-vs-numpy differences preserved deliberately: same `\\` (mldivide) vs
inverse and 1-based-vs-0-based indexing notes as transmission.py.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.linalg as sla

from negf_common import parse_parameters, orbital_offsets, build_sumSig, load_H0


def compute_dos(params: dict, H0: np.ndarray, eta: float):
    orbitals = params["Orbitals"]
    size_h = int(orbitals.sum())
    if size_h != H0.shape[0]:
        raise ValueError(f"Norb ({size_h}) != size(H0) ({H0.shape[0]})")

    n_atoms = orbitals.size
    sumSig = build_sumSig(params)
    bounds = orbital_offsets(orbitals)

    Energy = params["Energy"]
    NE = Energy.size
    DOS = np.full(NE, -1.0, dtype=np.float64)
    DOSAtom = np.full((n_atoms, NE), -1.0, dtype=np.float64)

    I = np.eye(size_h, dtype=np.complex128)

    for nE in range(NE):
        E = Energy[nE]
        A = (E + 1j * eta) * np.eye(size_h) - H0 - sumSig
        Gr = sla.solve(A, I)

        tempM = np.imag(np.diag(Gr))
        DOS[nE] = -tempM.sum() / np.pi

        for atom_1based in range(1, n_atoms + 1):
            lo, hi = bounds(atom_1based)
            DOSAtom[atom_1based - 1, nE] = -tempM[lo:hi].sum() / np.pi

    return Energy, DOS, DOSAtom


def process_run(strand: str, run_dir: Path, base_dir: Path = None):
    run_dir = Path(run_dir)
    base_dir = Path(base_dir) if base_dir else run_dir.parent

    params = parse_parameters(run_dir / "Parameters.txt")
    strand = params["strand"] or strand
    H0 = load_H0(strand, base_dir)

    eta = params["eta"]  # read from Parameters.txt; expected to be 0.0 in
    # this dataset (settled project policy per the task brief), but DOS_calc.m
    # itself does not hardcode/override it the way the transmission script
    # does, so this port doesn't either.
    Energy, DOS, DOSAtom = compute_dos(params, H0, eta=eta)

    gL, gR = params["gammaL"], params["gammaR"]
    out_name = f"DOS_{strand}_gammaL_{gL:g}_gammaR_{gR:g}.mat"
    out_path = run_dir / out_name
    sio.savemat(out_path, {
        "Energy": Energy.reshape(1, -1),
        "DOS": DOS.reshape(1, -1),
        "DOSAtom": DOSAtom,
    })
    print(f"Finished DOS! -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: dos_calc.py <strand_name> <run_dir> [base_dir]")
        sys.exit(1)
    strand_arg = sys.argv[1]
    run_dir_arg = Path(sys.argv[2])
    base_dir_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    process_run(strand_arg, run_dir_arg, base_dir_arg)
