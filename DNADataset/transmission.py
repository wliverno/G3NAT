#!/usr/bin/env python3
"""Python port of DNATransmission_Ballistic.m.

Computes ballistic transmission T(E) = Tr[GammaL Gr GammaR Ga] (summed over
all Lsite/Rsite atom pairs) via the wide-band approximation, and writes
Tran_<strand>_gammaL_<gL>_gammaR_<gR>.mat with the same `Energy`, `T`
variable names/shapes as the MATLAB script (both (1, NE) row vectors, to
stay readable by DNADataset/convert_to_pickle.py unchanged).

eta = 0 is hardcoded here exactly as in DNATransmission_Ballistic.m (line
~70, "eta = 0;" -- overriding whatever eta happens to be read from
Parameters.txt). This is settled project policy, not a tunable; do not
change it.

MATLAB-vs-numpy differences preserved deliberately:
  - `Gr = ((E + 1i*eta)*eye(sizeH) - H0 - sumSig) \\ eye(sizeH)` is a linear
    SOLVE against the identity (mldivide), not an explicit inverse.
    Replicated with `scipy.linalg.solve` against `np.eye(sizeH)`, not
    `np.linalg.inv`.
  - All site/atom indices in Parameters.txt are MATLAB 1-based; converted
    via `negf_common.orbital_offsets` to 0-based Python slice bounds.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.linalg as sla

from negf_common import parse_parameters, orbital_offsets, build_sumSig, load_H0


def compute_transmission(params: dict, H0: np.ndarray, eta: float = 0.0):
    orbitals = params["Orbitals"]
    size_h = int(orbitals.sum())
    if size_h != H0.shape[0]:
        raise ValueError(f"Norb ({size_h}) != size(H0) ({H0.shape[0]})")

    sumSig = build_sumSig(params)
    bounds = orbital_offsets(orbitals)

    Energy = params["Energy"]
    NE = Energy.size
    T = np.full(NE, -1.0, dtype=np.float64)

    I = np.eye(size_h, dtype=np.complex128)
    Lsite = params["Lsite"]
    Rsite = params["Rsite"]
    gammaL = params["gammaL"]
    gammaR = params["gammaR"]

    for nE in range(NE):
        E = Energy[nE]
        A = (E + 1j * eta) * np.eye(size_h) - H0 - sumSig
        Gr = sla.solve(A, I)
        Ga = Gr.conj().T

        Tmat = np.zeros((len(Lsite), len(Rsite)), dtype=np.float64)
        for ii, isite in enumerate(Lsite):
            li, hi_i = bounds(int(isite))
            Gammai = gammaL * np.eye(hi_i - li)
            for jj, jsite in enumerate(Rsite):
                lj, hi_j = bounds(int(jsite))
                Gammaj = gammaR * np.eye(hi_j - lj)
                block = Gammai @ Gr[li:hi_i, lj:hi_j] @ Gammaj @ Ga[lj:hi_j, li:hi_i]
                Tmat[ii, jj] = np.real(np.trace(block))

        T[nE] = Tmat.sum()

    return Energy, T


def process_run(strand: str, run_dir: Path, base_dir: Path = None):
    run_dir = Path(run_dir)
    base_dir = Path(base_dir) if base_dir else run_dir.parent

    params = parse_parameters(run_dir / "Parameters.txt")
    strand = params["strand"] or strand
    H0 = load_H0(strand, base_dir)

    eta = 0.0  # HARDCODED: settled project policy, matches
    # DNATransmission_Ballistic.m's `eta = 0;` override. Do NOT read this
    # from Parameters.txt or make it a CLI option.
    Energy, T = compute_transmission(params, H0, eta=eta)

    gL, gR = params["gammaL"], params["gammaR"]
    out_name = f"Tran_{strand}_gammaL_{gL:g}_gammaR_{gR:g}.mat"
    out_path = run_dir / out_name
    sio.savemat(out_path, {
        "Energy": Energy.reshape(1, -1),
        "T": T.reshape(1, -1),
    })
    print(f"Finished Ballistic Transmission! -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: transmission.py <strand_name> <run_dir> [base_dir]")
        sys.exit(1)
    strand_arg = sys.argv[1]
    run_dir_arg = Path(sys.argv[2])
    base_dir_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    process_run(strand_arg, run_dir_arg, base_dir_arg)
