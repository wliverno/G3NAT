"""Shared helpers for the Python ports of DNATransmission_Ballistic.m and
DOS_calc.m: Parameters.txt parsing and wide-band self-energy assembly.

Both MATLAB scripts share an (almost) identical parameter-parsing preamble
and identical `sumSig` construction; factored out here rather than
duplicated so the two match exactly.
"""
from pathlib import Path

import numpy as np


def parse_parameters(params_path: Path):
    """Parse Parameters.txt into the same fields readMAT.m's downstream
    scripts use. Returns a dict with keys: strand, Orbitals (1D int array,
    1 entry per atom), Energy (1D float array), Lsite, Rsite (1D int arrays,
    1-based atom indices), gammaL, gammaR (floats), Dsites (1D int array),
    eta (float), bprobe (float).

    Section layout of Parameters.txt (fixed, matches the MATLAB parser):
        <strand name>
        (blank)
        Orbitals set
        <values...>
        (blank)
        Energy Range
        <values...>
        (blank)
        Inject Site (atoms number)
        <values...>
        (blank)
        Extract Site (atoms number)
        <values...>
        (blank)
        GammaL
        <value>
        (blank)
        GammaR
        <value>
        (blank)
        Probes Site (atoms number)
        <values, possibly empty>
        (blank)
        Broadening (for DOS)
        <value>
        (blank)
        Probe (for Decoh)
        <value>
    """
    lines = Path(params_path).read_text(errors="replace").splitlines()
    strand = lines[0].strip()

    def _section(header):
        idx = lines.index(header)
        vals = []
        for line in lines[idx + 1:]:
            s = line.strip()
            if s == header or (s and any(c.isalpha() for c in s)):
                break
            if s:
                vals.append(s)
        return vals

    orbitals = np.array([int(float(x)) for x in _section("Orbitals set")])
    energy = np.array([float(x) for x in _section("Energy Range")])
    lsite = np.array([int(float(x)) for x in _section("Inject Site (atoms number)")])
    rsite = np.array([int(float(x)) for x in _section("Extract Site (atoms number)")])
    gammaL = float(_section("GammaL")[0])
    gammaR = float(_section("GammaR")[0])
    dsites_raw = _section("Probes Site (atoms number)")
    dsites = np.array([int(float(x)) for x in dsites_raw]) if dsites_raw else np.array([], dtype=int)
    eta = float(_section("Broadening (for DOS)")[0])
    bprobe = float(_section("Probe (for Decoh)")[0])

    return {
        "strand": strand, "Orbitals": orbitals, "Energy": energy,
        "Lsite": lsite, "Rsite": rsite, "gammaL": gammaL, "gammaR": gammaR,
        "Dsites": dsites, "eta": eta, "bprobe": bprobe,
    }


def orbital_offsets(orbitals: np.ndarray):
    """1-based atom index -> (start, end) 0-based Python slice bounds into
    the orbital-space Hamiltonian, i.e. atom `isite` (1-based) occupies
    H0[start:end, start:end]. Matches MATLAB's
    TempLen1 = sum(Orbitals(1:isite)) - Orbitals(isite)
    TempLen2 = sum(Orbitals(1:isite))
    then slicing H(TempLen1+1:TempLen2, ...).
    """
    cum = np.concatenate(([0], np.cumsum(orbitals)))

    def bounds(isite_1based: int):
        return int(cum[isite_1based - 1]), int(cum[isite_1based])

    return bounds


def build_sumSig(params: dict) -> np.ndarray:
    """Wide-band self-energy sumSig = -1i * sumSig / 2, matching both
    DNATransmission_Ballistic.m and DOS_calc.m exactly (eta is NOT folded
    in here; it only enters the (E + 1i*eta) term at the Green's-function
    step).
    """
    orbitals = params["Orbitals"]
    size_h = int(orbitals.sum())
    bounds = orbital_offsets(orbitals)

    sumSig = np.zeros((size_h, size_h), dtype=np.float64)
    sites = np.concatenate([params["Lsite"], params["Rsite"]])
    gamma = np.concatenate([
        np.full(len(params["Lsite"]), params["gammaL"]),
        np.full(len(params["Rsite"]), params["gammaR"]),
    ])
    for isite, g in zip(sites, gamma):
        lo, hi = bounds(int(isite))
        sumSig[lo:hi, lo:hi] = g * np.eye(hi - lo)

    return -1j * sumSig / 2.0


def load_H0(strand: str, work_dir: Path) -> np.ndarray:
    """Load the orthogonalized Hamiltonian for `strand`. Looks for
    `<strand>_H0.mat` (written by readmat_parse.process_strand) first, then
    falls back to `<strand>.mat` (the MATLAB in-place-overwrite layout) for
    compatibility with directories produced by the original MATLAB
    pipeline.
    """
    import scipy.io as sio

    work_dir = Path(work_dir)
    h0_path = work_dir / f"{strand}_H0.mat"
    legacy_path = work_dir / f"{strand}.mat"
    if h0_path.exists():
        d = sio.loadmat(h0_path)
        return np.asarray(d[strand], dtype=np.float64)
    if legacy_path.exists():
        d = sio.loadmat(legacy_path)
        return np.asarray(d[strand], dtype=np.float64)
    raise FileNotFoundError(f"Neither {h0_path} nor {legacy_path} found")
