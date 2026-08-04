#!/usr/bin/env python3
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# Canonical run definitions (matches combined_script.slurm)
RUN_MAP = {
    "run1": {"coupling_eV": 0.1, "contact_type": "same"},
    "run2": {"coupling_eV": 0.1, "contact_type": "cross"},
    "run3": {"coupling_eV": 0.6, "contact_type": "same"},
    "run4": {"coupling_eV": 0.6, "contact_type": "cross"},
}


def _read_int_block(lines, header: str, src):
    if header not in lines:
        raise ValueError(f"Header '{header}' not found in {src}")
    vals = []
    for line in lines[lines.index(header) + 1:]:
        line = line.strip()
        if not line or any(ch.isalpha() for ch in line):
            break
        vals.append(int(float(line)))
    return vals


def parse_parameters_txt(param_path: Path):
    """
    Extract contact atom indices from Parameters.txt:
      - left atoms = Inject Site (atoms number)
      - right atoms = Extract Site (atoms number)

    Returns (left_atoms, right_atoms) as Python lists of ints.
    """
    lines = param_path.read_text(errors="replace").splitlines()
    left_atoms = _read_int_block(lines, "Inject Site (atoms number)", param_path)
    right_atoms = _read_int_block(lines, "Extract Site (atoms number)", param_path)
    return left_atoms, right_atoms


def parse_orbitals_set(param_path: Path):
    """Per-atom basis-function counts, same order as the atom table.

    Written by TransportSetup.py from a hardcoded orbital_map that assumes
    B3LYP/6-31G(d,p) Cartesian polarization functions. Parsing it here lets the
    AO-to-atom boundaries be checked against the atom table rather than trusted.
    """
    lines = param_path.read_text(errors="replace").splitlines()
    return _read_int_block(lines, "Orbitals set", param_path)


def parse_pdb_atoms(pdb_path: Path):
    """Atom table in PDB file order, which is also the DOSAtom row order.

    Fixed-column PDB format (cols are 1-based in the spec, 0-based here):
      13-16 name, 18-20 resName, 22 chainID, 23-26 resSeq,
      31-38 x, 39-46 y, 47-54 z, 77-78 element

    NOTE: chainID is BLANK in these files (dnabuilder does not set it), so strand
    identity comes from resSeq ordering -- residues 1..L are the primary strand and
    L+1..2L the complementary. Do not rely on `chain` to separate strands.
    """
    element, name, resseq, resname, chain, xyz = [], [], [], [], [], []
    for line in pdb_path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name.append(line[12:16].strip())
        resname.append(line[17:20].strip())
        chain.append(line[21:22])
        resseq.append(int(line[22:26]))
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        el = line[76:78].strip()
        element.append(el if el else line[12:16].strip()[0])
    return {
        "element": element, "name": name, "resseq": resseq,
        "resname": resname, "chain": chain,
        "xyz": np.asarray(xyz, dtype=np.float64),
    }


def _mat_vars(mat_path: Path):
    d = loadmat(mat_path)
    return {k: np.asarray(v).squeeze() for k, v in d.items() if not k.startswith("__")}


def _pick_energy_key(vars_dict):
    # Common energy variable names
    for k in ["Egrid", "EGrid", "E_list", "Elist", "E", "energy", "Energy", "energies"]:
        if k in vars_dict:
            v = vars_dict[k]
            if v.ndim == 1 and v.size > 1 and np.issubdtype(v.dtype, np.number):
                return k

    # Fallback: pick a numeric 1D vector that looks monotonic
    candidates = []
    for k, v in vars_dict.items():
        if v.ndim == 1 and v.size > 10 and np.issubdtype(v.dtype, np.number):
            candidates.append(k)

    if not candidates:
        raise ValueError("No obvious energy vector found in mat file.")

    for k in candidates:
        v = vars_dict[k].astype(float)
        dv = np.diff(v)
        if np.all(dv >= 0) or np.all(dv <= 0):
            return k

    return candidates[0]


def _pick_value_key(vars_dict, energy_key, kind):
    # kind is "T" or "DOS"
    preferred = {
        "T": ["T", "Tran", "Trans", "Transmission", "transmission", "tran", "t"],
        "DOS": ["DOS", "dos", "DoS", "density", "density_of_states"],
    }[kind]

    for k in preferred:
        if k in vars_dict and k != energy_key:
            v = vars_dict[k]
            if v.ndim == 1 and np.issubdtype(v.dtype, np.number):
                return k

    # Fallback: choose another 1D numeric vector matching energy length
    E = vars_dict[energy_key]
    for k, v in vars_dict.items():
        if k == energy_key:
            continue
        if v.ndim == 1 and v.size == E.size and np.issubdtype(v.dtype, np.number):
            return k

    raise ValueError(f"Could not find {kind} vector matching energy length in mat file.")


def load_energy_and_values(mat_path: Path, kind: str):
    """
    Returns (E, V, energy_key, value_key)
    """
    vars_dict = _mat_vars(mat_path)
    ekey = _pick_energy_key(vars_dict)
    vkey = _pick_value_key(vars_dict, ekey, kind)

    E = vars_dict[ekey].astype(float).squeeze()
    V = vars_dict[vkey].astype(float).squeeze()

    if E.ndim != 1 or V.ndim != 1:
        raise ValueError(f"{mat_path} did not produce 1D vectors for E and {kind}")

    return E, V, ekey, vkey


def load_dosatom(mat_path: Path, n_energy: int):
    """Atom-resolved DOS, [n_atoms, n_energy], float64.

    Requires the key by NAME. The heuristic fallbacks in _pick_value_key exist for
    the 1-D DOS/T vectors, whose names vary across older runs; applying that kind of
    guessing to a 2-D array would silently return whatever else happened to be there.
    """
    raw = loadmat(mat_path)
    if "DOSAtom" not in raw:
        raise KeyError(f"DOSAtom not found in {mat_path}; keys: "
                       f"{sorted(k for k in raw if not k.startswith('__'))}")
    da = np.asarray(raw["DOSAtom"], dtype=np.float64)
    if da.ndim != 2:
        raise ValueError(f"DOSAtom in {mat_path} is {da.ndim}-D, expected 2-D")
    # MATLAB may store either orientation; normalise to (atoms, energy).
    # Degenerate case n_atoms == n_energy cannot be disambiguated by shape alone.
    # Measured across every PDB in the source dataset: atom counts range 250-510
    # and 0 sequences have exactly 201 atoms (the n_energy value seen in this
    # dataset's runs), so the ambiguity does not occur in practice here.
    if da.shape[0] == n_energy and da.shape[1] != n_energy:
        da = da.T
    if da.shape[1] != n_energy:
        raise ValueError(f"DOSAtom {da.shape} has no axis matching n_energy={n_energy}")
    return da


# Measured across 55 sequences / 22781 atoms of the source dataset: this mapping is
# one-to-one with no exceptions. B3LYP/6-31G(d,p) Cartesian polarization functions.
ELEMENT_ORBITALS = {"H": 5, "C": 15, "N": 15, "O": 15, "P": 19}


COMPLEMENT = {"a": "t", "t": "a", "g": "c", "c": "g"}


def reverse_complement(seq: str) -> str:
    """Complementary strand in PDB FILE order, which is the reverse complement.

    Residues 1..L of the PDB are the primary strand 5'->3'; residues L+1..2L are the
    complementary strand, also written 5'->3', which means they run antiparallel to
    the primary. So for 'aaac' the file's residues 5..8 are DG DT DT DT -> 'gttt',
    NOT the position-wise complement 'tttg'. Verified against
    tests/fixtures/dataset/aaac/aaac.pdb, and this matches the fallback already in
    g3nat/data/pickle.py:41 (`...[::-1]`), so old and new records agree.
    """
    return "".join(COMPLEMENT[c] for c in seq.lower())[::-1]


def build_record(seq_dir: Path, run: str, meta: dict):
    """Full schema for one (sequence, run), or None if it cannot be built."""
    seq = seq_dir.name
    rdir = seq_dir / run
    pdb_path = seq_dir / f"{seq}.pdb"
    gjf_path = seq_dir / f"{seq}.gjf"
    params_path = rdir / "Parameters.txt"
    dos_path = next(rdir.glob("DOS_*.mat"), None)
    tran_path = next(rdir.glob("Tran_*.mat"), None)

    for p in (pdb_path, gjf_path, params_path):
        if not p.exists():
            print(f"[DROP] {seq}/{run}: missing {p.name}")
            return None
    if dos_path is None or tran_path is None:
        print(f"[DROP] {seq}/{run}: missing DOS_*.mat or Tran_*.mat")
        return None

    left_atoms, right_atoms = parse_parameters_txt(params_path)
    E_dos, DOS_vals, _, _ = load_energy_and_values(dos_path, "DOS")
    E_tran, T_vals, _, _ = load_energy_and_values(tran_path, "T")
    if E_dos.size != E_tran.size or np.max(np.abs(E_dos - E_tran)) > 1e-9:
        print(f"[DROP] {seq}/{run}: DOS and Tran energy grids differ")
        return None

    Egrid = np.asarray(E_tran, dtype=np.float64)
    atoms = parse_pdb_atoms(pdb_path)

    return {
        "sequence": seq,
        "complementary_sequence": reverse_complement(seq),
        "gjf_text": gjf_path.read_text(errors="replace"),
        "contacts": {
            "left_atoms": left_atoms, "right_atoms": right_atoms,
            "coupling_eV": meta["coupling_eV"], "contact_type": meta["contact_type"],
        },
        "Egrid": Egrid,
        "T": np.asarray(T_vals, dtype=np.float64),
        "DOS": np.asarray(DOS_vals, dtype=np.float64),
        "DOSAtom": load_dosatom(dos_path, n_energy=Egrid.size),
        "atoms": atoms,
        "energy_reference_eV": float(Egrid.mean()),
    }


def validate_record(rec, n_orbitals):
    """Failure reasons; empty list means valid. The writer refuses to emit a
    record that fails, rather than producing subtly wrong files at scale.

    n_orbitals: per-atom basis-function counts, same order as rec["atoms"], from
    parse_orbitals_set(). REQUIRED -- not stored in the record itself (DOSAtom is
    already summed per atom, and the archive has no Fock/Overlap/H0 for
    AO-space data to be indexed against, so the counts have no consumer once
    validated), but they are the only thing that catches a permuted atom table, so
    they must still be checked every time. No default is given on purpose: an
    omitted argument must raise TypeError, not silently skip this check.
    """
    reasons = []
    n_atoms = len(rec["atoms"]["element"])
    n_E = rec["Egrid"].size

    if rec["DOSAtom"].shape != (n_atoms, n_E):
        reasons.append(f"DOSAtom {rec['DOSAtom'].shape} != ({n_atoms}, {n_E})")
    if not (rec["T"].size == rec["DOS"].size == n_E):
        reasons.append("Egrid/T/DOS length mismatch")
    if rec["DOSAtom"].shape == (n_atoms, n_E) and \
            not np.allclose(rec["DOSAtom"].sum(axis=0), rec["DOS"], rtol=1e-6, atol=0):
        reasons.append("DOSAtom does not sum to DOS")
    for key in ("T", "DOS", "DOSAtom"):
        if np.any(rec[key] < 0):
            reasons.append(f"{key} contains negative values (-1 checkpoint sentinel?)")
    for side in ("left_atoms", "right_atoms"):
        idx = rec["contacts"][side]
        if not idx:
            reasons.append(f"{side} is empty (no contact atoms)")
        elif min(idx) < 1 or max(idx) > n_atoms:
            reasons.append(f"{side} out of range 1..{n_atoms}")

    resseq = rec["atoms"].get("resseq")
    if resseq is not None and len(resseq) > 1:
        # Measured across 55 sequences: resseq is non-decreasing in file order in
        # every case, so this has zero false positives on real data. It catches any
        # cross-residue reordering, the realistic parser-bug class.
        if any(resseq[i + 1] < resseq[i] for i in range(len(resseq) - 1)):
            reasons.append("atom table is not grouped by residue in file order")

    orbs = n_orbitals
    if orbs is None or len(orbs) != n_atoms:
        reasons.append("n_orbitals argument missing or wrong length")
    else:
        # Catches a permuted atom table: orbital counts come from Parameters.txt in
        # file order, so they only line up with elements if the order is intact.
        # LIMITATION: C, N, and O all map to 15 orbitals, so a permutation confined
        # to swapping only C/N/O atoms among themselves is invisible to this check
        # in principle, not just in this implementation -- it does not fully verify
        # the atom mapping on its own. The resseq monotonicity check above provides
        # a partial, independent mitigation for that gap.
        for el, o in zip(rec["atoms"]["element"], orbs):
            expected = ELEMENT_ORBITALS.get(el)
            if expected is None:
                reasons.append(f"unknown element '{el}' not in ELEMENT_ORBITALS map")
                break
            if o != expected:
                reasons.append("atom order inconsistent with orbital counts")
                break
    return reasons


def main(seq_dir, out_dir=None):
    seq_dir = Path(seq_dir)
    out_dir = Path(out_dir) if out_dir else seq_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for run, meta in RUN_MAP.items():
        if not (seq_dir / run).exists():
            print(f"[DROP] {seq_dir.name}/{run}: run dir missing")
            continue
        try:
            rec = build_record(seq_dir, run, meta)
            orbs = (parse_orbitals_set(seq_dir / run / "Parameters.txt")
                    if rec is not None else None)
        except Exception as exc:                      # noqa: BLE001
            print(f"[DROP] {seq_dir.name}/{run}: {type(exc).__name__}: {exc}")
            continue
        if rec is None:
            continue
        reasons = validate_record(rec, orbs)
        if reasons:
            print(f"[DROP] {seq_dir.name}/{run}: {'; '.join(reasons)}")
            continue
        name = (f"{seq_dir.name}_{run}.pkl" if out_dir != seq_dir else f"{run}.pkl")
        with open(out_dir / name, "wb") as f:
            pickle.dump(rec, f)
        written += 1
        print(f"[OK] wrote {out_dir / name}")
    print(f"[SUMMARY] {seq_dir.name}: {written}/{len(RUN_MAP)} records written")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("--out-dir", default=None,
                    help="write <seq>_<run>.pkl here instead of <run>.pkl in seq_dir")
    a = ap.parse_args()
    main(a.seq_dir, a.out_dir)

