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


def parse_parameters_txt(param_path: Path):
    """
    Extract contact atom indices from Parameters.txt:
      - left atoms = Inject Site (atoms number)
      - right atoms = Extract Site (atoms number)

    Returns (left_atoms, right_atoms) as Python lists of ints.
    """
    text = param_path.read_text(errors="replace").splitlines()

    def read_int_block(header: str):
        if header not in text:
            raise ValueError(f"Header '{header}' not found in {param_path}")
        start = text.index(header) + 1
        vals = []
        for line in text[start:]:
            line = line.strip()
            if not line:
                break
            # Stop if we hit the next labeled section
            if any(ch.isalpha() for ch in line):
                break
            vals.append(int(float(line)))
        return vals

    left_atoms = read_int_block("Inject Site (atoms number)")
    right_atoms = read_int_block("Extract Site (atoms number)")
    return left_atoms, right_atoms


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


def main(seq_dir):
    seq_dir = Path(seq_dir)
    seq = seq_dir.name

    gjf_path = seq_dir / f"{seq}.gjf"
    if not gjf_path.exists():
        raise FileNotFoundError(f"Missing {gjf_path}")
    gjf_text = gjf_path.read_text(errors="replace")

    for run, meta in RUN_MAP.items():
        rdir = seq_dir / run
        if not rdir.exists():
            print(f"[WARN] {run} missing, skipping")
            continue

        params_path = rdir / "Parameters.txt"
        if not params_path.exists():
            raise FileNotFoundError(f"Missing {params_path}")

        left_atoms, right_atoms = parse_parameters_txt(params_path)

        # Load DOS and Transmission from MAT files, including energy grid
        dos_path = next(rdir.glob("DOS_*.mat"), None)
        tran_path = next(rdir.glob("Tran_*.mat"), None)
        if dos_path is None:
            raise FileNotFoundError(f"No DOS_*.mat found in {rdir}")
        if tran_path is None:
            raise FileNotFoundError(f"No Tran_*.mat found in {rdir}")

        E_dos, DOS_vals, ekey_dos, vkey_dos = load_energy_and_values(dos_path, "DOS")
        E_tran, T_vals, ekey_tran, vkey_tran = load_energy_and_values(tran_path, "T")

        # Prefer transmission energy grid as canonical, warn if mismatch
        if E_dos.size != E_tran.size or np.max(np.abs(E_dos - E_tran)) > 1e-9:
            print(f"[WARN] {run}: energy grids differ between {dos_path.name} and {tran_path.name}")

        Egrid = E_tran

        out = {
	    "sequence": seq,
            "gjf_text": gjf_text,
            "contacts": {
                "left_atoms": left_atoms,
                "right_atoms": right_atoms,
                "coupling_eV": meta["coupling_eV"],
                "contact_type": meta["contact_type"],
            },
            "Egrid": Egrid.tolist(),
            "T": T_vals.tolist(),
            "DOS": DOS_vals.tolist(),
        }

        out_path = seq_dir / f"{run}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(out, f)

        # Helpful debug print so you can confirm the correct MAT keys are being used
        print(f"[{run}] DOS mat: E={ekey_dos}, DOS={vkey_dos} | Tran mat: E={ekey_tran}, T={vkey_tran}")
        print(f"[{run}] wrote {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python convert_to_pickle.py <sequence_dir>")
        raise SystemExit(1)

    main(sys.argv[1])

