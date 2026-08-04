#!/usr/bin/env python3
"""Export the regenerated pickles to one portable HDF5.

The resulting archive ships with the preprint, not before. Reads the PICKLES, not
the .mat files, so the archive is a pure
transformation of exactly what we train on and cannot silently disagree with our
results. Group path is /<sequence>/<run>.
"""
import pickle
from pathlib import Path

import h5py
import numpy as np

STR = h5py.string_dtype(encoding="utf-8")
ARRAYS = ("Egrid", "T", "DOS", "DOSAtom")


def export(pickle_dir: Path, out_path: Path, attrs: dict) -> int:
    pickle_dir, out_path = Path(pickle_dir), Path(out_path)
    n = 0
    with h5py.File(out_path, "w") as h:
        for k, v in attrs.items():
            h.attrs[k] = v
        for p in sorted(pickle_dir.glob("*_run*.pkl")):
            with open(p, "rb") as f:
                rec = pickle.load(f)
            run = p.stem.rsplit("_", 1)[1]
            g = h.create_group(f"/{rec['sequence']}/{run}")
            for key in ARRAYS:
                g.create_dataset(key, data=np.asarray(rec[key], dtype=np.float64),
                                 compression="gzip", compression_opts=4)
            g.create_dataset("gjf_text", data=rec["gjf_text"], dtype=STR)
            g.create_dataset("complementary_sequence",
                             data=rec["complementary_sequence"], dtype=STR)
            a = g.create_group("atoms")
            for key in ("element", "name", "resname"):
                a.create_dataset(key, data=rec["atoms"][key], dtype=STR)
            a.create_dataset("resseq", data=np.asarray(rec["atoms"]["resseq"], dtype=np.int32))
            a.create_dataset("xyz", data=rec["atoms"]["xyz"], compression="gzip")
            c = rec["contacts"]
            g.create_dataset("left_atoms", data=np.asarray(c["left_atoms"], dtype=np.int32))
            g.create_dataset("right_atoms", data=np.asarray(c["right_atoms"], dtype=np.int32))
            g.attrs["coupling_eV"] = c["coupling_eV"]
            g.attrs["contact_type"] = c["contact_type"]
            g.attrs["energy_reference_eV"] = rec["energy_reference_eV"]
            n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle_dir")
    ap.add_argument("out_path")
    a = ap.parse_args()
    attrs = {
        "units_energy": "eV",
        "units_xyz": "Angstrom",
        "energy_convention": (
            "Egrid is RAW/absolute. Each record's grid is centred on THAT sequence's "
            "HOMO, so energy_reference_eV differs per record. WARNING: the reference "
            "is a composition proxy (AT-only vs GC-only sequences differ by 0.813 eV, "
            "13.6 sigma, zero overlap), so comparing a fixed RELATIVE energy across "
            "sequences reintroduces a base-composition confound."),
        "dos_definition": "DOS = -(1/pi) Im Tr(G^r); bare, no 2e^2/h",
        "transmission_definition": "T = Tr(Gamma_L G^r Gamma_R G^a); bare Landauer trace, no 2e^2/h",
        "strand_identity": ("PDB chainID is blank in these structures (the builder does not set "
                            "it) and is therefore NOT exported. Strand identity comes from resseq: "
                            "for a duplex of L base pairs, residues 1..L are the primary strand "
                            "5'->3' and residues L+1..2L are the complementary strand, also "
                            "written 5'->3' and therefore antiparallel to the primary. This is why "
                            "complementary_sequence is the REVERSE complement of sequence."),
        "spin": ("Spin-restricted closed-shell Fock (alpha only). DOS and T are ONE "
                 "spin-degenerate channel; double for total-electron DOS or conductance."),
        "contact_model": ("Sigma_L,R = -i*Gamma/2 * I; wide-band limit, energy-independent, "
                          "purely imaginary, no real part and no work function. Applied to "
                          "EVERY atomic orbital of EVERY atom in the terminal base. "
                          "coupling_eV is used for both leads (gammaL == gammaR). "
                          "There is therefore no physical Fermi level in this model."),
        "orthogonalization": "H0 = S^-1/2 F S^-1/2 (Lowdin symmetric)",
        "atom_index_base": "contacts left_atoms/right_atoms are 1-BASED into atoms/*",
        "geometry": ("Idealized NAB fiber B-DNA template (dnabuilder); no MD, no "
                     "per-sequence relaxation. Geometry is IDENTICAL across sequences "
                     "except for base identity, so all transport variation is electronic, "
                     "not conformational."),
        "regime": "Coherent, ballistic, zero-bias only.",
        "run_map": "run1=(0.1,same) run2=(0.1,cross) run3=(0.6,same) run4=(0.6,cross)",
        "limitations": ("Fock/Overlap/H0 are NOT included, so DOS/T cannot be independently "
                        "reproduced from this archive alone."),
    }
    print(f"wrote {export(a.pickle_dir, a.out_path, attrs)} records to {a.out_path}")
