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

RUN_MAP = {"run1": (0.1, "same"), "run2": (0.1, "cross"),
           "run3": (0.6, "same"), "run4": (0.6, "cross")}


def check_contacts(rec, run, where):
    """Each contact must be exactly one whole terminal residue; coupling/type must match RUN_MAP.

    same: residue 1 -> residue L; cross: residue 1 -> residue L+1. Returns (left, right) resseq.
    """
    rs = list(rec["atoms"]["resseq"])
    c = rec["contacts"]
    L = len(rec["sequence"])
    if any(ch != " " for ch in rec["atoms"]["chain"]):
        raise ValueError(f"{where}: atoms.chain is not all blank; the importer restores it as blank")
    if run not in RUN_MAP:
        raise ValueError(f"{where}: run {run!r} not in run map {sorted(RUN_MAP)}")
    coupling, ctype = RUN_MAP[run]
    if abs(c["coupling_eV"] - coupling) > 1e-12 or c["contact_type"] != ctype:
        raise ValueError(f"{where}: coupling/type ({c['coupling_eV']}, {c['contact_type']}) "
                         f"disagree with run map {run}={RUN_MAP[run]}")
    out = []
    for side, want in (("left_atoms", 1), ("right_atoms", L if ctype == "same" else L + 1)):
        idx = list(c[side])
        res = {rs[i - 1] for i in idx}
        whole = sorted(i + 1 for i, r in enumerate(rs) if r == want)
        if res != {want} or sorted(idx) != whole:
            raise ValueError(f"{where}: {side} is not exactly the whole residue {want} "
                             f"(got residues {sorted(res)})")
        out.append(want)
    return tuple(out)


def _write_record(h, p, split):
    with open(p, "rb") as f:
        rec = pickle.load(f)
    run = p.stem.rsplit("_", 1)[1]
    left_res, right_res = check_contacts(rec, run, p.name)
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
    g.attrs["split"] = split
    g.attrs["left_residue"] = left_res
    g.attrs["right_residue"] = right_res


def export(pickle_dir: Path, out_path: Path, attrs: dict, heldout_dirs=()) -> int:
    out_path = Path(out_path)
    sources = [(Path(pickle_dir), "train")] + [(Path(d), "heldout") for d in heldout_dirs]
    n = 0
    with h5py.File(out_path, "w") as h:
        for k, v in attrs.items():
            h.attrs[k] = v
        for d, split in sources:
            for p in sorted(d.glob("*_run*.pkl")):
                _write_record(h, p, split)
                n += 1
    return n


ROOT_ATTRS = {
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
                      "EVERY atomic orbital of EVERY atom in the terminal RESIDUE "
                      "(the full nucleotide: base, sugar and phosphate). "
                      "coupling_eV is used for both leads (gammaL == gammaR). "
                      "There is therefore no physical Fermi level in this model."),
    "level_of_theory": ("B3LYP/6-31G(d,p) in implicit water (SCRF continuum, "
                        "scrf=(solvent=water)); single-point on the idealized "
                        "geometry, no optimization."),
    "charge_and_multiplicity": ("Net charge -2(L-1) for an L-bp duplex (every "
                                "internal phosphodiester phosphate deprotonated, "
                                "free 5'/3'-OH termini, no counterions); "
                                "closed-shell singlet."),
    "orthogonalization": "H0 = S^-1/2 F S^-1/2 (Lowdin symmetric)",
    "atom_index_base": "contacts left_atoms/right_atoms are 1-BASED into atoms/*",
    "geometry": ("Idealized NAB fiber B-DNA template (dnabuilder); no MD, no "
                 "per-sequence relaxation. Geometry varies only through base identity "
                 "(e.g. twist SD 1.01 deg, rise SD 0.005 A over the training set), so "
                 "conformational and electronic effects are not separable in this dataset."),
    "regime": "Coherent, ballistic, zero-bias only.",
    "run_map": ("run1=(0.1 eV, same) run2=(0.1 eV, cross) run3=(0.6 eV, same) "
                "run4=(0.6 eV, cross). same: left contact = residue 1 (primary strand 5' end), "
                "right contact = residue L (primary strand 3' end). cross: left = residue 1, "
                "right = residue L+1 (complementary strand 5' end). Each run group also stores "
                "left_residue/right_residue (resseq) and left_atoms/right_atoms (1-based)."),
    "limitations": ("Fock and overlap matrices, with a per-row orbital map, are in the companion "
                    "file matrices.h5 (same sequence keys); H0 = S^-1/2 F S^-1/2 is not stored. "
                    "A contact's matrix rows are all rows whose basis atom lies in its residue."),
    "split": ("Each run group has attr split='train' (lengths 4-8, the training set) or "
              "'heldout' (lengths 12 and 16, never used in training)."),
    "license": "CC-BY-4.0",
}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle_dir")
    ap.add_argument("out_path")
    ap.add_argument("--heldout", nargs="*", default=[])
    a = ap.parse_args()
    n = export(a.pickle_dir, a.out_path, ROOT_ATTRS, heldout_dirs=a.heldout)
    print(f"wrote {n} records to {a.out_path}")
