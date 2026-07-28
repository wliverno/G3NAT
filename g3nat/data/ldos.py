"""Per-residue aggregation of atom-resolved LDOS.

The DFT output carries DOSAtom with shape [n_atoms, n_energy]. The tight-binding
model has one site per nucleobase. This module collapses the former onto the
latter using the PDB resseq column.

Verified over all 2077 records of pickle_files_v2 on 2026-07-26: resseq runs
contiguously 1..2L, residues 1..L match the primary sequence and L+1..2L the
complementary one, and the atom table is grouped by residue in file order.
See docs/dataset.md, section "Properties verified across the full set", for
the full per-check results. Combined with the graph node layout, this makes
H index = resseq - 1.

Units are LINEAR throughout. The log10 transform belongs to the caller, so that
it happens in the same place as the existing DOS and transmission transforms.
"""
import numpy as np

# Phosphate atoms. Every other backbone atom carries a prime (sugar), including
# the terminal hydroxyls HO5' and HO3'.
BACKBONE_EXACT = frozenset({"P", "OP1", "OP2"})


def is_backbone_atom(name: str) -> bool:
    """True for deoxyribose and phosphate atoms, False for nucleobase atoms.

    Primed names are the sugar; P/OP1/OP2 are the phosphate. Note that O4'
    (sugar) and O4 (thymine carbonyl) differ only by the prime.
    """
    stripped = name.strip()
    return ("'" in stripped) or (stripped in BACKBONE_EXACT)


def aggregate_by_residue(dosatom, resseq, atom_names=None, base_only=False):
    """Sum atom-resolved LDOS into per-residue LDOS.

    Args:
        dosatom: array [n_atoms, n_energy], linear units.
        resseq: length n_atoms, the PDB residue number of each atom.
        atom_names: length n_atoms PDB atom names. Required when base_only.
        base_only: when True, exclude sugar and phosphate atoms.

    Returns:
        array [n_residues, n_energy], float64, linear units. Row i corresponds
        to the i-th residue in ASCENDING resseq order, which is Hamiltonian
        index i.
    """
    dosatom = np.asarray(dosatom, dtype=np.float64)
    resseq = np.asarray(resseq)

    if dosatom.ndim != 2:
        raise ValueError(f"dosatom must be 2-D [n_atoms, n_energy], got shape {dosatom.shape}")
    if dosatom.shape[0] != resseq.shape[0]:
        raise ValueError(
            f"dosatom has {dosatom.shape[0]} rows but resseq has {resseq.shape[0]} entries"
        )

    if base_only:
        if atom_names is None:
            raise ValueError("atom_names is required when base_only=True")
        if len(atom_names) != dosatom.shape[0]:
            raise ValueError(
                f"atom_names has {len(atom_names)} entries but dosatom has "
                f"{dosatom.shape[0]} rows"
            )
        keep = np.fromiter(
            (not is_backbone_atom(n) for n in atom_names), dtype=bool, count=len(atom_names)
        )
    else:
        keep = np.ones(dosatom.shape[0], dtype=bool)

    residues = np.unique(resseq)  # np.unique returns sorted values
    out = np.zeros((residues.shape[0], dosatom.shape[1]), dtype=np.float64)
    for i, residue in enumerate(residues):
        mask = (resseq == residue) & keep
        out[i] = dosatom[mask].sum(axis=0)
    return out
