import importlib.util
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/dataset/aaac"


def _load_module():
    """DNADataset/ is not a package; load the script by path."""
    spec = importlib.util.spec_from_file_location(
        "convert_to_pickle", REPO / "DNADataset/convert_to_pickle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_pdb_atoms_shapes_and_order():
    m = _load_module()
    atoms = m.parse_pdb_atoms(FIXTURE / "aaac.pdb")
    n = len(atoms["element"])
    assert n == 253, f"aaac has 253 atoms, got {n}"
    for key in ("name", "resseq", "resname", "chain"):
        assert len(atoms[key]) == n
    assert atoms["xyz"].shape == (n, 3)
    assert atoms["xyz"].dtype == np.float64
    # First two atoms of aaac.pdb, verified against the file
    assert atoms["name"][0] == "HO5'" and atoms["element"][0] == "H"
    assert atoms["name"][1] == "O5'" and atoms["element"][1] == "O"
    assert np.allclose(atoms["xyz"][1], [0.427, -7.826, -1.788])


def test_residues_are_contiguous_and_cover_both_strands():
    m = _load_module()
    atoms = m.parse_pdb_atoms(FIXTURE / "aaac.pdb")
    res = sorted(set(atoms["resseq"]))
    # aaac is a 4-mer duplex -> 8 residues, numbered 1..8, primary then complementary
    assert res == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [atoms["resname"][atoms["resseq"].index(r)] for r in res] == \
        ["DA", "DA", "DA", "DC", "DG", "DT", "DT", "DT"]
