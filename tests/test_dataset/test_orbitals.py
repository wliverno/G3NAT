import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/dataset/aaac"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "convert_to_pickle", REPO / "DNADataset/convert_to_pickle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_orbitals_set_matches_atom_count_and_known_values():
    m = _load_module()
    orbs = m.parse_orbitals_set(FIXTURE / "run1/Parameters.txt")
    atoms = m.parse_pdb_atoms(FIXTURE / "aaac.pdb")
    assert len(orbs) == len(atoms["element"]) == 253
    # 6-31G(d,p): H -> 5, heavy atoms (C/N/O) -> 15. First six from the file.
    assert orbs[:6] == [5, 15, 15, 5, 5, 15]
    assert all(o > 0 for o in orbs)


def test_orbitals_consistent_with_element_identity():
    """H gets 5 basis functions, C/N/O get 15 under 6-31G(d,p)."""
    m = _load_module()
    orbs = m.parse_orbitals_set(FIXTURE / "run1/Parameters.txt")
    atoms = m.parse_pdb_atoms(FIXTURE / "aaac.pdb")
    for el, o in zip(atoms["element"], orbs):
        if el == "H":
            assert o == 5, f"H should have 5 orbitals, got {o}"
        elif el in ("C", "N", "O"):
            assert o == 15, f"{el} should have 15 orbitals, got {o}"
