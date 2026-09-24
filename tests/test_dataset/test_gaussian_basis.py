import importlib.util
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
HDR = REPO / "tests/fixtures/gaussian_dump/atatatatatat_header.txt"


def _mod():
    spec = importlib.util.spec_from_file_location("gb", REPO / "DNADataset/gaussian_basis.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_counts_per_element():
    gb = _mod()
    for el, n in {"H": 5, "C": 15, "N": 15, "O": 15, "P": 19}.items():
        assert len(gb.derive_basis([el])["atom_index"]) == n


def test_unknown_element_raises():
    gb = _mod()
    with pytest.raises(ValueError, match="Na"):
        gb.derive_basis(["C", "Na"])


def test_matches_gaussian_dump_exactly():
    gb = _mod()
    d = gb.parse_dump_header(HDR)
    els = [gb.ELEMENT_OF_Z[z] for z in d["atomic_numbers"]]
    b = gb.derive_basis(els)
    assert len(b["atom_index"]) == d["nbasis"] == 8778
    # function type codes, row for row
    assert b["type_code"].tolist() == d["ibftyp"]
    # shells: type sequence and shell -> atom map
    shells = sorted(set(zip(b["shell_index"].tolist(), b["atom_index"].tolist())))
    assert [a + 1 for _, a in shells] == d["shell_to_atom"]
    st = [int(b["shell_type"][b["shell_index"] == s][0]) for s, _ in shells]
    assert st == d["shell_types"]
