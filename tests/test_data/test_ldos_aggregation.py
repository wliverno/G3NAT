import numpy as np
import pytest

from g3nat.data.ldos import is_backbone_atom, aggregate_by_residue


def test_backbone_classification_covers_every_fixture_atom_name():
    # The 51 distinct atom names present in tests/fixtures/dataset/aaac/aaac.pdb.
    backbone = [
        "C1'", "C2'", "C3'", "C4'", "C5'", "H1'", "H2'", "H2''", "H3'", "H4'",
        "H5'", "H5''", "HO3'", "HO5'", "O3'", "O4'", "O5'", "P", "OP1", "OP2",
    ]
    base = [
        "C2", "C4", "C5", "C6", "C7", "C8", "H1", "H2", "H21", "H22", "H3",
        "H41", "H42", "H5", "H6", "H61", "H62", "H71", "H72", "H73",
        "N1", "N2", "N3", "N4", "N6", "N7", "N9", "O2", "O4", "O6",
    ]
    assert len(backbone) + len(base) == 50  # 50 named here; H8 added below
    for name in backbone:
        assert is_backbone_atom(name) is True, name
    for name in base + ["H8"]:
        assert is_backbone_atom(name) is False, name


def test_terminal_hydroxyls_are_backbone_not_dropped():
    # HO5'/HO3' are sugar hydroxyls on 5'/3' termini. They must classify as
    # backbone, and must still be counted in the whole-residue aggregation.
    assert is_backbone_atom("HO5'") is True
    assert is_backbone_atom("HO3'") is True

    dosatom = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float64)
    resseq = [1, 1]
    names = ["N9", "HO5'"]

    both = aggregate_by_residue(dosatom, resseq)
    assert both.shape == (1, 2)
    np.testing.assert_allclose(both, [[11.0, 22.0]])

    base = aggregate_by_residue(dosatom, resseq, names, base_only=True)
    np.testing.assert_allclose(base, [[1.0, 2.0]])


def test_names_are_stripped_before_classification():
    # PDB columns 13-16 are fixed width, so names arrive space padded.
    assert is_backbone_atom(" P  ") is True
    assert is_backbone_atom(" C1'") is True
    assert is_backbone_atom(" N9 ") is False


def test_aggregate_sums_rows_grouped_by_resseq():
    dosatom = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ], dtype=np.float64)
    resseq = [1, 1, 2, 2]

    out = aggregate_by_residue(dosatom, resseq)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, [[4.0, 6.0], [12.0, 14.0]])


def test_row_order_is_ascending_resseq_not_first_appearance():
    dosatom = np.array([[5.0], [1.0]], dtype=np.float64)
    resseq = [2, 1]

    out = aggregate_by_residue(dosatom, resseq)

    # Row 0 must be residue 1 (value 1.0), because H index = resseq - 1.
    np.testing.assert_allclose(out, [[1.0], [5.0]])


def test_base_only_excludes_backbone_atoms():
    dosatom = np.array([
        [1.0],   # N9    base
        [10.0],  # P     backbone
        [100.0], # C1'   backbone
        [2.0],   # C8    base
    ], dtype=np.float64)
    resseq = [1, 1, 1, 1]
    names = ["N9", "P", "C1'", "C8"]

    residue = aggregate_by_residue(dosatom, resseq)
    base_only = aggregate_by_residue(dosatom, resseq, names, base_only=True)

    np.testing.assert_allclose(residue, [[113.0]])
    np.testing.assert_allclose(base_only, [[3.0]])


def test_base_only_without_names_raises():
    dosatom = np.array([[1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="atom_names"):
        aggregate_by_residue(dosatom, [1], base_only=True)


def test_row_count_mismatch_raises():
    dosatom = np.array([[1.0], [2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="rows"):
        aggregate_by_residue(dosatom, [1, 1, 1])


def test_atom_names_length_mismatch_raises():
    dosatom = np.array([[1.0], [2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="atom_names"):
        aggregate_by_residue(dosatom, [1, 2], ["N9"], base_only=True)
