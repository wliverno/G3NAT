import pickle

import numpy as np
import pytest

from g3nat.data.pickle import load_single_pickle


def _write_record(tmp_path, with_dosatom=True):
    """A minimal 1-base-pair record, enough to exercise the loader."""
    n_e = 3
    record = {
        'sequence': 'a',
        'complementary_sequence': 't',
        'Egrid': np.array([-6.0, -5.0, -4.0]),
        'DOS': np.array([10.0, 100.0, 10.0]),
        'T': np.array([0.1, 1.0, 0.1]),
        'contacts': {'contact_type': 'cross', 'coupling_eV': 0.6},
    }
    if with_dosatom:
        # 3 atoms: residue 1 has a base atom and a backbone atom,
        # residue 2 has one base atom. Columns sum to DOS.
        record['DOSAtom'] = np.array([
            [1.0, 10.0, 1.0],   # N9  residue 1  base
            [2.0, 20.0, 2.0],   # P   residue 1  backbone
            [7.0, 70.0, 7.0],   # N1  residue 2  base
        ])
        record['atoms'] = {
            'resseq': [1, 1, 2],
            'name': ['N9', 'P', 'N1'],
        }
    path = tmp_path / "rec.pkl"
    with open(path, 'wb') as f:
        pickle.dump(record, f)
    return path


def test_loader_returns_both_aggregations_in_log10(tmp_path):
    path = _write_record(tmp_path, with_dosatom=True)

    out = load_single_pickle(str(path))

    assert out is not None
    assert out['ldos_residue'].shape == (2, 3)
    assert out['ldos_base_only'].shape == (2, 3)

    # residue 1 whole = 1+2 = 3; residue 2 whole = 7
    np.testing.assert_allclose(
        out['ldos_residue'], np.log10([[3.0, 30.0, 3.0], [7.0, 70.0, 7.0]])
    )
    # residue 1 base-only = 1 (P excluded); residue 2 base-only = 7
    np.testing.assert_allclose(
        out['ldos_base_only'], np.log10([[1.0, 10.0, 1.0], [7.0, 70.0, 7.0]])
    )


def test_residue_aggregation_reproduces_total_dos(tmp_path):
    # The invariant measured at 3.6e-15 over all 2077 v2 records, as a guard.
    path = _write_record(tmp_path, with_dosatom=True)

    out = load_single_pickle(str(path))

    summed = np.power(10.0, out['ldos_residue']).sum(axis=0)
    np.testing.assert_allclose(summed, np.power(10.0, out['dos']), rtol=1e-12)


def test_v1_record_without_dosatom_loads_with_none(tmp_path):
    # pickle_files/ has no DOSAtom. It must keep loading, not raise.
    path = _write_record(tmp_path, with_dosatom=False)

    out = load_single_pickle(str(path))

    assert out is not None
    assert out['ldos_residue'] is None
    assert out['ldos_base_only'] is None
    assert out['dos'] is not None


def test_nonpositive_aggregate_raises_rather_than_producing_nan(tmp_path):
    # log10 of a non-positive value yields -inf or nan silently. The measured
    # data is strictly positive (min 1.76e-10 over 4.85M values), so a
    # non-positive aggregate means something is wrong upstream. Fail loudly.
    record = {
        'sequence': 'a',
        'complementary_sequence': 't',
        'Egrid': np.array([-6.0, -5.0]),
        'DOS': np.array([1.0, 1.0]),
        'T': np.array([1.0, 1.0]),
        'contacts': {'contact_type': 'same', 'coupling_eV': 0.1},
        'DOSAtom': np.array([[0.0, 1.0], [1.0, 1.0]]),
        'atoms': {'resseq': [1, 2], 'name': ['N9', 'N1']},
    }
    path = tmp_path / "bad.pkl"
    with open(path, 'wb') as f:
        pickle.dump(record, f)

    with pytest.raises(ValueError, match="non-positive"):
        load_single_pickle(str(path))


def test_stored_complementary_sequence_is_uppercased(tmp_path):
    # v2 records store complementary_sequence in lowercase. BASE_FEATURES in
    # g3nat/graph/construction.py is keyed 'A'/'T'/'G'/'C' and that module does
    # no case normalisation, so a lowercase value reaching graph construction
    # raises KeyError. The fallback branch (no stored complementary_sequence)
    # is uppercase by accident, built from the already-uppercased primary
    # sequence -- which is why v1 data never hit this. _write_record's
    # 'complementary_sequence': 't' is deliberately lowercase, matching the
    # real v2 record shape; do not "fix" the fixture to uppercase or this
    # test stops proving anything.
    path = _write_record(tmp_path, with_dosatom=False)

    out = load_single_pickle(str(path))

    assert out['sequence'] == 'A'
    assert out['complementary_sequence'] == 'T'
