import pickle

import numpy as np
import pytest

from g3nat.data.pickle import load_pickle_directory


def _record(seq, comp, with_dosatom):
    rec = {
        'sequence': seq,
        'complementary_sequence': comp,
        'Egrid': np.array([-6.0, -5.0]),
        'DOS': np.array([3.0, 3.0]),
        'T': np.array([1.0, 1.0]),
        'contacts': {'contact_type': 'same', 'coupling_eV': 0.1},
    }
    if with_dosatom:
        rec['DOSAtom'] = np.array([[1.0, 1.0], [2.0, 2.0]])
        rec['atoms'] = {'resseq': [1, 2], 'name': ['N9', 'N1']}
    return rec


def _write(tmp_path, name, rec):
    with open(tmp_path / name, 'wb') as f:
        pickle.dump(rec, f)


def test_returns_seven_values_with_ldos_dict(tmp_path):
    _write(tmp_path, 'a.pkl', _record('a', 't', True))
    _write(tmp_path, 'b.pkl', _record('c', 'g', True))

    result = load_pickle_directory(str(tmp_path))

    assert len(result) == 7
    ldos_data = result[6]
    assert set(ldos_data) == {'residue', 'base_only'}
    assert len(ldos_data['residue']) == 2
    assert ldos_data['residue'][0].shape == (2, 2)


def test_returns_none_ldos_when_no_record_has_dosatom(tmp_path):
    _write(tmp_path, 'a.pkl', _record('a', 't', False))
    _write(tmp_path, 'b.pkl', _record('c', 'g', False))

    result = load_pickle_directory(str(tmp_path))

    assert len(result) == 7
    assert result[6] is None


def test_mixed_availability_raises(tmp_path):
    # A ragged dataset -- some samples with an LDOS target and some without --
    # cannot be trained coherently. Fail loudly rather than silently dropping.
    _write(tmp_path, 'a.pkl', _record('a', 't', True))
    _write(tmp_path, 'b.pkl', _record('c', 'g', False))

    with pytest.raises(ValueError, match="some records"):
        load_pickle_directory(str(tmp_path))
