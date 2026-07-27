"""Published HDF5 archive and pickle_files_v2 must aggregate identically.

g3nat_dna_transport.h5 and pickle_files_v2/ are two serialisations of the same
2077 records. If they ever disagree, downstream consumers of the published
archive get different LDOS numbers than what the paper reports from the
pickles. This test is the only check that catches that.

Layout assumed below was read directly from DNADataset/export_hdf5.py (not
guessed) and cross-checked against the existing round-trip test
tests/test_dataset/test_export_hdf5.py:
    /<sequence>/<run>/DOSAtom          dataset, float64, [n_atoms, n_energy]
    /<sequence>/<run>/atoms/resseq     dataset, int32,   [n_atoms]
DOSAtom and the atoms subgroup sit directly under the run group (the
exporter flattens "contacts" into attrs plus left_atoms/right_atoms
datasets, but DOSAtom/atoms are not part of that flattening and were never
nested under it).
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from g3nat.data.ldos import aggregate_by_residue

H5_PATH = Path('g3nat_dna_transport.h5')
PKL_DIR = Path('pickle_files_v2')

pytestmark = pytest.mark.skipif(
    not (H5_PATH.exists() and PKL_DIR.exists()),
    reason="published archive or pickle_files_v2 not present in this checkout",
)


def test_one_record_aggregates_identically_from_both_sources():
    h5py = pytest.importorskip("h5py")

    pkl_path = sorted(PKL_DIR.glob('*.pkl'))[0]
    with open(pkl_path, 'rb') as f:
        record = pickle.load(f)

    seq = record['sequence']
    run = pkl_path.stem.rsplit('_', 1)[1]

    from_pickle = aggregate_by_residue(
        record['DOSAtom'], record['atoms']['resseq']
    )

    with h5py.File(H5_PATH, 'r') as h5:
        group = h5[seq][run]
        dosatom = np.asarray(group['DOSAtom'])
        resseq = np.asarray(group['atoms']['resseq'])

    from_hdf5 = aggregate_by_residue(dosatom, resseq)

    assert from_pickle.shape == from_hdf5.shape
    np.testing.assert_allclose(from_pickle, from_hdf5, rtol=1e-12, atol=0.0)
