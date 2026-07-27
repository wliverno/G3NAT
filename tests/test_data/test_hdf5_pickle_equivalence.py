"""Published HDF5 archive and pickle_files_v2 must aggregate identically.

g3nat_dna_transport.h5 and pickle_files_v2/ are two serialisations of the same
2077 records. If they ever disagree, downstream consumers of the published
archive get different LDOS numbers than what the paper reports from the
pickles.

Coverage is split in two, because a single spot check cannot rule out a
partial or truncated export:
  - EVERY run group in the archive is checked for presence and shape
    (DOSAtom is 2-D with the right energy-grid width, atoms/resseq has the
    matching atom count), and the total group count is checked against the
    number of *_run*.pkl files on disk. This uses HDF5 metadata only -- no
    pickle reads, no array loads -- so it is cheap even over 2077 records.
  - A SAMPLE of records (first, middle, last of the sorted file list) is
    checked for exact aggregated LDOS values against the pickle source.
    This is the expensive check (loads full arrays from both sources), so
    it is not run over the whole archive.
Neither check alone establishes both properties; together they do.

Layout assumed below was read directly from DNADataset/export_hdf5.py (not
guessed) and cross-checked against the existing round-trip test
tests/test_dataset/test_export_hdf5.py:
    /<sequence>/<run>/Egrid            dataset, float64, [n_energy]
    /<sequence>/<run>/DOSAtom          dataset, float64, [n_atoms, n_energy]
    /<sequence>/<run>/atoms/resseq     dataset, int32,   [n_atoms]
DOSAtom and the atoms subgroup sit directly under the run group (the
exporter flattens "contacts" into attrs plus left_atoms/right_atoms
datasets, but DOSAtom/atoms are not part of that flattening and were never
nested under it).

The pickle glob deliberately matches the exporter's own pattern,
*_run*.pkl, not the looser *.pkl -- so a stray non-record .pkl file (a
cache artifact, a partial write) cannot be picked up here when the exporter
never processed it either.
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


def _iter_run_groups(h5):
    """Yield (sequence, run, group) for every run group in the archive."""
    for seq in h5.keys():
        seq_group = h5[seq]
        for run in seq_group.keys():
            yield seq, run, seq_group[run]


def test_every_record_is_present_in_the_archive_with_the_expected_shape():
    # Breadth, cheap: the value-comparison test below only checks a sample,
    # but a partial or truncated export would leave most records missing or
    # mis-shaped while that sample still matched. This walks all of them
    # using HDF5 metadata only -- no pickle reads, no array loads.
    h5py = pytest.importorskip("h5py")

    n_groups = 0
    with h5py.File(H5_PATH, 'r') as h5:
        for seq, run, group in _iter_run_groups(h5):
            n_groups += 1

            assert 'DOSAtom' in group, f"{seq}/{run} missing DOSAtom"
            dosatom = group['DOSAtom']
            assert dosatom.ndim == 2, (
                f"{seq}/{run} DOSAtom is not 2-D: shape {dosatom.shape}"
            )

            assert 'Egrid' in group, f"{seq}/{run} missing Egrid"
            energy_len = group['Egrid'].shape[0]
            assert dosatom.shape[1] == energy_len, (
                f"{seq}/{run} DOSAtom second axis {dosatom.shape[1]} != "
                f"Egrid length {energy_len}"
            )

            assert 'atoms' in group and 'resseq' in group['atoms'], (
                f"{seq}/{run} missing atoms/resseq"
            )
            resseq_len = group['atoms']['resseq'].shape[0]
            assert resseq_len == dosatom.shape[0], (
                f"{seq}/{run} atoms/resseq length {resseq_len} != "
                f"DOSAtom first axis {dosatom.shape[0]}"
            )

    n_pkl = len(list(PKL_DIR.glob('*_run*.pkl')))
    assert n_groups > 0, "no run groups found in the archive -- H5_PATH or PKL_DIR may be misconfigured"
    assert n_groups == n_pkl, (
        f"archive has {n_groups} run groups but pickle_files_v2 has {n_pkl} "
        f"*_run*.pkl files -- export may be truncated or partial"
    )


def test_sampled_records_aggregate_identically_from_both_sources():
    # Depth: first, middle and last of the sorted file list, not just the
    # first -- a truncated or partially-written export is far more likely to
    # damage records at the end (or in the middle) than the very first one.
    h5py = pytest.importorskip("h5py")

    pkl_paths = sorted(PKL_DIR.glob('*_run*.pkl'))
    sample = [pkl_paths[0], pkl_paths[len(pkl_paths) // 2], pkl_paths[-1]]

    with h5py.File(H5_PATH, 'r') as h5:
        for pkl_path in sample:
            with open(pkl_path, 'rb') as f:
                record = pickle.load(f)

            seq = record['sequence']
            run = pkl_path.stem.rsplit('_', 1)[1]

            from_pickle = aggregate_by_residue(
                record['DOSAtom'], record['atoms']['resseq']
            )

            group = h5[seq][run]
            dosatom = np.asarray(group['DOSAtom'])
            resseq = np.asarray(group['atoms']['resseq'])

            from_hdf5 = aggregate_by_residue(dosatom, resseq)

            assert from_pickle.shape == from_hdf5.shape, (
                f"{seq}/{run}: shape mismatch {from_pickle.shape} vs "
                f"{from_hdf5.shape}"
            )
            np.testing.assert_allclose(
                from_pickle, from_hdf5, rtol=1e-12, atol=0.0
            )
