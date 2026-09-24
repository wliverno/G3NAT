import importlib.util
from pathlib import Path
import h5py
import numpy as np
import pytest
from scipy.io import savemat

REPO = Path(__file__).resolve().parents[2]
_A = np.random.default_rng(0).normal(size=(10, 10))
_F = (_A + _A.T) / 2


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"DNADataset/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _job(tmp_path, seq, elements, asym=False):
    F = _F.copy()
    if asym:
        F[0, 1] += 1e-3
    S = np.eye(10) + 0.01 * _F / np.abs(_F).max()
    savemat(tmp_path / f"{seq}_Fock.mat", {f"{seq}_Fock": F})
    savemat(tmp_path / f"{seq}_Overlap.mat", {f"{seq}_Overlap": S})
    return (seq, elements, tmp_path / f"{seq}_Fock.mat", tmp_path / f"{seq}_Overlap.mat", None)


def test_writes_matrices_and_basis(tmp_path):
    em = _mod("export_matrices_hdf5")
    job = _job(tmp_path, "hh", ["H", "H"])
    out = tmp_path / "m.h5"
    assert em.export_matrices([job], out, {"units_fock": "Hartree"}) == 1
    with h5py.File(out, "r") as h:
        assert h["/hh/fock"].shape == (10, 10)
        np.testing.assert_array_equal(h["/hh/fock"][:], _F)
        assert h["/hh/basis/component"][:].tolist()[:5] == [b"S", b"S", b"PX", b"PY", b"PZ"]
        assert h["/hh"].attrs["label_source"] == "derived"


def test_rejects_asymmetric(tmp_path):
    em = _mod("export_matrices_hdf5")
    job = _job(tmp_path, "hh", ["H", "H"], asym=True)
    with pytest.raises(ValueError, match="hh: .*symmetric"):
        em.export_matrices([job], tmp_path / "m.h5", {})


def test_rejects_size_mismatch(tmp_path):
    em = _mod("export_matrices_hdf5")
    job = _job(tmp_path, "hh", ["H", "H", "H"])   # 15 labels, 10x10 matrices
    with pytest.raises(ValueError, match="hh: .*15"):
        em.export_matrices([job], tmp_path / "m.h5", {})
