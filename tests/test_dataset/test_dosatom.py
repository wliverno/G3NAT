import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/dataset/aaac"
DOSMAT = FIXTURE / "run1/DOS_aaac_gammaL_0.1_gammaR_0.1.mat"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "convert_to_pickle", REPO / "DNADataset/convert_to_pickle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dosatom_shape_and_dtype():
    m = _load_module()
    da = m.load_dosatom(DOSMAT, n_energy=201)
    assert da.shape == (253, 201), f"expected (atoms, energy), got {da.shape}"
    assert da.dtype == np.float64


def test_dosatom_transposes_when_stored_energy_by_atoms(tmp_path):
    """Covers the orientation-normalisation branch.

    The real fixture's DOSAtom already ships as (atoms, energy), so it never
    exercises the transpose line. This writes a synthetic .mat with DOSAtom
    stored the OTHER way round -- (energy, atoms), using distinct non-square
    dimensions matching the real data (201 vs 253) and non-symmetric values --
    and checks load_dosatom both reorients it and preserves the values
    (a shape-only check would pass even if the transpose logic were wrong).
    """
    m = _load_module()
    n_energy, n_atoms = 201, 253
    stored = np.arange(n_energy * n_atoms, dtype=np.float64).reshape(n_energy, n_atoms)
    mat_path = tmp_path / "synthetic_DOSAtom_transposed.mat"
    savemat(mat_path, {"DOSAtom": stored})

    da = m.load_dosatom(mat_path, n_energy=n_energy)

    assert da.shape == (n_atoms, n_energy)
    assert da.dtype == np.float64
    np.testing.assert_array_equal(da, stored.T)


def test_dosatom_sums_to_total_dos():
    """The invariant that makes per-atom LDOS trustworthy.

    Measured on this exact file: sum over atoms equals the 1-D DOS vector to a
    ratio of 1.0000 at every energy. A mis-picked key or a transposed array
    breaks this immediately.
    """
    m = _load_module()
    E, DOS, _, _ = m.load_energy_and_values(DOSMAT, "DOS")
    da = m.load_dosatom(DOSMAT, n_energy=len(E))
    assert np.allclose(da.sum(axis=0), DOS, rtol=1e-9, atol=0)


def test_missing_dosatom_key_raises_rather_than_guessing():
    """The pre-existing heuristics fall back to 'any vector of matching length'.
    For DOSAtom that would silently return the wrong array."""
    m = _load_module()
    with pytest.raises(KeyError, match="DOSAtom"):
        m.load_dosatom(FIXTURE / "run1/Tran_aaac_gammaL_0.1_gammaR_0.1.mat", n_energy=201)
