# tests/test_dataset/test_loader_compat.py
"""The new schema is additive. If this fails, the training path is broken and
every recorded result becomes incomparable."""
import importlib.util
import pickle
from pathlib import Path
import numpy as np

from g3nat.data.pickle import load_single_pickle

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/dataset/aaac"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "convert_to_pickle", REPO / "DNADataset/convert_to_pickle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_existing_loader_reads_new_schema(tmp_path):
    # coupling_eV=0.6 / contact_type="cross" deliberately differ from
    # g3nat/data/pickle.py:54-55's own fallback defaults (0.1 / "same"). If
    # build_record ever dropped or renamed the "contacts" dict, the loader would
    # silently fall back to those defaults and this test must go red -- with the
    # old 0.1/"same" fixture values it would have stayed green by coincidence.
    m = _load_module()
    rec = m.build_record(FIXTURE, "run1", {"coupling_eV": 0.6, "contact_type": "cross"})
    p = tmp_path / "aaac_run1.pkl"
    with open(p, "wb") as f:
        pickle.dump(rec, f)

    d = load_single_pickle(str(p))
    assert d is not None, "loader rejected the new schema"
    assert d["sequence"] == "AAAC"
    assert d["coupling"] == 0.6
    assert d["contact_type"] == "cross"
    # ndarray inputs must survive the loader's np.array(...) + log10 unchanged
    assert np.all(np.isfinite(d["dos"]))
    assert np.all(np.isfinite(d["transmission"]))
    # loader centres the grid; ours ships raw, so centring must still land on [-1, 1]
    assert np.isclose(d["energy_grid"].min(), -1.0, atol=1e-6)
    assert np.isclose(d["energy_grid"].max(), 1.0, atol=1e-6)
