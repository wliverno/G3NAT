import copy
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/dataset/aaac"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"DNADataset/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _as_run(rec, run, ex):
    r = copy.deepcopy(rec)
    cpl, ct = ex.RUN_MAP[run]
    r["contacts"]["coupling_eV"] = cpl
    r["contacts"]["contact_type"] = ct
    L = len(r["sequence"])
    want = L if ct == "same" else L + 1
    r["contacts"]["right_atoms"] = [i + 1 for i, x in enumerate(r["atoms"]["resseq"]) if x == want]
    return r


@pytest.fixture
def archive(tmp_path):
    cp, ex = _mod("convert_to_pickle"), _mod("export_hdf5")
    pdir = tmp_path / "pkl"; pdir.mkdir()
    recs = {}
    rec1 = cp.build_record(FIXTURE, "run1", {"coupling_eV": 0.1, "contact_type": "same"})
    for run in ex.RUN_MAP:
        rec = rec1 if run == "run1" else _as_run(rec1, run, ex)
        with open(pdir / f"aaac_{run}.pkl", "wb") as f:
            pickle.dump(rec, f)
        recs[run] = rec
    out = tmp_path / "t.h5"
    ex.export(pdir, out, ex.ROOT_ATTRS)
    return out, recs


def test_import_roundtrip_exact(archive, tmp_path):
    im, va = _mod("import_hdf5"), _mod("verify_archive")
    out, recs = archive
    dst = tmp_path / "back"
    assert im.import_archive(out, dst) == 4
    for run, rec in recs.items():
        with open(dst / f"aaac_{run}.pkl", "rb") as f:
            back = pickle.load(f)
        assert va.records_equal(rec, back) == []
        assert list(back) == list(rec)                      # key order
        assert type(back["atoms"]["element"][0]) is str      # not bytes
        assert type(back["contacts"]["left_atoms"][0]) is int


def test_import_refuses_overwrite(archive, tmp_path):
    im = _mod("import_hdf5")
    out, _ = archive
    dst = tmp_path / "back"; dst.mkdir()
    (dst / "aaac_run1.pkl").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        im.import_archive(out, dst)
    assert (dst / "aaac_run1.pkl").read_bytes() == b"x"


def test_records_equal_detects_differences(archive):
    va = _mod("verify_archive")
    _, recs = archive
    a = recs["run1"]
    b = pickle.loads(pickle.dumps(a))
    b["T"] = b["T"].copy(); b["T"][0] = np.nextafter(b["T"][0], 1.0)
    b["atoms"]["resseq"] = np.asarray(b["atoms"]["resseq"])
    diffs = va.records_equal(a, b)
    assert any("T" in d for d in diffs) and any("resseq" in d for d in diffs)
