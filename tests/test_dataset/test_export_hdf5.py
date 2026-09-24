import importlib.util
import pickle
from pathlib import Path
import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/dataset/aaac"


def _mod(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / f"DNADataset/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


def _decode_list(arr):
    return [_decode(s) for s in arr]


def test_roundtrip_is_lossless(tmp_path):
    cp = _mod("convert_to_pickle")
    ex = _mod("export_hdf5")
    rec = cp.build_record(FIXTURE, "run1", {"coupling_eV": 0.1, "contact_type": "same"})
    pdir = tmp_path / "pkl"; pdir.mkdir()
    with open(pdir / "aaac_run1.pkl", "wb") as f:
        pickle.dump(rec, f)

    out = tmp_path / "d.h5"
    assert ex.export(pdir, out, {"units_energy": "eV"}) == 1

    with h5py.File(out, "r") as h:
        g = h["/aaac/run1"]

        # Numeric arrays: bit-exact values AND dtype, not just values.
        for key in ("Egrid", "T", "DOS", "DOSAtom"):
            np.testing.assert_array_equal(g[key][:], rec[key])
            assert g[key].dtype == np.float64

        np.testing.assert_array_equal(g["atoms/xyz"][:], rec["atoms"]["xyz"])
        assert g["atoms/xyz"].dtype == np.float64

        np.testing.assert_array_equal(
            g["atoms/resseq"][:], np.asarray(rec["atoms"]["resseq"], dtype=np.int32))
        assert g["atoms/resseq"].dtype == np.int32

        np.testing.assert_array_equal(
            g["left_atoms"][:], np.asarray(rec["contacts"]["left_atoms"], dtype=np.int32))
        assert g["left_atoms"].dtype == np.int32
        np.testing.assert_array_equal(
            g["right_atoms"][:], np.asarray(rec["contacts"]["right_atoms"], dtype=np.int32))
        assert g["right_atoms"].dtype == np.int32

        # Strings: exact equality after decoding, full lists not spot checks.
        assert _decode(g["gjf_text"][()]) == rec["gjf_text"]
        assert _decode(g["complementary_sequence"][()]) == rec["complementary_sequence"]
        assert _decode_list(g["atoms/element"][:]) == rec["atoms"]["element"]
        assert _decode_list(g["atoms/name"][:]) == rec["atoms"]["name"]
        assert _decode_list(g["atoms/resname"][:]) == rec["atoms"]["resname"]

        # atoms/chain is deliberately NOT exported (blank in source PDBs, no
        # consumer); strand identity is carried by the root attr instead.
        assert "chain" not in g["atoms"]

        # Attrs: exact equality.
        assert g.attrs["coupling_eV"] == rec["contacts"]["coupling_eV"]
        assert g.attrs["contact_type"] == rec["contacts"]["contact_type"]
        assert g.attrs["energy_reference_eV"] == rec["energy_reference_eV"]

        assert h.attrs["units_energy"] == "eV"


import copy
import pytest


def _fixture_rec():
    cp = _mod("convert_to_pickle")
    return cp.build_record(FIXTURE, "run1", {"coupling_eV": 0.1, "contact_type": "same"})


def _as_run(rec, run, ex):
    r = copy.deepcopy(rec)
    cpl, ct = ex.RUN_MAP[run]
    r["contacts"]["coupling_eV"] = cpl
    r["contacts"]["contact_type"] = ct
    L = len(r["sequence"])
    want = L if ct == "same" else L + 1
    r["contacts"]["right_atoms"] = [i + 1 for i, x in enumerate(r["atoms"]["resseq"]) if x == want]
    return r


def _write(pdir, name, rec):
    pdir.mkdir(exist_ok=True)
    with open(pdir / name, "wb") as f:
        pickle.dump(rec, f)


def test_export_heldout_split_and_residues(tmp_path):
    ex = _mod("export_hdf5")
    rec = _fixture_rec()
    _write(tmp_path / "train", "aaac_run1.pkl", rec)
    held = copy.deepcopy(rec); held["sequence"] = "aaaa"   # distinct group name
    _write(tmp_path / "held", "aaaa_run1.pkl", held)
    out = tmp_path / "t.h5"
    assert ex.export(tmp_path / "train", out, {"units_energy": "eV"},
                     heldout_dirs=[tmp_path / "held"]) == 2
    L = len(rec["sequence"])
    with h5py.File(out, "r") as h:
        assert h["/aaac/run1"].attrs["split"] == "train"
        assert h["/aaaa/run1"].attrs["split"] == "heldout"
        assert h["/aaac/run1"].attrs["left_residue"] == 1
        assert h["/aaac/run1"].attrs["right_residue"] == L        # same: 1 -> L


def test_check_contacts_cross(tmp_path):
    ex = _mod("export_hdf5")
    rec = _as_run(_fixture_rec(), "run2", ex)
    L = len(rec["sequence"])
    assert ex.check_contacts(rec, "run2", "x") == (1, L + 1)


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["contacts"]["left_atoms"].pop(), "whole residue"),
    (lambda r: r["contacts"].__setitem__("coupling_eV", 0.6), "run map"),
    (lambda r: r["contacts"].__setitem__("contact_type", "cross"), "run map"),
    (lambda r: r["atoms"]["chain"].__setitem__(0, "A"), "chain"),
])
def test_export_rejects_bad_contacts(tmp_path, mutate, msg):
    ex = _mod("export_hdf5")
    rec = _fixture_rec()
    mutate(rec)
    _write(tmp_path / "p", "aaac_run1.pkl", rec)
    with pytest.raises(ValueError, match=msg) as e:
        ex.export(tmp_path / "p", tmp_path / "t.h5", {})
    assert "aaac_run1.pkl" in str(e.value)


def test_root_attrs_are_current():
    ex = _mod("export_hdf5")
    a = ex.ROOT_ATTRS
    assert "NOT included" not in a["limitations"]
    assert "matrices.h5" in a["limitations"]
    assert "5' end" in a["run_map"] and "complementary" in a["run_map"]
    assert a["license"] == "CC-BY-4.0"
