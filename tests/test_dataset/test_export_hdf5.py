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
