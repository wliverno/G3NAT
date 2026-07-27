import importlib.util
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/dataset/aaac"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "convert_to_pickle", REPO / "DNADataset/convert_to_pickle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec():
    """Returns (module, record, n_orbitals). n_orbitals is parsed the same way
    main() does -- separately from build_record, since it is no longer stored
    in the record itself -- and passed as validate_record's required 2nd arg."""
    m = _load_module()
    rec = m.build_record(FIXTURE, "run1",
                         {"coupling_eV": 0.1, "contact_type": "same"})
    orbs = m.parse_orbitals_set(FIXTURE / "run1" / "Parameters.txt")
    return m, rec, orbs


def test_record_has_all_schema_fields():
    _, rec, _ = _rec()
    for k in ("sequence", "complementary_sequence", "gjf_text", "contacts",
              "Egrid", "T", "DOS", "DOSAtom", "atoms", "energy_reference_eV"):
        assert k in rec, f"missing field {k}"
    assert rec["sequence"] == "aaac"
    assert rec["complementary_sequence"] == "gttt"
    assert rec["contacts"]["coupling_eV"] == 0.1
    assert rec["contacts"]["contact_type"] == "same"


def test_arrays_are_float64_ndarrays_not_lists():
    _, rec, _ = _rec()
    for k in ("Egrid", "T", "DOS", "DOSAtom"):
        assert isinstance(rec[k], np.ndarray), f"{k} should be ndarray"
        assert rec[k].dtype == np.float64


def test_energy_reference_is_the_raw_grid_mean_and_grid_stays_absolute():
    _, rec, _ = _rec()
    assert np.isclose(rec["energy_reference_eV"], rec["Egrid"].mean())
    # raw/absolute, NOT centred -- centring must stay reversible
    assert abs(rec["Egrid"].mean()) > 1.0


def test_validate_accepts_the_real_record():
    m, rec, orbs = _rec()
    assert m.validate_record(rec, orbs) == []


def test_validate_rejects_minus_one_sentinels():
    """Both MATLAB drivers checkpoint by writing -1 and resuming from the last
    non-(-1) index. A walltime-killed run leaves real -1 values that look like
    ordinary floats. T and DOS are non-negative, so any negative is corruption."""
    m, rec, orbs = _rec()
    bad = dict(rec)
    bad["T"] = rec["T"].copy()
    bad["T"][17] = -1.0
    reasons = m.validate_record(bad, orbs)
    assert any("-1" in r or "negative" in r for r in reasons), reasons


def test_validate_rejects_dosatom_row_count_mismatch():
    m, rec, orbs = _rec()
    bad = dict(rec)
    bad["DOSAtom"] = rec["DOSAtom"][:-1]
    assert m.validate_record(bad, orbs) != []


def test_validate_rejects_permuted_atom_table():
    """THE test. A scrambled atom order still sums correctly and still has the
    right shapes -- only the per-atom identity is wrong, which is exactly the
    failure that produces plausible-but-wrong data. Detected via the orbital
    counts (now a separate argument, not stored in the record), which must stay
    consistent with element identity: orbs stays in original Parameters.txt file
    order while rec["atoms"] gets scrambled, so the mismatch still fires."""
    m, rec, orbs = _rec()
    bad = dict(rec)
    atoms = {k: (list(v) if isinstance(v, list) else v.copy())
             for k, v in rec["atoms"].items()}
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(atoms["element"]))
    for k in ("element", "name", "resseq", "resname", "chain"):
        atoms[k] = [atoms[k][i] for i in perm]
    atoms["xyz"] = atoms["xyz"][perm]
    bad["atoms"] = atoms
    assert m.validate_record(bad, orbs) != [], \
        "a permuted atom table MUST be rejected, or the mapping is untested"


def test_validate_rejects_phosphorus_orbital_corruption():
    """This is the test that would have caught Critical 2: the old validator's
    branch only covered 'H' and 'C'/'N'/'O', so a corrupted phosphorus orbital
    count passed silently -- every phosphate backbone atom was unvalidated.
    ELEMENT_ORBITALS now covers P -> 19, so this must be rejected. n_orbitals is
    now the validate_record ARGUMENT (not part of rec["atoms"]), so the
    corruption is made there."""
    m, rec, orbs = _rec()
    bad_orbs = list(orbs)
    p_idx = rec["atoms"]["element"].index("P")
    assert bad_orbs[p_idx] == 19
    bad_orbs[p_idx] = 15
    reasons = m.validate_record(rec, bad_orbs)
    assert reasons != [], "a corrupted P orbital count MUST be rejected"


def test_validate_rejects_unknown_element():
    """An element absent from ELEMENT_ORBITALS must be a failure reason, not a
    silent pass -- it means either a parser fault or data this validator was
    never measured against. element stays in rec["atoms"] (n_orbitals moved out,
    element did not), so the corruption target is unchanged."""
    m, rec, orbs = _rec()
    bad = dict(rec)
    atoms = {k: (list(v) if isinstance(v, list) else v.copy())
             for k, v in rec["atoms"].items()}
    atoms["element"][0] = "X"
    bad["atoms"] = atoms
    reasons = m.validate_record(bad, orbs)
    assert any("unknown element" in r for r in reasons), reasons


def test_validate_rejects_non_monotonic_resseq():
    """resseq is measured non-decreasing in file order across 55/55 sequences, so
    this check has zero false positives on real data and catches cross-residue
    reordering -- the realistic parser-bug class. Reversing resseq alone (leaving
    element/name/xyz untouched) isolates this check from the orbital-consistency
    and permutation checks above."""
    m, rec, orbs = _rec()
    bad = dict(rec)
    atoms = {k: (list(v) if isinstance(v, list) else v.copy())
             for k, v in rec["atoms"].items()}
    atoms["resseq"] = list(reversed(atoms["resseq"]))
    bad["atoms"] = atoms
    reasons = m.validate_record(bad, orbs)
    assert any("resseq" in r or "residue" in r for r in reasons), reasons


def test_validate_rejects_empty_left_atoms():
    """The old check `if idx and (min(idx) < 1 or max(idx) > n_atoms)` short-
    circuits when idx is []: an empty contact list identifies no atoms at all
    (physically meaningless) but passed validation silently. An empty side must
    now be its own explicit failure reason, distinct from the out-of-range one,
    and the non-empty range check must keep working exactly as before."""
    m, rec, orbs = _rec()
    bad = dict(rec)
    contacts = dict(rec["contacts"])
    contacts["left_atoms"] = []
    bad["contacts"] = contacts
    reasons = m.validate_record(bad, orbs)
    assert any("left_atoms" in r and "empty" in r for r in reasons), reasons


def test_validate_rejects_negative_dosatom_directly():
    """DOSAtom must be checked directly for the -1 sentinel, not only via the
    sum-to-DOS tolerance. DOS is adjusted by the same delta so the sum invariant
    still holds exactly -- if a reason still names DOSAtom, it is the direct
    per-value check firing, not the indirect sum-mismatch check."""
    m, rec, orbs = _rec()
    bad = dict(rec)
    bad["DOSAtom"] = rec["DOSAtom"].copy()
    bad["DOS"] = rec["DOS"].copy()
    i, j = 0, 0
    delta = bad["DOSAtom"][i, j] - (-1.0)
    bad["DOSAtom"][i, j] = -1.0
    bad["DOS"][j] -= delta
    assert np.allclose(bad["DOSAtom"].sum(axis=0), bad["DOS"], rtol=1e-6, atol=0)
    reasons = m.validate_record(bad, orbs)
    assert any("DOSAtom" in r for r in reasons), reasons


def test_atoms_dict_does_not_contain_n_orbitals():
    """n_orbitals must not ship in the record: DOSAtom is already per-atom (summed
    over AOs), and the published archive has no Fock/Overlap/H0 for AO-space data
    to be indexed against, so the counts have no consumer once validated."""
    _, rec, _ = _rec()
    assert "n_orbitals" not in rec["atoms"]


def test_validate_record_requires_n_orbitals_argument():
    """No default is allowed -- an omitted argument must raise TypeError, not
    silently skip the orbital-consistency check."""
    m, rec, _ = _rec()
    with pytest.raises(TypeError):
        m.validate_record(rec)
