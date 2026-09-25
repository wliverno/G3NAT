"""The guards that run before any fit: validate_structure, merge_ci, and the
cell/seed indexing that response_array depends on.

Oracles: the design's own factor-level tables (so the expected key set is
recomputed, not retyped), hand lookups of individual values, and direct
recomputation of the medians and companion means merge_ci is supposed to
produce.
"""
import copy

import numpy as np
import pytest

import doe_v3 as D
from conftest import make_runs, make_ci, in_cell


# ---------------------------------------------------------------------------
# validate_structure
# ---------------------------------------------------------------------------

def test_the_complete_design_passes_and_has_the_documented_counts():
    runs = make_runs()
    D.validate_structure(runs)
    assert len(runs) == 120 == D.N_TOTAL_RUNS
    assert sum(1 for r in runs.values() if r['family'] == 'ham') == 96
    assert sum(1 for r in runs.values() if r['family'] == 'blind') == 24


def test_a_missing_run_aborts():
    runs = make_runs()
    del runs[next(k for k, v in runs.items() if v['family'] == 'ham')]
    with pytest.raises(D.StructureError):
        D.validate_structure(runs)


def test_a_duplicated_cell_seed_key_aborts_even_at_the_right_total():
    """Two names carrying the same (cell, seed) key while the total stays 120:
    only the factor-level SET comparison can catch this, not the row count."""
    runs = make_runs()
    ham = sorted(k for k, v in runs.items() if v['family'] == 'ham')
    runs[ham[1]] = copy.deepcopy(runs[ham[0]])
    with pytest.raises(D.StructureError) as e:
        D.validate_structure(runs)
    assert 'structure does not match' in str(e.value)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
@pytest.mark.parametrize('resp', ['l12_dos', 'l16_transmission'])
def test_a_non_finite_response_aborts(bad, resp):
    runs = make_runs()
    runs[next(iter(runs))][resp] = bad
    with pytest.raises(D.StructureError) as e:
        D.validate_structure(runs)
    assert 'non-finite' in str(e.value)


def test_a_missing_response_key_aborts():
    runs = make_runs()
    del runs[next(iter(runs))]['l12_dos']
    with pytest.raises(D.StructureError) as e:
        D.validate_structure(runs)
    assert 'missing response' in str(e.value)


def test_a_level_outside_the_design_aborts():
    runs = make_runs()
    k = next(k for k, v in runs.items() if v['family'] == 'ham')
    runs[k]['n_orb'] = 3
    with pytest.raises(D.StructureError) as e:
        D.validate_structure(runs)
    assert 'structure does not match' in str(e.value)


@pytest.mark.parametrize('key', ['onsite_near', 'onsite_far',
                                 'onsite_n_graphs'])
def test_a_missing_onsite_key_on_a_ham_row_aborts(key):
    runs = make_runs()
    del runs[next(k for k, v in runs.items() if v['family'] == 'ham')][key]
    with pytest.raises(D.StructureError):
        D.validate_structure(runs)


def test_a_non_finite_onsite_value_aborts():
    runs = make_runs()
    runs[next(k for k, v in runs.items()
              if v['family'] == 'ham')]['onsite_far'] = float('nan')
    with pytest.raises(D.StructureError):
        D.validate_structure(runs)


def test_sub_n_comparisons_must_be_identical_across_all_120_rows():
    runs = make_runs()
    runs[next(iter(runs))]['sub_n_comparisons'] = 32
    with pytest.raises(D.StructureError) as e:
        D.validate_structure(runs)
    assert 'sub_n_comparisons' in str(e.value)


def test_an_unknown_family_label_aborts():
    runs = make_runs()
    runs[next(iter(runs))]['family'] = 'hamiltonian'
    with pytest.raises(D.StructureError):
        D.validate_structure(runs)


def test_a_string_valued_response_escapes_as_a_typeerror():
    """DOCUMENTED, not endorsed: np.isfinite on a str raises TypeError, so a
    string-valued response does not reach the FATAL StructureError path. It
    still aborts the run loudly, which is the property that matters."""
    runs = make_runs()
    runs[next(iter(runs))]['l12_dos'] = 'NaN'
    with pytest.raises(TypeError):
        D.validate_structure(runs)


def test_a_stray_legacy_sub_t_key_is_read_tolerantly_and_never_analyzed():
    """Decision 2026-08-26: old reports carrying sub_t are read tolerantly, even
    when the stray value is non-finite, and the value is simply not analyzed."""
    runs = make_runs()
    for row in runs.values():
        row['sub_t'] = float('nan')
    D.validate_structure(runs)
    assert 'sub_t' not in D.RESPONSES_COMMON
    fa = D.analyze_family(runs, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          list(D.RESPONSES_COMMON))
    assert 'sub_t' not in fa.responses


# ---------------------------------------------------------------------------
# merge_ci
# ---------------------------------------------------------------------------

def test_merge_ci_attaches_per_run_medians_and_the_companion_verbatim():
    """Oracle: np.median of each raw vector, recomputed here; the companion is
    copied field for field with no re-derivation. Blind rows and the caller's
    own dict must be left untouched."""
    runs = make_runs()
    ci = make_ci(runs)
    merged = D.merge_ci(runs, ci)
    for name, row in merged.items():
        if row['family'] != 'ham':
            assert 'ci_interior_mode' not in row
            continue
        v = ci['runs'][name]['vectors']
        assert row['ci_interior_mode'] == pytest.approx(
            float(np.median(v['D_interior_mode'])))
        assert row['ci_interior_total'] == pytest.approx(
            float(np.median(v['D_interior_total'])))
        assert row['ci_companion'] == pytest.approx(
            ci['runs'][name]['companion'])
        assert set(row['ci_companion']) == set(D.CI_COMPANION_FIELDS)
    assert 'ci_interior_mode' not in next(
        v for v in runs.values() if v['family'] == 'ham')


def test_merge_ci_missing_entry_for_a_ham_run_aborts():
    runs = make_runs()
    ci = make_ci(runs)
    ci['runs'].pop(next(iter(ci['runs'])))
    with pytest.raises(D.StructureError):
        D.merge_ci(runs, ci)


def test_merge_ci_non_finite_vector_aborts():
    runs = make_runs()
    ci = make_ci(runs)
    ci['runs'][next(iter(ci['runs']))]['vectors']['D_interior_mode'][0] = \
        float('nan')
    with pytest.raises(D.StructureError):
        D.merge_ci(runs, ci)


def test_merge_ci_mismatched_vector_lengths_abort():
    runs = make_runs()
    ci = make_ci(runs)
    ci['runs'][next(iter(ci['runs']))]['vectors']['D_interior_total'] = \
        [0.1, 0.2]
    with pytest.raises(D.StructureError):
        D.merge_ci(runs, ci)


def test_merge_ci_empty_vector_aborts_but_a_single_exact_zero_is_data():
    """The guard is `if not vec`, so the boundary matters: an EMPTY vector is
    fatal, a vector holding one exact 0.0 is a legitimate measurement and must
    pass (no clamping, no discarding)."""
    runs = make_runs()
    ci = make_ci(runs)
    ci['runs'][next(iter(ci['runs']))]['vectors']['D_interior_mode'] = []
    with pytest.raises(D.StructureError):
        D.merge_ci(runs, ci)

    ci2 = make_ci(runs)
    k = next(iter(ci2['runs']))
    ci2['runs'][k]['vectors']['D_interior_mode'] = [0.0]
    ci2['runs'][k]['vectors']['D_interior_total'] = [0.0]
    assert D.merge_ci(runs, ci2)[k]['ci_interior_mode'] == 0.0


def test_merge_ci_missing_companion_object_aborts():
    runs = make_runs()
    ci = make_ci(runs)
    del ci['runs'][next(iter(ci['runs']))]['companion']
    with pytest.raises(D.StructureError):
        D.merge_ci(runs, ci)


def test_merge_ci_missing_companion_field_aborts():
    runs = make_runs()
    ci = make_ci(runs)
    del ci['runs'][next(iter(ci['runs']))]['companion'][
        D.CI_COMPANION_FIELDS[0]]
    with pytest.raises(D.StructureError):
        D.merge_ci(runs, ci)


def test_merge_ci_non_finite_companion_field_aborts():
    runs = make_runs()
    ci = make_ci(runs)
    ci['runs'][next(iter(ci['runs']))]['companion'][
        D.CI_COMPANION_FIELDS[1]] = float('inf')
    with pytest.raises(D.StructureError):
        D.merge_ci(runs, ci)


# ---------------------------------------------------------------------------
# response_array / analyze_response indexing
# ---------------------------------------------------------------------------

def test_response_array_indexing_matches_a_hand_lookup():
    """Oracle: for every (cell, seed) the value is looked up by scanning the
    rows, independently of the level-index arithmetic under test."""
    runs = make_runs(seed=4)
    fam = D.collect_family_runs(runs, 'ham')
    cells, Y = D.response_array(fam, D.HAM_FACTORS, D.HAM_LEVELS, 'l12_dos')
    assert cells == D.build_cells(D.HAM_FACTORS, D.HAM_LEVELS)
    assert Y.shape == (32, 3) and np.isfinite(Y).all()
    for ci, cell in enumerate(cells):
        for si, s in enumerate(D.SEEDS):
            want = [r['l12_dos'] for r in fam.values()
                    if r['seed'] == s and in_cell(r, cell, D.HAM_FACTORS)]
            assert len(want) == 1
            assert Y[ci, si] == want[0]


def test_response_array_leaves_a_silent_hole_if_validation_is_bypassed():
    """Shows why validate_structure is load-bearing rather than decorative: on
    a duplicated cell response_array silently overwrites and the missing cell
    becomes NaN, with no exception anywhere."""
    runs = make_runs()
    ham = sorted(k for k, v in runs.items() if v['family'] == 'ham')
    runs[ham[0]] = copy.deepcopy(runs[ham[1]])
    fam = D.collect_family_runs(runs, 'ham')
    _c, Y = D.response_array(fam, D.HAM_FACTORS, D.HAM_LEVELS, 'l12_dos')
    assert np.isnan(Y).any()


def test_analyze_response_flattening_agrees_with_the_design_row_order():
    """Oracle: plant a response that depends on geometry only. Then the
    geometry SS must be exactly 96 * (half the gap)^2 = 96*25 and every other
    term's SS must be zero -- which only holds if y and the design rows are in
    the same order."""
    def resp(r, cell, s, family):
        if r == 'l12_transmission':
            return 10.0 * (cell['geometry'] + 1) + 0.001 * D.SEEDS.index(s)
        return None

    fam = D.collect_family_runs(make_runs(resp_fn=resp), 'ham')
    rr = D.analyze_response(fam, D.HAM_FACTORS, D.HAM_LEVELS,
                            'l12_transmission')
    assert rr.status == 'ok'
    d = {t.term: t for t in rr.terms}
    assert d['geometry'].ss == pytest.approx(96 * 25.0, rel=1e-6)
    for name, t in d.items():
        if name != 'geometry':
            assert abs(t.ss) < 1e-9, name
    assert rr.best_cell['geometry'] == 0 and rr.worst_cell['geometry'] == 1
    assert rr.best_mean == pytest.approx(10.001)
    assert rr.worst_mean == pytest.approx(20.001)
    assert rr.best_worst_ratio == pytest.approx(20.001 / 10.001)


def test_entered_levels_are_read_off_the_data_not_the_design_table():
    """The 'ALWAYS CHECK THIS LINE' guard is only useful if it reports what was
    actually in the report, so it must be derived from the rows."""
    fa = D.analyze_family(make_runs(seed=11), 'ham', D.HAM_FACTORS,
                          D.HAM_LEVELS, ['l12_dos'])
    for f in D.HAM_FACTORS:
        assert fa.entered_levels[f] == sorted(D.HAM_LEVELS[f], key=str)


def test_best_worst_ratio_is_signed_and_can_be_nan():
    """DOCUMENTED (clarity, not a numeric defect): best_worst_ratio is printed
    unconditionally as worst/best. On a sign-changing response it is negative,
    and on a zero best mean it is nan. Oracle: constructed cells with a
    negative and a zero minimum."""
    def sign_flip(r, cell, s, family):
        if r == 'onsite_near':
            return (-1.0 if cell['geometry'] else 1.0) + 0.001 * D.SEEDS.index(s)
        return None

    fam = D.collect_family_runs(make_runs(resp_fn=sign_flip), 'ham')
    rr = D.analyze_response(fam, D.HAM_FACTORS, D.HAM_LEVELS, 'onsite_near')
    assert rr.best_mean < 0 < rr.worst_mean
    assert rr.best_worst_ratio < 0

    def zero_best(r, cell, s, family):
        if r == 'onsite_near':
            return (0.0 if cell['geometry'] == 0 else 1.0) + (
                0.001 if s == D.SEEDS[0] else -0.0005)
        return None

    fam2 = D.collect_family_runs(make_runs(resp_fn=zero_best), 'ham')
    rr2 = D.analyze_response(fam2, D.HAM_FACTORS, D.HAM_LEVELS, 'onsite_near')
    if rr2.best_mean == 0:
        assert np.isnan(rr2.best_worst_ratio)
