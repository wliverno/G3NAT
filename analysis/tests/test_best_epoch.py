"""The training-time channel: the selected epoch, its log10, and its own BH
family.

Oracles: np.log10 recomputed on the cache values, hand-built cache corruptions,
and a planted multiplicative epoch effect.
"""
import inspect

import numpy as np
import pytest

import doe_v3 as D
from conftest import make_runs, make_ci, make_epochs, run_analysis


def test_merge_attaches_the_raw_and_log10_epoch_without_mutating_the_input():
    """Oracle: log10 of the cache's own value, recomputed here."""
    runs = make_runs(seed=40)
    cache = make_epochs(runs, seed=400)
    merged = D.merge_best_epoch(runs, cache)
    assert len(merged) == 120
    for name, row in merged.items():
        assert row['best_epoch'] == pytest.approx(cache[name]['best_epoch'])
        assert row['log10_best_epoch'] == pytest.approx(
            np.log10(cache[name]['best_epoch']))
    assert 'best_epoch' not in next(iter(runs.values()))


def test_a_cache_written_against_a_different_run_set_aborts():
    runs = make_runs(seed=41)
    cache = make_epochs(runs)
    cache.pop(next(iter(cache)))
    with pytest.raises(D.StructureError) as e:
        D.merge_best_epoch(runs, cache)
    assert 'stale' in str(e.value)


@pytest.mark.parametrize('factor', ['geometry', 'num_layers', 'supervision'])
def test_a_cache_disagreeing_on_any_shared_factor_aborts(factor):
    runs = make_runs(seed=42)
    cache = make_epochs(runs)
    k = next(iter(cache))
    cache[k][factor] = 'WRONG' if factor == 'supervision' else 99
    with pytest.raises(D.StructureError) as e:
        D.merge_best_epoch(runs, cache)
    assert 'disagrees' in str(e.value)


@pytest.mark.parametrize('bad', [0, -5])
def test_a_non_positive_selected_epoch_aborts(bad):
    runs = make_runs(seed=43)
    cache = make_epochs(runs)
    cache[next(iter(cache))]['best_epoch'] = bad
    with pytest.raises(D.StructureError):
        D.merge_best_epoch(runs, cache)


def test_n_orb_is_cross_checked_on_ham_rows_only():
    """The blind family has no n_orb, so a bogus n_orb on a blind cache entry
    is correctly ignored while the same corruption on a ham entry aborts."""
    runs = make_runs(seed=44)
    cache = make_epochs(runs)
    blind = next(k for k, v in runs.items() if v['family'] == 'blind')
    cache[blind]['n_orb'] = 99
    D.merge_best_epoch(runs, cache)

    cache2 = make_epochs(runs)
    ham = next(k for k, v in runs.items() if v['family'] == 'ham')
    cache2[ham]['n_orb'] = 99
    with pytest.raises(D.StructureError):
        D.merge_best_epoch(runs, cache2)


def test_the_arm_table_is_a_mean_over_cell_means_with_their_range():
    """Oracle: each arm's value is the mean of its 8 cell means and the range
    is the min/max of those same cell means, so lo <= mean <= hi always."""
    runs = make_runs(seed=49)
    merged = D.merge_best_epoch(runs, make_epochs(runs, seed=409))
    arm, n_cells = D.best_epoch_arm_table(merged, 'ham', D.HAM_FACTORS,
                                          D.HAM_LEVELS)
    assert n_cells == 8
    assert set(arm) == set(D.HAM_LEVELS['supervision'])
    _c, Y = D.response_array(D.collect_family_runs(merged, 'ham'),
                             D.HAM_FACTORS, D.HAM_LEVELS, 'best_epoch')
    cells = D.build_cells(D.HAM_FACTORS, D.HAM_LEVELS)
    means = Y.mean(axis=1)
    for lv, (mean, lo, hi) in arm.items():
        sel = [i for i, c in enumerate(cells) if str(c['supervision']) == lv]
        assert mean == pytest.approx(float(means[sel].mean()))
        assert lo == pytest.approx(float(means[sel].min()))
        assert hi == pytest.approx(float(means[sel].max()))
        assert lo <= mean <= hi


def test_a_planted_epoch_effect_is_recovered_and_survives_bh():
    """Oracle: runs at num_layers=4 get 4x the epoch, a factor-of-four shift on
    a log scale that no seed jitter of this size can mimic."""
    runs = make_runs(seed=48)
    rng = np.random.default_rng(408)
    cache = make_epochs(runs, epoch_fn=lambda r: int(
        400 * (4.0 if r['num_layers'] == 4 else 1.0)
        * np.exp(0.05 * rng.standard_normal())))
    merged = D.merge_best_epoch(runs, cache)
    fa = D.analyze_family(merged, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          list(D.BEST_EPOCH_RESPONSES))
    t = next(x for x in fa.responses['log10_best_epoch'].terms
             if x.term == 'num_layers')
    assert t.p < 1e-6
    assert ('log10_best_epoch', 'num_layers', t.p) in fa.bh_survivors


def test_the_printed_best_epoch_family_size_is_derived_not_hardcoded():
    """FIX (label defect 3). print_best_epoch_section used to state the family
    size as the literal prose '15 terms x 2 responses = 30 for ham, 7 x 2 = 14
    for blind', which could disagree with the BH line printed a few lines
    below it. Oracle: no literal family-size arithmetic may appear in the
    function's source, and the numbers it prints must equal the m on each
    family's own BENJAMINI-HOCHBERG line."""
    src = inspect.getsource(D.print_best_epoch_section)
    assert '15 terms x 2 responses = 30' not in src
    assert '7 x 2 = 14' not in src

    runs = make_runs(seed=46)
    merged = D.merge_best_epoch(runs, make_epochs(runs, seed=406))
    lines = []
    res = D.print_best_epoch_section(lines.append, merged)
    txt = '\n'.join(lines)

    assert res['ham'].bh_m == 30 and res['blind'].bh_m == 14
    model_line = next(l for l in lines if l.startswith('Model:'))
    assert '= 30 for ham' in model_line and '= 14 for blind' in model_line
    bh = [l for l in lines
          if l.startswith("BENJAMINI-HOCHBERG (q=0.05) over this family's")]
    assert bh[0].endswith('m=30') and bh[1].endswith('m=14')
    assert 'DESCRIPTIVE RESPONSE -- DIRECTION CARRIES NO QUALITY JUDGEMENT' \
        in txt
    assert 'RESPONSE: best_epoch' in txt and 'RESPONSE: log10_best_epoch' in txt
    assert 'MEAN SELECTED EPOCH BY SUPERVISION ARM -- ham' in txt
    assert 'MEAN SELECTED EPOCH BY SUPERVISION ARM -- blind' in txt


def test_best_epoch_joins_its_own_dispersion_family():
    """Oracle: with the cache present the ham dispersion family gains one
    response (7 location + best_epoch = 8) and the blind one gains one
    (5 + 1 = 6); the fitted model itself is unchanged."""
    runs = make_runs(seed=47)
    ci = make_ci(runs)
    cache = make_epochs(runs, seed=407)
    lines = []
    res = run_analysis({'runs': runs}, ci, cache, lines.append)
    assert 'DISPERSION RESPONSE: best_epoch' in '\n'.join(lines)
    assert res['ham_dispersion'].bh_m == 10 * 8
    assert res['blind_dispersion'].bh_m == 3 * 6
    assert res['ham_dispersion'].df_res == 13
    assert res['blind_dispersion'].df_res == 4


@pytest.mark.xfail(strict=True, reason=(
    "finding 5: best_epoch and log10_best_epoch are one test counted twice "
    "(m=30 should be 15); decision pending"))
def test_best_epoch_and_its_own_log_are_not_pooled_as_two_independent_tests():
    """BEST_EPOCH_RESPONSES pools a quantity and its own base-10 log into a
    single BH family, m = 15 x 2 = 30 for ham. A monotone transform of a
    response is not a second test: the 30 tests are ~15 distinct ones and the
    BH threshold is about half what the information content justifies -- the
    same 'a near-copy would only pad the family' argument the script itself
    uses to keep ci_interior_total out of the ham family.

    Oracle: the two responses' per-term p-value orderings are identical, so
    they carry the same evidence; the family should be m=15.

    Left as a strict xfail: changing it changes a published survivor list and
    is a design decision, not the test suite's.
    """
    runs = make_runs(seed=99)
    rng = np.random.default_rng(99)
    cache = make_epochs(runs, epoch_fn=lambda r: int(
        200 + 60 * r['num_layers'] + 40 * r['geometry']
        + rng.integers(1, 30)))
    merged = D.merge_best_epoch(runs, cache)
    fa = D.analyze_family(merged, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          list(D.BEST_EPOCH_RESPONSES))
    p_raw = {t: p for r, t, p in fa.all_tested if r == 'best_epoch'}
    p_log = {t: p for r, t, p in fa.all_tested if r == 'log10_best_epoch'}
    terms = sorted(p_raw)
    assert list(np.argsort([p_raw[t] for t in terms])) == \
        list(np.argsort([p_log[t] for t in terms])), "the two agree term by term"
    assert fa.bh_m == 15, (
        f"m={fa.bh_m}: a response and its own log10 are pooled as two "
        f"independent tests")
