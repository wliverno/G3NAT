"""Head-to-head: SELECT on L=12, REPORT on L=16, and the matched-selection-
pressure resampling.

Oracles: cell means recomputed from response_array (never read back off the
HeadToHead object), an independent reimplementation of the resampling loop
with the same analysis seed, and a plant that points l12 and l16 in opposite
directions so any look-ahead at l16 is visible.
"""
import inspect

import numpy as np
import pytest

import doe_v3 as D
from conftest import make_runs


def _cell_means(runs, family, factors, levels, response):
    fam = D.collect_family_runs(runs, family)
    cells, Y = D.response_array(fam, factors, levels, response)
    return cells, Y.mean(axis=1)


def test_the_reported_cell_is_the_l12_argmin_and_its_value_is_that_cell_l16():
    """Oracle: argmin of the l12 cell means and that cell's l16 mean, both
    recomputed here from response_array."""
    runs = make_runs(seed=5)
    h = D.head_to_head(runs)
    for family, factors, levels, cell_attr, val_attr in (
            ('ham', D.HAM_FACTORS, D.HAM_LEVELS,
             'best_ham_cell_on_l12', 'best_ham_l16'),
            ('blind', D.BLIND_FACTORS, D.BLIND_LEVELS,
             'best_blind_cell_on_l12', 'best_blind_l16')):
        cells, m12 = _cell_means(runs, family, factors, levels,
                                 'l12_transmission')
        _c, m16 = _cell_means(runs, family, factors, levels,
                              'l16_transmission')
        i = int(np.argmin(m12))
        assert getattr(h, cell_attr) == D.cell_name(factors, cells[i])
        assert getattr(h, val_attr) == pytest.approx(float(m16[i]))


def test_selection_does_not_peek_at_l16():
    """Oracle: l12 and l16 are planted in OPPOSITE directions, so the l12-best
    cell is the l16-WORST cell. If selection secretly used l16 the reported
    value would be the minimum instead of the maximum."""
    def resp(r, cell, s, family):
        rank = (cell['num_layers'] + 3 * cell['geometry']
                + 7 * cell.get('n_orb', 0)
                + 20 * (D.HAM_LEVELS['supervision'].index(cell['supervision'])
                        if family == 'ham'
                        else D.BLIND_LEVELS['supervision'].index(
                            cell['supervision'])))
        if r == 'l12_transmission':
            return float(rank)
        if r == 'l16_transmission':
            return float(-rank)
        return None

    runs = make_runs(resp_fn=resp)
    h = D.head_to_head(runs)
    assert h.best_ham_l16 == max(h.ham_cell_means_l16.values())
    assert h.best_ham_l16 == h.ham_cell_means_l16[h.best_ham_cell_on_l12]
    assert h.best_blind_l16 == max(h.blind_cell_means_l16.values())
    assert h.best_blind_l16 == h.blind_cell_means_l16[h.best_blind_cell_on_l12]


def test_a_planted_optimum_is_the_cell_that_gets_selected():
    """Oracle: one supervision arm is given a large NEGATIVE (better) shift on
    l12_transmission, so the selected cell must come from that arm."""
    runs = make_runs(seed=7, resp_fn=lambda r, c, s, f: (
        1.0 - 5.0 if (r == 'l12_transmission' and f == 'ham'
                      and c['supervision'] == 'ldosonly') else None))
    h = D.head_to_head(runs)
    assert 'supervision=ldosonly' in h.best_ham_cell_on_l12


def test_the_resampling_is_reproducible_and_reimplementable():
    """Oracle: the loop rewritten here with the same fixed analysis seed --
    8 of 32 without replacement, select that draw's own l12 argmin, report its
    l16 mean."""
    runs = make_runs(seed=101)
    h = D.head_to_head(runs)
    assert np.array_equal(h.resample_l16_distribution,
                          D.head_to_head(runs).resample_l16_distribution)
    assert h.resample_seed == D.RESAMPLE_SEED == 20260825
    assert h.n_resamples == D.N_RESAMPLES == 10000
    assert len(h.resample_l16_distribution) == 10000

    _c, m12 = _cell_means(runs, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          'l12_transmission')
    _c, m16 = _cell_means(runs, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                          'l16_transmission')
    rng = np.random.default_rng(D.RESAMPLE_SEED)
    want = np.empty(D.N_RESAMPLES)
    for i in range(D.N_RESAMPLES):
        idx = rng.choice(32, size=D.N_BLIND_CELLS, replace=False)
        want[i] = m16[idx[np.argmin(m12[idx])]]
    assert np.array_equal(h.resample_l16_distribution, want)

    # every drawn value is some ham cell's l16 mean: no l12 value leaks through
    assert set(np.round(h.resample_l16_distribution, 12)) <= set(
        np.round(m16, 12))


def test_the_draws_are_matched_to_the_blind_family_size_and_can_hit_the_global_best():
    """FIX (comment defect 4). Each draw is 8 of 32 without replacement -- the
    same number of cells the blind family has -- so the globally best cell is
    present in about a quarter of the draws and IS selected whenever it is.
    The comment used to say each draw 'never' selects the global best.

    Oracle: the hypergeometric expectation, 8/32 = 0.25, measured on the
    distribution itself; and the source comment must not claim otherwise.
    """
    runs = make_runs(seed=12)
    h = D.head_to_head(runs)
    frac = float(np.mean(h.resample_l16_distribution == h.best_ham_l16))
    assert 0.15 < frac < 0.40, (
        f"global-best l16 selected in {frac:.3f} of draws; 8/32 = 0.25")
    src = inspect.getsource(D.head_to_head)
    assert 'never the global best' not in src
    assert 'not forced to the global best' in src


def test_the_printed_blind_wins_fraction_matches_its_own_label():
    """Oracle: the label says 'the blind cell's l16 value is still better
    (lower) than the resampled ham draw', which is mean(draw > blind).
    Recomputed from the distribution and parsed back off the printed line."""
    runs = make_runs(seed=13)
    h = D.head_to_head(runs)
    lines = []
    D.print_head_to_head(lines.append, h)
    line = next(l for l in lines if 'still better (lower)' in l)
    printed = float(line.strip().split(':')[-1])
    assert printed == pytest.approx(
        float(np.mean(h.resample_l16_distribution > h.best_blind_l16)),
        abs=5e-5)


def test_the_fraction_is_one_when_blind_is_uniformly_better():
    """Oracle: every blind l16 is 100 lower than every ham l16, so every draw
    must lose and the printed fraction must be exactly 1."""
    def resp(r, cell, s, family):
        if r in ('l12_transmission', 'l16_transmission'):
            base = 100.0 if family == 'ham' else 1.0
            return base + cell['num_layers'] + cell['geometry']
        return None

    h = D.head_to_head(make_runs(resp_fn=resp))
    assert float(np.mean(h.resample_l16_distribution > h.best_blind_l16)) == 1.0
