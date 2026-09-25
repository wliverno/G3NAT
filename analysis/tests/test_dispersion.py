"""The dispersion channel: y = max - min over the 3 seeds, one observation per
cell, reduced model, own BH family.

Oracles: hand-computed ranges, the closed-form 2^3 factorial contrast formula
for the blind family, and spread surfaces constructed to an exactly known
shape.
"""
import itertools

import numpy as np
import pytest
from scipy import stats as sps

import doe_v3 as D
from conftest import cells_of, make_runs, make_ci


def make_spread_rows(family, factors, levels, response, spread_for_cell,
                     base=1.0):
    """Rows whose three seed values are base-d/2, base, base+d/2, so the cell
    spread is EXACTLY spread_for_cell(cell) with no noise to contaminate it."""
    rows = {}
    for cell in cells_of(factors, levels):
        d = spread_for_cell(cell)
        for s, v in zip(D.SEEDS, (base - d / 2.0, base, base + d / 2.0)):
            name = ('disp_' + family + '_'
                    + '_'.join(f"{f}{cell[f]}" for f in factors) + f"_s{s}")
            row = {'family': family, 'seed': s, response: float(v)}
            row.update(cell)
            rows[name] = row
    return rows


def test_cell_spreads_is_the_range_over_seeds_not_a_standard_deviation():
    """Oracle: max-min of each row, by hand. Explicitly NOT the SD and not a
    log; an exact-zero spread is kept, never floored."""
    Y = np.array([[1.0, 4.0, 2.0], [-3.0, -3.0, -3.0], [0.5, 0.25, 1.75]])
    got = D.cell_spreads(Y)
    assert got.tolist() == [3.0, 0.0, 1.5]
    assert not np.allclose(got, Y.std(axis=1))


def test_dispersion_model_order_cells_terms_and_residual_df():
    """Oracle: ham uses mains+two-way on 32 cells (10 terms, 19 parameters,
    df_res 13); blind uses mains only on 8 cells (3 terms, 4 parameters,
    df_res 4). Each family's BH family is its own terms x its own responses."""
    runs = D.merge_ci(make_runs(seed=5), make_ci(make_runs(seed=5)))
    ham = D.analyze_family_dispersion(runs, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                                      list(D.DISPERSION_RESPONSES))
    assert (ham.max_order, ham.n_cells, ham.n_terms, ham.df_res) == (2, 32, 10, 13)
    assert ham.bh_m == 10 * len(D.DISPERSION_RESPONSES)

    blind = D.analyze_family_dispersion(runs, 'blind', D.BLIND_FACTORS,
                                        D.BLIND_LEVELS,
                                        list(D.BLIND_DISPERSION_RESPONSES))
    assert (blind.max_order, blind.n_cells, blind.n_terms, blind.df_res) == \
        (1, 8, 3, 4)
    assert blind.bh_m == 3 * len(D.BLIND_DISPERSION_RESPONSES)


def test_blind_dispersion_p_values_match_the_closed_form_contrast_formula():
    """Oracle: a mains-only model on a 2^3 factorial with one observation per
    cell. Each main-effect SS is 4*sum((arm mean - grand)^2) and the residual
    is the total minus the three mains, on 4 df."""
    runs = make_runs(seed=17)
    fam = D.collect_family_runs(runs, 'blind')
    dr = D.analyze_dispersion_response(fam, D.BLIND_FACTORS, D.BLIND_LEVELS,
                                       'l12_transmission', 1)
    _c, Y = D.response_array(fam, D.BLIND_FACTORS, D.BLIND_LEVELS,
                             'l12_transmission')
    S = (Y.max(axis=1) - Y.min(axis=1)).reshape(2, 2, 2)
    grand = S.mean()
    ss = {'supervision': 4 * ((S.mean(axis=(1, 2)) - grand) ** 2).sum(),
          'num_layers': 4 * ((S.mean(axis=(0, 2)) - grand) ** 2).sum(),
          'geometry': 4 * ((S.mean(axis=(0, 1)) - grand) ** 2).sum()}
    ms_res = (((S - grand) ** 2).sum() - sum(ss.values())) / 4
    got = {t.term: t for t in dr.terms}
    for k, v in ss.items():
        assert got[k].df == 1
        assert got[k].p == pytest.approx(float(sps.f.sf(v / ms_res, 1, 4)),
                                         rel=1e-8)


def test_a_factor_that_doubles_the_spread_is_significant_and_survives_bh():
    """Oracle: geometry=1 cells are given exactly twice the spread of geometry=0
    cells, with a small jitter; no other factor is involved."""
    rng = np.random.default_rng(200)

    def spread_fn(cell):
        base = 0.10 + 0.01 * float(rng.standard_normal())
        return 2 * base if cell['geometry'] == 1 else base

    rows = make_spread_rows('ham', D.HAM_FACTORS, D.HAM_LEVELS,
                            'l16_transmission', spread_fn)
    dr = D.analyze_dispersion_response(rows, D.HAM_FACTORS, D.HAM_LEVELS,
                                       'l16_transmission', 2)
    assert dr.status == 'ok'
    assert next(t for t in dr.terms if t.term == 'geometry').p < 0.01
    fd = D.analyze_family_dispersion(rows, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                                     ['l16_transmission'])
    assert 'geometry' in {t for _r, t, _p in fd.bh_survivors}


def test_an_identical_spread_in_every_cell_is_reported_as_constant():
    """Oracle: every cell spread is 0.15, so the spread surface really is
    constant and the most/least stable cells must both report 0.15."""
    rows = make_spread_rows('ham', D.HAM_FACTORS, D.HAM_LEVELS,
                            'l12_transmission', lambda c: 0.15)
    dr = D.analyze_dispersion_response(rows, D.HAM_FACTORS, D.HAM_LEVELS,
                                       'l12_transmission', 2)
    assert dr.status == 'constant'
    assert dr.terms == []
    assert dr.constant_value == pytest.approx(0.15)
    assert dr.most_stable_spread == pytest.approx(0.15)
    assert dr.least_stable_spread == pytest.approx(0.15)
    fd = D.analyze_family_dispersion(rows, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                                     ['l12_transmission'])
    assert fd.bh_k_star == 0 and fd.bh_survivors == []


def test_an_exactly_additive_spread_surface_is_not_called_constant():
    """FIX (label defect 1, dispersion face). The reduced ham model is
    mains+two-way, so a spread surface that is exactly additive in the four
    factors leaves zero residual variance with 32 DIFFERENT spreads. That must
    report zero residual variance, not a constant spread.

    Oracle: the spreads are built additively here and there are 16 distinct
    values, so any 'constant' claim is false by construction."""
    coef = {'supervision': {'dos': 0.0, 'ldos': 1.0, 'ldosonly': 2.0,
                            'tonly': 3.0},
            'n_orb': {1: 0.0, 2: 0.5},
            'num_layers': {2: 0.0, 4: 0.25},
            'geometry': {0: 0.0, 1: 0.125}}
    rows = make_spread_rows(
        'ham', D.HAM_FACTORS, D.HAM_LEVELS, 'l12_dos',
        lambda c: sum(coef[f][c[f]] for f in D.HAM_FACTORS), base=10.0)
    _c, Y = D.response_array(rows, D.HAM_FACTORS, D.HAM_LEVELS, 'l12_dos')
    spreads = set(np.round(Y.max(axis=1) - Y.min(axis=1), 12))
    assert len(spreads) > 1, "sanity: the spread surface is NOT constant"

    dr = D.analyze_dispersion_response(rows, D.HAM_FACTORS, D.HAM_LEVELS,
                                       'l12_dos', 2)
    assert dr.status == 'zero_residual_variance'
    assert dr.constant_value is None
    assert dr.most_stable_spread == pytest.approx(min(spreads))
    assert dr.least_stable_spread == pytest.approx(max(spreads))

    lines = []
    fd = D.analyze_family_dispersion(rows, 'ham', D.HAM_FACTORS, D.HAM_LEVELS,
                                     ['l12_dos'])
    D.print_family_dispersion(lines.append, fd)
    txt = '\n'.join(lines)
    assert 'ZERO RESIDUAL VARIANCE' in txt
    assert 'CONSTANT spread across all cells' not in txt
    assert fd.bh_m == 0


def test_exact_zero_spreads_enter_the_fit_and_are_never_dropped():
    """No clamping, no cutoff: half the cells are given an exactly zero spread
    and must still be observations, with a finite p on every term."""
    rng = np.random.default_rng(201)
    rows = make_spread_rows(
        'ham', D.HAM_FACTORS, D.HAM_LEVELS, 'val_transmission_at_selection',
        lambda c: 0.0 if c['geometry'] == 0
        else 0.08 + 0.01 * float(rng.standard_normal()))
    dr = D.analyze_dispersion_response(
        rows, D.HAM_FACTORS, D.HAM_LEVELS, 'val_transmission_at_selection', 2)
    assert dr.status == 'ok'
    assert dr.n_cells == 32
    assert dr.most_stable_spread == pytest.approx(0.0)
    for t in dr.terms:
        assert np.isfinite(t.p)


def test_a_model_order_too_high_for_the_family_aborts():
    """Oracle: order 3 on the blind family's 8 cells needs 8 parameters, so
    df_res would be 0 and no dispersion p-value would be defined."""
    fam = D.collect_family_runs(make_runs(seed=6), 'blind')
    with pytest.raises(D.StructureError):
        D.analyze_dispersion_response(fam, D.BLIND_FACTORS, D.BLIND_LEVELS,
                                      'l12_transmission', 3)


def test_the_dispersion_response_list_is_the_location_list_minus_descriptives():
    """Ruling 2026-09-11: the dispersion channel covers exactly the responses
    the LOCATION channel F-tests, so the two lists are kept identical by
    construction rather than by hand, and the descriptive-only responses never
    enter. The three transmission responses stay first so the pre-extension
    blocks print in the same order."""
    location = (set(D.RESPONSES_COMMON) | {D.HAM_EXTRA_RESPONSE}
                | set(D.CI_TESTED_RESPONSES))
    assert set(D.DISPERSION_RESPONSES) == location
    assert set(D.BLIND_DISPERSION_RESPONSES) == set(D.RESPONSES_COMMON)
    for r in D.CI_DESCRIPTIVE_ONLY + D.ONSITE_DESCRIPTIVE_ONLY:
        assert r not in D.DISPERSION_RESPONSES
    assert D.DISPERSION_RESPONSES[:3] == (
        'val_transmission_at_selection', 'l12_transmission',
        'l16_transmission')


def test_dispersion_responses_are_filtered_to_what_every_row_actually_carries():
    """Oracle: without a CI merge no row has ci_interior_mode, so it must drop
    out of the entered list; after a merge it must appear, and its
    descriptive-only twin must not."""
    runs = make_runs(seed=34)
    ham = D.collect_family_runs(runs, 'ham')
    assert D.dispersion_responses_for('ham', ham) == [
        'val_transmission_at_selection', 'l12_transmission',
        'l16_transmission', 'l12_dos', 'l16_dos', 'onsite_near']
    assert D.dispersion_responses_for('blind',
                                      D.collect_family_runs(runs, 'blind')) == [
        'val_transmission_at_selection', 'l12_transmission',
        'l16_transmission', 'l12_dos', 'l16_dos']

    merged = D.merge_ci(runs, make_ci(runs))
    got = D.dispersion_responses_for('ham',
                                     D.collect_family_runs(merged, 'ham'))
    assert got[-1] == 'ci_interior_mode'
    assert 'ci_interior_total' not in got
    assert D.analyze_family_dispersion(merged, 'ham', D.HAM_FACTORS,
                                       D.HAM_LEVELS, got).bh_m == 10 * 7


def test_per_supervision_arm_mean_spread_is_a_plain_average_over_its_cells():
    """Oracle: each arm's 8 ham cells are given one exact spread, so the arm
    mean must be that value."""
    want = {'dos': 0.1, 'ldos': 0.2, 'ldosonly': 0.3, 'tonly': 0.4}
    rows = make_spread_rows('ham', D.HAM_FACTORS, D.HAM_LEVELS,
                            'l16_transmission',
                            lambda c: want[c['supervision']])
    dr = D.analyze_dispersion_response(rows, D.HAM_FACTORS, D.HAM_LEVELS,
                                       'l16_transmission', 2)
    assert dr.arm_n_cells == 8
    assert dr.arm_mean_spread == {k: pytest.approx(v) for k, v in want.items()}


def test_an_empty_dispersion_response_list_reports_zero_cells():
    """DOCUMENTED (cosmetic): n_cells/df_res/n_terms are assigned inside the
    per-response loop, so with no responses the header advertises n=0 cells."""
    fd = D.analyze_family_dispersion(make_runs(seed=8), 'blind',
                                     D.BLIND_FACTORS, D.BLIND_LEVELS, [])
    assert (fd.n_cells, fd.df_res, fd.n_terms) == (0, 0, 0)
    lines = []
    D.print_family_dispersion(lines.append, fd)
    assert 'n=0 cells' in '\n'.join(lines)
