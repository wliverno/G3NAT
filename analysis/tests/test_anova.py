"""Type III drop-one F-tests, and the zero-residual-variance guard.

Oracles: a closed-form balanced two-way ANOVA written from marginal means, an
independently computed sequential (Type I) decomposition, projection-matrix
residual sums, scipy's F survival function recomputed in the test, and one
exact rational-arithmetic residual sum of squares.
"""
import itertools
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats as sps

import doe_v3 as D
from conftest import cells_of, make_runs


def _two_way_oracle(y, a, b, r):
    """Textbook balanced two-way ANOVA with replication, from marginal means."""
    Y = y.reshape(a, b, r)
    grand = Y.mean()
    mA, mB, mAB = Y.mean(axis=(1, 2)), Y.mean(axis=(0, 2)), Y.mean(axis=2)
    ss = {'A': b * r * ((mA - grand) ** 2).sum(),
          'B': a * r * ((mB - grand) ** 2).sum(),
          'A:B': r * ((mAB - mA[:, None] - mB[None, :] + grand) ** 2).sum()}
    ss_res = ((Y - mAB[:, :, None]) ** 2).sum()
    df_res = a * b * (r - 1)
    ms_res = ss_res / df_res
    df = {'A': a - 1, 'B': b - 1, 'A:B': (a - 1) * (b - 1)}
    out = {k: (df[k], ss[k], (ss[k] / df[k]) / ms_res,
               float(sps.f.sf((ss[k] / df[k]) / ms_res, df[k], df_res)))
           for k in ss}
    out['_res'] = (df_res, ss_res, ms_res)
    return out


def test_type3_matches_the_closed_form_balanced_two_way_anova():
    """Oracle: 4x2 with 3 replicates, SS/F/p from marginal means only. The
    4-level factor is deliberate -- supervision carries 3 df, so a 1-df-per-
    term shortcut would not reproduce these numbers."""
    factors, levels = ('A', 'B'), {'A': (0, 1, 2, 3), 'B': (10, 11)}
    obs = [c for c in cells_of(factors, levels) for _ in D.SEEDS]
    y = np.random.default_rng(7).normal(5.0, 2.0, size=len(obs))
    X, terms = D.build_design(obs, factors, levels)
    res, ss_res, df_res, ms_res = D.fit_type3(X, terms, y)

    o = _two_way_oracle(y, 4, 2, 3)
    assert df_res == o['_res'][0] == 16
    assert ss_res == pytest.approx(o['_res'][1], rel=1e-10)
    assert ms_res == pytest.approx(o['_res'][2], rel=1e-10)
    got = {t.term: t for t in res}
    for key in ('A', 'B', 'A:B'):
        df, ss, F, p = o[key]
        assert got[key].df == df
        assert got[key].ss == pytest.approx(ss, rel=1e-8)
        assert got[key].F == pytest.approx(F, rel=1e-8)
        assert got[key].p == pytest.approx(p, rel=1e-8)


def test_type3_equals_type1_on_the_balanced_ham_design():
    """Oracle: the sequential (Type I) decomposition, computed from scratch.
    On a balanced complete design the two must coincide term by term, and the
    term SS plus the residual must reconstruct the corrected total SS."""
    obs = [c for c in cells_of(D.HAM_FACTORS, D.HAM_LEVELS) for _ in D.SEEDS]
    rng = np.random.default_rng(11)
    y = rng.normal(size=len(obs)) + np.array(
        [3.0 * (c['geometry'] == 1) + 1.0 * (c['supervision'] == 'tonly')
         for c in obs])
    X, terms = D.build_design(obs, D.HAM_FACTORS, D.HAM_LEVELS)
    res, ss_res, df_res, _ms = D.fit_type3(X, terms, y)

    def sse(cols):
        b, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        r = y - X[:, cols] @ b
        return float(r @ r)

    cols, prev, t1 = [0], None, {}
    prev = sse(cols)
    for name, sl in terms:
        cols = cols + list(range(sl.start, sl.stop))
        cur = sse(cols)
        t1[name] = prev - cur
        prev = cur

    assert df_res == 96 - 32 == 64
    for t in res:
        assert t.ss == pytest.approx(t1[t.term], rel=1e-8, abs=1e-10), t.term
    total = float(((y - y.mean()) ** 2).sum())
    assert sum(t.ss for t in res) + ss_res == pytest.approx(total, rel=1e-10)
    assert sum(t.df for t in res) == 31


def test_type3_is_drop_one_and_independent_of_term_order():
    """Oracle: RSS(model without this term) - RSS(full), formed here with
    projection matrices rather than the script's own lstsq refits."""
    factors = ('A', 'B', 'C')
    levels = {'A': (0, 1, 2, 3), 'B': (0, 1), 'C': (0, 1)}
    obs = [c for c in cells_of(factors, levels) for _ in D.SEEDS]
    y = np.random.default_rng(11).normal(0.0, 1.0, size=len(obs))
    X, terms = D.build_design(obs, factors, levels)
    res, *_ = D.fit_type3(X, terms, y)

    P = X @ np.linalg.pinv(X)
    rss_full = float(y @ (y - P @ y))
    for name, sl in terms:
        keep = [j for j in range(X.shape[1]) if not sl.start <= j < sl.stop]
        Xr = X[:, keep]
        Pr = Xr @ np.linalg.pinv(Xr)
        want = float(y @ (y - Pr @ y)) - rss_full
        got = next(t for t in res if t.term == name)
        assert got.ss == pytest.approx(want, rel=1e-8, abs=1e-12)


def test_p_values_are_upper_tail_finite_and_in_the_unit_interval():
    """Oracle: sps.f.sf(F, df, df_res), recomputed over 25 random responses,
    plus a directional check (a dominating effect gives p ~ 0, a null term
    does not). Guards against a two-sided or lower-tail p."""
    factors, levels = ('A', 'B'), {'A': (0, 1, 2, 3), 'B': (0, 1)}
    obs = [c for c in cells_of(factors, levels) for _ in D.SEEDS]
    X, terms = D.build_design(obs, factors, levels)
    for s in range(25):
        y = np.random.default_rng(1000 + s).normal(0.0, 1.0, size=len(obs))
        res, _ss, df_res, _ms = D.fit_type3(X, terms, y)
        for t in res:
            assert 0.0 <= t.p <= 1.0 and np.isfinite(t.p)
            assert t.p == pytest.approx(float(sps.f.sf(t.F, t.df, df_res)),
                                        rel=1e-9, abs=1e-12)

    rng = np.random.default_rng(3)
    y = np.array([100.0 * c['A'] + rng.normal(0, 0.01) for c in obs])
    res, _ss, df_res, _ms = D.fit_type3(X, terms, y)
    d = {t.term: t for t in res}
    assert d['A'].p < 1e-20
    assert d['B'].p > 0.01


def test_saturated_model_with_one_observation_per_cell_has_no_residual_df():
    """Oracle: 32 parameters on 32 observations leaves df_res = 0, so no F
    ratio exists and the fit must refuse rather than divide by zero."""
    cells = D.build_cells(D.HAM_FACTORS, D.HAM_LEVELS)
    X, terms = D.build_design(cells, D.HAM_FACTORS, D.HAM_LEVELS)
    assert X.shape == (32, 32)
    with pytest.raises(ValueError):
        D.fit_type3(X, terms, np.arange(32, dtype=float))


def test_unbalanced_data_is_rejected_before_any_fitting():
    factors, levels = ('A', 'B'), {'A': (0, 1), 'B': (0, 1)}
    cells = cells_of(factors, levels)
    obs = [c for c in cells for _ in range(3)] + [cells[0]]
    y = np.random.default_rng(1).normal(size=len(obs))
    X, terms = D.build_design(obs, factors, levels)
    with pytest.raises(AssertionError):
        D.fit_type3(X, terms, y)


def test_a_genuinely_constant_response_raises_and_is_labelled_constant():
    """Oracle: every observation identical -> the corrected total SS is exactly
    zero, so there is nothing to test and the response really is constant."""
    obs = [c for c in cells_of(D.BLIND_FACTORS, D.BLIND_LEVELS)
           for _ in D.SEEDS]
    X, terms = D.build_design(obs, D.BLIND_FACTORS, D.BLIND_LEVELS)
    with pytest.raises(D.ConstantResponseError):
        D.fit_type3(X, terms, np.ones(len(obs)))

    runs = make_runs(resp_fn=lambda r, c, s, f:
                     7.5 if r == 'l12_dos' else None)
    fam = D.collect_family_runs(runs, 'ham')
    rr = D.analyze_response(fam, D.HAM_FACTORS, D.HAM_LEVELS, 'l12_dos')
    assert rr.status == 'constant'
    assert rr.constant_value == 7.5


def test_zero_replicate_variance_raises_but_is_not_called_constant():
    """FIX (label defect 1). The guard fires on zero RESIDUAL variance, which
    is what happens when the three seeds agree exactly inside every cell. That
    is not a constant response: here the response takes 32 distinct values.
    Oracle: the set of observed values has 32 members."""
    runs = make_runs(resp_fn=lambda r, c, s, f: (
        1.0 + D.build_cells(D.HAM_FACTORS, D.HAM_LEVELS).index(c)
        if (r == 'l12_dos' and f == 'ham') else None))
    fam = D.collect_family_runs(runs, 'ham')
    observed = {row['l12_dos'] for row in fam.values()}
    assert len(observed) == 32, "sanity: the response is NOT constant"

    rr = D.analyze_response(fam, D.HAM_FACTORS, D.HAM_LEVELS, 'l12_dos')
    assert rr.status == 'zero_replicate_variance'
    assert rr.terms == []
    assert rr.constant_value is None
    assert rr.best_mean == min(observed)
    assert rr.worst_mean == max(observed)


def test_the_zero_variance_guard_is_scale_free():
    """FIX (label defect 1). F statistics are invariant under y -> c*y for any
    c > 0, so the verdict and the F values must be too. The old guard,
    ms_res <= max(mean(y**2), 1.0) * 1e-20, was an ABSOLUTE tolerance for any
    response whose values are below 1, and declared small-magnitude responses
    constant. Oracle: the same F values at three scales spanning 1e24."""
    factors, levels = ('A', 'B'), {'A': (0, 1, 2, 3), 'B': (0, 1)}
    obs = [c for c in cells_of(factors, levels) for _ in D.SEEDS]
    X, terms = D.build_design(obs, factors, levels)
    y = np.random.default_rng(13).normal(5.0, 1.0, size=len(obs))

    base, *_ = D.fit_type3(X, terms, y)
    for c in (1e-12, 1e12):
        scaled, *_ = D.fit_type3(X, terms, y * c)
        assert [t.F for t in scaled] == pytest.approx([t.F for t in base],
                                                      rel=1e-6)
        assert [t.p for t in scaled] == pytest.approx([t.p for t in base],
                                                      rel=1e-6)


def test_small_but_genuine_replicate_variance_is_not_declared_constant():
    """FIX (label defect 1). A response with mean 1000 and a seed spread of
    1e-9 has genuine variance six orders of magnitude above float64's relative
    resolution, so an F-test IS defined. Oracle: the residual sum of squares
    about the exact cell means, computed in rational arithmetic, is strictly
    positive."""
    factors, levels = ('A', 'B'), {'A': (0, 1, 2, 3), 'B': (0, 1)}
    obs = [c for c in cells_of(factors, levels) for _ in D.SEEDS]
    rng = np.random.default_rng(99)
    y = np.array([1000.0 + 3.0 * c['A'] for c in obs]) + rng.normal(
        0, 1e-9, size=len(obs))

    fy = [Fraction(v) for v in y]
    rss = Fraction(0)
    for c0 in range(8):
        grp = fy[3 * c0:3 * c0 + 3]
        m = sum(grp) / 3
        rss += sum((g - m) ** 2 for g in grp)
    assert rss > 0

    X, terms = D.build_design(obs, factors, levels)
    res, *_ = D.fit_type3(X, terms, y)
    assert all(np.isfinite(t.F) for t in res)


def test_type3_sums_of_squares_are_never_negative():
    """Oracle: a sum of squares cannot be negative. ss_term is formed by
    SUBTRACTING two independently computed lstsq residual sums, so nothing
    structurally prevents a small negative value for a null term; 2000
    zero-effect fits are scanned for one."""
    factors, levels = ('A', 'B'), {'A': (0, 1, 2, 3), 'B': (0, 1)}
    obs = [c for c in cells_of(factors, levels) for _ in D.SEEDS]
    X, terms = D.build_design(obs, factors, levels)
    worst = np.inf
    for s in range(2000):
        rng = np.random.default_rng(s)
        y = np.array([1000.0 + 3.0 * c['A'] for c in obs]) + rng.normal(
            0, 1e-3, size=len(obs))
        res, *_ = D.fit_type3(X, terms, y)
        worst = min(worst, min(t.ss for t in res))
    assert worst >= 0.0, f"most negative Type III SS observed: {worst:.3g}"


def test_a_planted_main_effect_dominates_every_other_term():
    """Oracle: a +5 shift on one supervision arm against a 0.02 seed wobble.
    With 14 other terms tested at raw alpha one landing under 0.05 by chance is
    expected, so the check is on the F ratio margin, not on the other p's."""
    obs = [c for c in cells_of(D.HAM_FACTORS, D.HAM_LEVELS) for _ in D.SEEDS]
    X, terms = D.build_design(obs, D.HAM_FACTORS, D.HAM_LEVELS)
    rng = np.random.default_rng(0)
    y = np.array([5.0 if c['supervision'] == 'tonly' else 0.0 for c in obs])
    y = y + 0.02 * rng.standard_normal(len(y))
    res, *_ = D.fit_type3(X, terms, y)
    sup = next(t for t in res if t.term == 'supervision')
    assert sup.p < 1e-6
    assert sup.F > 100 * max(t.F for t in res if t.term != 'supervision')
