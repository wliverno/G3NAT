"""Benjamini-Hochberg step-up over one family of tests.

Oracle: a from-scratch step-up written from the definition
k* = max{ i : p_(i) <= q*i/m }, reject the k* smallest, plus hand-worked
vectors whose arithmetic is spelled out in the docstrings.
"""
import numpy as np
import pytest

import doe_v3 as D


def bh_oracle(p, q=0.05):
    """Independent step-up BH: returns (k*, threshold, rejected index set)."""
    p = list(p)
    m = len(p)
    if m == 0:
        return 0, 0.0, set()
    order = sorted(range(m), key=lambda j: p[j])
    k = 0
    for i in range(1, m + 1):
        if p[order[i - 1]] <= q * i / m:
            k = i
    return k, (q * k / m if k else 0.0), set(order[:k])


def _check(pv, q=0.05):
    k, thr, mask = D.benjamini_hochberg(pv, q=q)
    ek, ethr, eset = bh_oracle(pv, q=q)
    assert k == ek, f"k*: code={k} oracle={ek} for {pv}"
    assert thr == pytest.approx(ethr)
    assert set(np.flatnonzero(mask).tolist()) == eset
    return k, thr, mask


@pytest.mark.parametrize('pv', [
    [],
    [0.04],
    [0.06],
    [0.9, 0.5, 0.6, 0.77, 0.4],
    [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
    [0.30, 0.001, 0.55, 0.004, 0.02, 0.80, 0.09, 0.005, 0.15, 0.03],
    [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216],
    [0.05, 0.05],
    [0.04, 0.04],
    [0.01, 0.02, 0.02],
    [0.9, 0.8, 0.7],
    [0.0, 0.0, 1.0],
    [0.005] * 20,
])
def test_bh_matches_the_independent_step_up_oracle(pv):
    _check(pv)


def test_bh_on_the_empty_family_is_a_defined_no_op():
    k, thr, mask = D.benjamini_hochberg([])
    assert k == 0 and thr == 0.0 and mask.shape == (0,)


def test_bh_textbook_ten_value_case_worked_by_hand():
    """Hand arithmetic, m=10, q=0.05. sorted p: .001 .004 .005 .02 .03 .09 .15
    .30 .55 .80; critical q*i/m: .005 .010 .015 .020 .025 .030 ...
    i=1..4 pass, i=5 (.03 > .025) fails and nothing later passes -> k*=4."""
    pv = [0.30, 0.001, 0.55, 0.004, 0.02, 0.80, 0.09, 0.005, 0.15, 0.03]
    k, thr, mask = _check(pv)
    assert k == 4
    assert thr == pytest.approx(0.02)
    assert sorted(p for p, m in zip(pv, mask) if m) == pytest.approx(
        [0.001, 0.004, 0.005, 0.02])


def test_bh_retains_a_p_value_that_fails_its_own_threshold():
    """The step-up trap, worked by hand. m=3, q=0.05, thresholds .01667 .03333
    .05 for sorted p .009 .045 .049: rank 2 FAILS (.045 > .03333) but rank 3
    passes (.049 <= .05), so k*=3 and the failing .045 is still rejected. A
    step-DOWN implementation would stop at rank 1."""
    k, thr, mask = _check([0.045, 0.009, 0.049])
    assert k == 3 and mask.all()
    assert thr == pytest.approx(0.05)


def test_bh_retains_below_a_later_crossing_in_a_longer_family():
    """m=5, q=0.05, thresholds .01 .02 .03 .04 .05 for sorted p .001 .040 .041
    .042 .049: only rank 1 and rank 5 pass, so k*=5 and everything is kept even
    though p_(2)=.040 exceeds its own .02."""
    pv = [0.001, 0.040, 0.041, 0.042, 0.049]
    k, _thr, mask = _check(pv)
    assert k == 5 and mask.all()
    assert 0.040 > 0.05 * 2 / 5


def test_bh_never_splits_a_tie_group():
    """Oracle: if the lower rank of a tie group passes, the higher rank of the
    same value passes a LARGER threshold, so k* can never cut through a tie.
    Checked on fixed vectors and on 200 randomized all-duplicated vectors."""
    for pv in ([0.02, 0.02, 0.02, 0.9, 0.9], [0.01] * 4,
               [0.03, 0.03, 0.001, 0.5, 0.5, 0.5], [0.05, 0.05]):
        _k, _thr, mask = _check(pv)
        arr = np.asarray(pv)
        for v in set(pv):
            sel = mask[arr == v]
            assert sel.all() or not sel.any(), f"tie at p={v} split in {pv}"

    rng = np.random.default_rng(3)
    for _ in range(200):
        base = np.round(rng.uniform(0, 0.2, size=6), 3)
        pv = np.concatenate([base, base]).tolist()
        _k, _thr, mask = D.benjamini_hochberg(pv, q=0.05)
        arr = np.asarray(pv)
        for v in set(pv):
            sel = mask[arr == v]
            assert sel.all() or not sel.any()


def test_bh_threshold_partitions_survivors_from_non_survivors():
    """Oracle: thr = q*k*/m, and the survivor set is exactly the p-values at or
    below it -- so the printed threshold is the rule's own threshold, not a
    nominal alpha. 200 random families of 30."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        pv = rng.uniform(0, 1, size=30)
        k, thr, mask = D.benjamini_hochberg(pv, q=0.05)
        if k:
            assert thr == pytest.approx(0.05 * k / 30)
            assert (pv[mask] <= thr + 1e-15).all()
            assert (pv[~mask] > thr - 1e-15).all()
            assert pv[mask].max() <= pv[~mask].min() if (~mask).any() else True
