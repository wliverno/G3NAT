"""Factor coding and design-matrix construction.

Oracles: closed-form contrast algebra (sum-to-zero columns, rank, the
row-wise Kronecker structure of an interaction block, the product-of-(k-1)
term degrees of freedom) and X'X block-diagonality. Nothing here is compared
against a value the script produced.
"""
import itertools

import numpy as np
import pytest

import doe_v3 as D
from conftest import cells_of


@pytest.mark.parametrize('k', [2, 3, 4, 5])
def test_deviation_columns_are_sum_to_zero_and_full_rank(k):
    """Oracle: a deviation (sum-to-zero) basis has k-1 columns, each summing to
    zero over the levels, and together with the intercept spans all k levels."""
    C = D.deviation_columns(k)
    assert C.shape == (k, k - 1)
    assert np.allclose(C.sum(axis=0), 0.0)
    assert np.linalg.matrix_rank(C) == k - 1
    assert np.linalg.matrix_rank(np.hstack([np.ones((k, 1)), C])) == k


def test_deviation_columns_exact_values_for_k4():
    """Oracle: the identity-over-last-row-of-minus-ones matrix, written out."""
    assert np.array_equal(D.deviation_columns(4),
                          np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.],
                                    [-1., -1., -1.]]))


@pytest.mark.parametrize('factors,levels', [
    (D.HAM_FACTORS, D.HAM_LEVELS), (D.BLIND_FACTORS, D.BLIND_LEVELS)])
def test_design_has_one_intercept_and_the_analytic_column_count(factors, levels):
    """Oracle: 1 + sum over non-empty subsets S of prod_{f in S}(k_f - 1)
    columns and 2**len(factors) - 1 terms, with the term slices tiling the
    matrix exactly once in order."""
    obs = [c for c in cells_of(factors, levels) for _ in D.SEEDS]
    X, terms = D.build_design(obs, factors, levels)

    expect_cols, expect_terms = 1, 0
    for r in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, r):
            d = 1
            for f in combo:
                d *= len(levels[f]) - 1
            expect_cols += d
            expect_terms += 1
    assert X.shape[1] == expect_cols
    assert len(terms) == expect_terms == 2 ** len(factors) - 1
    assert np.linalg.matrix_rank(X) == X.shape[1]

    # exactly one all-ones column, and it is column 0
    ones = [j for j in range(X.shape[1]) if np.allclose(X[:, j], 1.0)]
    assert ones == [0]

    pos = 1
    for _n, sl in terms:
        assert sl.start == pos
        pos = sl.stop
    assert pos == X.shape[1]


def test_term_degrees_of_freedom_are_the_product_of_k_minus_one():
    """Oracle: df(S) = prod_{f in S}(k_f - 1); they must sum to n_cells - 1."""
    cells = D.build_cells(D.HAM_FACTORS, D.HAM_LEVELS)
    _X, terms = D.build_design(cells, D.HAM_FACTORS, D.HAM_LEVELS)
    got = {n: sl.stop - sl.start for n, sl in terms}
    for r in range(1, len(D.HAM_FACTORS) + 1):
        for combo in itertools.combinations(D.HAM_FACTORS, r):
            d = 1
            for f in combo:
                d *= len(D.HAM_LEVELS[f]) - 1
            assert got[':'.join(combo)] == d
    assert sum(got.values()) == 31
    dfs = got
    assert dfs['supervision'] == 3 and dfs['n_orb'] == 1
    assert dfs['supervision:n_orb'] == 3
    assert dfs['n_orb:num_layers:geometry'] == 1
    assert dfs['supervision:n_orb:num_layers:geometry'] == 3


def test_interaction_blocks_are_row_wise_kronecker_products_of_parents():
    """Oracle: each interaction block is rebuilt here from deviation_columns
    alone, for every order up to the full four-way, and its column (i,j,...) is
    independently checked as the elementwise product of the parent columns."""
    cells = cells_of(D.HAM_FACTORS, D.HAM_LEVELS)
    X, terms = D.build_design(cells, D.HAM_FACTORS, D.HAM_LEVELS)
    tmap = dict(terms)

    main = {}
    for f in D.HAM_FACTORS:
        C = D.deviation_columns(len(D.HAM_LEVELS[f]))
        idx = [D.HAM_LEVELS[f].index(c[f]) for c in cells]
        main[f] = C[idx]
        assert np.allclose(X[:, tmap[f]], main[f]), f

    for k in range(2, len(D.HAM_FACTORS) + 1):
        for combo in itertools.combinations(D.HAM_FACTORS, k):
            expect = main[combo[0]]
            for f in combo[1:]:
                expect = np.einsum('ni,nj->nij', expect, main[f]).reshape(
                    len(cells), -1)
            got = X[:, tmap[':'.join(combo)]]
            assert got.shape == expect.shape
            assert np.allclose(got, expect), ':'.join(combo)

    A, B = main['supervision'], main['n_orb']
    AB = X[:, tmap['supervision:n_orb']]
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            assert np.allclose(AB[:, i * B.shape[1] + j], A[:, i] * B[:, j])


@pytest.mark.parametrize('factors,levels', [
    (D.HAM_FACTORS, D.HAM_LEVELS), (D.BLIND_FACTORS, D.BLIND_LEVELS)])
def test_balanced_design_is_block_orthogonal_replicated_and_unreplicated(
        factors, levels):
    """Oracle: on a balanced complete design X'X is exactly block diagonal with
    respect to (intercept + term blocks) -- which is what makes Type I = Type
    II = Type III. Checked directly, not via assert_block_orthogonal alone."""
    cells = cells_of(factors, levels)
    for obs in (cells, [c for c in cells for _ in D.SEEDS]):
        X, terms = D.build_design(obs, factors, levels)
        D.assert_block_orthogonal(X, terms)
        blocks = [slice(0, 1)] + [sl for _n, sl in terms]
        G = X.T @ X
        for i, si in enumerate(blocks):
            for j, sj in enumerate(blocks):
                if i != j:
                    assert np.abs(G[si, sj]).max() < 1e-9


def test_one_extra_replicate_destroys_block_orthogonality_and_raises():
    """Oracle: an unbalanced design cannot be block orthogonal, so the guard
    that licenses the Type III shortcut must fire."""
    cells = cells_of(D.BLIND_FACTORS, D.BLIND_LEVELS)
    obs = [c for c in cells for _ in D.SEEDS] + [cells[0]]
    X, terms = D.build_design(obs, D.BLIND_FACTORS, D.BLIND_LEVELS)
    with pytest.raises(AssertionError):
        D.assert_block_orthogonal(X, terms)


def test_build_design_limited_matches_the_fixed_dispersion_model_orders():
    """Oracle: mains+two-way on 4x2x2x2 is 4+6 terms and 1+6+12=19 parameters
    (df_res 13 on 32 cells); mains-only on 2x2x2 is 3 terms, 4 parameters
    (df_res 4 on 8 cells). Both counted by hand here."""
    cells = D.build_cells(D.HAM_FACTORS, D.HAM_LEVELS)
    X, terms = D.build_design_limited(cells, D.HAM_FACTORS, D.HAM_LEVELS,
                                      D.DISPERSION_MAX_ORDER['ham'])
    assert D.DISPERSION_MAX_ORDER['ham'] == 2
    assert len(terms) == 10
    assert X.shape == (32, 19)
    assert np.linalg.matrix_rank(X) == 19
    assert len(cells) - X.shape[1] == 13
    D.assert_block_orthogonal(X, terms)

    bcells = D.build_cells(D.BLIND_FACTORS, D.BLIND_LEVELS)
    Xb, tb = D.build_design_limited(bcells, D.BLIND_FACTORS, D.BLIND_LEVELS,
                                    D.DISPERSION_MAX_ORDER['blind'])
    assert D.DISPERSION_MAX_ORDER['blind'] == 1
    assert len(tb) == 3
    assert Xb.shape == (8, 4)
    assert len(bcells) - Xb.shape[1] == 4


def test_build_cells_is_itertools_product_order_last_factor_fastest():
    """Oracle: itertools.product order, restated here as an explicit pattern."""
    cells = D.build_cells(D.HAM_FACTORS, D.HAM_LEVELS)
    assert len(cells) == 32
    assert [c['geometry'] for c in cells[:4]] == [0, 1, 0, 1]
    assert len({c['supervision'] for c in cells[:8]}) == 1
    assert cells[0]['supervision'] == 'dos'
    assert cells[8]['supervision'] == 'ldos'


def test_hardcoded_run_counts_agree_with_the_level_tables():
    """Oracle: the counts recomputed from HAM_LEVELS/BLIND_LEVELS/SEEDS."""
    n_ham = 1
    for f in D.HAM_FACTORS:
        n_ham *= len(D.HAM_LEVELS[f])
    n_blind = 1
    for f in D.BLIND_FACTORS:
        n_blind *= len(D.BLIND_LEVELS[f])
    assert D.N_HAM_CELLS == n_ham == 32
    assert D.N_BLIND_CELLS == n_blind == 8
    assert D.N_HAM_RUNS == n_ham * len(D.SEEDS) == 96
    assert D.N_BLIND_RUNS == n_blind * len(D.SEEDS) == 24
    assert D.N_TOTAL_RUNS == 120
    assert len(set(D.SEEDS)) == 3
