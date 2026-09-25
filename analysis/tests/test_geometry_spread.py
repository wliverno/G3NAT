"""Unit tests for geometry_spread.py's edge-row collection and stats table.

Oracles: hand-built tiny caches with known bp_pars/step_pars/centroids, so the
expected per-feature means/SDs and edge counts are computed independently of
the module under test, not copied from a real geometry cache.
"""
import numpy as np
import pytest

import geometry_spread as G


def _entry(step_pars, bp_pars, primary_centroids, comp_centroids):
    return {
        'step_pars': np.asarray(step_pars, dtype=float),
        'bp_pars': np.asarray(bp_pars, dtype=float),
        'primary_centroids': np.asarray(primary_centroids, dtype=float),
        'comp_centroids': np.asarray(comp_centroids, dtype=float),
    }


def _full_duplex_entry(length, step_val, bp_val):
    """A full (no-gap) duplex of the given length: length-1 backbone steps and
    `length` hbond pairs, all rows equal to the given 6-vectors, centroids on
    the x-axis one unit apart so centroid_distance is always 1.0."""
    pc = [[float(i), 0.0, 0.0] for i in range(length)]
    cc = [[float(i), 0.0, 0.0] for i in range(length)]
    step = [step_val] * max(0, length - 1)
    bp = [bp_val] * length
    return _entry(step, bp, pc, cc)


def test_edge_counts_for_a_single_full_duplex():
    """length L: primary backbone L-1, complementary backbone L-1 (since the
    complementary strand is also full length here), hbond L. So backbone total
    = 2*(L-1), pair total = L."""
    e = _full_duplex_entry(4, [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])
    cache = {'seqa': e}
    back, pair = G.collect_rows(cache)
    assert back.shape == (2 * (4 - 1), 7)
    assert pair.shape == (4, 7)


def test_edge_counts_sum_across_sequences():
    cache = {
        'a': _full_duplex_entry(4, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
        'b': _full_duplex_entry(5, [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    }
    back, pair = G.collect_rows(cache)
    assert back.shape[0] == 2 * (4 - 1) + 2 * (5 - 1)
    assert pair.shape[0] == 4 + 5


def test_centroid_distance_is_the_first_column_and_stepbp_fill_the_rest():
    e = _full_duplex_entry(3, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    back, pair = G.collect_rows({'s': e})
    # primary/comp centroids are both unit-spaced on the x-axis (same axis),
    # so every adjacent-step backbone distance is 1.0. Pair i connects
    # primary i to comp (ncomp-1-i), so its distance is |i - (ncomp-1-i)|:
    # for ncomp=3, that is [2.0, 0.0, 2.0].
    assert np.allclose(back[:, 0], 1.0)
    assert np.allclose(pair[:, 0], [2.0, 0.0, 2.0])
    assert np.allclose(back[:, 1:], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert np.allclose(pair[:, 1:], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_mean_and_std_over_two_sequences_matches_hand_computation():
    cache = {
        'a': _full_duplex_entry(3, [0.0] * 6, [0.0] * 6),
        'b': _full_duplex_entry(3, [2.0] * 6, [2.0] * 6),
    }
    back, pair = G.collect_rows(cache)
    table = G.stats_table(back, G.BACKBONE_NAMES)
    # half the step rows are all-0, half all-2 -> mean 1.0, population SD 1.0
    assert table['shift'] == pytest.approx((1.0, 1.0))
    assert table['rise'] == pytest.approx((1.0, 1.0))
    ptable = G.stats_table(pair, G.PAIR_NAMES)
    assert ptable['propeller'] == pytest.approx((1.0, 1.0))


def test_missing_complementary_backbone_when_strands_differ_in_length():
    """A gapped duplex (comp strand shorter) still yields the primary backbone
    edges and only the hbond pairs that both strands actually have -- exactly
    what assemble_graph_geometry defines, since this script must not re-derive
    the edge definition."""
    pc = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    cc = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # only 2 comp bases, so 1 comp step
    e = _entry(step_pars=[[1] * 6, [2] * 6], bp_pars=[[9] * 6, [9] * 6, [9] * 6],
               primary_centroids=pc, comp_centroids=cc)
    back, pair = G.collect_rows({'s': e})
    # primary backbone: 2 steps; complementary backbone: 1 step -> 3 total
    assert back.shape[0] == 3
    # hbond: min(bp_pars rows=3, n=3, ncomp=2) = 2
    assert pair.shape[0] == 2


def test_column_names_match_the_documented_order():
    assert G.BACKBONE_NAMES == ['centroid_distance', 'shift', 'slide', 'rise',
                                'tilt', 'roll', 'twist']
    assert G.PAIR_NAMES == ['centroid_distance', 'shear', 'stretch', 'stagger',
                            'buckle', 'propeller', 'opening']
