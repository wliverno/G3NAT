"""compute_norm_stats(cache, sequences=...) must restrict the stats to the given
sequences (e.g. train split only), so held-out geometry cannot leak into the
z-score normalization. Added retroactively for a filter that landed in 1633ca5
with no test coverage.
"""
import numpy as np
from g3nat.graph.geometry import compute_norm_stats


def _entry(bp_val, step_val, n=4):
    """A cache entry whose backbone/hbond 7-tuples are all `step_val`/`bp_val`
    (up to the centroid-distance slot 0, which comes from the centroids)."""
    return {
        "bp_pars": np.full((n, 6), bp_val, dtype=float),
        "step_pars": np.full((n - 1, 6), step_val, dtype=float),
        "primary_centroids": np.array([[0.0, 0.0, i * 3.4] for i in range(n)], dtype=float),
        "comp_centroids": np.array([[6.0, 0.0, (n - 1 - i) * 3.4] for i in range(n)], dtype=float),
    }


def _cache():
    # Two sequences whose bp/step geometry values differ substantially.
    return {
        "acgt": _entry(bp_val=1.0, step_val=2.0),
        "ggcc": _entry(bp_val=100.0, step_val=200.0),
    }


def test_filtered_stats_differ_from_unfiltered():
    cache = _cache()
    unfiltered = compute_norm_stats(cache)
    filtered = compute_norm_stats(cache, sequences=["acgt"])

    # unfiltered mixes both sequences -> mean is pulled toward the ggcc values;
    # filtered to acgt alone must not be.
    assert not np.allclose(unfiltered["backbone"]["mean"], filtered["backbone"]["mean"])
    assert not np.allclose(unfiltered["hbond"]["mean"], filtered["hbond"]["mean"])


def test_filtered_stats_match_singleton_cache():
    cache = _cache()
    filtered = compute_norm_stats(cache, sequences=["acgt"])
    singleton = compute_norm_stats({"acgt": cache["acgt"]})

    for edge_type in ("backbone", "hbond"):
        assert np.allclose(filtered[edge_type]["mean"], singleton[edge_type]["mean"])
        assert np.allclose(filtered[edge_type]["std"], singleton[edge_type]["std"])


def test_filter_is_case_insensitive_and_ignores_absent_sequences():
    cache = _cache()
    filtered_upper = compute_norm_stats(cache, sequences=["ACGT", "not_in_cache"])
    filtered_lower = compute_norm_stats(cache, sequences=["acgt"])

    for edge_type in ("backbone", "hbond"):
        assert np.allclose(filtered_upper[edge_type]["mean"], filtered_lower[edge_type]["mean"])
