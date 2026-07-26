# tests/test_evaluation/test_physicality.py
import numpy as np
from g3nat.evaluation.physicality import (onsite_metrics, eig_metrics,
                                          coupling_bandwidth, baseline_distinctness, is_physical_win)

def test_onsite_frac_in_window():
    d = np.array([-0.5, 0.5, -33.0, 2.0])
    m = onsite_metrics(d)
    assert abs(m['frac_in_window'] - 0.5) < 1e-9
    assert m['min'] == -33.0 and m['max'] == 2.0

def test_eig_in_window_counts_eigenvalues():
    H = np.diag([-0.2, 0.3, -20.0])  # eig = diagonal
    assert abs(eig_metrics(H)['frac_eig_in_window'] - 2/3) < 1e-9

def test_coupling_bandwidth_is_max_offdiag():
    H = np.array([[0.0, 0.7], [0.7, 0.0]])
    assert abs(coupling_bandwidth(H) - 0.7) < 1e-9

def test_distinctness_flags_collapse():
    collapsed = np.array([[-0.5], [-0.5], [-0.49], [-0.5]])
    distinct = np.array([[-0.49], [-1.39], [0.0], [-1.12]])
    assert baseline_distinctness(collapsed)['min_pairwise'] < 0.05
    # NOTE: brief specified > 0.4, but the true min pairwise gap among the Roche
    # baseline values (-0.49, -1.39, 0.0, -1.12) is |T - C| = |-1.39 - (-1.12)| = 0.27,
    # so > 0.4 is unreachable with this fixture. Lowered to 0.2 (still well above the
    # collapsed case and below the true 0.27) -- see task-5-report.md for detail.
    assert baseline_distinctness(distinct)['min_pairwise'] > 0.2

def test_win_requires_both_to_improve():
    before = {'frac_in_window': 0.6, 'frac_eig_in_window': 0.6}
    shifted = {'frac_in_window': 0.9, 'frac_eig_in_window': 0.6}   # onsite up, eig flat
    real = {'frac_in_window': 0.9, 'frac_eig_in_window': 0.85}
    assert is_physical_win(before, shifted) is False
    assert is_physical_win(before, real) is True
