"""The best checkpoint must be the argmin of the run's OWN objective.

L = a*T + c*[ b*LDOS + (1-b)*DOS ]. Selecting every arm on a fixed metric
containing DOS is wrong wherever DOS is not trained: T only has loss_c=0, and
LDOS+T has loss_b=1 which zeroes the DOS half. Measured on the previous campaign,
the T-only arm's saved checkpoint sat 21.1% off its own optimum and some cells
were saved at epoch 6 of 15,000.
"""
import pytest

from g3nat.training.selection import resolve_selection_metric

T, D, L = 'val_transmission', 'val_dos', 'val_ldos_residue'


def test_dos_plus_t():
    name, w = resolve_selection_metric(1.0, 0.0, 1.0)
    assert w == {T: 1.0, D: 1.0}
    assert L not in w


def test_ldos_dos_t():
    name, w = resolve_selection_metric(1.0, 0.5, 1.0)
    assert w == pytest.approx({T: 1.0, L: 0.5, D: 0.5})


def test_ldos_plus_t_has_no_dos_term():
    """b=1 zeroes the DOS half. A DOS term here is the exact defect being fixed."""
    name, w = resolve_selection_metric(1.0, 1.0, 1.0)
    assert w == pytest.approx({T: 1.0, L: 1.0})
    assert D not in w, 'b=1 means DOS is not trained; it must not select on DOS'


def test_t_only_has_neither_dos_nor_ldos():
    """c=0 zeroes the whole bracket."""
    name, w = resolve_selection_metric(1.0, 0.0, 0.0)
    assert w == {T: 1.0}
    assert D not in w and L not in w


def test_zero_weight_terms_are_dropped_not_kept_at_zero():
    """A term at weight 0 contributes nothing but would still let a nan in that
    metric poison the sum. Drop it."""
    _n, w = resolve_selection_metric(1.0, 0.0, 0.0)
    assert all(v > 0 for v in w.values())


def test_a_run_that_trains_nothing_is_rejected():
    with pytest.raises(ValueError):
        resolve_selection_metric(0.0, 0.0, 0.0)


def test_name_is_stable_and_descriptive():
    n1, _ = resolve_selection_metric(1.0, 1.0, 1.0)
    n2, _ = resolve_selection_metric(1.0, 1.0, 1.0)
    assert n1 == n2
    assert 'transmission' in n1 and 'ldos' in n1 and 'dos+' not in n1
