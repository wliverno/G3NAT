"""The best checkpoint must be the argmin of the run's OWN objective.

L = a*T + c*[ b*LDOS + (1-b)*DOS ]. Selecting every arm on a fixed metric
containing DOS is wrong wherever DOS is not trained: T only has loss_c=0, and
LDOS+T has loss_b=1 which zeroes the DOS half. Measured on the previous campaign,
the T-only arm's saved checkpoint sat 21.1% off its own optimum and some cells
were saved at epoch 6 of 15,000.
"""
import pytest

import g3nat.training.selection as selection
from g3nat.training.selection import (LDOS_RESIDUE, resolve_selection_metric,
                                      selection_value)

T, D, L = 'val_transmission', 'val_dos', 'val_ldos_residue'


def test_the_ldos_constant_is_named_for_the_target_it_holds():
    """`LDOS` was a generic name holding a target-SPECIFIC value: the 'residue'
    key only. `base_only` runs key off val_ldos_base_only, so anything that read
    the old constant as "the LDOS key" was silently wrong for that target."""
    assert LDOS_RESIDUE == 'val_ldos_residue'
    assert not hasattr(selection, 'LDOS'), \
        'the ambiguous generic name is back'


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


# ------------------------------------------------- selection follows ldos_target

def test_base_only_runs_select_on_the_key_they_actually_measure():
    """`--ldos_target base_only` is SUPPORTED but not exercised by campaign v2 or the
    v3 matrix -- this closes a latent trap, it does not fix a live run. The trainer
    records the measured LDOS under val_ldos_{ldos_target} and pins the OTHER variant
    to nan, so a base_only run selected on val_ldos_residue would have a nan
    criterion every epoch, never update best_unweighted, and publish nothing."""
    name, w = resolve_selection_metric(1.0, 1.0, 1.0, ldos_target='base_only')
    assert w == pytest.approx({T: 1.0, 'val_ldos_base_only': 1.0})
    assert L not in w
    assert 'ldos' in name


def test_ldos_target_defaults_to_residue():
    """Three-argument callers keep the behaviour they had."""
    assert (resolve_selection_metric(1.0, 1.0, 1.0)
            == resolve_selection_metric(1.0, 1.0, 1.0, ldos_target='residue'))


def test_a_base_only_run_gets_a_finite_selection_value_from_a_real_entry():
    """THE POINT OF THE FIX. `entry` is shaped exactly the way
    Trainer._validate_epoch builds it for ldos_target='base_only': the measured
    value under val_ldos_base_only, nan under val_ldos_residue."""
    entry = {'val_transmission': 0.4, 'val_dos': 0.2,
             'val_ldos_residue': float('nan'), 'val_ldos_base_only': 0.3}
    _n, w = resolve_selection_metric(1.0, 1.0, 1.0, ldos_target='base_only')
    got = selection_value(entry, w)
    assert got == pytest.approx(0.7), 'a base_only run cannot select on a nan'


def test_an_unknown_ldos_target_is_rejected():
    """Silently building val_ldos_typo would give a KeyError thousands of epochs
    later, inside the epoch loop, instead of at construction."""
    with pytest.raises(ValueError):
        resolve_selection_metric(1.0, 1.0, 1.0, ldos_target='base-only')


def test_ldos_target_is_irrelevant_when_ldos_is_untrained():
    """b=0 drops the LDOS term entirely, so the target cannot matter."""
    assert (resolve_selection_metric(1.0, 0.0, 1.0, ldos_target='base_only')
            == resolve_selection_metric(1.0, 0.0, 1.0, ldos_target='residue'))
