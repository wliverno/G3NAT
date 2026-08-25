"""The weights snapshot must follow the run's OWN objective.

An earlier draft of this change edited a line that feeds only a nan counter,
leaving selection on val_dos_t_unweighted while the checkpoint recorded otherwise.
This test fails if that happens: the two criteria are made to disagree about which
epoch is best, and the run must pick its own.
"""
import ast
import os

import numpy as np
import pytest

from g3nat.training.selection import resolve_selection_metric, selection_value

_TRAINER = os.path.join(os.path.dirname(__file__), '..', '..',
                        'g3nat', 'training', 'trainer.py')


def test_own_objective_and_the_old_fixed_metric_pick_different_epochs():
    """A T-only run: loss_c=0, so DOS is untrained and must not count.

    Epoch 0 has the better DOS+T sum; epoch 1 has the better transmission. A run
    training only transmission must choose epoch 1.
    """
    history = [
        {'val_transmission': 0.90, 'val_dos': 0.10,
         'val_ldos_residue': float('nan'), 'val_dos_t_unweighted': 1.00},
        {'val_transmission': 0.40, 'val_dos': 0.80,
         'val_ldos_residue': float('nan'), 'val_dos_t_unweighted': 1.20},
    ]
    _name, weights = resolve_selection_metric(1.0, 0.0, 0.0)
    own = [selection_value(h, weights) for h in history]
    old = [h['val_dos_t_unweighted'] for h in history]
    assert int(np.argmin(own)) == 1, 'own objective must prefer epoch 1'
    assert int(np.argmin(old)) == 0, 'fixture is pointless unless they disagree'


def test_an_ldos_arm_is_not_selected_on_dos():
    """loss_b=1 zeroes the DOS half of the bracket, so a huge DOS must not matter."""
    _name, weights = resolve_selection_metric(1.0, 1.0, 1.0)
    entry = {'val_transmission': 1.0, 'val_dos': 1e6, 'val_ldos_residue': 2.0}
    assert selection_value(entry, weights) == pytest.approx(3.0)


def _fit_body():
    """The statements of Trainer.fit, which is where the snapshot is taken."""
    with open(_TRAINER) as fh:
        tree = ast.parse(fh.read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == 'fit')


def test_the_snapshot_branch_reads_the_resolved_objective_not_a_fixed_key():
    """THE 198-vs-512 GUARD.

    `best_unweighted['state_dict'] = ...` is the assignment that decides which
    weights get published. The `metric` it is gated on must come from
    selection_value(), not from a hardcoded metric_history key -- editing the
    nan-counter line at the bottom of _validate_epoch instead would leave
    selection on val_dos_t_unweighted while the checkpoint claimed otherwise.
    """
    fit = _fit_body()
    calls = {n.func.id for n in ast.walk(fit)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert 'selection_value' in calls, \
        'Trainer.fit never calls selection_value; the real selection path is untouched'
    src = ast.unparse(fit)
    assert 'val_dos_t_unweighted' not in src, \
        'Trainer.fit still selects on the fixed metric val_dos_t_unweighted'
