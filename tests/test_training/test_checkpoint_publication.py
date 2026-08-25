"""Guards around scripts/train.py's checkpoint bookkeeping.

These are the paths a PREEMPTIBLE run exercises on every requeue, and the ones
whose failure is silent: a lost best-weights file, a best that is never
republished, a "Training complete!" printed over a run that produced nothing.
"""
import importlib.util
import math
import os

import torch

_spec = importlib.util.spec_from_file_location(
    "train_script",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py"))
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def _touch(path, payload=None):
    torch.save(payload if payload is not None else {'x': 1}, path)


# ---------------------------------------------------------------- I5: stale best

def test_stale_best_survives_when_a_latest_checkpoint_is_present(tmp_path):
    """RESUME case. Deleting here would throw away a preempted run's best weights
    -- the single line this whole gate exists for."""
    d = str(tmp_path)
    _touch(os.path.join(d, 'checkpoint_latest.pth'))
    _touch(os.path.join(d, 'checkpoint_best.pth'))
    removed = train_script.maybe_clear_stale_best(d)
    assert removed is False
    assert os.path.exists(os.path.join(d, 'checkpoint_best.pth')), \
        "a resuming run just lost its own best weights"


def test_stale_best_is_removed_when_no_latest_checkpoint_exists(tmp_path):
    """FRESH run in a reused dir: the leftover best belongs to another config and
    would be republished under this run's args."""
    d = str(tmp_path)
    _touch(os.path.join(d, 'checkpoint_best.pth'))
    removed = train_script.maybe_clear_stale_best(d)
    assert removed is True
    assert not os.path.exists(os.path.join(d, 'checkpoint_best.pth'))


def test_nothing_to_clear_is_a_no_op(tmp_path):
    assert train_script.maybe_clear_stale_best(str(tmp_path)) is False


# ------------------------------------------------- I6: seeding the running best

def test_best_value_is_seeded_from_disk_not_from_history_ahead_of_it(tmp_path):
    """checkpoint_latest.pth (which carries metric_history) is written BEFORE
    checkpoint_best.pth. A kill between the two leaves the history ahead of the
    weights actually on disk; seeding from the history then sets a bar better than
    the stored weights, and the genuinely-better later epoch is never republished."""
    d = str(tmp_path)
    _touch(os.path.join(d, 'checkpoint_best.pth'),
           {'selection_value': 0.50, 'selection_metric': 'val_dos_t_unweighted'})
    history = [{'val_dos_t_unweighted': 0.80},
               {'val_dos_t_unweighted': 0.50},
               {'val_dos_t_unweighted': 0.20}]  # epoch that never reached disk
    assert train_script.seed_best_value(d, history) == 0.50, \
        "seeded from the history minimum (0.20), which is better than the weights on disk"


def test_best_value_falls_back_to_history_when_no_best_on_disk(tmp_path):
    history = [{'val_dos_t_unweighted': 0.80}, {'val_dos_t_unweighted': 0.20}]
    assert train_script.seed_best_value(str(tmp_path), history) == 0.20


def test_best_value_falls_back_when_the_key_is_absent(tmp_path):
    """Checkpoints written before 'selection_value' existed."""
    d = str(tmp_path)
    _touch(os.path.join(d, 'checkpoint_best.pth'), {'epoch': 3})
    history = [{'val_dos_t_unweighted': 0.20}]
    assert train_script.seed_best_value(d, history) == 0.20


def test_best_value_is_inf_with_nothing_to_seed_from(tmp_path):
    assert math.isinf(train_script.seed_best_value(str(tmp_path), []))
    assert math.isinf(train_script.seed_best_value(
        str(tmp_path), [{'val_dos_t_unweighted': float('nan')}]))


# --------------------------------------------- C2: no best checkpoint published

def test_warning_when_the_selection_metric_was_nonfinite_every_epoch(tmp_path):
    best = os.path.join(str(tmp_path), 'checkpoint_best.pth')
    # nan_selection_metric_total is CUMULATIVE and counts the run's OWN criterion.
    history = [{'nan_selection_metric_total': float(i + 1)} for i in range(4)]
    msg = train_script.best_publication_warning(best, history,
                                                selection_metric_name='transmission*1')
    assert msg is not None, \
        "the run published no best checkpoint and said nothing about it"
    assert 'WARNING' in msg and 'NON-FINITE' in msg and 'transmission*1' in msg
    assert '4' in msg


def test_warning_when_no_validation_epoch_ran(tmp_path):
    msg = train_script.best_publication_warning(
        os.path.join(str(tmp_path), 'checkpoint_best.pth'), [])
    assert msg is not None and 'WARNING' in msg


def test_warning_names_a_partial_nonfinite_cause(tmp_path):
    history = [{'nan_selection_metric_total': 1.0},
               {'nan_selection_metric_total': 1.0}]
    msg = train_script.best_publication_warning(
        os.path.join(str(tmp_path), 'checkpoint_best.pth'), history)
    assert msg is not None and '1 of 2' in msg


def test_no_warning_when_a_best_checkpoint_exists(tmp_path):
    best = os.path.join(str(tmp_path), 'checkpoint_best.pth')
    _touch(best, {'selection_value': 0.1})
    history = [{'nan_selection_metric_total': 0.0}]
    assert train_script.best_publication_warning(best, history) is None


def test_the_diagnosis_counts_the_runs_own_criterion_not_val_dos_t(tmp_path):
    """THE WRONG-CAUSE BUG. For an LDOS+T run whose own criterion is nan every
    epoch, the fixed metric val_dos_t_unweighted can be FINITE every epoch.
    Counting the old key tells the operator the metric was fine and blames the
    checkpoint callback -- entirely the wrong subsystem."""
    history = [{'val_dos_t_unweighted': 0.5,
                'nan_selection_metric_total': float(i + 1)} for i in range(3)]
    msg = train_script.best_publication_warning(
        os.path.join(str(tmp_path), 'checkpoint_best.pth'), history,
        selection_metric_name='ldos*1+transmission*1')
    assert 'NON-FINITE in all 3 epochs' in msg, \
        'diagnosed from val_dos_t_unweighted, which was finite the whole run'
    assert 'checkpoint callback' not in msg, 'blamed the wrong subsystem'
    assert 'ldos*1+transmission*1' in msg, \
        'the message must name the metric actually in use'


def test_the_diagnosis_falls_back_for_histories_without_the_counter(tmp_path):
    """Pre-v3 metric_history carries no nan_selection_metric_total key."""
    history = [{'val_dos_t_unweighted': float('nan')},
               {'val_dos_t_unweighted': 0.5}]
    msg = train_script.best_publication_warning(
        os.path.join(str(tmp_path), 'checkpoint_best.pth'), history)
    assert msg is not None and '1 of 2' in msg


# ------------------------------------------- C2 (metadata half): the new counter

def test_metric_history_carries_the_nonfinite_selection_counter():
    from g3nat.training.trainer import Trainer
    from g3nat.training.config import TrainingConfig
    import torch.nn as nn

    class _NanOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def forward(self, batch):
            out = self.lin(batch.x_feat) * float('nan')
            return out[:, :1].expand(-1, 4), out[:, 1:].expand(-1, 4)

    class _B:
        def __init__(self):
            self.x_feat = torch.randn(3, 2)
            self.dos = torch.randn(12)
            self.transmission = torch.randn(12)

        def to(self, device):
            return self

    trainer = Trainer(_NanOut(), TrainingConfig(num_epochs=1, learning_rate=1e-3))
    trainer._validate_epoch([_B()], 0)
    trainer._validate_epoch([_B()], 1)
    assert trainer.metric_history[-1]['nan_selection_metric_total'] == 2.0


# ------------------- Ruling B: the requeue fallback must use the RUN'S criterion

def _ldos_t_history():
    """LDOS+T (loss_a=1, loss_b=1, loss_c=1): the run's own criterion is T+LDOS.

    Built so the two criteria disagree AND the old one is the SMALLER of the two,
    which is the dangerous direction: a bar seeded at 0.60 is stricter than
    anything this run can reach, so checkpoint_best.pth would never be written
    again for the rest of the run.
    """
    return [{'val_transmission': 0.50, 'val_dos': 0.10, 'val_ldos_residue': 0.20,
             'val_dos_t_unweighted': 0.60},
            {'val_transmission': 0.40, 'val_dos': 0.90, 'val_ldos_residue': 0.90,
             'val_dos_t_unweighted': 1.30}]


def test_ldos_arm_fallback_is_seeded_from_its_own_objective_not_dos_t(tmp_path):
    """No best checkpoint on disk -- the path a run preempted before its first
    best write actually lands on. These runs are preemptible, so it is live."""
    from g3nat.training.selection import resolve_selection_metric

    _name, weights = resolve_selection_metric(1.0, 1.0, 1.0)
    history = _ldos_t_history()
    seeded = train_script.seed_best_value(str(tmp_path), history,
                                          selection_weights=weights)
    assert seeded == 0.70, (
        "seeded from val_dos_t_unweighted (0.60), a bar this run's own T+LDOS "
        "objective can never beat -- no best checkpoint for the rest of the run")
    assert min(m['val_dos_t_unweighted'] for m in history) == 0.60, \
        'fixture is pointless unless the two criteria disagree'


def test_t_only_fallback_ignores_the_untrained_dos_term(tmp_path):
    from g3nat.training.selection import resolve_selection_metric

    _name, weights = resolve_selection_metric(1.0, 0.0, 0.0)
    history = [{'val_transmission': 0.90, 'val_dos': 0.10,
                'val_ldos_residue': float('nan'), 'val_dos_t_unweighted': 1.00},
               {'val_transmission': 0.40, 'val_dos': 0.80,
                'val_ldos_residue': float('nan'), 'val_dos_t_unweighted': 1.20}]
    assert train_script.seed_best_value(str(tmp_path), history,
                                        selection_weights=weights) == 0.40, \
        'a T-only run must not have its resume bar set by an untrained DOS term'


def test_fallback_skips_entries_missing_a_contributing_term(tmp_path):
    """Pre-v3 histories do not carry every key. A resume must not crash on them."""
    from g3nat.training.selection import resolve_selection_metric

    _name, weights = resolve_selection_metric(1.0, 1.0, 1.0)
    history = [{'val_transmission': 0.5},                       # no LDOS key
               {'val_transmission': 0.4, 'val_ldos_residue': None},
               {'val_transmission': 0.4, 'val_ldos_residue': 0.3}]
    assert train_script.seed_best_value(str(tmp_path), history,
                                        selection_weights=weights) == 0.70


def test_the_resume_call_site_passes_the_runs_resolved_weights():
    """seed_best_value defaults to the legacy criterion when given no weights, so
    a call site that forgets them is silently wrong -- exactly the defect this
    fixes. Pin that main() passes them."""
    import ast

    path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'train.py')
    with open(path) as fh:
        tree = ast.parse(fh.read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == 'seed_best_value']
    assert calls, 'seed_best_value is never called'
    for call in calls:
        passed = [kw.arg for kw in call.keywords] + ['_pos'] * len(call.args)
        assert 'selection_weights' in passed or len(call.args) >= 3, \
            'seed_best_value called without the run\'s resolved selection weights'


# ------------- Finding 4: the PREFERRED branch must also match the run's criterion

def test_a_best_checkpoint_written_under_a_DIFFERENT_criterion_is_not_trusted(tmp_path):
    """A pre-v3 checkpoint_best.pth in a reused checkpoint dir stores a
    val_dos_t_unweighted number under the key 'selection_value'. Seeding a v3 run's
    bar from it compares two different quantities -- the same defect Ruling B fixed
    in the fallback, one branch up."""
    from g3nat.training.selection import resolve_selection_metric

    d = str(tmp_path)
    _touch(os.path.join(d, 'checkpoint_best.pth'),
           {'selection_value': 0.50, 'selection_metric': 'val_dos_t_unweighted'})
    _name, weights = resolve_selection_metric(1.0, 1.0, 1.0)   # T+LDOS
    history = _ldos_t_history()
    assert train_script.seed_best_value(d, history, selection_weights=weights) == 0.70, \
        "seeded the T+LDOS bar from a stored T+DOS number"


def test_a_best_checkpoint_written_under_the_SAME_criterion_is_trusted(tmp_path):
    """The converse: matching recorded weights means the stored value is the right
    quantity and must still win over the history minimum."""
    from g3nat.training.selection import resolve_selection_metric

    d = str(tmp_path)
    _name, weights = resolve_selection_metric(1.0, 1.0, 1.0)
    _touch(os.path.join(d, 'checkpoint_best.pth'),
           {'selection_value': 0.90, 'selection_metric': _name,
            'selection_weights': dict(weights)})
    assert train_script.seed_best_value(d, _ldos_t_history(),
                                        selection_weights=weights) == 0.90


# --------- Finding 3: the SNAPSHOT itself, not just the helper, follows the run

def test_the_weights_snapshot_epoch_follows_the_runs_own_criterion():
    """BEHAVIOURAL 198-vs-512 GUARD.

    Runs Trainer.fit for two epochs with stubbed train/validate, over a history in
    which the run's own criterion (T only, loss_c=0) and the old fixed metric
    val_dos_t_unweighted prefer DIFFERENT epochs. best_unweighted['epoch'] is the
    state that becomes checkpoint_best.pth, so this fails if the snapshot is still
    gated on the fixed metric -- including the case where selection_value is called
    somewhere else in fit purely for logging.
    """
    import torch.nn as nn
    from g3nat.training.config import TrainingConfig
    from g3nat.training.trainer import Trainer

    entries = [
        {'val_transmission': 0.90, 'val_dos': 0.10, 'val_dos_t_unweighted': 1.00},
        {'val_transmission': 0.40, 'val_dos': 0.80, 'val_dos_t_unweighted': 1.20},
    ]
    own = [e['val_transmission'] for e in entries]
    old = [e['val_dos_t_unweighted'] for e in entries]
    assert own.index(min(own)) == 1 and old.index(min(old)) == 0, \
        'fixture is pointless unless the two criteria disagree'

    cfg = TrainingConfig(num_epochs=2, warmup_epochs=0, device='cpu',
                         loss_a=1.0, loss_b=0.0, loss_c=0.0)      # T only
    trainer = Trainer(nn.Linear(1, 1), cfg)
    trainer._train_epoch = lambda loader: 0.0

    def _fake_validate(loader, epoch):
        trainer.metric_history.append(entries[epoch])
        return float(epoch)

    trainer._validate_epoch = _fake_validate
    trainer.fit([], [])

    assert trainer.best_unweighted['epoch'] == 1, (
        "the published weights came from epoch %r -- the epoch preferred by the "
        "fixed metric val_dos_t_unweighted, not by this run's own objective"
        % trainer.best_unweighted['epoch'])
    assert trainer.best_unweighted['value'] == 0.40
    assert trainer.best_unweighted['state_dict'] is not None
