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
    history = [{'val_dos_t_unweighted': float('nan')} for _ in range(4)]
    msg = train_script.best_publication_warning(best, history)
    assert msg is not None, \
        "the run published no best checkpoint and said nothing about it"
    assert 'WARNING' in msg and 'NON-FINITE' in msg and 'val_dos_t_unweighted' in msg
    assert '4' in msg


def test_warning_when_no_validation_epoch_ran(tmp_path):
    msg = train_script.best_publication_warning(
        os.path.join(str(tmp_path), 'checkpoint_best.pth'), [])
    assert msg is not None and 'WARNING' in msg


def test_warning_names_a_partial_nonfinite_cause(tmp_path):
    history = [{'val_dos_t_unweighted': float('nan')},
               {'val_dos_t_unweighted': 0.5}]
    msg = train_script.best_publication_warning(
        os.path.join(str(tmp_path), 'checkpoint_best.pth'), history)
    assert msg is not None and '1 of 2' in msg


def test_no_warning_when_a_best_checkpoint_exists(tmp_path):
    best = os.path.join(str(tmp_path), 'checkpoint_best.pth')
    _touch(best, {'selection_value': 0.1})
    history = [{'val_dos_t_unweighted': 0.1}]
    assert train_script.best_publication_warning(best, history) is None


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
