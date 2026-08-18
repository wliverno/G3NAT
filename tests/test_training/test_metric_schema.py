"""The launch-frozen metric schema, and the whole-window transmission metric.

EXPECTED_METRIC_KEYS freezes what `_validate_epoch` records. Anything not recorded
per epoch is unavailable at every epoch but one, and re-deriving it means re-running
the campaign -- so schema drift has to fail loudly rather than silently.

`val_transmission` is the Huber over the WHOLE energy window. There is deliberately
no threshold-restricted companion: a metric that discards the deep tail discards the
region the length-extrapolation claim rests on. Any tail-versus-resonance split is a
question for analysis time, computed from full data.
"""

import pytest
import torch
import torch.nn as nn

from g3nat.training import trainer as trainer_mod
from g3nat.training.trainer import (
    EXPECTED_METRIC_KEYS,
    Trainer,
)
from g3nat.training.config import TrainingConfig


class _Const(nn.Module):
    """Predicts log10 DOS = log10 T = 0 everywhere, for 2 sequences x 4 energies."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        base = torch.zeros(2, 4) + self.p
        return base, base


class _Batch:
    def __init__(self, t_target):
        self.dos = torch.zeros(8)
        if torch.is_tensor(t_target):
            self.transmission = t_target
        else:
            self.transmission = torch.full((8,), float(t_target))

    def to(self, device):
        return self


def _trainer():
    return Trainer(_Const(), TrainingConfig(num_epochs=1, warmup_epochs=0, device='cpu'))


def _run(batch):
    tr = _trainer()
    tr._validate_epoch([batch], epoch=0)
    return tr.metric_history[-1]


def _huber(residual):
    """Scalar HuberLoss (delta=1) of a single residual, for hand-checked values."""
    a = abs(residual)
    return 0.5 * a * a if a <= 1.0 else a - 0.5


def test_entry_matches_frozen_schema_exactly():
    entry = _run(_Batch(-3.0))
    assert set(entry.keys()) == set(EXPECTED_METRIC_KEYS)


def test_frozen_set_covers_every_recorded_diagnostic():
    # The keys later tasks added: absent from the frozen set means absent from
    # every campaign artifact.
    for key in (
        'epoch', 'val_dos', 'val_dos_shape', 'val_transmission',
        'val_dos_t_unweighted',
        'val_dos_t_shape_unweighted', 'val_ldos_residue', 'val_ldos_base_only',
        'val_ldos_shape_residue', 'val_ldos_shape_base_only',
        'val_ldos_localization_gap', 'floored_frac_dos', 'floored_frac_t',
        'floored_frac_ldos', 'neg_frac_dos', 'neg_frac_t', 'neg_frac_ldos',
        'nan_skipped_total', 'nan_selection_metric_total',
    ):
        assert key in EXPECTED_METRIC_KEYS, key


def test_schema_drift_raises_naming_missing_and_extra_separately():
    tr = _trainer()
    drifted = frozenset(set(EXPECTED_METRIC_KEYS) - {'val_dos'} | {'val_invented'})
    original = trainer_mod.EXPECTED_METRIC_KEYS
    trainer_mod.EXPECTED_METRIC_KEYS = drifted
    try:
        with pytest.raises(AssertionError) as exc:
            tr._validate_epoch([_Batch(-3.0)], epoch=0)
    finally:
        trainer_mod.EXPECTED_METRIC_KEYS = original
    message = str(exc.value)
    # 'val_invented' is expected-but-absent (missing); 'val_dos' is emitted-but-
    # unexpected (extra). Reversing the two sets must not pass.
    missing_part, _, extra_part = message.partition('extra=')
    assert 'val_invented' in missing_part and 'val_dos' not in missing_part
    assert 'val_dos' in extra_part and 'val_invented' not in extra_part


def test_transmission_metric_covers_the_whole_window_including_the_deep_tail():
    # Sequence 0 sits in the deep tunnelling tail, sequence 1 at a resonance.
    # Prediction is 0 everywhere, so residuals are +20 (tail) and +3 (resonance),
    # both outside Huber's quadratic region. Every point is counted: the tail
    # dominates the average precisely because it is not discarded.
    target = torch.tensor([-20.0] * 4 + [-3.0] * 4)
    entry = _run(_Batch(target))
    assert entry['val_transmission'] == pytest.approx(
        0.5 * (_huber(20.0) + _huber(3.0)), rel=1e-6)
