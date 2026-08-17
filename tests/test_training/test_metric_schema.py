"""Transport-restricted transmission metric and the launch-frozen metric schema.

`val_transmission_appreciable` restricts the transmission Huber to the energy
points where the TARGET exceeds APPRECIABLE_T_LOG10, i.e. where current actually
flows (docs/dataset.md: roughly half of every spectrum is deep tunnelling that no
transport measurement resolves, yet it carries half the error budget of the
whole-window metric).

EXPECTED_METRIC_KEYS freezes what `_validate_epoch` records. Anything not recorded
per epoch is unavailable at every epoch but one, and re-deriving it means re-running
the campaign -- so schema drift has to fail loudly rather than silently.
"""

import math

import pytest
import torch
import torch.nn as nn

from g3nat.training import trainer as trainer_mod
from g3nat.training.trainer import (
    APPRECIABLE_T_LOG10,
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
        'val_transmission_appreciable', 'val_dos_t_unweighted',
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


def test_threshold_constant_is_the_documented_value():
    assert APPRECIABLE_T_LOG10 == -16.0


def test_appreciable_metric_masks_the_deep_tail():
    above = _run(_Batch(-3.0))['val_transmission_appreciable']   # all points appreciable
    assert above == above  # not nan
    below = _run(_Batch(-20.0))['val_transmission_appreciable']  # none appreciable
    assert below != below  # nan


def test_appreciable_metric_uses_only_the_appreciable_points():
    # Sequence 0 sits in the deep tail, sequence 1 at an appreciable resonance.
    target = torch.tensor([-20.0] * 4 + [-3.0] * 4)
    entry = _run(_Batch(target))
    # Prediction is 0 everywhere, so residuals are +20 (tail) and +3 (appreciable),
    # both outside Huber's quadratic region.
    assert entry['val_transmission_appreciable'] == pytest.approx(_huber(3.0), rel=1e-6)
    # Whole-window average over all 8 points -- the number the restricted metric
    # exists to be read alongside, and a different number.
    assert entry['val_transmission'] == pytest.approx(
        0.5 * (_huber(20.0) + _huber(3.0)), rel=1e-6)
    assert entry['val_transmission_appreciable'] < entry['val_transmission']


def test_threshold_is_strict_so_a_target_at_the_floor_does_not_qualify():
    at = _run(_Batch(APPRECIABLE_T_LOG10))['val_transmission_appreciable']
    assert math.isnan(at)
    just_above = _run(_Batch(APPRECIABLE_T_LOG10 + 0.5))['val_transmission_appreciable']
    assert not math.isnan(just_above)


def test_appreciable_metric_averages_over_batches_not_over_points():
    tr = _trainer()
    # Two batches: one wholly appreciable, one wholly tail. Only the first
    # contributes, so the epoch value is that batch's value, not a value diluted
    # by the batch that had nothing to measure.
    tr._validate_epoch([_Batch(-3.0), _Batch(-20.0)], epoch=0)
    entry = tr.metric_history[-1]
    assert entry['val_transmission_appreciable'] == pytest.approx(_huber(3.0), rel=1e-6)
