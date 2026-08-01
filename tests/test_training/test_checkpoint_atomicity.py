"""A preempted checkpoint write must not destroy the previous checkpoint.

scripts/run_ldos_phases.sh runs on preemptible partitions with --requeue, so
SLURM killing the process mid-write is an EXPECTED event, not an edge case. On
2026-07-31 job 37966189_0 was preempted during a checkpoint write, leaving a
truncated zip; the requeued attempt then died on resume with

    RuntimeError: PytorchStreamReader failed reading zip archive:
    failed finding central directory

and the cell lost roughly three hours of training. Writing to a temporary file
and renaming makes the replacement atomic, so an interrupted write leaves the
previous checkpoint intact and the requeue resumes from it.
"""
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from g3nat.training.callbacks import save_checkpoint

ENERGY_GRID = np.linspace(-1.0, 1.0, 5)


@pytest.fixture
def fixture(tmp_path):
    model = nn.Linear(4, 3)
    opt = torch.optim.Adam(model.parameters())
    return model, opt, str(tmp_path / 'checkpoint_latest.pth')


def _save(fixture, epoch):
    model, opt, path = fixture
    save_checkpoint(model, opt, epoch, [0.5], [0.6], {'seed': 1}, ENERGY_GRID, path)


def test_checkpoint_is_written_and_loadable(fixture):
    _save(fixture, 7)
    ck = torch.load(fixture[2], map_location='cpu', weights_only=False)
    assert ck['epoch'] == 7


def test_no_temporary_file_is_left_behind(fixture):
    _save(fixture, 1)
    leftovers = [f for f in os.listdir(os.path.dirname(fixture[2]))
                 if f != 'checkpoint_latest.pth']
    assert leftovers == [], f"stray files: {leftovers}"


def test_failed_write_leaves_the_previous_checkpoint_intact(fixture, monkeypatch):
    """The regression that cost job 37966189_0 three hours of training."""
    model, opt, path = fixture
    _save(fixture, 100)
    assert torch.load(path, map_location='cpu', weights_only=False)['epoch'] == 100

    real_save = torch.save

    def die_midway(obj, f, *a, **k):
        # save_checkpoint passes a PATH, not a handle. Truncate whatever path it
        # hands us, exactly as a kill during serialization does, then fail.
        # Non-atomic: that path IS the destination, so the destination is
        # corrupted. Atomic: it is a temp file, and the destination is untouched.
        # (An earlier version of this test branched on hasattr(f, 'write'), never
        # fired, and passed against the unfixed code.)
        assert isinstance(f, (str, bytes, os.PathLike)), \
            "test assumes save_checkpoint passes a path; update it if that changes"
        with open(f, 'wb') as fh:
            fh.write(b'PK\x03\x04truncated')
        raise RuntimeError('preempted mid-write')

    monkeypatch.setattr(torch, 'save', die_midway)
    with pytest.raises(RuntimeError):
        _save(fixture, 200)
    monkeypatch.setattr(torch, 'save', real_save)

    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['epoch'] == 100, "previous checkpoint was destroyed by a failed write"


def test_failed_write_does_not_strand_a_temp_file(fixture, monkeypatch):
    _save(fixture, 1)
    def die(obj, f, *a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(torch, 'save', die)
    with pytest.raises(RuntimeError):
        _save(fixture, 2)
    leftovers = [f for f in os.listdir(os.path.dirname(fixture[2]))
                 if f != 'checkpoint_latest.pth']
    assert leftovers == [], f"stray files after failed write: {leftovers}"


def test_metric_history_survives_the_atomic_path(fixture):
    """metric_history is load-bearing for every trade-off figure; make sure the
    rewrite did not drop it or start writing a None-valued key."""
    model, opt, path = fixture
    save_checkpoint(model, opt, 3, [0.1], [0.2], {}, ENERGY_GRID, path,
                    metric_history=[{'val_dos': 0.5}])
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['metric_history'] == [{'val_dos': 0.5}]

    save_checkpoint(model, opt, 4, [0.1], [0.2], {}, ENERGY_GRID, path)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert 'metric_history' not in ck
