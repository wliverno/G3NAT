import math
import torch
import torch.nn as nn
from g3nat.training.trainer import Trainer, train_model
from g3nat.training.config import TrainingConfig


class _NanEveryOther(nn.Module):
    """Tiny stand-in model: forward returns (dos_pred, t_pred); every 2nd batch NaN."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        out = self.lin(batch.x_feat)  # [B, 2]
        dos = out[:, :1].expand(-1, 4)
        t = out[:, 1:].expand(-1, 4)
        if self.calls % 2 == 0:
            dos = dos * float('nan')
        return dos, t


class _AlwaysNan(nn.Module):
    """Every batch produces a non-finite loss."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, batch):
        out = self.lin(batch.x_feat) * float('nan')
        return out[:, :1].expand(-1, 4), out[:, 1:].expand(-1, 4)


class _FiniteLossInfGrad(nn.Module):
    """Finite forward and finite loss, but a hook forces a NON-FINITE gradient.

    This is the branch the loss-only NaN tests never reach: without the
    isfinite(grad_norm) check in _train_epoch, clip_grad_norm_ happily returns an
    inf norm, scales the gradients by max_norm/inf, and the optimizer steps anyway.
    """
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.lin.weight.register_hook(lambda g: torch.full_like(g, float('inf')))

    def forward(self, batch):
        out = self.lin(batch.x_feat)
        return out[:, :1].expand(-1, 4), out[:, 1:].expand(-1, 4)


class _Clean(nn.Module):
    """Finite everywhere; never triggers a skip."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, batch):
        out = self.lin(batch.x_feat)
        return out[:, :1].expand(-1, 4), out[:, 1:].expand(-1, 4)


class _Batch:
    def __init__(self, n):
        self.x_feat = torch.randn(n, 2)
        self.dos = torch.randn(n * 4)
        self.transmission = torch.randn(n * 4)

    def to(self, device):
        return self


def test_nan_batch_is_skipped_and_counted_and_params_stay_finite():
    torch.manual_seed(0)
    model = _NanEveryOther()
    trainer = Trainer(model, TrainingConfig(num_epochs=1, learning_rate=1e-2))
    loader = [_Batch(3) for _ in range(6)]
    loss = trainer._train_epoch(loader)
    assert trainer.nan_skipped_total == 3
    assert math.isfinite(loss)
    for p in model.parameters():
        assert torch.isfinite(p).all(), "a NaN step reached the optimizer"


def test_finite_loss_with_nonfinite_gradient_is_skipped_and_no_step_is_taken():
    """C1/review: the grad-norm branch, which the loss-only cases never exercise."""
    torch.manual_seed(0)
    model = _FiniteLossInfGrad()
    trainer = Trainer(model, TrainingConfig(num_epochs=1, learning_rate=1e-2))
    loader = [_Batch(3) for _ in range(4)]
    trainer._train_epoch(loader)
    assert trainer.nan_skipped_total == 4, (
        "a non-finite GRADIENT with a finite loss was not counted as a skip")
    # Adam allocates per-parameter state on its first step; empty state proves no
    # step was ever taken.
    assert len(trainer.optimizer.state) == 0, "the optimizer stepped on an inf gradient"
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_fully_skipped_epoch_reports_nan_not_zero():
    """C1: `train_loss /= max(1, n_used)` reported 0.0 -- a perfect fit -- for an
    epoch in which not one batch was usable."""
    torch.manual_seed(0)
    model = _AlwaysNan()
    trainer = Trainer(model, TrainingConfig(num_epochs=1, learning_rate=1e-2))
    loader = [_Batch(3) for _ in range(5)]
    loss = trainer._train_epoch(loader)
    assert trainer.nan_skipped_total == 5
    assert math.isnan(loss), (
        f"a fully skipped epoch reported {loss!r}; 0.0 reads as a perfect fit at "
        "the exact epoch the model was most broken")


def test_nan_counters_are_reseeded_from_metric_history_on_resume():
    """I4: these runs are preemptible, so train_model's resume path runs constantly.
    Seeding only the history left the cumulative counters at 0 after every requeue."""
    torch.manual_seed(0)
    model = _Clean()
    loader = [_Batch(3) for _ in range(3)]
    seeded = [{'epoch': 11, 'val_dos_t_unweighted': 1.0,
               'nan_skipped_total': 7.0, 'nan_selection_metric_total': 5.0}]
    out = []
    train_model(model=model, train_loader=loader, val_loader=loader,
                num_epochs=1, learning_rate=1e-3, device='cpu', warmup_epochs=0,
                metric_history=seeded, metric_history_out=out)
    assert out[-1]['nan_skipped_total'] == 7.0, (
        "nan_skipped_total reset to 0 on resume; the cumulative series jumps "
        "backwards at every preemption")
    assert out[-1]['nan_selection_metric_total'] == 5.0
