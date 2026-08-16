import math
import torch
import torch.nn as nn
from g3nat.training.trainer import Trainer
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
