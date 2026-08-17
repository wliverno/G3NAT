import copy
import torch
import torch.nn as nn
from g3nat.training.trainer import Trainer
from g3nat.training.config import TrainingConfig


class _Scripted(nn.Module):
    """Model whose val unweighted metric follows a scripted curve via a scale param."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        out = batch.base * self.scale
        return out, out


class _Batch:
    def __init__(self):
        self.base = torch.ones(2, 4)
        self.dos = torch.zeros(8)
        self.transmission = torch.zeros(8)

    def to(self, device):
        return self


def test_best_unweighted_tracks_the_true_minimum_epoch():
    model = _Scripted()
    cfg = TrainingConfig(num_epochs=5, learning_rate=0.0, warmup_epochs=0)
    trainer = Trainer(model, cfg)
    # Script the parameter per epoch: dip at epoch 2, rise after.
    values = [1.0, 0.5, 0.1, 0.7, 0.9]

    def progress(epoch, tr, vl):
        with torch.no_grad():
            model.scale.fill_(values[min(epoch + 1, 4)])

    with torch.no_grad():
        model.scale.fill_(values[0])
    trainer.fit([_Batch()], [_Batch()], progress_callback=progress)
    assert trainer.best_unweighted['epoch'] == 2
    saved_scale = trainer.best_unweighted['state_dict']['scale'].item()
    assert abs(saved_scale - 0.1) < 1e-6, "stored weights are not from the best epoch"


def test_final_checkpoint_reports_actual_last_epoch():
    """The final save must report the epoch the loop ACTUALLY reached.

    Written with start_epoch=5 > num_epochs=3 -- an already-finished range, which
    is what a requeue of a completed cell looks like. The loop body never executes,
    so the honest answer is start_epoch - 1 == 4. `num_epochs - 1` would report 2,
    rewinding the recorded epoch by two and making the next resume redo finished
    work. (The original version of this test used num_epochs=3/start_epoch=0, where
    num_epochs - 1 and the true last epoch are both 2 -- it could not discriminate.)
    """
    model = _Scripted()
    cfg = TrainingConfig(num_epochs=3, learning_rate=0.0, warmup_epochs=0,
                         checkpoint_frequency=100)  # periodic save never fires
    trainer = Trainer(model, cfg)
    seen = {}

    def cb(model_, opt, epoch, tr, vl, metric_history=None, best_state=None):
        seen['epoch'] = epoch
        seen['best_state'] = best_state

    trainer.fit([_Batch()], [_Batch()], checkpoint_callback=cb, start_epoch=5)
    assert seen['epoch'] == 4, (
        f"final save reported epoch {seen['epoch']}; expected start_epoch - 1 == 4")
    assert seen['best_state'] is not None and 'epoch' in seen['best_state']
