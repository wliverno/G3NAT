"""loss_c: the DOS-family weight, added 2026-08-10 (willll).

total = a*T + c*(b*LDOS + (1-b)*DOS). c=1 must reproduce every run on record
bit-for-bit; c=0 is transmission-only training, the previously unreachable arm
(b and 1-b sum to 1, so the DOS family always carried weight 1 before).
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from g3nat.data.datasets import create_dna_dataset
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.training.config import TrainingConfig
from g3nat.training.trainer import Trainer


def _dataset(with_ldos):
    seqs = ['AAAC', 'AAAG']
    n_e = 11
    egrid = np.linspace(-1, 1, n_e)
    ldos_data = None
    if with_ldos:
        arrays = [np.full((8, n_e), -1.0) for _ in seqs]
        ldos_data = {'residue': arrays, 'base_only': arrays}
    return create_dna_dataset(
        sequences=seqs,
        dos_data=np.full((2, n_e), -1.0),
        transmission_data=np.full((2, n_e), -3.0),
        energy_grid=egrid,
        complementary_sequences=['GTTT', 'CTTT'],
        ldos_data=ldos_data,
    )


def _trainer(loss_a=1.0, loss_b=0.0, loss_c=1.0):
    model = DNATransportHamiltonianGNN(
        energy_grid=np.linspace(-1, 1, 11), hidden_dim=16, num_layers=1, n_orb=1
    )
    config = TrainingConfig(num_epochs=1, batch_size=2, device='cpu',
                            loss_a=loss_a, loss_b=loss_b, loss_c=loss_c,
                            shape_loss=False)
    return Trainer(model, config)


def _losses(trainer, with_ldos=True):
    dataset = _dataset(with_ldos=with_ldos)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    dos_pred, trans_pred = trainer.model(batch)
    return trainer._compute_losses(batch, dos_pred, trans_pred)


def test_default_is_one_and_reproduces_history():
    assert TrainingConfig().loss_c == 1.0
    losses = _losses(_trainer(loss_b=0.0, loss_c=1.0))
    torch.testing.assert_close(losses['total'],
                               losses['dos'] + losses['transmission'])


def test_c_one_b_positive_reproduces_the_three_term_formula():
    losses = _losses(_trainer(loss_a=2.0, loss_b=0.25, loss_c=1.0))
    expected = 2.0 * losses['transmission'] + 0.25 * losses['ldos'] \
        + 0.75 * losses['dos']
    torch.testing.assert_close(losses['total'], expected)


def test_c_zero_total_is_transmission_only():
    losses = _losses(_trainer(loss_a=1.0, loss_b=0.0, loss_c=0.0))
    torch.testing.assert_close(losses['total'], losses['transmission'])
    assert losses['ldos'] is None


def test_c_zero_needs_no_ldos_target_even_at_b_positive():
    # At c=0 the whole DOS family is skipped by branch, so b>0 must NOT raise
    # the missing-LDOS-target error on a v1-style dataset.
    losses = _losses(_trainer(loss_b=0.5, loss_c=0.0), with_ldos=False)
    torch.testing.assert_close(losses['total'], losses['transmission'])


def test_c_zero_never_enters_the_ldos_path(monkeypatch):
    import g3nat.training.trainer as trainer_mod

    def _boom(*args, **kwargs):
        raise AssertionError("site_ldos_log10 must not be called when loss_c == 0")

    monkeypatch.setattr(trainer_mod, "site_ldos_log10", _boom)
    losses = _losses(_trainer(loss_b=0.5, loss_c=0.0))
    assert losses['ldos'] is None
    assert torch.isfinite(losses['total'])


def test_c_zero_still_measures_the_dos_diagnostics():
    # The arm is trained on T only, but held-out DOS must remain measured --
    # that is the whole point of running it.
    losses = _losses(_trainer(loss_c=0.0))
    assert torch.isfinite(losses['dos'])
    assert torch.isfinite(losses['dos_shape'])
    assert torch.isfinite(losses['dos_t_unweighted'])


def test_c_scales_the_dos_family():
    t_half = _trainer(loss_b=0.0, loss_c=0.5)
    losses = _losses(t_half)
    expected = losses['transmission'] + 0.5 * losses['dos']
    torch.testing.assert_close(losses['total'], expected)
