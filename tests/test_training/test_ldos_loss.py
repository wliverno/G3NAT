import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from g3nat.data.datasets import create_dna_dataset
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.training.config import TrainingConfig
from g3nat.training.trainer import Trainer


def _dataset(with_ldos):
    # UPPERCASE is required: BASE_FEATURES is keyed 'A'/'T'/'G'/'C' and
    # g3nat/graph/construction.py does no case normalization. The real pipeline
    # only works because load_single_pickle calls .upper() (pickle.py:37).
    seqs = ['AAAC', 'AAAG']
    n_e = 11
    egrid = np.linspace(-1, 1, n_e)
    ldos_data = None
    if with_ldos:
        arrays = [np.full((8, n_e), -1.0) for _ in seqs]
        ldos_data = {'residue': arrays, 'base_only': arrays}
    return create_dna_dataset(
        sequences=seqs,
        dos_data=np.zeros((2, n_e)),
        transmission_data=np.zeros((2, n_e)),
        energy_grid=egrid,
        complementary_sequences=['GTTT', 'CTTT'],
        ldos_data=ldos_data,
    )


def _trainer(loss_a=1.0, loss_b=0.0):
    model = DNATransportHamiltonianGNN(
        energy_grid=np.linspace(-1, 1, 11), hidden_dim=16, num_layers=1, n_orb=1
    )
    config = TrainingConfig(num_epochs=1, batch_size=2, device='cpu',
                            loss_a=loss_a, loss_b=loss_b)
    return Trainer(model, config)


def test_config_defaults_reproduce_todays_weights():
    config = TrainingConfig()
    assert config.loss_a == 1.0
    assert config.loss_b == 0.0


def test_b_zero_total_equals_dos_plus_transmission():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=1.0, loss_b=0.0)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    expected = losses['dos'] + losses['transmission']
    torch.testing.assert_close(losses['total'], expected)


def test_b_zero_does_not_compute_an_ldos_term():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.0)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert losses['ldos'] is None


def test_b_positive_composes_all_three_terms():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=2.0, loss_b=0.25)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert losses['ldos'] is not None
    expected = (2.0 * losses['transmission']
                + 0.25 * losses['ldos']
                + 0.75 * losses['dos'])
    torch.testing.assert_close(losses['total'], expected)


def test_unweighted_diagnostic_ignores_the_weights():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=3.0, loss_b=0.5)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    torch.testing.assert_close(
        losses['dos_t_unweighted'], losses['dos'] + losses['transmission']
    )


def test_b_positive_without_a_target_raises_a_named_error():
    dataset = _dataset(with_ldos=False)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.5)

    dos_pred, trans_pred = trainer.model(batch)
    with pytest.raises(ValueError, match="no LDOS target"):
        trainer._compute_losses(batch, dos_pred, trans_pred)


def test_b_zero_trains_on_data_without_any_ldos_target():
    dataset = _dataset(with_ldos=False)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.0)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert torch.isfinite(losses['total'])
