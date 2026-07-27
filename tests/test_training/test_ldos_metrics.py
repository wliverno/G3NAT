import math

import numpy as np
import torch

from g3nat.data.datasets import create_dna_dataset
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.training.config import TrainingConfig
from g3nat.training.trainer import Trainer
from torch_geometric.loader import DataLoader


def _loader(with_ldos):
    n_e = 11
    ldos_data = None
    # UPPERCASE is required: BASE_FEATURES is keyed 'A'/'T'/'G'/'C' and
    # g3nat/graph/construction.py does no case normalization. The real pipeline
    # only works because load_single_pickle calls .upper() (pickle.py:37).
    seqs = ['AAAC', 'AAAG']
    if with_ldos:
        arrays = [np.full((8, n_e), -1.0) for _ in seqs]
        ldos_data = {'residue': arrays, 'base_only': arrays}
    dataset = create_dna_dataset(
        sequences=seqs,
        dos_data=np.zeros((2, n_e)),
        transmission_data=np.zeros((2, n_e)),
        energy_grid=np.linspace(-1, 1, n_e),
        complementary_sequences=['GTTT', 'CTTT'],
        ldos_data=ldos_data,
    )
    return DataLoader(dataset, batch_size=2)


def _trainer(loss_b=0.0):
    model = DNATransportHamiltonianGNN(
        energy_grid=np.linspace(-1, 1, 11), hidden_dim=16, num_layers=1, n_orb=1
    )
    config = TrainingConfig(num_epochs=1, batch_size=2, device='cpu', loss_b=loss_b)
    return Trainer(model, config)


def test_ldos_agreement_reported_even_at_b_zero():
    # The metric is governed by presence of a target; loss_b governs only the
    # loss. At b=0 the LDOS term is untrained but still measured -- that is the
    # Phase A reference the whole experiment is denominated against.
    trainer = _trainer(loss_b=0.0)
    loader = _loader(with_ldos=True)

    trainer.fit(loader, loader)

    entry = trainer.metric_history[-1]
    assert math.isfinite(entry['val_ldos_residue'])
    assert math.isfinite(entry['val_dos_t_unweighted'])


def test_metric_skipped_not_crashed_on_v1_data():
    # pickle_files/ carries no DOSAtom. PyG drops the None attribute entirely,
    # so an unguarded batch.ldos raises AttributeError rather than returning None.
    trainer = _trainer(loss_b=0.0)
    loader = _loader(with_ldos=False)

    trainer.fit(loader, loader)

    entry = trainer.metric_history[-1]
    assert math.isnan(entry['val_ldos_residue'])
    assert math.isfinite(entry['val_dos_t_unweighted'])
