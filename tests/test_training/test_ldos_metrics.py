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


def _trainer(loss_b=0.0, ldos_target='residue'):
    model = DNATransportHamiltonianGNN(
        energy_grid=np.linspace(-1, 1, 11), hidden_dim=16, num_layers=1, n_orb=1
    )
    config = TrainingConfig(num_epochs=1, batch_size=2, device='cpu', loss_b=loss_b,
                            ldos_target=ldos_target)
    return Trainer(model, config)


def test_ldos_agreement_reported_even_at_b_zero():
    # The metric is governed by presence of a target; loss_b governs only the
    # loss. At b=0 the LDOS term is untrained but still measured -- that is the
    # Phase A reference the whole experiment is denominated against.
    # ldos_target defaults to 'residue', so the measured value lands under
    # val_ldos_residue and val_ldos_base_only stays nan -- pinned here so a
    # regression to a hardcoded key is caught in either direction.
    trainer = _trainer(loss_b=0.0)
    loader = _loader(with_ldos=True)

    trainer.fit(loader, loader)

    entry = trainer.metric_history[-1]
    assert math.isfinite(entry['val_ldos_residue'])
    assert math.isnan(entry['val_ldos_base_only'])
    assert math.isfinite(entry['val_dos_t_unweighted'])
    assert entry['epoch'] == 0


def test_ldos_agreement_keyed_by_base_only_target():
    # ldos_target='base_only' (Phase C of the LDOS experiment) must land the
    # measured value under val_ldos_base_only, with val_ldos_residue nan --
    # the mirror image of the default-target case above. Before this fix,
    # _validate_epoch unconditionally wrote the measured value under the
    # literal key 'val_ldos_residue' regardless of ldos_target, so Phase C
    # cells would silently report their base_only numbers under the residue
    # key and val_ldos_base_only would read nan.
    trainer = _trainer(loss_b=0.0, ldos_target='base_only')
    loader = _loader(with_ldos=True)

    trainer.fit(loader, loader)

    entry = trainer.metric_history[-1]
    assert math.isfinite(entry['val_ldos_base_only'])
    assert math.isnan(entry['val_ldos_residue'])
    assert math.isfinite(entry['val_dos_t_unweighted'])
    assert entry['epoch'] == 0


def test_metric_skipped_not_crashed_on_v1_data():
    # pickle_files/ carries no DOSAtom. PyG drops the None attribute entirely,
    # so an unguarded batch.ldos raises AttributeError rather than returning None.
    trainer = _trainer(loss_b=0.0)
    loader = _loader(with_ldos=False)

    trainer.fit(loader, loader)

    entry = trainer.metric_history[-1]
    assert math.isnan(entry['val_ldos_residue'])
    assert math.isfinite(entry['val_dos_t_unweighted'])
    assert entry['epoch'] == 0


def test_metric_history_epoch_is_absolute_not_list_index():
    # On a resumed run, fit() is called with start_epoch > 0 and
    # metric_history starts empty (nothing carried over in this test), so
    # the FIRST entry appended lands at list index 0 but
    # must record absolute epoch 5, not 0. A consumer that aligns by epoch
    # number (e.g. matching argmin(val_losses) to the epoch that produced it)
    # stays correct only if this key is present and absolute; aligning by list
    # position would silently read the wrong epoch's diagnostics once
    # metric_history and val_losses have different lengths after a resume.
    trainer = _trainer(loss_b=0.0)
    trainer.config.num_epochs = 7
    loader = _loader(with_ldos=True)

    trainer.fit(loader, loader, start_epoch=5)

    assert len(trainer.metric_history) == 2
    assert trainer.metric_history[0]['epoch'] == 5
    assert trainer.metric_history[1]['epoch'] == 6


def test_floor_diagnostics_recorded_including_ldos_and_negatives():
    # R1/R5: the LDOS floor fraction is recorded alongside DOS/T, and the
    # negative fraction is tracked SEPARATELY from underflow (negative DOS is
    # the non-Hermiticity pathology, not smallness). These live on the model as
    # tensors to avoid a CUDA sync per forward; the trainer converts once here.
    trainer = _trainer(loss_b=0.0)
    loader = _loader(with_ldos=True)

    trainer.fit(loader, loader)

    entry = trainer.metric_history[-1]
    for key in ('floored_frac_dos', 'floored_frac_t', 'floored_frac_ldos',
                'neg_frac_dos', 'neg_frac_t', 'neg_frac_ldos'):
        assert key in entry, f"missing diagnostic {key}"
        assert isinstance(entry[key], float)
        assert math.isfinite(entry[key]), f"{key} was never populated"
        assert 0.0 <= entry[key] <= 1.0
