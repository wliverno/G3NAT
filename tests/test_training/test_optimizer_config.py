"""The optimizer choice is now configurable. The default MUST stay Adam(weight_decay=1e-5),
because every recorded result in docs/model-results.md was produced with it -- a silent change
would invalidate the whole comparison history rather than just the next run.

Why adamw exists: Loshchilov & Hutter, ICLR 2019 (arXiv:1711.05101) show Adam's `weight_decay`
is folded into the gradient and then rescaled by Adam's per-parameter adaptive rates, so it is
not true weight decay and the effective regularization is weaker and parameter-dependent. See
docs/references.md.
"""
import numpy as np
import pytest
import torch

import g3nat
from g3nat.training import TrainingConfig
from g3nat.training.trainer import Trainer


def _trainer(**cfg):
    torch.manual_seed(0)
    model = g3nat.DNATransportHamiltonianGNN(
        hidden_dim=32, num_layers=2, num_heads=2,
        energy_grid=np.linspace(-1, 1, 20), n_orb=1)
    return Trainer(model, TrainingConfig(device='cpu', **cfg))


def test_default_is_unchanged_adam_1e5():
    """Guards the comparison history. If this fails, prior results are not comparable."""
    opt = _trainer().optimizer
    assert type(opt).__name__ == 'Adam'
    assert opt.param_groups[0]['weight_decay'] == 1e-5


def test_adamw_selected_and_decay_applied():
    opt = _trainer(optimizer='adamw', weight_decay=0.01).optimizer
    assert type(opt).__name__ == 'AdamW'
    assert opt.param_groups[0]['weight_decay'] == 0.01


def test_adam_still_selectable_with_custom_decay():
    opt = _trainer(optimizer='adam', weight_decay=0.1).optimizer
    assert type(opt).__name__ == 'Adam'
    assert opt.param_groups[0]['weight_decay'] == 0.1


def test_unknown_optimizer_raises_rather_than_silently_defaulting():
    """Silently falling back would make a typo in a sweep look like a real result."""
    with pytest.raises(ValueError, match="unknown optimizer"):
        _trainer(optimizer='sgd')


def test_case_insensitive():
    assert type(_trainer(optimizer='AdamW').optimizer).__name__ == 'AdamW'
