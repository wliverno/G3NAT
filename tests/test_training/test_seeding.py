"""Model initialization must be reproducible independently of the data split.

Until 2026-07-31 the only seed in the codebase was --split_seed, which controls
the grouped train/val split and nothing else. Model init was left to torch's
global RNG, so a "3 seed" sweep varied the HELD-OUT SET, not the initialization,
and every cross-seed spread on record mixes the two. Separating them is a
precondition for asking whether a loss term makes the recovered H more
reproducible, which requires holding the split fixed and varying init.
"""
import numpy as np
import pytest
import torch

from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.training.utils import set_init_seed

ENERGY_GRID = np.linspace(-1.0, 1.0, 21)


def build():
    return DNATransportHamiltonianGNN(
        hidden_dim=16, num_layers=2, num_heads=2,
        energy_grid=ENERGY_GRID, n_orb=1, conv_type='gat')


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def test_same_seed_gives_identical_initialization():
    set_init_seed(1234)
    a = flat_params(build())
    set_init_seed(1234)
    b = flat_params(build())
    assert a.numel() > 0
    assert torch.equal(a, b)


def test_different_seed_gives_different_initialization():
    set_init_seed(1234)
    a = flat_params(build())
    set_init_seed(5678)
    b = flat_params(build())
    assert a.shape == b.shape
    assert not torch.equal(a, b)


def test_none_does_not_seed_and_reports_it():
    """None must leave the RNGs untouched, so historical runs stay reproducible
    in the only sense they ever were: not at all."""
    assert set_init_seed(None) is False
    a = flat_params(build())
    b = flat_params(build())
    assert not torch.equal(a, b), "unseeded builds should differ"


def test_seeding_reports_that_it_happened():
    assert set_init_seed(7) is True


def test_seed_covers_numpy_and_python_rngs_not_just_torch():
    """Init is not the only consumer of randomness -- anything added later that
    reaches for numpy or random must be covered too, or this seed silently means
    less than it claims."""
    import random
    set_init_seed(99)
    t0, n0, p0 = torch.rand(3), np.random.rand(3), [random.random() for _ in range(3)]
    set_init_seed(99)
    t1, n1, p1 = torch.rand(3), np.random.rand(3), [random.random() for _ in range(3)]
    assert torch.equal(t0, t1)
    assert np.array_equal(n0, n1)
    assert p0 == p1


def test_split_seed_and_init_seed_are_independent():
    """The whole point: same init, different split, must give the same weights at
    construction time. If these were ever coupled, a split sweep would silently be
    an init sweep as well -- which is the defect being fixed."""
    from g3nat.data.splits import grouped_split
    seqs = ['AAAC', 'GTTT', 'ACGT', 'TGCA', 'AACC', 'GGTT'] * 4

    set_init_seed(11)
    a = flat_params(build())
    grouped_split(seqs, test_size=0.2, seed=42)

    set_init_seed(11)
    b = flat_params(build())
    grouped_split(seqs, test_size=0.2, seed=43)

    assert torch.equal(a, b)
