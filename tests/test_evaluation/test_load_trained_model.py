"""load_trained_model must reconstruct a model that actually accepts its own weights.

The direct model's output width is the number of energy points. It was rebuilt from
args['num_energy_points'], which only applies to the synthetic tight-binding data
source -- for pickle data the grid comes from the data and that arg keeps its
default of 100. So every direct-model checkpoint trained on the 201-point DFT grid
failed to load with

    size mismatch for dos_proj.3.weight: copying a param with shape
    torch.Size([201, 128]) ... current model is torch.Size([100, 128])

which made the whole evaluation path unusable for the physics-blind control. The
energy grid is stored in the checkpoint and is authoritative; use its length.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from g3nat.evaluation.inference import load_trained_model
from g3nat.models.standard import DNATransportGNN
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN

N_ENERGY = 201
GRID = np.linspace(-1.0, 1.0, N_ENERGY)


def _save(tmp_path, model, args, name):
    p = tmp_path / name
    torch.save({'model_state_dict': model.state_dict(),
                'energy_grid': GRID, 'args': args}, p)
    return str(p)


def test_direct_model_roundtrips_on_a_201_point_grid(tmp_path):
    """The exact failure: grid length 201, num_energy_points left at its default."""
    m = DNATransportGNN(hidden_dim=64, num_layers=2, num_heads=2,
                        output_dim=N_ENERGY, dropout=0.0, conv_type='gat')
    args = {'model_type': 'standard', 'hidden_dim': 64, 'num_layers': 2,
            'num_heads': 2, 'conv_type': 'gat', 'dropout': 0.0,
            'num_energy_points': 100}          # <-- the stale default that caused it
    path = _save(tmp_path, m, args, 'standard_pickle_model_best.pth')
    loaded, grid, _ = load_trained_model(path, device='cpu')
    assert len(grid) == N_ENERGY
    assert loaded.dos_proj[-1].out_features == N_ENERGY
    assert loaded.transmission_proj[-1].out_features == N_ENERGY


def test_direct_model_weights_are_actually_restored(tmp_path):
    """Shapes matching is not enough -- confirm the values came across."""
    m = DNATransportGNN(hidden_dim=64, num_layers=2, num_heads=2,
                        output_dim=N_ENERGY, dropout=0.0, conv_type='gat')
    with torch.no_grad():
        m.dos_proj[-1].bias.fill_(0.4242)
    args = {'model_type': 'standard', 'hidden_dim': 64, 'num_layers': 2,
            'num_heads': 2, 'conv_type': 'gat', 'dropout': 0.0,
            'num_energy_points': 100}
    loaded, _, _ = load_trained_model(_save(tmp_path, m, args, 'standard_x.pth'), device='cpu')
    assert torch.allclose(loaded.dos_proj[-1].bias,
                          torch.full((N_ENERGY,), 0.4242), atol=1e-6)


def test_hamiltonian_model_still_roundtrips(tmp_path):
    """The fix must not disturb the path that already worked."""
    m = DNATransportHamiltonianGNN(hidden_dim=64, num_layers=2, num_heads=2,
                                   energy_grid=GRID, n_orb=2, conv_type='gat')
    args = {'model_type': 'hamiltonian', 'hidden_dim': 64, 'num_layers': 2,
            'num_heads': 2, 'conv_type': 'gat', 'n_orb': 2,
            'solver_type': 'complex'}
    loaded, grid, _ = load_trained_model(
        _save(tmp_path, m, args, 'hamiltonian_pickle_model_best.pth'), device='cpu')
    assert len(grid) == N_ENERGY


def test_grid_length_wins_over_a_contradictory_num_energy_points(tmp_path):
    """If the two disagree, the stored grid is authoritative -- it is what the
    weights were actually trained against."""
    m = DNATransportGNN(hidden_dim=32, num_layers=1, num_heads=2,
                        output_dim=N_ENERGY, dropout=0.0, conv_type='gat')
    args = {'model_type': 'standard', 'hidden_dim': 32, 'num_layers': 1,
            'num_heads': 2, 'conv_type': 'gat', 'dropout': 0.0,
            'num_energy_points': 17}           # deliberately absurd
    loaded, _, _ = load_trained_model(_save(tmp_path, m, args, 'standard_y.pth'), device='cpu')
    assert loaded.dos_proj[-1].out_features == N_ENERGY
