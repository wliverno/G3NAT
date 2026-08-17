"""predict_sequence and dos_map must not silently amputate the geometry channel.

A geometry-trained checkpoint (use_geometry=True in its stored args) built its
evaluation graph with no geometry argument, so sequence_to_graph filled in an
all-zero geometry mask -- the model ran without error and produced plausible-
looking, silently wrong numbers. This is spec B8 (Task 7): a missing cache or a
cache miss for the requested sequence must raise ValueError instead.
"""
import numpy as np
import pytest
import torch

from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.evaluation import inference

N_ENERGY = 8
GRID = np.linspace(-1.0, 1.0, N_ENERGY)


def _geom_model_path(tmp_path):
    stats = {'backbone': {'mean': np.zeros(7), 'std': np.ones(7)},
             'hbond': {'mean': np.zeros(7), 'std': np.ones(7)}}
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=GRID, n_orb=1,
                                   use_geometry=True, geom_norm_stats=stats)
    path = tmp_path / 'geom_model.pth'
    torch.save({'model_state_dict': m.state_dict(),
                'args': {'model_type': 'hamiltonian', 'hidden_dim': 16,
                         'num_layers': 1, 'num_heads': 2, 'n_orb': 1,
                         'use_geometry': True, 'solver_type': 'complex',
                         'log_floor': 1e-38, 'complex_eta': 1e-12,
                         'use_log_outputs': True, 'enforce_hermiticity': True,
                         'conv_type': 'gat'},
                'energy_grid': GRID}, path)
    return str(path)


def _non_geom_model_path(tmp_path):
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=GRID, n_orb=1, use_geometry=False)
    path = tmp_path / 'no_geom_model.pth'
    torch.save({'model_state_dict': m.state_dict(),
                'args': {'model_type': 'hamiltonian', 'hidden_dim': 16,
                         'num_layers': 1, 'num_heads': 2, 'n_orb': 1,
                         'use_geometry': False, 'solver_type': 'complex',
                         'log_floor': 1e-38, 'complex_eta': 1e-12,
                         'use_log_outputs': True, 'enforce_hermiticity': True,
                         'conv_type': 'gat'},
                'energy_grid': GRID}, path)
    return str(path)


def test_geom_model_without_cache_raises(tmp_path):
    path = _geom_model_path(tmp_path)
    model, grid, _ = inference.load_trained_model(path, device='cpu')
    with pytest.raises(ValueError, match='geometry'):
        inference.predict_sequence(model, 'GATT', 'AATC')


def test_geom_model_with_cache_miss_raises(tmp_path):
    path = _geom_model_path(tmp_path)
    model, grid, _ = inference.load_trained_model(path, device='cpu')
    with pytest.raises(ValueError, match='GATT'):
        inference.predict_sequence(model, 'GATT', 'AATC',
                                   geometry_cache={'aaaa': object()})


def test_geom_model_with_cache_hit_succeeds(tmp_path):
    path = _geom_model_path(tmp_path)
    model, grid, _ = inference.load_trained_model(path, device='cpu')
    # Minimal but structurally valid geometry entry (4-base sequence): 4 primary
    # and 4 complementary centroids, 3 backbone steps, 4 base-pair params -- the
    # shape assemble_graph_geometry (g3nat/graph/geometry.py) expects.
    geom_entry = {
        'bp_pars': np.zeros((4, 6)),
        'step_pars': np.zeros((3, 6)),
        'primary_centroids': np.arange(12, dtype=float).reshape(4, 3),
        'comp_centroids': np.arange(12, 24, dtype=float).reshape(4, 3),
    }
    cache = {'gatt': geom_entry}
    dos_pred, t_pred = inference.predict_sequence(
        model, 'GATT', 'AATC', geometry_cache=cache)
    assert dos_pred.shape == (N_ENERGY,)
    assert t_pred.shape == (N_ENERGY,)


def test_non_geom_model_ignores_geometry_cache_argument(tmp_path):
    """A non-geometry checkpoint must not be forced to require a cache -- the
    argument is simply ignored when the model was not trained with it."""
    path = _non_geom_model_path(tmp_path)
    model, grid, _ = inference.load_trained_model(path, device='cpu')
    dos_pred, t_pred = inference.predict_sequence(model, 'GATT', 'AATC')
    assert dos_pred.shape == (N_ENERGY,)
    assert t_pred.shape == (N_ENERGY,)
