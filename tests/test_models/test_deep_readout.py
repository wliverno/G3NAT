"""Deeper-readout probe: one extra hidden Linear in each readout perceptron.

DNATransportHamiltonianGNNDeep and DNATransportGNNDeep are DUPLICATE subclasses
so the campaign code path (the base classes) is untouched. These tests pin down
the architecture delta, forward-shape parity, checkpoint round-tripping through
load_trained_model, and that the base-class parameter counts are the recorded
ones.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from g3nat.models import DNATransportGNN, DNATransportHamiltonianGNN
from g3nat.models.hamiltonian_deep import DNATransportHamiltonianGNNDeep
from g3nat.models.standard_deep import DNATransportGNNDeep
from g3nat.graph import sequence_to_graph

N_ENERGY = 41
GRID = np.linspace(-1.0, 1.0, N_ENERGY)
HIDDEN = 32
N_ORB = 2


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def _linears(seq):
    return [m for m in seq if isinstance(m, nn.Linear)]


def _batch():
    # Same length in both graphs: the Hamiltonian model requires it per batch.
    g1 = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)
    g2 = sequence_to_graph("GGAT", "GGAT", 0, 3, 0.1, 0.1)
    return Batch.from_data_list([g1, g2])


def _ham(cls, **kw):
    torch.manual_seed(0)
    return cls(hidden_dim=HIDDEN, num_layers=2, num_heads=2, n_orb=N_ORB,
               conv_type='gat', energy_grid=GRID, **kw)


def _direct(cls, **kw):
    torch.manual_seed(0)
    return cls(hidden_dim=HIDDEN, num_layers=2, num_heads=2,
               output_dim=N_ENERGY, dropout=0.0, conv_type='gat', **kw)


# ---------------------------------------------------------------- (a) ----
def test_deep_hamiltonian_readout_has_three_linears_and_expected_param_delta():
    base = _ham(DNATransportHamiltonianGNN)
    deep = _ham(DNATransportHamiltonianGNNDeep)
    assert isinstance(deep, DNATransportHamiltonianGNN)
    for proj in (deep.onsite_proj, deep.coupling_proj):
        lins = _linears(proj)
        assert len(lins) == 3
        assert lins[-1].out_features == N_ORB ** 2
    assert _linears(deep.coupling_proj)[0].in_features == 3 * HIDDEN
    extra = 2 * (HIDDEN * HIDDEN + HIDDEN)
    assert _n_params(deep) - _n_params(base) == extra


def test_deep_hamiltonian_final_layers_are_near_zero_initialized():
    deep = _ham(DNATransportHamiltonianGNNDeep)
    for proj in (deep.onsite_proj, deep.coupling_proj):
        last = _linears(proj)[-1]
        assert last.weight.abs().max().item() < 0.1
        assert torch.all(last.bias == 0)


# ---------------------------------------------------------------- (b) ----
def test_deep_hamiltonian_forward_matches_base_shapes_and_is_finite():
    base = _ham(DNATransportHamiltonianGNN).eval()
    deep = _ham(DNATransportHamiltonianGNNDeep).eval()
    batch = _batch()
    with torch.no_grad():
        dos_b, t_b = base(batch)
        dos_d, t_d = deep(batch)
    assert dos_d.shape == dos_b.shape == (2, N_ENERGY)
    assert t_d.shape == t_b.shape == (2, N_ENERGY)
    assert torch.isfinite(dos_d).all() and torch.isfinite(t_d).all()


# ---------------------------------------------------------------- (c) ----
def test_deep_direct_readout_has_three_linears_and_expected_param_delta():
    base = _direct(DNATransportGNN)
    deep = _direct(DNATransportGNNDeep)
    assert isinstance(deep, DNATransportGNN)
    for proj in (deep.dos_proj, deep.transmission_proj):
        lins = _linears(proj)
        assert len(lins) == 3
        assert [l.in_features for l in lins] == [HIDDEN, HIDDEN, HIDDEN // 2]
        assert lins[-1].out_features == N_ENERGY
    extra = 2 * (HIDDEN * HIDDEN + HIDDEN)
    assert _n_params(deep) - _n_params(base) == extra


def test_deep_direct_forward_matches_base_shapes():
    base = _direct(DNATransportGNN).eval()
    deep = _direct(DNATransportGNNDeep).eval()
    batch = _batch()
    with torch.no_grad():
        dos_b, t_b = base(batch)
        dos_d, t_d = deep(batch)
    assert dos_d.shape == dos_b.shape == (2, N_ENERGY)
    assert t_d.shape == t_b.shape == (2, N_ENERGY)
    assert torch.isfinite(dos_d).all() and torch.isfinite(t_d).all()


# ---------------------------------------------------------------- (d) ----
def _save(tmp_path, model, args, name):
    p = tmp_path / name
    torch.save({'model_state_dict': model.state_dict(),
                'energy_grid': GRID, 'args': args}, p)
    return str(p)


def test_deep_hamiltonian_roundtrips_through_load_trained_model(tmp_path):
    from g3nat.evaluation.inference import load_trained_model
    m = _ham(DNATransportHamiltonianGNNDeep, log_floor=1e-25, floor_mode='clamp')
    with torch.no_grad():
        for proj in (m.onsite_proj, m.coupling_proj):
            for lin in _linears(proj):
                lin.weight.normal_(std=0.05)
                lin.bias.normal_(std=0.05)
    args = {'model_type': 'hamiltonian_deep', 'hidden_dim': HIDDEN,
            'num_layers': 2, 'num_heads': 2, 'n_orb': N_ORB, 'conv_type': 'gat',
            'solver_type': 'complex', 'log_floor': 1e-25, 'floor_mode': 'clamp',
            'use_log_outputs': True, 'enforce_hermiticity': True,
            'use_geometry': False, 'per_base_onsite': False}
    path = _save(tmp_path, m, args, 'hamiltonian_deep_pickle_model.pth')
    loaded, grid, _ = load_trained_model(path, device='cpu')
    assert type(loaded) is DNATransportHamiltonianGNNDeep
    assert len(_linears(loaded.onsite_proj)) == 3
    assert len(grid) == N_ENERGY
    batch = _batch()
    m.eval()
    with torch.no_grad():
        dos_a, t_a = m(batch)
        dos_b, t_b = loaded(batch)
    assert torch.allclose(dos_a, dos_b, atol=1e-6)
    assert torch.allclose(t_a, t_b, atol=1e-6)


def test_deep_direct_roundtrips_through_load_trained_model(tmp_path):
    from g3nat.evaluation.inference import load_trained_model
    m = _direct(DNATransportGNNDeep)
    with torch.no_grad():
        m.dos_proj[-1].bias.fill_(0.4242)
    args = {'model_type': 'standard_deep', 'hidden_dim': HIDDEN,
            'num_layers': 2, 'num_heads': 2, 'conv_type': 'gat', 'dropout': 0.0,
            'use_geometry': False, 'num_energy_points': 100}
    path = _save(tmp_path, m, args, 'standard_deep_pickle_model.pth')
    loaded, grid, _ = load_trained_model(path, device='cpu')
    assert type(loaded) is DNATransportGNNDeep
    assert len(_linears(loaded.dos_proj)) == 3
    assert loaded.dos_proj[-1].out_features == N_ENERGY
    batch = _batch()
    m.eval()
    with torch.no_grad():
        dos_a, t_a = m(batch)
        dos_b, t_b = loaded(batch)
    assert torch.allclose(dos_a, dos_b, atol=1e-6)
    assert torch.allclose(t_a, t_b, atol=1e-6)


# ---------------------------------------------------------------- (e) ----
def test_base_class_parameter_counts_are_the_recorded_ones():
    torch.manual_seed(0)
    ham = DNATransportHamiltonianGNN(hidden_dim=256, num_layers=4, num_heads=4,
                                     n_orb=2, conv_type='gat', energy_grid=GRID)
    direct = DNATransportGNN(hidden_dim=256, num_layers=2, num_heads=4,
                             output_dim=201, dropout=0.0, conv_type='gat')
    assert _n_params(ham) == 533248 + 264712
    assert _n_params(direct) == 268032 + 117650
