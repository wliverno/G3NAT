"""Boolean per-base onsite head (spec F1), and legacy alpha-checkpoint mapping.

The continuous alpha mix is GONE. `per_base_onsite=False` (the default) is the old
alpha=0 path and must be byte-identical to the historical default; True is the old
alpha=1, a pure learned 4-entry-per-base table. Old checkpoints with a fractional or
learned alpha cannot be expressed by this model and must fail loudly.
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from g3nat import DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph

EG = np.linspace(-3, 3, 40)


def _build(seed=0, **kw):
    torch.manual_seed(seed)
    return DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
                                      energy_grid=EG, n_orb=1, conv_type='gat', **kw)


def _H(model, seq='ACGT', comp='ACGT'):
    data = sequence_to_graph(seq, comp, left_contact_positions=0,
                             right_contact_positions=len(seq) - 1)
    with torch.no_grad():
        model(Batch.from_data_list([data]))
    return model.H[0]


# ---- API surface -----------------------------------------------------------

def test_per_base_onsite_true_creates_baseline_param():
    m = _build(seed=0, per_base_onsite=True)
    assert hasattr(m, 'onsite_baseline')
    assert m.onsite_baseline.shape == (4, 1)
    assert m.onsite_baseline.requires_grad
    # no alpha state survives
    names = set(dict(m.named_parameters())) | set(dict(m.named_buffers()))
    assert not any('alpha' in n for n in names), sorted(n for n in names if 'alpha' in n)


def test_per_base_onsite_false_is_default_and_adds_no_params():
    m0 = _build(seed=0)
    assert m0.per_base_onsite is False
    assert not hasattr(m0, 'onsite_baseline')


def test_default_off_is_byte_identical():
    """Explicit off must reproduce the historical default exactly (same RNG stream)."""
    a = _build(seed=0)
    b = _build(seed=0, per_base_onsite=False)
    ka, kb = dict(a.named_parameters()), dict(b.named_parameters())
    assert ka.keys() == kb.keys()
    assert len(ka) > 0
    for k in ka:
        assert torch.equal(ka[k], kb[k]), f"param {k} differs -> RNG stream perturbed"
    # ... and turning the head ON may only ADD a parameter, never move an existing
    # draw (the optional-modules-last construction order, Task 8).
    on = dict(_build(seed=0, per_base_onsite=True).named_parameters())
    assert set(ka) < set(on)
    for k in ka:
        assert torch.equal(ka[k], on[k]), \
            f"param {k} shifted when the per-base onsite head was turned on"


def test_old_alpha_kwargs_are_gone():
    for bad in ({'structured_onsite': True}, {'alpha_value': 1.0},
                {'alpha_mode': 'learned'}, {'alpha_granularity': 'per_base'},
                {'alpha_init': 0.9}):
        with pytest.raises(TypeError):
            _build(seed=0, **bad)


# ---- the mixing head itself ------------------------------------------------

def test_mix_onsite_is_pure_table_when_on():
    m = _build(seed=0, per_base_onsite=True)
    feats = torch.randn(3, 32)
    onehot = torch.eye(4)[[0, 2, 1]]
    out = m._mix_onsite(feats, onehot)
    expected = onehot @ m.onsite_baseline
    assert torch.allclose(out, expected)
    # and it is NOT the context head (would be vacuous if the two coincided)
    assert not torch.allclose(out, m.onsite_proj(feats))


def test_mix_onsite_is_the_context_head_when_off():
    m = _build(seed=0)
    feats = torch.randn(3, 32)
    onehot = torch.eye(4)[[0, 2, 1]]
    assert torch.allclose(m._mix_onsite(feats, onehot), m.onsite_proj(feats))


def test_onsite_equals_baseline_per_base_in_H():
    m = _build(seed=2, per_base_onsite=True)
    with torch.no_grad():
        m.onsite_baseline.copy_(torch.tensor([[-0.5], [-1.4], [0.0], [-1.1]]))  # A,T,G,C
    H = _H(m, seq='AACC', comp='GGTT')          # primary strand bases A,A,C,C
    diag = torch.diag(H)[:4]
    assert torch.allclose(diag, torch.tensor([-0.5, -0.5, -1.1, -1.1]), atol=1e-5)


def test_baseline_indexed_by_identity_not_position():
    m = _build(seed=3, per_base_onsite=True)
    with torch.no_grad():
        m.onsite_baseline.copy_(torch.tensor([[10.0], [20.0], [30.0], [40.0]]))  # A,T,G,C
    g1 = sequence_to_graph('GA', 'TC', left_contact_positions=0, right_contact_positions=1)
    g2 = sequence_to_graph('AG', 'CT', left_contact_positions=0, right_contact_positions=1)
    with torch.no_grad():
        m(Batch.from_data_list([g1, g2]))
    H = m.H
    assert torch.allclose(torch.diag(H[0])[:2], torch.tensor([30.0, 10.0]), atol=1e-4)  # G,A
    assert torch.allclose(torch.diag(H[1])[:2], torch.tensor([10.0, 30.0]), atol=1e-4)  # A,G


def test_gradient_flows_to_baseline():
    m = _build(seed=4, per_base_onsite=True)
    m(Batch.from_data_list([sequence_to_graph('ACGT', 'ACGT', 0, 3)]))
    (m.H ** 2).sum().backward()
    assert m.onsite_baseline.grad is not None and m.onsite_baseline.grad.abs().sum() > 0


# ---- checkpoint loading ----------------------------------------------------

def _save(tmp_path, name, state_dict, args):
    p = tmp_path / name
    torch.save({'model_state_dict': state_dict, 'energy_grid': EG, 'args': args}, p)
    return str(p)


_BASE_ARGS = {'hidden_dim': 32, 'num_layers': 2, 'num_heads': 2, 'n_orb': 1,
              'conv_type': 'gat'}


def test_new_checkpoint_roundtrip(tmp_path):
    from g3nat.evaluation.inference import load_trained_model
    m = _build(seed=6, per_base_onsite=True)
    with torch.no_grad():
        m.onsite_baseline.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    path = _save(tmp_path, 'new.pth', m.state_dict(),
                 dict(_BASE_ARGS, per_base_onsite=True))
    loaded, _, _ = load_trained_model(path, device='cpu')
    assert loaded.per_base_onsite is True
    assert torch.equal(loaded.onsite_baseline, m.onsite_baseline)


def test_legacy_alpha1_checkpoint_maps_to_per_base_onsite(tmp_path):
    """alpha=1 was the pure-table arm: it IS per_base_onsite=True, and its
    baseline weights must survive the load."""
    from g3nat.evaluation.inference import load_trained_model
    m = _build(seed=6, per_base_onsite=True)
    with torch.no_grad():
        m.onsite_baseline.copy_(torch.tensor([[0.1], [0.2], [0.3], [0.4]]))
    sd = dict(m.state_dict())
    sd['onsite_alpha_fixed'] = torch.tensor([1.0])       # legacy buffer
    path = _save(tmp_path, 'a1.pth', sd,
                 dict(_BASE_ARGS, structured_onsite=True, alpha_mode='fixed',
                      alpha_value=1.0, alpha_granularity='global', alpha_init=0.9))
    loaded, _, _ = load_trained_model(path, device='cpu')
    assert loaded.per_base_onsite is True
    assert torch.allclose(loaded.onsite_baseline,
                          torch.tensor([[0.1], [0.2], [0.3], [0.4]]))


def test_legacy_alpha0_checkpoint_maps_to_off(tmp_path):
    """alpha=0 is the default context path; the unused baseline/alpha state in the
    old state_dict must be dropped rather than crashing the strict load."""
    from g3nat.evaluation.inference import load_trained_model
    m = _build(seed=7)
    sd = dict(m.state_dict())
    sd['onsite_baseline'] = torch.zeros(4, 1)
    sd['onsite_alpha_fixed'] = torch.tensor([0.0])
    path = _save(tmp_path, 'a0.pth', sd,
                 dict(_BASE_ARGS, structured_onsite=True, alpha_mode='fixed',
                      alpha_value=0.0, alpha_granularity='global', alpha_init=0.9))
    loaded, _, _ = load_trained_model(path, device='cpu')
    assert loaded.per_base_onsite is False
    assert not hasattr(loaded, 'onsite_baseline')


@pytest.mark.parametrize('extra', [
    {'alpha_mode': 'fixed', 'alpha_value': 0.5},
    {'alpha_mode': 'fixed', 'alpha_value': 0.9},
    {'alpha_mode': 'learned', 'alpha_value': 0.0, 'alpha_granularity': 'per_base'},
])
def test_legacy_fractional_or_learned_alpha_raises(tmp_path, extra):
    """0 < alpha < 1 and learned alpha are unrepresentable now. Fail loudly, naming
    the file, instead of silently loading a different model."""
    from g3nat.evaluation.inference import load_trained_model
    m = _build(seed=8, per_base_onsite=True)
    sd = dict(m.state_dict())
    sd['onsite_alpha_fixed'] = torch.tensor([0.5])
    legacy = dict(_BASE_ARGS, structured_onsite=True, alpha_granularity='global',
                  alpha_init=0.9)
    legacy.update(extra)
    path = _save(tmp_path, 'frac.pth', sd, legacy)
    with pytest.raises(ValueError) as e:
        load_trained_model(path, device='cpu')
    msg = str(e.value)
    assert 'frac.pth' in msg
    assert 'alpha' in msg
