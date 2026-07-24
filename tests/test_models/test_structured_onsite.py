import torch
import numpy as np
from g3nat import DNATransportHamiltonianGNN

EG = np.linspace(-3, 3, 40)

def _build(seed=0, **kw):
    torch.manual_seed(seed)
    return DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
                                      energy_grid=EG, n_orb=1, conv_type='gat', **kw)

def test_default_off_is_byte_identical():
    a = _build(seed=0)                          # current model
    b = _build(seed=0, structured_onsite=False) # explicit off
    ka, kb = dict(a.named_parameters()), dict(b.named_parameters())
    assert ka.keys() == kb.keys()               # no new params
    for k in ka:
        assert torch.equal(ka[k], kb[k]), f"param {k} differs -> RNG stream perturbed"

def test_on_adds_baseline_and_alpha():
    m = _build(seed=0, structured_onsite=True, alpha_mode='fixed', alpha_value=1.0)
    assert m.onsite_baseline.shape == (4, 1)
    assert torch.allclose(m._onsite_alpha(), torch.ones(1))   # EXACT 1.0, no sigmoid drift

def test_fixed_alpha_zero_is_exact():
    m = _build(seed=0, structured_onsite=True, alpha_mode='fixed', alpha_value=0.0)
    assert torch.equal(m._onsite_alpha(), torch.zeros(1))

def test_per_base_learned_alpha_has_four_values():
    m = _build(seed=0, structured_onsite=True, alpha_granularity='per_base', alpha_mode='learned')
    assert m._onsite_alpha().shape == (4,)
    assert (m._onsite_alpha() > 0).all() and (m._onsite_alpha() < 1).all()
