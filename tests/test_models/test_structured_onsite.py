import torch
import numpy as np
from torch_geometric.data import Batch
from g3nat import DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph

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


def _H(model, seq='ACGT', comp='ACGT'):
    data = sequence_to_graph(seq, comp, left_contact_positions=0, right_contact_positions=len(seq)-1)
    with torch.no_grad():
        model(Batch.from_data_list([data]))
    return model.H[0]

def test_alpha0_matches_current_model():
    off = _build(seed=1)
    on0 = _build(seed=1, structured_onsite=True, alpha_mode='fixed', alpha_value=0.0)
    # onsite_proj weights identical (built after gated params; same seed prefix) -> H equal
    assert torch.allclose(_H(off), _H(on0), atol=1e-6)

def test_alpha1_onsite_equals_baseline_per_base():
    m = _build(seed=2, structured_onsite=True, alpha_mode='fixed', alpha_value=1.0)
    # set distinct baselines so we can read them off the diagonal
    with torch.no_grad():
        m.onsite_baseline.copy_(torch.tensor([[-0.5], [-1.4], [0.0], [-1.1]]))  # A,T,G,C
    H = _H(m, seq='AACC', comp='GGTT')          # primary strand bases A,A,C,C
    diag = torch.diag(H)[:4]
    assert torch.allclose(diag, torch.tensor([-0.5, -0.5, -1.1, -1.1]), atol=1e-5)

def test_baseline_indexed_by_identity_not_position():
    # two graphs, local position 0 is 'G' in one and 'A' in the other
    m = _build(seed=3, structured_onsite=True, alpha_mode='fixed', alpha_value=1.0)
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
    m = _build(seed=4, structured_onsite=True, alpha_mode='fixed', alpha_value=0.5)
    out = m(Batch.from_data_list([sequence_to_graph('ACGT','ACGT',0,3)]))
    loss = (m.H ** 2).sum()
    loss.backward()
    assert m.onsite_baseline.grad is not None and m.onsite_baseline.grad.abs().sum() > 0
