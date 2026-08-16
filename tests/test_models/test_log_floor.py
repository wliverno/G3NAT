import numpy as np
import torch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def _model(log_floor):
    grid = np.linspace(-1.0, 1.0, 8)
    return DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                      energy_grid=grid, n_orb=1,
                                      solver_type='complex', log_floor=log_floor)


def test_deep_tail_keeps_gradient():
    # Weakly coupled 4-site chain: transmission lands at 1e-27..1e-20, i.e.
    # entirely BELOW the old 1e-16 clamp, which is where DFT transmission
    # targets (down to 6.7e-19) actually live. The plan's original H = 0 is a
    # stationary point of T for structural reasons -- T ~ |Gr_0,N-1|^2 is
    # second order in the couplings, so dT/dH vanishes at H = 0 under ANY
    # floor -- and therefore cannot distinguish a clamp from a smooth eps.
    m = _model(1e-38)
    H = torch.zeros(1, 4, 4)
    for i in range(3):
        H[0, i, i + 1] = 1e-4
        H[0, i + 1, i] = 1e-4
    H.requires_grad_(True)
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    T, DOS, _ = m.NEGFProjectionComplex(H, GL, GR)
    assert T.max().item() < -16.0, (
        "test setup is not in the deep tail: max log10 T = "
        f"{T.max().item()}, needs to be below the old 1e-16 clamp")
    # A target 1.5 decades below the OLD 1e-16 clamp must still pull gradient.
    loss = ((T - (-17.5)) ** 2).mean()
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    assert H.grad.abs().max() > 0, "deep-tail gradient is dead"


def test_floor_never_binds_above_eps_and_fractions_are_recorded():
    m = _model(1e-38)
    H = torch.randn(1, 4, 4) * 0.1
    H = 0.5 * (H + H.transpose(-1, -2))
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    T, DOS, _ = m.NEGFProjectionComplex(H, GL, GR)
    assert torch.isfinite(T).all() and torch.isfinite(DOS).all()
    assert 0.0 <= m.last_floored_frac_t <= 1.0
    assert 0.0 <= m.last_floored_frac_dos <= 1.0


def test_smooth_floor_matches_plain_log10_when_far_from_floor():
    m = _model(1e-38)
    x = torch.tensor([1e-3, 1.0, 10.0])
    out = m._log10_floored(x)
    assert torch.allclose(out, torch.log10(x), atol=1e-6)
