import numpy as np
import pytest
import torch

from g3nat.models.hamiltonian import (DNATransportHamiltonianGNN,
                                      _floor_diagnostics, log10_floored,
                                      site_ldos_log10)


def _model(log_floor, floor_mode='smooth'):
    grid = np.linspace(-1.0, 1.0, 8)
    return DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                      energy_grid=grid, n_orb=1,
                                      solver_type='complex', log_floor=log_floor,
                                      floor_mode=floor_mode)


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


def test_floor_never_binds_on_a_well_conditioned_forward():
    # Previously this asserted that a mean of booleans lay in [0, 1] -- true by
    # construction and therefore vacuous. On this well-conditioned, strongly
    # coupled Hermitian case NOTHING underflows 1e-38 and NOTHING is negative,
    # so the exact expected value of all four diagnostics is 0.
    m = _model(1e-38)
    torch.manual_seed(0)
    H = torch.randn(1, 4, 4) * 0.1
    H = 0.5 * (H + H.transpose(-1, -2))
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    T, DOS, _ = m.NEGFProjectionComplex(H, GL, GR)
    assert torch.isfinite(T).all() and torch.isfinite(DOS).all()
    assert float(m.last_floored_frac_t) == 0.0
    assert float(m.last_floored_frac_dos) == 0.0
    assert float(m.last_neg_frac_t) == 0.0
    assert float(m.last_neg_frac_dos) == 0.0


def test_diagnostics_are_tensors_not_python_floats():
    # Deliberate: converting inside the forward costs a CUDA sync per batch,
    # including in the training loop. The conversion happens once per epoch in
    # Trainer._validate_epoch instead.
    m = _model(1e-38)
    H = torch.zeros(1, 4, 4)
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    m.NEGFProjectionComplex(H, GL, GR)
    assert isinstance(m.last_floored_frac_t, torch.Tensor)
    assert isinstance(m.last_neg_frac_dos, torch.Tensor)
    assert m.last_floored_frac_t.requires_grad is False


def test_underflow_and_negative_fractions_are_separate():
    # The old single fraction was (x < eps), which counted negative values as
    # "floored". Negative DOS is not smallness -- it is the signature of a
    # non-Hermitian H, exactly what the Hermiticity guard exists to catch.
    x = torch.tensor([-1.0, -2.0, 1e-40, 1.0])
    under, neg = _floor_diagnostics(x, 1e-38)
    assert float(neg) == 0.5
    assert float(under) == 0.25


def test_smooth_floor_matches_plain_log10_when_far_from_floor():
    m = _model(1e-38)
    x = torch.tensor([1e-3, 1.0, 10.0])
    out = m._log10_floored(x)
    assert torch.allclose(out, torch.log10(x), atol=1e-6)


def test_clamp_and_smooth_modes_differ_below_the_floor():
    x = torch.tensor([1e-20], dtype=torch.float64, requires_grad=True)
    clamped = log10_floored(x, 1e-16, 'clamp')
    assert float(clamped) == pytest.approx(-16.0, abs=1e-12)
    clamped.sum().backward()
    assert float(x.grad) == 0.0, "the hard clamp must kill the gradient"

    y = torch.tensor([1e-20], dtype=torch.float64, requires_grad=True)
    smooth = log10_floored(y, 1e-16, 'smooth')
    # log10(1e-20 + 1e-16): biased, but NOT pinned to -16, and differentiable.
    assert float(smooth) == pytest.approx(float(np.log10(1e-20 + 1e-16)), abs=1e-12)
    assert float(smooth) != pytest.approx(-16.0, abs=1e-9)
    smooth.sum().backward()
    assert float(y.grad) != 0.0


def test_default_floor_mode_is_the_legacy_clamp():
    # A checkpoint whose args predate floor_mode must reproduce its old numbers
    # bit-for-bit, so the CONSTRUCTOR default is 'clamp', not 'smooth'.
    m = DNATransportHamiltonianGNN(hidden_dim=8, num_layers=1, num_heads=2,
                                   energy_grid=np.linspace(-1, 1, 4), n_orb=1)
    assert m.floor_mode == 'clamp'
    assert m.log_floor == 1e-16
    out = m._log10_floored(torch.tensor([1e-25]))
    assert float(out) == pytest.approx(-16.0, abs=1e-5)


def test_invalid_floor_mode_rejected():
    with pytest.raises(ValueError, match="floor_mode"):
        DNATransportHamiltonianGNN(hidden_dim=8, num_layers=1, num_heads=2,
                                   energy_grid=np.linspace(-1, 1, 4), n_orb=1,
                                   floor_mode='hard')


def test_ldos_floor_is_smooth_and_keeps_gradient_below_the_floor():
    # R1: site_ldos_log10 used to hard-clamp unconditionally. With log_floor
    # moved to 1e-38 a floored per-site point read -38 with ZERO gradient and
    # dumped ~37 into the Huber LDOS term, which feeds checkpoint selection.
    raw = torch.tensor([[[1e-40]]], dtype=torch.float64, requires_grad=True)
    out = site_ldos_log10(raw, n_sites=1, log_floor=1e-38, floor_mode='smooth')
    expected = float(np.log10(1e-40 + 1e-38))
    assert float(out) == pytest.approx(expected, abs=1e-12)
    assert float(out) != pytest.approx(-38.0, abs=1e-6), "still hard-clamping"
    out.sum().backward()
    assert float(raw.grad) != 0.0


def test_ldos_clamp_mode_still_available_for_legacy():
    raw = torch.tensor([[[1e-40]]], dtype=torch.float64)
    out = site_ldos_log10(raw, n_sites=1, log_floor=1e-38, floor_mode='clamp')
    assert float(out) == pytest.approx(-38.0, abs=1e-12)


def test_ldos_diagnostics_measure_the_site_sum():
    # Two sites, n_orb=2. Site 0 sums to -1.0 (negative: unphysical), site 1
    # sums to 1e-40 (underflow). Per-ORBITAL diagnostics would give different
    # fractions, and the floor itself is applied to the site sum.
    raw = torch.tensor([[[-2.0, 1.0, 5e-41, 5e-41]]], dtype=torch.float64)
    out, under, neg = site_ldos_log10(raw, n_sites=2, log_floor=1e-38,
                                      floor_mode='smooth',
                                      return_diagnostics=True)
    assert out.shape == (1, 1, 2)
    assert float(neg) == 0.5
    assert float(under) == 0.5


def test_model_records_ldos_fractions():
    m = _model(1e-38)
    m.ldos = torch.tensor([[[1.0, 2.0]]])
    out = m.site_ldos_log10_recorded(n_sites=2)
    assert out.shape == (1, 1, 2)
    assert isinstance(m.last_floored_frac_ldos, torch.Tensor)
    assert float(m.last_floored_frac_ldos) == 0.0
    assert float(m.last_neg_frac_ldos) == 0.0
