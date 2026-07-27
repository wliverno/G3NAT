import numpy as np
import pytest
import torch

from g3nat.models.hamiltonian import site_ldos_log10


def test_n_orb_one_is_a_pure_log_transform():
    raw = torch.tensor([[[1.0, 10.0, 100.0]]])  # [1, 1, 3]

    out = site_ldos_log10(raw, n_sites=3, log_floor=1e-16)

    assert out.shape == (1, 1, 3)
    torch.testing.assert_close(out, torch.tensor([[[0.0, 1.0, 2.0]]]))


def test_multi_orbital_sums_within_each_site():
    # Layout is site-major / orbital-minor: row_orb[i, o] = i*n_orb + o (see
    # the row_orb index formula inside construct_hamiltonian_from_graph).
    # n_sites=3, n_orb=2 (unequal, on purpose): with equal n_sites/n_orb a
    # reshape that groups the axes in the wrong order is undetectable because
    # the flat vector looks the same either way. Site 0 owns columns 0,1;
    # site 1 owns 2,3; site 2 owns 4,5.
    raw = torch.tensor([[[1.0, 9.0, 30.0, 70.0, 500.0, 500.0]]])  # [1, 1, 6]

    out = site_ldos_log10(raw, n_sites=3, log_floor=1e-16)

    assert out.shape == (1, 1, 3)
    # Per-site sums are 10, 100, 1000 -- clean powers of ten. A wrong
    # (n_orb, n_sites) axis order groups different elements together and
    # would not land on clean values.
    torch.testing.assert_close(out, torch.tensor([[[1.0, 2.0, 3.0]]]))


def test_log_floor_guards_nonpositive_predictions():
    # The model can in principle emit a non-positive diagonal. log10(0) is -inf
    # and would poison the loss; the existing NaN guard prevents that.
    raw = torch.tensor([[[0.0, 1.0]]])

    out = site_ldos_log10(raw, n_sites=2, log_floor=1e-16)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(-16.0))


def test_clamp_applies_after_the_orbital_sum_not_before():
    # log_floor must guard the SITE SUM, not each orbital individually. A
    # per-orbital pre-sum clamp would floor a negative orbital contribution
    # to log_floor before summing, silently discarding it instead of letting
    # it offset a positive one in the same site.
    raw = torch.tensor([[[-100.0, 200.0]]])  # [1, 1, 2]: one site, n_orb=2

    out = site_ldos_log10(raw, n_sites=1, log_floor=1e-16)

    # Correct (clamp after sum): log10(clamp(-100 + 200, min=1e-16))
    #   == log10(100) == 2.0
    # Buggy (clamp before sum): log10(clamp(-100, min=1e-16) + 200)
    #   ~= log10(200) ~= 2.301 -- would fail this assertion.
    torch.testing.assert_close(out, torch.tensor([[[2.0]]]))


def test_accepts_a_noncontiguous_diagonal_and_returns_correct_values():
    # torch.diagonal returns a non-contiguous view, and this is the realistic
    # shape the production caller passes in (see ldos_lin as constructed
    # from a diagonal inside NEGFProjection / NEGFProjectionComplex).
    #
    # Measured behavior (contrary to an earlier version of this test and
    # comment): .view() does NOT actually raise on this input. Contiguity
    # gates MERGING dimensions; a trailing SPLIT -- which is all
    # site_ldos_log10 does to the width axis -- never requires it. So this
    # test does not exercise a reshape-vs-view crash. What it does exercise:
    # that a non-contiguous, diagonal-derived tensor is consumed without
    # error and its values come out correct, not scrambled or copied wrong.
    gr = torch.arange(2 * 3 * 4 * 4, dtype=torch.float).reshape(2, 3, 4, 4)
    raw = torch.diagonal(gr, dim1=-2, dim2=-1)
    assert not raw.is_contiguous()

    out = site_ldos_log10(raw, n_sites=4, log_floor=1e-16)

    assert out.shape == (2, 3, 4)
    # n_sites == width here, so n_orb == 1 and the per-site "sum" over a
    # group of one is the identity: the output must equal the clamped
    # log10 of each diagonal entry exactly, in the original order.
    expected = torch.log10(torch.clamp(raw, min=1e-16))
    torch.testing.assert_close(out, expected)


def test_indivisible_site_count_raises():
    raw = torch.ones(1, 1, 5)
    with pytest.raises(ValueError, match="not divisible"):
        site_ldos_log10(raw, n_sites=2, log_floor=1e-16)


def test_sum_of_linear_site_ldos_equals_dos_for_n_orb_one():
    # The model's DOS is the trace of the same matrix whose diagonal is ldos
    # (dos_raw vs ldos_lin, both derived from Gr_imag inside NEGFProjection),
    # so the sum over sites IS the DOS.
    raw = torch.rand(2, 5, 6) + 0.1

    out = site_ldos_log10(raw, n_sites=6, log_floor=1e-16)

    torch.testing.assert_close(
        torch.pow(10.0, out).sum(dim=-1), raw.sum(dim=-1), rtol=1e-5, atol=1e-6
    )
