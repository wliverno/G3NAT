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
    # Layout is site-major / orbital-minor: row_orb[i, o] = i*n_orb + o
    # (hamiltonian.py:401-403). Site 0 owns columns 0,1; site 1 owns 2,3.
    raw = torch.tensor([[[1.0, 9.0, 50.0, 50.0]]])  # [1, 1, 4]

    out = site_ldos_log10(raw, n_sites=2, log_floor=1e-16)

    assert out.shape == (1, 1, 2)
    torch.testing.assert_close(out, torch.tensor([[[1.0, 2.0]]]))


def test_log_floor_guards_nonpositive_predictions():
    # The model can in principle emit a non-positive diagonal. log10(0) is -inf
    # and would poison the loss; the existing NaN guard prevents that.
    raw = torch.tensor([[[0.0, 1.0]]])

    out = site_ldos_log10(raw, n_sites=2, log_floor=1e-16)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(-16.0))


def test_works_on_a_noncontiguous_input():
    # torch.diagonal returns a non-contiguous view; .view() would raise on it
    # while .reshape() copies. Guard against a regression to .view().
    gr = torch.arange(2 * 3 * 4 * 4, dtype=torch.float).reshape(2, 3, 4, 4)
    raw = torch.diagonal(gr, dim1=-2, dim2=-1).abs() + 1.0
    assert not raw.is_contiguous()

    out = site_ldos_log10(raw, n_sites=4, log_floor=1e-16)

    assert out.shape == (2, 3, 4)


def test_indivisible_site_count_raises():
    raw = torch.ones(1, 1, 5)
    with pytest.raises(ValueError, match="not divisible"):
        site_ldos_log10(raw, n_sites=2, log_floor=1e-16)


def test_sum_of_linear_site_ldos_equals_dos_for_n_orb_one():
    # The model's DOS is the trace of the same matrix whose diagonal is ldos
    # (hamiltonian.py:547 vs :555), so the sum over sites IS the DOS.
    raw = torch.rand(2, 5, 6) + 0.1

    out = site_ldos_log10(raw, n_sites=6, log_floor=1e-16)

    torch.testing.assert_close(
        torch.pow(10.0, out).sum(dim=-1), raw.sum(dim=-1), rtol=1e-5, atol=1e-6
    )
