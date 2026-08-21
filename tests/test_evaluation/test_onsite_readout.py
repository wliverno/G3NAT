"""The site-mean onsite readout is basis-invariant; the splitting is what it drops.

These pin a claim that was got WRONG once in the other direction: the readout was
reported as n_orb-blind and liable to manufacture a fake n_orb effect. It is not
-- mean(diag(block)) == mean(eigenvalues(block)) because the trace is the sum of
the eigenvalues. The tests below make that non-negotiable, and separately pin the
limitation that IS real (the discarded level splitting).
"""
import numpy as np
import pytest

from g3nat.evaluation.physicality import onsite_block_eigs


def _blocks(n_sites, n_orb, seed=0):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n_sites * n_orb, n_sites * n_orb))
    return 0.5 * (H + H.T)


def _site_mean_via_diag(H, n_orb, n_sites):
    return np.array([np.mean(np.diag(H[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb]))
                     for s in range(n_sites)])


@pytest.mark.parametrize('n_orb', [1, 2, 3])
def test_site_mean_equals_mean_of_block_eigenvalues(n_orb):
    """The identity that makes the readout invariant: trace = sum of eigenvalues."""
    H = _blocks(4, n_orb)
    got = _site_mean_via_diag(H, n_orb, 4)
    want = onsite_block_eigs(H, n_orb).mean(axis=1)
    assert np.abs(got - want).max() < 1e-12


def test_site_mean_survives_a_basis_rotation_of_each_block():
    """The operational statement of invariance: rotate each site's orbital basis
    and the site-mean must not move. A readout that used raw diagonal entries as
    individual levels WOULD move here -- that is the failure B14 describes."""
    n_orb, n_sites = 2, 4
    H = _blocks(n_sites, n_orb, seed=3)
    before = _site_mean_via_diag(H, n_orb, n_sites)

    rng = np.random.default_rng(11)
    Hr = H.copy()
    for s in range(n_sites):
        th = rng.uniform(0.2, np.pi - 0.2)          # avoid the identity rotation
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        blk = H[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb]
        Hr[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb] = R.T @ blk @ R

    after = _site_mean_via_diag(Hr, n_orb, n_sites)
    assert np.abs(before - after).max() < 1e-12

    # The rotation must actually have changed the raw diagonal, or the test is
    # vacuous -- it would pass against a rotation that did nothing.
    raw_before = np.diag(H)
    raw_after = np.diag(Hr)
    assert np.abs(raw_before - raw_after).max() > 1e-3


def test_individual_diagonal_entries_are_NOT_invariant():
    """The converse, pinned so the distinction cannot be lost: what B14 guards
    against is real, it just does not apply to a mean."""
    n_orb, n_sites = 2, 3
    H = _blocks(n_sites, n_orb, seed=5)
    rng = np.random.default_rng(7)
    Hr = H.copy()
    for s in range(n_sites):
        th = 0.7
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        blk = H[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb]
        Hr[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb] = R.T @ blk @ R
    assert np.abs(np.diag(H) - np.diag(Hr)).max() > 1e-3


def test_block_eigenvalues_are_themselves_invariant():
    """The basis-invariant per-level quantity, for code that needs levels rather
    than their mean."""
    n_orb, n_sites = 2, 3
    H = _blocks(n_sites, n_orb, seed=9)
    e0 = onsite_block_eigs(H, n_orb)
    Hr = H.copy()
    for s in range(n_sites):
        th = 1.1
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        blk = H[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb]
        Hr[s*n_orb:(s+1)*n_orb, s*n_orb:(s+1)*n_orb] = R.T @ blk @ R
    e1 = onsite_block_eigs(Hr, n_orb)
    assert np.abs(np.sort(e0, axis=1) - np.sort(e1, axis=1)).max() < 1e-12


def test_splitting_is_zero_at_n_orb_1_and_nonzero_above():
    """The real limitation: the mean discards a quantity that exists only at
    n_orb>1. It therefore cannot be a factorial response across an n_orb factor."""
    e1 = onsite_block_eigs(_blocks(4, 1), 1)
    assert e1.shape[1] == 1
    assert np.ptp(e1, axis=1).max() == 0.0

    e2 = onsite_block_eigs(_blocks(4, 2, seed=1), 2)
    assert np.ptp(e2, axis=1).min() > 0.0


def test_two_sites_can_share_a_mean_while_differing_in_splitting():
    """Constructed so the mean is IDENTICAL and the splitting is not -- the
    concrete case where eta2-on-the-mean cannot see a difference that exists."""
    a = np.diag([1.0, 3.0])          # mean 2.0, splitting 2.0
    b = np.diag([1.9, 2.1])          # mean 2.0, splitting 0.2
    H = np.zeros((4, 4))
    H[0:2, 0:2] = a
    H[2:4, 2:4] = b
    e = onsite_block_eigs(H, 2)
    assert e[0].mean() == pytest.approx(e[1].mean())
    assert np.ptp(e[0]) == pytest.approx(2.0)
    assert np.ptp(e[1]) == pytest.approx(0.2)
