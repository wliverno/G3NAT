import numpy as np
from g3nat.evaluation.physicality import onsite_block_eigs, coupling_block_bandwidth


def test_block_eigs_reduce_to_diag_at_norb1():
    H = np.diag([0.1, -0.2, 0.3])
    eigs = onsite_block_eigs(H, n_orb=1)
    assert eigs.shape == (3, 1)
    assert np.allclose(sorted(eigs.ravel()), sorted([0.1, -0.2, 0.3]))


def test_block_eigs_at_norb2_use_the_full_block():
    # One site, two orbitals: onsite block [[0, 0.5], [0.5, 0]] -> eigs +/-0.5.
    # Naive diag(H) would report (0, 0) and misclassify 0.5 as a coupling.
    H = np.array([[0.0, 0.5], [0.5, 0.0]])
    eigs = onsite_block_eigs(H, n_orb=2)
    assert eigs.shape == (1, 2)
    assert np.allclose(sorted(eigs.ravel()), [-0.5, 0.5])


def test_coupling_block_bandwidth_excludes_intra_site_block():
    H = np.array([[0.0, 0.5, 0.1, 0.0],
                  [0.5, 0.0, 0.0, 0.1],
                  [0.1, 0.0, 0.0, 0.5],
                  [0.0, 0.1, 0.5, 0.0]])
    # n_orb=2, 2 sites: the 0.5s are INTRA-site, only the 0.1 blocks are couplings.
    bw = coupling_block_bandwidth(H, n_orb=2)
    assert abs(bw - np.linalg.norm(np.array([[0.1, 0.0], [0.0, 0.1]]))) < 1e-12
