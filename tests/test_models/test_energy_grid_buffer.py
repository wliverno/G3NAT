import numpy as np
import torch

from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.utils.physics import calculate_NEGF


def test_energy_grid_buffer_holds_the_grid_values_in_the_source_dtype():
    # The previous version of this test asserted only that a buffer named
    # 'energy_grid_t' existed and that the solver returned the right SHAPE --
    # both true for a buffer of zeros in the wrong dtype. Assert the values.
    grid = np.linspace(-1, 1, 8)
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=grid, n_orb=1)
    buf = dict(m.named_buffers())
    assert 'energy_grid_t' in buf
    # Source dtype (float64 from np.linspace), NOT a forced float32: a float64
    # consumer would otherwise silently get a grid rounded to ~2.9e-8 eV.
    assert buf['energy_grid_t'].dtype == torch.float64
    assert torch.equal(buf['energy_grid_t'], torch.tensor(grid, dtype=torch.float64))
    # Non-persistent, so old and new checkpoints stay mutually loadable.
    assert 'energy_grid_t' not in m.state_dict()


def test_solver_uses_the_actual_grid_values_at_nonzero_H():
    # At H = 0 with a symmetric grid, several wrong grids (zeros, a sign flip)
    # produce plausible output. Use a nonzero, energy-asymmetric H and an
    # asymmetric grid, and compare against an INDEPENDENT numpy NEGF evaluated
    # on the explicitly constructed grid.
    grid = np.array([-1.7, -0.6, 0.15, 0.9, 2.3], dtype=np.float64)
    H = np.array([
        [0.3, 0.15, 0.0],
        [0.15, -0.8, 0.15],
        [0.0, 0.15, 1.1],
    ], dtype=np.float64)
    GammaL = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    GammaR = np.array([0.0, 0.0, 0.1], dtype=np.float64)

    T_np, DOS_np = calculate_NEGF(H, GammaL, GammaR, grid)

    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=grid, n_orb=1)
    with torch.no_grad():
        T, DOS, _ = m.NEGFProjectionComplex(
            torch.tensor(H, dtype=torch.float64),
            torch.tensor(GammaL, dtype=torch.float64),
            torch.tensor(GammaR, dtype=torch.float64))
    assert T.shape == (5,) and DOS.shape == (5,)

    # Default floor semantics are the legacy clamp at 1e-16; every value here is
    # far above it, so both sides are plain log10.
    assert T_np.min() > 1e-10 and DOS_np.min() > 1e-10
    np.testing.assert_allclose(T.numpy(), np.log10(T_np), atol=1e-9)
    np.testing.assert_allclose(DOS.numpy(), np.log10(DOS_np), atol=1e-9)
