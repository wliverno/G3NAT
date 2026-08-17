import numpy as np
import pytest
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def test_non_hermitian_norb2_refused():
    grid = np.linspace(-1, 1, 8)
    with pytest.raises(ValueError, match='hermiticity'):
        DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=grid, n_orb=2,
                                   enforce_hermiticity=False)


def test_non_hermitian_norb1_still_allowed():
    grid = np.linspace(-1, 1, 8)
    DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                               energy_grid=grid, n_orb=1,
                               enforce_hermiticity=False)  # no raise (it is a no-op)
