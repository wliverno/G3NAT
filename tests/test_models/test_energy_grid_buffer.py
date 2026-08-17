import numpy as np
import torch
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN


def test_energy_grid_is_a_buffer_and_solver_output_unchanged():
    grid = np.linspace(-1, 1, 8)
    m = DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                   energy_grid=grid, n_orb=1)
    assert 'energy_grid_t' in dict(m.named_buffers())
    H = torch.zeros(1, 4, 4)
    GL = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    GR = torch.tensor([[0.0, 0.0, 0.0, 0.1]])
    T, DOS, _ = m.NEGFProjectionComplex(H, GL, GR)
    assert T.shape == (1, 8) and DOS.shape == (1, 8)
