import numpy as np
import torch
from torch_geometric.data import Batch

from g3nat.models import DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph


def _H_for(model, seq, n_orb):
    """Run the model on a single-strand sequence and return its Hamiltonian [H_size, H_size]."""
    data = sequence_to_graph(
        seq, left_contact_positions=0, right_contact_positions=len(seq) - 1
    )
    batch = Batch.from_data_list([data])
    model.eval()
    with torch.no_grad():
        model(batch)
    return model.H[0]  # H_size = len(seq) * n_orb


def _coupling_block(H, i, j, n_orb):
    """Off-diagonal coupling block between DNA nodes i and j."""
    return H[i * n_orb:(i + 1) * n_orb, j * n_orb:(j + 1) * n_orb]


def test_backbone_coupling_depends_on_base_norb1():
    torch.manual_seed(0)
    model = DNATransportHamiltonianGNN(
        hidden_dim=32, num_layers=2, num_heads=2, n_orb=1,
        energy_grid=np.linspace(-1.0, 1.0, 5),
    )
    HA = _H_for(model, "AAAA", n_orb=1)
    HG = _H_for(model, "GAAA", n_orb=1)  # differs only at primary position 0
    cA = _coupling_block(HA, 0, 1, 1)
    cG = _coupling_block(HG, 0, 1, 1)
    assert not torch.allclose(cA, cG, atol=1e-6), (
        "backbone coupling H[0,1] identical despite a different base at position 0 "
        "-> coupling head is base-blind"
    )


def test_backbone_coupling_depends_on_base_norb2():
    torch.manual_seed(0)
    model = DNATransportHamiltonianGNN(
        hidden_dim=32, num_layers=2, num_heads=2, n_orb=2,
        energy_grid=np.linspace(-1.0, 1.0, 5),
    )
    HA = _H_for(model, "AAAA", n_orb=2)
    HG = _H_for(model, "GAAA", n_orb=2)
    cA = _coupling_block(HA, 0, 1, 2)
    cG = _coupling_block(HG, 0, 1, 2)
    assert not torch.allclose(cA, cG, atol=1e-6), (
        "n_orb>1 backbone coupling block identical despite different base -> base-blind"
    )
