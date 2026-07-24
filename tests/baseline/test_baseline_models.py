# tests/baseline/test_baseline_models.py
import torch
import numpy as np
from g3nat.models import DNATransportGNN, DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph
from ._util import check_or_capture

def test_capture_standard_model():
    """Capture DNATransportGNN predictions."""
    torch.manual_seed(42)

    # Create model
    model = DNATransportGNN(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        output_dim=100,
        dropout=0.1
    )
    model.eval()

    # Create test graph
    graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)

    # Forward pass
    with torch.no_grad():
        dos_pred, trans_pred = model(graph)

    baseline = {
        'dos': dos_pred.clone(),
        'transmission': trans_pred.clone(),
        'model_state': {k: v.clone() for k, v in model.state_dict().items()}
    }

    check_or_capture("model_standard.pkl", baseline)

def test_capture_hamiltonian_model():
    """Capture DNATransportHamiltonianGNN predictions."""
    torch.manual_seed(42)

    energy_grid = np.linspace(-1, 1, 50)
    model = DNATransportHamiltonianGNN(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        energy_grid=energy_grid,
        n_orb=1,
        solver_type='frobenius'
    )
    model.eval()

    graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)

    with torch.no_grad():
        dos_pred, trans_pred = model(graph)

    baseline = {
        'dos': dos_pred.clone(),
        'transmission': trans_pred.clone()
    }

    check_or_capture("model_hamiltonian.pkl", baseline)

if __name__ == "__main__":
    test_capture_standard_model()
    test_capture_hamiltonian_model()
    print("Model baselines captured")
