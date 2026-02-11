# tests/baseline/test_baseline_models.py
import torch
import pickle
import numpy as np
from pathlib import Path
from g3nat.models import DNATransportGNN, DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph

BASELINE_DIR = Path(__file__).parent / "outputs"
BASELINE_DIR.mkdir(exist_ok=True)

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

    with open(BASELINE_DIR / "model_standard.pkl", "wb") as f:
        pickle.dump(baseline, f)

    print(f"Standard model baseline: DOS shape {dos_pred.shape}, Trans shape {trans_pred.shape}")

def test_capture_hamiltonian_model():
    """Capture DNATransportHamiltonianGNN predictions."""
    torch.manual_seed(42)

    energy_grid = np.linspace(-1, 1, 50)
    model = DNATransportHamiltonianGNN(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        energy_grid=energy_grid,
        dropout=0.0,
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

    with open(BASELINE_DIR / "model_hamiltonian.pkl", "wb") as f:
        pickle.dump(baseline, f)

    print(f"Hamiltonian model baseline: DOS shape {dos_pred.shape}")

if __name__ == "__main__":
    test_capture_standard_model()
    test_capture_hamiltonian_model()
    print("Model baselines captured")
