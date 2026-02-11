# tests/baseline/test_baseline_data.py
import pickle
import numpy as np
from pathlib import Path
from g3nat.data import create_dna_dataset

BASELINE_DIR = Path(__file__).parent / "outputs"
BASELINE_DIR.mkdir(exist_ok=True)

def test_capture_dataset_creation():
    """Capture current dataset creation behavior."""
    np.random.seed(42)

    sequences = ["ACGT", "TGCA", "AAAA"]
    complementary = ["ACGT", "TGCA", "TTTT"]
    dos_data = np.random.rand(3, 100)
    trans_data = np.random.rand(3, 100)
    energy_grid = np.linspace(-3, 3, 100)

    dataset = create_dna_dataset(
        sequences=sequences,
        dos_data=dos_data,
        transmission_data=trans_data,
        energy_grid=energy_grid,
        complementary_sequences=complementary
    )

    # Capture first graph
    graph0 = dataset[0]
    baseline = {
        'num_samples': len(dataset),
        'graph0_x': graph0.x.clone(),
        'graph0_edge_index': graph0.edge_index.clone(),
        'graph0_dos': graph0.dos.clone(),
        'graph0_transmission': graph0.transmission.clone()
    }

    with open(BASELINE_DIR / "dataset.pkl", "wb") as f:
        pickle.dump(baseline, f)

    print(f"Dataset baseline: {len(dataset)} samples")

if __name__ == "__main__":
    test_capture_dataset_creation()
    print("Data baseline captured")
