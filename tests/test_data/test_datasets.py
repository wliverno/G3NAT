"""
Test dataset creation against baseline.
"""
import pytest
import pickle
import torch
import numpy as np
from pathlib import Path

# Import from new module location
from g3nat.data import create_dna_dataset


@pytest.fixture
def baseline_data():
    """Load baseline dataset."""
    baseline_path = Path(__file__).parent.parent / "baseline" / "outputs" / "dataset.pkl"
    with open(baseline_path, 'rb') as f:
        return pickle.load(f)


def test_create_dna_dataset_matches_baseline(baseline_data):
    """Test that create_dna_dataset produces output matching baseline."""
    # Use same input parameters as baseline creation
    np.random.seed(42)

    sequences = ["ACGT", "TGCA", "AAAA"]
    complementary = ["ACGT", "TGCA", "TTTT"]
    dos_data = np.random.rand(3, 100)
    trans_data = np.random.rand(3, 100)
    energy_grid = np.linspace(-3, 3, 100)

    # Recreate dataset using new module
    dataset = create_dna_dataset(
        sequences=sequences,
        dos_data=dos_data,
        transmission_data=trans_data,
        energy_grid=energy_grid,
        complementary_sequences=complementary
    )

    # Compare basic properties
    assert len(dataset) == baseline_data['num_samples'], "Dataset length mismatch"

    # Compare first graph (which is what baseline captured)
    graph0 = dataset[0]

    # Compare graph structure
    assert torch.equal(graph0.x, baseline_data['graph0_x']), "Sample 0: Node features mismatch"
    assert torch.equal(graph0.edge_index, baseline_data['graph0_edge_index']), "Sample 0: Edge index mismatch"

    # Compare targets
    assert torch.allclose(graph0.dos, baseline_data['graph0_dos']), "Sample 0: DOS mismatch"
    assert torch.allclose(graph0.transmission, baseline_data['graph0_transmission']), "Sample 0: Transmission mismatch"
