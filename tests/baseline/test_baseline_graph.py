# tests/baseline/test_baseline_graph.py
import sys
sys.path.insert(0, '.')

import torch
import pickle
from pathlib import Path
from dataset import sequence_to_graph

BASELINE_DIR = Path("tests/baseline/outputs")
BASELINE_DIR.mkdir(exist_ok=True)

def test_capture_sequence_to_graph_simple():
    """Capture current sequence_to_graph behavior for simple sequence."""
    graph = sequence_to_graph(
        primary_sequence="ACGT",
        complementary_sequence="ACGT",
        left_contact_positions=0,
        right_contact_positions=3,
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )

    baseline = {
        'x': graph.x.clone(),
        'edge_index': graph.edge_index.clone(),
        'edge_attr': graph.edge_attr.clone(),
        'num_nodes': graph.x.size(0),
        'num_edges': graph.edge_index.size(1)
    }

    with open(BASELINE_DIR / "graph_simple.pkl", "wb") as f:
        pickle.dump(baseline, f)

    print(f"Captured baseline: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} edges")

def test_capture_sequence_to_graph_cross_contacts():
    """Capture behavior with cross contacts (complementary strand)."""
    graph = sequence_to_graph(
        primary_sequence="ACGTACGT",
        complementary_sequence="ACGTACGT",
        left_contact_positions=0,
        right_contact_positions=('complementary', 0),
        left_contact_coupling=0.1,
        right_contact_coupling=0.6
    )

    baseline = {
        'x': graph.x.clone(),
        'edge_index': graph.edge_index.clone(),
        'edge_attr': graph.edge_attr.clone()
    }

    with open(BASELINE_DIR / "graph_cross.pkl", "wb") as f:
        pickle.dump(baseline, f)

if __name__ == "__main__":
    test_capture_sequence_to_graph_simple()
    test_capture_sequence_to_graph_cross_contacts()
    print("Baselines captured successfully")
