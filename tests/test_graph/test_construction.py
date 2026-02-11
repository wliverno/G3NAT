# tests/test_graph/test_construction.py
import sys
sys.path.insert(0, '.')

import torch
import pickle
import pytest
from pathlib import Path
from g3nat.graph import sequence_to_graph

BASELINE_DIR = Path("tests/baseline/outputs")

def test_sequence_to_graph_simple():
    """Test simple sequence case against baseline."""
    # Load baseline
    with open(BASELINE_DIR / "graph_simple.pkl", "rb") as f:
        baseline = pickle.load(f)

    # Generate graph with new implementation
    graph = sequence_to_graph(
        primary_sequence="ACGT",
        complementary_sequence="ACGT",
        left_contact_positions=0,
        right_contact_positions=3,
        left_contact_coupling=0.1,
        right_contact_coupling=0.1
    )

    # Verify exact match
    assert torch.equal(graph.x, baseline['x']), "Node features do not match baseline"
    assert torch.equal(graph.edge_index, baseline['edge_index']), "Edge indices do not match baseline"
    assert torch.equal(graph.edge_attr, baseline['edge_attr']), "Edge attributes do not match baseline"
    assert graph.x.size(0) == baseline['num_nodes'], "Number of nodes does not match baseline"
    assert graph.edge_index.size(1) == baseline['num_edges'], "Number of edges does not match baseline"

    print(f"[OK] Simple test passed: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} edges")

def test_sequence_to_graph_cross_contacts():
    """Test cross-strand contacts against baseline."""
    # Load baseline
    with open(BASELINE_DIR / "graph_cross.pkl", "rb") as f:
        baseline = pickle.load(f)

    # Generate graph with new implementation
    graph = sequence_to_graph(
        primary_sequence="ACGTACGT",
        complementary_sequence="ACGTACGT",
        left_contact_positions=0,
        right_contact_positions=('complementary', 0),
        left_contact_coupling=0.1,
        right_contact_coupling=0.6
    )

    # Verify exact match
    assert torch.equal(graph.x, baseline['x']), "Node features do not match baseline"
    assert torch.equal(graph.edge_index, baseline['edge_index']), "Edge indices do not match baseline"
    assert torch.equal(graph.edge_attr, baseline['edge_attr']), "Edge attributes do not match baseline"
    assert graph.x.size(0) == baseline['num_nodes'], "Number of nodes does not match baseline"
    assert graph.edge_index.size(1) == baseline['num_edges'], "Number of edges does not match baseline"

    print(f"[OK] Cross-contacts test passed: {graph.x.size(0)} nodes, {graph.edge_index.size(1)} edges")

if __name__ == "__main__":
    test_sequence_to_graph_simple()
    test_sequence_to_graph_cross_contacts()
    print("All graph construction tests passed!")
