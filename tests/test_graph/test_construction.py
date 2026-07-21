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


def test_edge_geom_default_absent():
    """No geometry supplied -> zero tensors, all masked 0, correct shapes."""
    d = sequence_to_graph("ACGT", "ACGT")
    n_edges = d.edge_index.shape[1]
    assert d.edge_geom.shape == (n_edges, 7)
    assert d.edge_geom_mask.shape == (n_edges, 1)
    assert torch.count_nonzero(d.edge_geom_mask) == 0
    assert torch.count_nonzero(d.edge_geom) == 0


def test_edge_geom_lands_on_backbone_and_hbond_only():
    """With geometry supplied: backbone+hbond edges masked 1, contacts masked 0."""
    import numpy as np
    entry = {
        "bp_pars": np.arange(4 * 6).reshape(4, 6).astype(float),
        "step_pars": (np.arange(3 * 6).reshape(3, 6) + 100).astype(float),
        "primary_centroids": np.array([[0, 0, i * 3.4] for i in range(4)], float),
        "comp_centroids": np.array([[6, 0, (3 - i) * 3.4] for i in range(4)], float),
    }
    d = sequence_to_graph("ACGT", "ACGT", geometry=entry)
    ea = d.edge_attr
    contact = ea[:, 2] == 1
    bb = ea[:, 0] == 1
    hb = ea[:, 1] == 1
    # contacts never carry geometry
    assert torch.count_nonzero(d.edge_geom_mask[contact]) == 0
    # every backbone and hbond edge of a full duplex carries geometry
    assert torch.all(d.edge_geom_mask[bb] == 1)
    assert torch.all(d.edge_geom_mask[hb] == 1)
    # backbone slot 0 is a positive stacking distance (~3.4)
    assert torch.all(d.edge_geom[bb][:, 0] > 0)
    # hbond slot 0 is the ~6 A atom-centroid distance, not the ~0.09 A degeneracy
    assert torch.all(d.edge_geom[hb][:, 0] > 4.0)
