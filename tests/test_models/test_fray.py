import sys
sys.path.insert(0, '.')
import numpy as np
import torch
from torch_geometric.data import Batch
from g3nat.graph.construction import sequence_to_graph
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.evaluation.fray import (
    terminal_backbone_rows, terminal_h_indices, region_masks,
    run_fray_sweep, sweep_metrics,
)


def _entry(n=4):
    return {"bp_pars": np.zeros((n, 6)), "step_pars": np.ones((n - 1, 6)),
            "primary_centroids": np.array([[0, 0, i * 3.4] for i in range(n)], float),
            "comp_centroids": np.array([[6, 0, (n - 1 - i) * 3.4] for i in range(n)], float)}


def _geo_model():
    torch.manual_seed(0)
    stats = {"backbone": {"mean": [0] * 7, "std": [1] * 7},
             "hbond": {"mean": [0] * 7, "std": [1] * 7}}
    return DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
                                      energy_grid=np.linspace(-3, 3, 20), conv_type='gat',
                                      use_geometry=True, geom_norm_stats=stats)


# --- Task 1: pure helpers ---

def test_terminal_backbone_rows_point_at_last_primary_step():
    g = sequence_to_graph("ACGT", "ACGT", geometry=_entry(4))
    rows = terminal_backbone_rows(g, 4)
    assert len(rows) == 2                       # both directed copies
    ei = g.edge_index
    for r in rows:
        assert g.edge_attr[r, 0] == 1           # backbone
        assert {int(ei[0, r]), int(ei[1, r])} == {4, 5}   # nodes N, N+1 (primary pos 2,3)


def test_terminal_h_indices():
    assert terminal_h_indices(4) == (3, 2)


def test_region_masks_partition_and_terminal():
    m = region_masks(8, 4)                       # duplex ACGT: M=8, primary 0..3, comp 4..7
    assert m["diag"].sum() == 8
    assert np.array_equal(m["diag"] | m["offdiag"], np.ones((8, 8), bool))
    assert not np.any(m["diag"] & m["offdiag"])
    assert m["terminal_local"][3, 7] and m["terminal_local"][2, 0]
    assert not m["terminal_local"][4, 5]
    assert not np.any(m["distal"] & m["terminal_local"])
    assert not np.any(m["distal"] & m["diag"])
    assert np.array_equal(m["primary"] | m["comp"] | m["cross"], np.ones((8, 8), bool))


# --- Task 2: sweep + metrics ---

def test_sweep_starts_at_unmorphed_and_moves():
    g = sequence_to_graph("ACGT", "ACGT", geometry=_entry(4))
    m = _geo_model()
    with torch.no_grad():
        for p in m.geom_encoder.parameters():
            p.add_(0.1)                          # make geometry actually influence H
    rows = terminal_backbone_rows(g, 4)
    deltas = np.linspace(0, 3, 5)
    H = run_fray_sweep(m, g, rows, deltas)
    assert H.shape[0] == 5 and H.shape[1] == H.shape[2] == 8
    mets = sweep_metrics(H, deltas, 8, 4)
    assert mets["fro"][0] == 0.0                 # delta=0 -> no change vs itself
    assert mets["fro"][-1] > 0                   # morph moves H somewhere
    assert mets["argmax_ij"].shape == (5, 2)
    for k in ("diag", "distal", "terminal_local", "primary", "comp", "cross"):
        assert (mets["region"][k] >= 0).all()


def test_sweep_does_not_mutate_caller_graph():
    g = sequence_to_graph("ACGT", "ACGT", geometry=_entry(4))
    rows = terminal_backbone_rows(g, 4)
    before = g.edge_geom.clone()
    run_fray_sweep(_geo_model(), g, rows, np.linspace(0, 2, 3))
    assert torch.equal(g.edge_geom, before)      # restored after sweep
