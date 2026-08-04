# tests/test_models/test_forward.py
"""Test that models produce unchanged predictions after refactoring."""
import sys
sys.path.insert(0, '.')

import torch
import pickle
import pytest
import numpy as np
from pathlib import Path
from g3nat.models import DNATransportGNN, DNATransportHamiltonianGNN
from g3nat.graph import sequence_to_graph

BASELINE_DIR = Path("tests/baseline/outputs")


def test_standard_model_unchanged():
    """Verify DNATransportGNN produces identical predictions to baseline."""
    # Load baseline
    with open(BASELINE_DIR / "model_standard.pkl", "rb") as f:
        baseline = pickle.load(f)

    # Create model with same seed and architecture
    torch.manual_seed(42)
    model = DNATransportGNN(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        output_dim=100,
        dropout=0.1
    )

    # Load baseline weights
    model.load_state_dict(baseline['model_state'])
    model.eval()

    # Create test graph
    graph = sequence_to_graph("ACGT", "ACGT", 0, 3, 0.1, 0.1)

    # Forward pass
    with torch.no_grad():
        dos_pred, trans_pred = model(graph)

    # Compare to baseline
    assert torch.allclose(dos_pred, baseline['dos'], atol=1e-6), \
        f"DOS mismatch: max diff = {(dos_pred - baseline['dos']).abs().max()}"
    assert torch.allclose(trans_pred, baseline['transmission'], atol=1e-6), \
        f"Transmission mismatch: max diff = {(trans_pred - baseline['transmission']).abs().max()}"

    print(f"Standard model test passed: DOS shape {dos_pred.shape}, Trans shape {trans_pred.shape}")


def _hmodel(**kw):
    torch.manual_seed(0)
    return DNATransportHamiltonianGNN(
        hidden_dim=32, num_layers=2, num_heads=2, n_orb=1, conv_type='gat',
        energy_grid=np.linspace(-3, 3, 20), **kw)


def test_use_geometry_false_is_structurally_identical():
    """use_geometry=False adds no params/buffers -> existing checkpoints load."""
    from torch_geometric.data import Batch
    m = _hmodel(use_geometry=False)
    names = [n for n, _ in m.named_parameters()] + [n for n, _ in m.named_buffers()]
    assert not any('geom' in n for n in names)
    g = sequence_to_graph("ACGT", "ACGT")
    dos, trans = m(Batch.from_data_list([g]))
    assert torch.isfinite(dos).all() and torch.isfinite(trans).all()


def test_geometry_changes_output():
    """With use_geometry=True, changing an edge's geometry changes the prediction."""
    from torch_geometric.data import Batch
    m = _hmodel(use_geometry=True)
    # move geom_encoder off its near-zero init so geometry can influence the output
    with torch.no_grad():
        for p in m.geom_encoder.parameters():
            p.add_(0.1)
    g = sequence_to_graph("ACGT", "ACGT")            # all masks 0 -> no geometry
    with torch.no_grad():
        out0 = m(Batch.from_data_list([g]))
    g2 = g.clone()
    bb = (g2.edge_attr[:, 0] == 1).nonzero(as_tuple=True)[0][0]
    g2.edge_geom[bb] = torch.tensor([3.4, 1., 2., 3., 4., 5., 6.])
    g2.edge_geom_mask[bb] = 1.0
    with torch.no_grad():
        out1 = m(Batch.from_data_list([g2]))
    assert not torch.allclose(out0[1], out1[1])      # transmission responds to geometry


def test_geometry_checkpoint_roundtrip(tmp_path):
    """A use_geometry=True model saves and reloads via load_trained_model (buffers restored)."""
    from g3nat.evaluation.inference import load_trained_model
    egrid = np.linspace(-3, 3, 20)
    stats = {"backbone": {"mean": [0.5] * 7, "std": [1.5] * 7},
             "hbond": {"mean": [0.3] * 7, "std": [2.0] * 7}}
    m = DNATransportHamiltonianGNN(hidden_dim=32, num_layers=2, num_heads=2,
                                   energy_grid=egrid, conv_type='gat',
                                   use_geometry=True, geom_norm_stats=stats)
    p = str(tmp_path / "geom_model.pth")
    torch.save({'model_state_dict': m.state_dict(),
                'args': {'hidden_dim': 32, 'num_layers': 2, 'num_heads': 2,
                         'n_orb': 1, 'conv_type': 'gat', 'use_geometry': True},
                'energy_grid': egrid}, p)
    m2, eg, dev = load_trained_model(p, device='cpu')
    # per-type norm buffers restored from the checkpoint, not left at identity defaults
    assert torch.allclose(m2.geom_mean, m.geom_mean)
    assert torch.allclose(m2.geom_std, m.geom_std)
