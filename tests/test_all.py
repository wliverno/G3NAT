import numpy as np
import torch
from torch_geometric.data import Batch

from dataset import sequence_to_graph
from models import DNATransportGNN, DNATransportHamiltonianGNN
from visualize_dna_graph import visualize_dna_graph
from data_generator import calculate_NEGF


def test_sequence_to_graph_single_strand():
    seq = "ATGCA"
    graph = sequence_to_graph(
        primary_sequence=seq,
        left_contact_positions=0,
        right_contact_positions=len(seq) - 1,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2,
    )

    assert graph.x is not None and graph.edge_index is not None and graph.edge_attr is not None
    # 2 contacts + len(seq) bases
    assert graph.x.shape[0] == len(seq) + 2
    # Node features = 4, edge features = 5 per current implementation
    assert graph.x.shape[1] == 4
    assert graph.edge_attr.shape[1] == 5

    # Contact edges should have contact flag at index 2 and coupling at index 4
    contact_mask = graph.edge_attr[:, 2] == 1
    assert contact_mask.any(), "Expected at least one contact edge"
    contact_couplings = graph.edge_attr[contact_mask, 4]
    assert (contact_couplings > 0).any()


def test_sequence_to_graph_double_strand_with_blanks():
    primary = "ACGTA"
    complementary = "TG___"  # some blanks
    graph = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=0,
        right_contact_positions=len(primary) - 1,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2,
    )
    # Ensure graph is constructed and has reasonable shapes
    assert graph.x.shape[0] >= len(primary) + 2
    assert graph.edge_index.shape[1] > 0


def test_dna_transport_gnn_forward():
    seq = "ATGCA"
    graph = sequence_to_graph(
        primary_sequence=seq,
        left_contact_positions=0,
        right_contact_positions=len(seq) - 1,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2,
    )
    batch = Batch.from_data_list([graph])

    model = DNATransportGNN(hidden_dim=64, num_layers=2, num_heads=2, output_dim=50, dropout=0.1)
    model.eval()
    with torch.no_grad():
        dos_pred, trans_pred = model(batch)
    assert dos_pred.shape == trans_pred.shape == (1, 50)


def test_hamiltonian_gnn_forward():
    seq = "ATGCA"
    graph = sequence_to_graph(
        primary_sequence=seq,
        left_contact_positions=0,
        right_contact_positions=len(seq) - 1,
        left_contact_coupling=0.1,
        right_contact_coupling=0.2,
    )
    batch = Batch.from_data_list([graph])

    energy_grid = np.linspace(-2, 2, 20)
    model = DNATransportHamiltonianGNN(hidden_dim=64, num_layers=2, num_heads=2, energy_grid=energy_grid, dropout=0.1, n_orb=1)
    model.eval()
    with torch.no_grad():
        dos_pred, trans_pred = model(batch)
    assert dos_pred.shape == trans_pred.shape == (1, len(energy_grid))


def test_negf_consistency_with_generator():
    """Ensure the model's NEGFProjection matches data_generator.calculate_NEGF math.
    Compares log10-transformed DOS and Transmission for the same H, ΓL, ΓR, and energy grid.
    """
    # Define a small, well-conditioned Hamiltonian and gamma vectors
    H = np.array([
        [0.0, 0.1, 0.0],
        [0.1, -1.0, 0.1],
        [0.0, 0.1, 0.5],
    ], dtype=np.float64)
    GammaL = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    GammaR = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    energy_grid = np.linspace(-2.0, 2.0, 40)

    # Reference using numpy generator (raw, not log10)
    T_np, DOS_np = calculate_NEGF(H, GammaL, GammaR, energy_grid)
    T_np_log = np.log10(np.clip(T_np, 1e-16, None))
    DOS_np_log = np.log10(np.clip(DOS_np, 1e-16, None))

    # Model's NEGFProjection (returns log10 already)
    model = DNATransportHamiltonianGNN(hidden_dim=8, num_layers=1, num_heads=1,
                                       energy_grid=energy_grid, dropout=0.0, n_orb=1)
    H_t = torch.tensor(H, dtype=torch.float64)
    gL_t = torch.tensor(GammaL, dtype=torch.float64)
    gR_t = torch.tensor(GammaR, dtype=torch.float64)
    with torch.no_grad():
        T_t, DOS_t, _ = model.NEGFProjection(H_t, gL_t, gR_t)
    T_t = T_t.cpu().numpy()
    DOS_t = DOS_t.cpu().numpy()

    # Allow small numerical differences due to different solve strategies
    assert np.allclose(T_t, T_np_log, atol=5e-4, rtol=1e-4), "Transmission mismatch between model and generator"
    assert np.allclose(DOS_t, DOS_np_log, atol=5e-4, rtol=1e-4), "DOS mismatch between model and generator"


def test_visualization_smoke():
    primary = "ACGC"
    complementary = "GCGT"
    graph = sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=("primary", [0]),
        right_contact_positions=("complementary", [0]),
        left_contact_coupling=0.1,
        right_contact_coupling=0.1,
    )
    fig, ax = visualize_dna_graph(graph, primary_sequence=primary, complementary_sequence=complementary)
    assert fig is not None and ax is not None
    # Close the figure to avoid backend issues in CI
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass