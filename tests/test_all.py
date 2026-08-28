import numpy as np
import torch
from torch_geometric.data import Batch

from g3nat.graph import sequence_to_graph
from g3nat.models import DNATransportGNN, DNATransportHamiltonianGNN
from g3nat.visualization import visualize_dna_graph
from g3nat.utils.physics import calculate_NEGF


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
    model = DNATransportHamiltonianGNN(hidden_dim=64, num_layers=2, num_heads=2, energy_grid=energy_grid, n_orb=1)
    model.eval()
    with torch.no_grad():
        dos_pred, trans_pred = model(batch)
    assert dos_pred.shape == trans_pred.shape == (1, len(energy_grid))


def test_negf_consistency_with_generator():
    """Ensure the model's NEGFProjection matches calculate_NEGF math.
    Compares log10-transformed DOS and Transmission for the same H, GammaL, GammaR, and energy grid.
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

    # Reference using numpy (returns transmission, dos)
    T_np, DOS_np = calculate_NEGF(H, GammaL, GammaR, energy_grid)
    T_np_log = np.log10(np.clip(T_np, 1e-16, None))
    DOS_np_log = np.log10(np.clip(DOS_np, 1e-16, None))

    # Model's NEGFProjection (returns T, DOS, H in log10)
    model = DNATransportHamiltonianGNN(hidden_dim=8, num_layers=1, num_heads=1,
                                       energy_grid=energy_grid, n_orb=1)
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


def _viz_test_graph():
    """Same tiny graph used by test_visualization_smoke, factored out for the
    font/scale/ax tests below."""
    primary = "ACGC"
    complementary = "GCGT"
    return sequence_to_graph(
        primary_sequence=primary,
        complementary_sequence=complementary,
        left_contact_positions=("primary", [0]),
        right_contact_positions=("complementary", [0]),
        left_contact_coupling=0.1,
        right_contact_coupling=0.1,
    ), primary, complementary


def _all_text_artists(ax):
    """Every text artist visualize_dna_graph draws on ax: node/contact labels,
    edge coupling labels, title, axis labels, and legend entries."""
    texts = list(ax.texts)
    texts.append(ax.title)
    texts.append(ax.xaxis.label)
    texts.append(ax.yaxis.label)
    legend = ax.get_legend()
    if legend is not None:
        texts.extend(legend.get_texts())
    return texts


def test_visualization_default_call_unchanged():
    """Backward-compat guard for the default call (figsize/node_size/font_size
    all left at their defaults).

    Node-label and legend-text sizes were ALREADY driven by font_size=10
    before this change and must stay at 10.0 exactly.

    Title/x-label/y-label/edge-coupling-labels were previously hardcoded to
    14/12/12/8 regardless of font_size -- i.e. they did not obey the
    font_size parameter at all. Making every text element obey font_size
    (the point of this change) means that, at the default font_size=10,
    those four now render at 10.0 too instead of their old hardcoded
    values. This is an intentional, disclosed behavior change: the old
    values were an inconsistency being fixed, not a contract to preserve.
    """
    graph, primary, complementary = _viz_test_graph()
    fig, ax = visualize_dna_graph(graph, primary_sequence=primary, complementary_sequence=complementary)
    try:
        assert tuple(fig.get_size_inches()) == (8.0, 12.0)

        texts = _all_text_artists(ax)
        # Same set of text artists as measured on the pre-change code:
        # 10 base/contact node labels + 12 edge labels (2 coupling values +
        # 10 empty-string backbone/h-bond labels) + title + xlabel + ylabel
        # + 6 legend entries = 31.
        assert len(texts) == 31

        node_label_texts = [t for t in ax.texts if t.get_text() in
                             ('L', 'R', 'A', 'T', 'G', 'C')]
        assert len(node_label_texts) == 10
        assert all(t.get_fontsize() == 10.0 for t in node_label_texts)

        legend = ax.get_legend()
        assert legend is not None
        assert all(t.get_fontsize() == 10.0 for t in legend.get_texts())

        # Harmonized (previously hardcoded 14/12/12/8): now all == font_size.
        assert ax.title.get_fontsize() == 10.0
        assert ax.xaxis.label.get_fontsize() == 10.0
        assert ax.yaxis.label.get_fontsize() == 10.0
        edge_label_texts = [t for t in ax.texts if t.get_text() not in
                             ('L', 'R', 'A', 'T', 'G', 'C')]
        assert len(edge_label_texts) == 12
        assert all(t.get_fontsize() == 10.0 for t in edge_label_texts)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_visualization_ax_param_draws_into_given_axes():
    graph, primary, complementary = _viz_test_graph()
    import matplotlib.pyplot as plt
    fig_before = plt.gcf().number
    n_figs_before = len(plt.get_fignums())

    existing_fig, existing_ax = plt.subplots(figsize=(5, 5))
    try:
        n_figs_after_create = len(plt.get_fignums())
        assert n_figs_after_create == n_figs_before + 1

        returned_fig, returned_ax = visualize_dna_graph(
            graph, primary_sequence=primary, complementary_sequence=complementary,
            ax=existing_ax)

        assert returned_ax is existing_ax
        assert returned_fig is existing_fig
        # No new figure was created by passing ax=.
        assert len(plt.get_fignums()) == n_figs_after_create
        # Text was actually drawn into the given axes.
        assert len(_all_text_artists(existing_ax)) > 0
    finally:
        plt.close(existing_fig)


def test_visualization_scale_doubles_font_and_node_size():
    graph, primary, complementary = _viz_test_graph()
    fig1, ax1 = visualize_dna_graph(graph, primary_sequence=primary,
                                     complementary_sequence=complementary,
                                     node_size=700, font_size=9, scale=1.0)
    fig2, ax2 = visualize_dna_graph(graph, primary_sequence=primary,
                                     complementary_sequence=complementary,
                                     node_size=700, font_size=9, scale=2.0)
    try:
        texts1 = _all_text_artists(ax1)
        texts2 = _all_text_artists(ax2)
        assert len(texts1) == len(texts2)
        for t1, t2 in zip(texts1, texts2):
            assert t2.get_fontsize() == 2.0 * t1.get_fontsize()

        # Node marker sizes (PathCollection) should also double.
        node_coll1 = ax1.collections[0]
        node_coll2 = ax2.collections[0]
        sizes1 = np.asarray(node_coll1.get_sizes())
        sizes2 = np.asarray(node_coll2.get_sizes())
        assert np.allclose(sizes2, 2.0 * sizes1)

        # Edge line widths should also double.
        edge_coll1 = ax1.collections[1]
        edge_coll2 = ax2.collections[1]
        lw1 = np.asarray(edge_coll1.get_linewidth())
        lw2 = np.asarray(edge_coll2.get_linewidth())
        assert np.allclose(lw2, 2.0 * lw1)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig1)
        plt.close(fig2)


def test_visualization_single_font_size_applies_to_all_text():
    graph, primary, complementary = _viz_test_graph()
    fig, ax = visualize_dna_graph(graph, primary_sequence=primary,
                                   complementary_sequence=complementary,
                                   font_size=13)
    try:
        texts = _all_text_artists(ax)
        assert len(texts) > 0
        assert all(t.get_fontsize() == 13.0 for t in texts)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
