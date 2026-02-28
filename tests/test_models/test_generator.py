import os
import torch
import pytest


def test_optimizer_output_shapes():
    """Forward returns correct shapes."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=5)
    one_hot, scaled = opt()

    assert one_hot.shape == (5, 4), f"Expected (5, 4), got {one_hot.shape}"
    assert scaled.shape == (5, 4), f"Expected (5, 4), got {scaled.shape}"


def test_optimizer_straight_through():
    """Forward produces one-hot vectors in training mode (hard ST)."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=5)
    opt.train()
    one_hot, _ = opt()

    # Each row should sum to 1 and have exactly one 1.0
    assert torch.allclose(one_hot.sum(dim=-1), torch.ones(5), atol=1e-5), \
        "One-hot rows should sum to 1"
    assert (one_hot.max(dim=-1).values == 1.0).all(), \
        "Each position should have a 1.0 (hard one-hot)"


def test_optimizer_deterministic_eval():
    """Eval mode is deterministic (argmax, no sampling)."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=5)
    opt.eval()
    with torch.no_grad():
        out1, _ = opt()
        out2, _ = opt()

    assert torch.equal(out1, out2), "Eval mode should be deterministic"


def test_optimizer_has_gamma_beta():
    """Optimizer has learnable gamma and beta parameters."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)

    assert hasattr(opt, 'gamma'), "Should have gamma parameter"
    assert hasattr(opt, 'beta'), "Should have beta parameter"
    assert opt.gamma.shape == (4,), f"gamma shape should be (4,), got {opt.gamma.shape}"
    assert opt.beta.shape == (4,), f"beta shape should be (4,), got {opt.beta.shape}"
    assert torch.allclose(opt.gamma.data, torch.ones(4)), "gamma should init to 1"
    assert torch.allclose(opt.beta.data, torch.zeros(4)), "beta should init to 0"


def test_optimizer_instance_norm():
    """Forward applies per-channel instance normalization."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=10)
    opt.eval()
    with torch.no_grad():
        _, scaled = opt()

    # With gamma=1 and beta=0, scaled logits should have ~zero mean and
    # ~unit variance per channel (across positions)
    for j in range(4):
        col = scaled[:, j]
        assert abs(col.mean().item()) < 1e-4, \
            f"Channel {j} mean should be ~0, got {col.mean().item()}"
        assert abs(col.var(unbiased=False).item() - 1.0) < 1e-4, \
            f"Channel {j} var should be ~1, got {col.var(unbiased=False).item()}"


def test_decode_sequence_valid_dna():
    """decode_sequence returns a string of valid DNA bases with correct length."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=6)
    opt.eval()
    with torch.no_grad():
        one_hot, _ = opt()

    seq = opt.decode_sequence(one_hot)

    assert len(seq) == 6, f"Expected length 6, got {len(seq)}"
    assert all(b in 'ATGC' for b in seq), f"Invalid base in: {seq}"


def test_get_complement():
    """get_complement returns correct Watson-Crick reverse complement."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)

    # ATGC -> complement: TACG -> reverse: GCAT
    assert opt.get_complement("ATGC") == "GCAT"
    assert opt.get_complement("AAAA") == "TTTT"
    # GCGC -> complement: CGCG -> reverse: GCGC (palindrome)
    assert opt.get_complement("GCGC") == "GCGC"


def test_get_complement_inverse():
    """Reverse complement of reverse complement is the original sequence."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)
    original = "ATCG"
    assert opt.get_complement(opt.get_complement(original)) == original


def test_build_graph_topology():
    """Single-stranded graph from soft features has same topology as hard graph."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.graph import sequence_to_graph

    opt = SequenceOptimizer(seq_length=4)
    one_hot, _ = opt()
    soft_graph = opt.build_graph_with_soft_features(one_hot)

    hard_graph = sequence_to_graph("AAAA")

    assert soft_graph.edge_index.shape == hard_graph.edge_index.shape, \
        f"Edge shape mismatch: {soft_graph.edge_index.shape} vs {hard_graph.edge_index.shape}"
    assert torch.equal(soft_graph.edge_index, hard_graph.edge_index), \
        "Edge topology should match"
    assert torch.equal(soft_graph.edge_attr, hard_graph.edge_attr), \
        "Edge attributes should match"
    assert soft_graph.x.shape == hard_graph.x.shape, \
        f"Node feature shape mismatch: {soft_graph.x.shape} vs {hard_graph.x.shape}"


def test_build_graph_soft_features():
    """Soft features appear in correct node positions."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)
    one_hot, _ = opt()
    soft_graph = opt.build_graph_with_soft_features(one_hot)

    # Primary strand features (indices 2..5) should be the one_hot values
    assert torch.allclose(soft_graph.x[2:6, :4], one_hot, atol=1e-6), \
        "Primary strand features should be the ST one-hot outputs"

    # Contact node features should be zeros
    assert torch.equal(soft_graph.x[0], torch.zeros(4)), "Left contact should be zeros"
    assert torch.equal(soft_graph.x[1], torch.zeros(4)), "Right contact should be zeros"


def test_compute_loss_l1_norm():
    """Loss is negative L1 norm of transmission difference."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)

    t_single = torch.tensor([[1.0, 2.0, 3.0]])
    t_double = torch.tensor([[0.0, 0.0, 0.0]])

    loss = opt.compute_loss(t_single, t_double)

    expected = -torch.norm(t_single - t_double, p=1)
    assert torch.allclose(loss, expected, atol=1e-5), \
        f"Expected {expected}, got {loss}"


def test_compute_loss_with_mask():
    """Energy mask restricts loss to selected energy points."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)

    t_single = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    t_double = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([1.0, 0.0, 0.0, 1.0])

    loss = opt.compute_loss(t_single, t_double, energy_mask=mask)

    expected = -torch.norm((t_single - t_double) * mask, p=1)
    assert torch.allclose(loss, expected, atol=1e-5), \
        f"Expected {expected}, got {loss}"


def test_optimize_loss_decreases():
    """Losses trend downward over training."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    losses = opt.optimize(predictor, num_steps=50, lr=0.1, log_every=50)

    assert len(losses) == 50
    # Compare average of first 5 vs last 5 (more robust than single points)
    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    assert late_avg < early_avg, \
        f"Loss should decrease: early avg {early_avg:.4f} vs late avg {late_avg:.4f}"


def test_optimize_gradients_flow():
    """Logits have non-None grad after an optimization step."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    opt.optimize(predictor, num_steps=1, lr=0.1, log_every=10)

    assert opt.logits.grad is not None, "Logits should have gradients"
    assert opt.logits.grad.abs().sum() > 0, "Gradients should be non-zero"


def test_optimize_gamma_gradients_flow():
    """Gamma has non-None grad after an optimization step (adaptive entropy)."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    opt.optimize(predictor, num_steps=1, lr=0.1, log_every=10)

    assert opt.gamma.grad is not None, "Gamma should have gradients"
    assert opt.gamma.grad.abs().sum() > 0, "Gamma gradients should be non-zero"


def test_optimize_gamma_changes():
    """Gamma values change during training (adaptive entropy is learning)."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    gamma_init = opt.gamma.data.clone()

    opt.optimize(predictor, num_steps=50, lr=0.1, log_every=100)

    assert not torch.equal(opt.gamma.data, gamma_init), \
        "Gamma should change during training (adaptive entropy)"


def test_optimizer_importable():
    """SequenceOptimizer is importable from top-level package."""
    from g3nat import SequenceOptimizer
    from g3nat.models import SequenceOptimizer as Opt2
    from g3nat.models.generator import SequenceOptimizer as Opt3

    assert SequenceOptimizer is Opt2
    assert SequenceOptimizer is Opt3


MODEL_PATH = "trained_models/hamiltonian_2000x_4to10BP_5000epoch.pth"


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason=f"Trained model not found at {MODEL_PATH}"
)
def test_optimize_with_trained_model():
    """Integration test: deterministic sequence improves after optimization."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.evaluation import load_trained_model

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor, energy_grid, device = load_trained_model(MODEL_PATH)

    # Evaluate deterministic sequence BEFORE training
    predictor.requires_grad_(False)
    predictor.eval()
    opt.eval()
    with torch.no_grad():
        oh, _ = opt()
        seq = opt.decode_sequence(oh)
        gs = opt.build_graph_with_soft_features(oh)
        gd = opt.build_graph_with_soft_features(oh, opt.get_complement(seq))
        _, ts = predictor(gs)
        _, td = predictor(gd)
        loss_before = opt.compute_loss(ts, td).item()

    losses = opt.optimize(predictor, num_steps=100, lr=0.1, log_every=25)
    assert len(losses) == 100

    # Evaluate deterministic sequence AFTER training
    opt.eval()
    with torch.no_grad():
        oh, _ = opt()
        seq = opt.decode_sequence(oh)
        gs = opt.build_graph_with_soft_features(oh)
        gd = opt.build_graph_with_soft_features(oh, opt.get_complement(seq))
        _, ts = predictor(gs)
        _, td = predictor(gd)
        loss_after = opt.compute_loss(ts, td).item()

    assert loss_after < loss_before, \
        f"Deterministic loss should improve: before {loss_before:.4f} vs after {loss_after:.4f}"
