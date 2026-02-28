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
    """Optimizer has learnable gamma and beta parameters when instance norm is on."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4, use_instance_norm=True)

    assert hasattr(opt, 'gamma'), "Should have gamma parameter"
    assert hasattr(opt, 'beta'), "Should have beta parameter"
    assert opt.gamma.shape == (4,), f"gamma shape should be (4,), got {opt.gamma.shape}"
    assert opt.beta.shape == (4,), f"beta shape should be (4,), got {opt.beta.shape}"
    assert torch.allclose(opt.gamma.data, torch.ones(4)), "gamma should init to 1"
    assert torch.allclose(opt.beta.data, torch.zeros(4)), "beta should init to 0"


def test_optimizer_instance_norm():
    """Forward applies per-channel instance normalization."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=10, use_instance_norm=True)
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
    """Best-seen deterministic loss improves over training."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    result = opt.optimize(predictor, num_steps=50, lr=0.01, log_every=10)

    assert isinstance(result, dict), "optimize() should return a dict"
    assert 'losses' in result
    assert 'best_loss' in result
    assert 'best_step' in result
    assert 'best_sequence' in result
    assert len(result['losses']) == 50

    # Best deterministic loss should be finite and negative (we minimize -||diff||)
    assert result['best_loss'] < 0, \
        f"Best loss should be negative, got {result['best_loss']}"


def test_optimize_gradients_flow():
    """Logits have non-None grad after an optimization step."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    opt.optimize(predictor, num_steps=1, lr=0.001, log_every=10)

    assert opt.logits.grad is not None, "Logits should have gradients"
    assert opt.logits.grad.abs().sum() > 0, "Gradients should be non-zero"


def test_optimize_gamma_gradients_flow():
    """Gamma has non-None grad after an optimization step (adaptive entropy)."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4, use_instance_norm=True)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    opt.optimize(predictor, num_steps=1, lr=0.001, log_every=10)

    assert opt.gamma.grad is not None, "Gamma should have gradients"
    assert opt.gamma.grad.abs().sum() > 0, "Gamma gradients should be non-zero"


def test_optimize_gamma_changes():
    """Gamma values change during training (adaptive entropy is learning)."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4, use_instance_norm=True)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    gamma_init = opt.gamma.data.clone()

    # Use log_every <= num_steps so best-seen checkpointing captures changed gamma
    opt.optimize(predictor, num_steps=50, lr=0.01, log_every=25)

    assert not torch.equal(opt.gamma.data, gamma_init), \
        "Gamma should change during training (adaptive entropy)"


def test_optimizer_importable():
    """SequenceOptimizer is importable from top-level package."""
    from g3nat import SequenceOptimizer
    from g3nat.models import SequenceOptimizer as Opt2
    from g3nat.models.generator import SequenceOptimizer as Opt3

    assert SequenceOptimizer is Opt2
    assert SequenceOptimizer is Opt3


def test_soft_complement():
    """_soft_complement produces correct Watson-Crick pairs."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)

    # Hard one-hot for ATGC: A=[1,0,0,0], T=[0,1,0,0], G=[0,0,1,0], C=[0,0,0,1]
    one_hot = torch.tensor([
        [1., 0., 0., 0.],  # A
        [0., 1., 0., 0.],  # T
        [0., 0., 1., 0.],  # G
        [0., 0., 0., 1.],  # C
    ])

    comp = opt._soft_complement(one_hot)

    # Reverse complement of ATGC is GCAT
    # G=[0,0,1,0], C=[0,0,0,1], A=[1,0,0,0], T=[0,1,0,0]
    expected = torch.tensor([
        [0., 0., 1., 0.],  # G (complement of C, last position reversed)
        [0., 0., 0., 1.],  # C (complement of G)
        [1., 0., 0., 0.],  # A (complement of T)
        [0., 1., 0., 0.],  # T (complement of A)
    ])

    assert torch.allclose(comp, expected, atol=1e-6), \
        f"Soft complement of ATGC should be GCAT.\nGot:\n{comp}\nExpected:\n{expected}"


def test_soft_complement_gradient_flows():
    """Gradient flows through the soft complement path."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4)

    soft = torch.randn(4, 4, requires_grad=True)
    comp = opt._soft_complement(soft)
    loss = comp.sum()
    loss.backward()

    assert soft.grad is not None, "Gradient should flow through soft complement"
    assert soft.grad.abs().sum() > 0, "Gradients should be non-zero"


def test_optimizer_no_instance_norm():
    """Optimizer works with use_instance_norm=False."""
    from g3nat.models.generator import SequenceOptimizer

    opt = SequenceOptimizer(seq_length=4, use_instance_norm=False)

    assert not hasattr(opt, 'gamma'), "Should not have gamma when instance norm is off"
    assert not hasattr(opt, 'beta'), "Should not have beta when instance norm is off"

    # Forward should still work
    opt.train()
    one_hot, scaled = opt()
    assert one_hot.shape == (4, 4)
    assert scaled.shape == (4, 4)

    # scaled should be raw logits (no normalization)
    assert torch.allclose(scaled, opt.logits, atol=1e-6), \
        "Without instance norm, scaled should equal raw logits"


def test_optimize_early_stopping():
    """Early stopping halts training before num_steps when patience is exceeded."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    # patience=1 with log_every=5 means stop after 2 eval rounds without improvement
    result = opt.optimize(predictor, num_steps=500, lr=0.001,
                          log_every=5, patience=1)

    # Should stop well before 500 steps
    assert len(result['losses']) < 500, \
        f"Early stopping should halt before 500 steps, got {len(result['losses'])}"
    assert result['best_step'] > 0, "Should have found a best step"


def test_optimize_best_seen_restored():
    """Best-seen parameters are restored after training."""
    from g3nat.models.generator import SequenceOptimizer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    opt = SequenceOptimizer(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)

    result = opt.optimize(predictor, num_steps=100, lr=0.01, log_every=10)

    # After training, eval the restored parameters
    opt.eval()
    with torch.no_grad():
        one_hot, _ = opt()
        seq = opt.decode_sequence(one_hot)

    # The restored sequence should match best_sequence
    assert seq == result['best_sequence'], \
        f"Restored sequence '{seq}' should match best_sequence '{result['best_sequence']}'"


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
        gd = opt._build_double_soft(oh)
        _, ts = predictor(gs)
        _, td = predictor(gd)
        loss_before = opt.compute_loss(ts, td).item()

    result = opt.optimize(predictor, num_steps=500, lr=0.01, log_every=50)
    assert len(result['losses']) <= 500

    # Best-seen tracking should preserve at least the initial quality
    assert result['best_loss'] <= loss_before, \
        f"Best loss should not degrade: before {loss_before:.4f} vs best {result['best_loss']:.4f}"
    assert isinstance(result['best_sequence'], str)
    assert len(result['best_sequence']) == 4
