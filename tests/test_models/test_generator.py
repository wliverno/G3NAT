import torch
import pytest


def test_generator_output_shapes():
    """Generator produces correct output shapes."""
    from g3nat.models.generator import DNASequenceGenerator

    gen = DNASequenceGenerator(seq_length=5, latent_dim=16, hidden_dim=32, num_hidden_layers=2)
    soft_bases, logits = gen(batch_size=3)

    assert soft_bases.shape == (3, 5, 4), f"Expected (3, 5, 4), got {soft_bases.shape}"
    assert logits.shape == (3, 5, 4), f"Expected (3, 5, 4), got {logits.shape}"


def test_generator_softmax_sums_to_one():
    """Soft bases from Gumbel-Softmax sum to ~1.0 per position."""
    from g3nat.models.generator import DNASequenceGenerator

    gen = DNASequenceGenerator(seq_length=5, latent_dim=16, hidden_dim=32)
    soft_bases, _ = gen(batch_size=2)

    sums = soft_bases.sum(dim=-1)  # [2, 5]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Softmax sums not close to 1.0: {sums}"


def test_generator_accepts_explicit_z():
    """Generator uses provided latent vector instead of sampling."""
    from g3nat.models.generator import DNASequenceGenerator

    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16)
    z = torch.randn(2, 8)
    soft_bases, logits = gen(z=z)

    assert soft_bases.shape == (2, 4, 4)


def test_generator_deterministic_with_same_z():
    """Same z input produces same output."""
    from g3nat.models.generator import DNASequenceGenerator

    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16)
    gen.eval()
    z = torch.randn(1, 8)

    with torch.no_grad():
        out1, _ = gen(z=z)
        out2, _ = gen(z=z)

    assert torch.allclose(out1, out2), "Same z should produce same output in eval mode"


def test_decode_sequences_returns_valid_dna():
    """decode_sequences returns strings of valid DNA bases with correct length."""
    from g3nat.models.generator import DNASequenceGenerator

    gen = DNASequenceGenerator(seq_length=6, latent_dim=8, hidden_dim=16)
    gen.eval()
    with torch.no_grad():
        soft_bases, _ = gen(batch_size=3)

    sequences = gen.decode_sequences(soft_bases)

    assert len(sequences) == 3
    for seq in sequences:
        assert len(seq) == 6, f"Expected length 6, got {len(seq)}"
        assert all(b in 'ATGC' for b in seq), f"Invalid base in: {seq}"


def test_get_complement():
    """get_complement returns correct Watson-Crick complement."""
    from g3nat.models.generator import DNASequenceGenerator

    gen = DNASequenceGenerator(seq_length=4)

    assert gen.get_complement("ATGC") == "TACG"
    assert gen.get_complement("AAAA") == "TTTT"
    assert gen.get_complement("GCGC") == "CGCG"


def test_get_complement_inverse():
    """Complement of complement is the original sequence."""
    from g3nat.models.generator import DNASequenceGenerator

    gen = DNASequenceGenerator(seq_length=4)
    original = "ATCG"
    assert gen.get_complement(gen.get_complement(original)) == original


def test_build_single_stranded_graph_preserves_topology():
    """Single-stranded graph from soft features has same topology as hard graph."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.graph import sequence_to_graph
    from g3nat.models import DNATransportGNN

    seq_length = 4
    gen = DNASequenceGenerator(seq_length=seq_length, latent_dim=8, hidden_dim=16)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor)

    soft_bases, _ = gen(batch_size=1)
    soft_graph = trainer.build_graph_with_soft_features(soft_bases[0])

    # Compare with hard graph of same length
    hard_graph = sequence_to_graph("AAAA")  # dummy sequence, same length

    assert soft_graph.edge_index.shape == hard_graph.edge_index.shape, \
        f"Edge shape mismatch: {soft_graph.edge_index.shape} vs {hard_graph.edge_index.shape}"
    assert torch.equal(soft_graph.edge_index, hard_graph.edge_index), \
        "Edge topology should match between soft and hard single-stranded graphs"
    assert torch.equal(soft_graph.edge_attr, hard_graph.edge_attr), \
        "Edge attributes should match"
    assert soft_graph.x.shape == hard_graph.x.shape, \
        f"Node feature shape mismatch: {soft_graph.x.shape} vs {hard_graph.x.shape}"


def test_build_single_stranded_graph_has_soft_features():
    """Single-stranded graph node features are differentiable (from soft bases)."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.models import DNATransportGNN

    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor)

    soft_bases, _ = gen(batch_size=1)
    soft_graph = trainer.build_graph_with_soft_features(soft_bases[0])

    # Primary strand features (indices 2..5) should be the soft bases
    assert torch.allclose(soft_graph.x[2:6, :4], soft_bases[0], atol=1e-6), \
        "Primary strand features should be the soft Gumbel-Softmax outputs"

    # Contact node features should be zeros
    assert torch.equal(soft_graph.x[0], torch.zeros(4)), "Left contact should be zeros"
    assert torch.equal(soft_graph.x[1], torch.zeros(4)), "Right contact should be zeros"


def test_build_double_stranded_graph_topology():
    """Double-stranded graph has correct topology matching sequence_to_graph."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.graph import sequence_to_graph
    from g3nat.models import DNATransportGNN

    seq_length = 4
    gen = DNASequenceGenerator(seq_length=seq_length, latent_dim=8, hidden_dim=16)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor)

    gen.eval()
    with torch.no_grad():
        soft_bases, _ = gen(batch_size=1)

    sequences = gen.decode_sequences(soft_bases)
    complement = gen.get_complement(sequences[0])

    soft_graph = trainer.build_graph_with_soft_features(soft_bases[0], complement)
    hard_graph = sequence_to_graph(sequences[0], complement)

    assert soft_graph.x.shape == hard_graph.x.shape, \
        f"Node count mismatch: {soft_graph.x.shape} vs {hard_graph.x.shape}"
    assert soft_graph.edge_index.shape == hard_graph.edge_index.shape, \
        f"Edge shape mismatch: {soft_graph.edge_index.shape} vs {hard_graph.edge_index.shape}"
    assert torch.equal(soft_graph.edge_index, hard_graph.edge_index), \
        "Edge topology should match"


def test_compute_loss_returns_negative_diff():
    """Loss is negative L2 norm of transmission difference."""
    from g3nat.models.generator import GeneratorTrainer, DNASequenceGenerator
    from g3nat.models import DNATransportGNN

    gen = DNASequenceGenerator(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor)

    t_single = torch.tensor([[1.0, 2.0, 3.0]])
    t_double = torch.tensor([[0.0, 0.0, 0.0]])

    loss = trainer.compute_loss(t_single, t_double)

    expected_diff = torch.norm(t_single - t_double, p=2)
    assert torch.allclose(loss, -expected_diff, atol=1e-5), \
        f"Expected {-expected_diff}, got {loss}"


def test_compute_loss_identical_transmissions():
    """Loss is zero when transmissions are identical."""
    from g3nat.models.generator import GeneratorTrainer, DNASequenceGenerator
    from g3nat.models import DNATransportGNN

    gen = DNASequenceGenerator(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor)

    t = torch.tensor([[1.0, 2.0, 3.0]])
    loss = trainer.compute_loss(t, t)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), \
        f"Expected 0.0 for identical transmissions, got {loss}"


def test_compute_loss_with_energy_mask():
    """Energy mask restricts loss to selected energy points."""
    from g3nat.models.generator import GeneratorTrainer, DNASequenceGenerator
    from g3nat.models import DNATransportGNN

    gen = DNASequenceGenerator(seq_length=4)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor)

    t_single = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    t_double = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Only first and last energy points

    loss = trainer.compute_loss(t_single, t_double, energy_mask=mask)

    masked_diff = torch.norm((t_single - t_double) * mask, p=2)
    assert torch.allclose(loss, -masked_diff, atol=1e-5), \
        f"Expected {-masked_diff}, got {loss}"


def test_train_step_returns_loss_and_sequences():
    """train_step returns a loss value and generated sequences."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.models import DNATransportGNN

    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor, lr=1e-3)

    loss_val, sequences = trainer.train_step()

    assert isinstance(loss_val, float), f"Expected float loss, got {type(loss_val)}"
    assert isinstance(sequences, list), f"Expected list of sequences, got {type(sequences)}"
    assert len(sequences) == 1, f"Expected 1 sequence, got {len(sequences)}"
    assert len(sequences[0]) == 4, f"Expected length 4, got {len(sequences[0])}"
    assert all(b in 'ATGC' for b in sequences[0])


def test_train_step_generator_has_gradients():
    """After train_step, generator parameters should have received gradients."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.models import DNATransportGNN

    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor, lr=1e-3)

    trainer.train_step()

    has_grad = False
    for p in gen.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "Generator should have non-zero gradients after train_step"


def test_train_step_predictor_frozen():
    """Predictor parameters should not have gradients after train_step."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.models import DNATransportGNN

    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor, lr=1e-3)

    trainer.train_step()

    for p in predictor.parameters():
        assert not p.requires_grad, "Predictor params should be frozen"


def test_train_loop_loss_changes():
    """Training for multiple steps should change the loss."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(42)
    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor, lr=1e-2)

    losses = trainer.train(num_steps=20, log_every=10)

    assert len(losses) == 20, f"Expected 20 loss values, got {len(losses)}"
    # Loss should change over training (not stuck)
    assert losses[0] != losses[-1], "Loss should change during training"


def test_train_loop_sequences_change():
    """Generated sequences should evolve during training."""
    from g3nat.models.generator import DNASequenceGenerator, GeneratorTrainer
    from g3nat.models import DNATransportGNN

    torch.manual_seed(0)
    gen = DNASequenceGenerator(seq_length=4, latent_dim=8, hidden_dim=16, tau=0.5)
    predictor = DNATransportGNN(hidden_dim=16, num_layers=1, num_heads=1, output_dim=10)
    trainer = GeneratorTrainer(gen, predictor, lr=1e-2)

    # Get sequence before training
    gen.eval()
    with torch.no_grad():
        soft_before, _ = gen(batch_size=1)
    seq_before = gen.decode_sequences(soft_before)[0]

    # Train
    trainer.train(num_steps=50, log_every=50)

    # Get sequence after training
    gen.eval()
    with torch.no_grad():
        soft_after, _ = gen(batch_size=1)
    seq_after = gen.decode_sequences(soft_after)[0]

    # Sequences may or may not change (depends on optimization landscape)
    # But the soft bases should definitely be different
    assert not torch.allclose(soft_before, soft_after, atol=1e-3), \
        "Soft bases should change after training"


def test_generator_importable_from_top_level():
    """Generator classes should be importable from g3nat package."""
    from g3nat import DNASequenceGenerator
    from g3nat.models import DNASequenceGenerator as Gen2
    from g3nat.models.generator import GeneratorTrainer

    assert DNASequenceGenerator is Gen2
