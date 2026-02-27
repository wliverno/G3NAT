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
        "Edge topology should match"
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
