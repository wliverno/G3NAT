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
