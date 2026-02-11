#!/usr/bin/env python3
"""End-to-end integration test for G3NAT pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np

import g3nat
from g3nat.data import generate_tight_binding_data, create_dna_dataset
from g3nat.training import train_model, LengthBucketBatchSampler
from g3nat.utils import setup_device

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def test_full_pipeline():
    """
    Test complete training pipeline with TB data.

    Tests:
    - Generate tight-binding data
    - Create dataset
    - Train model for 2 epochs
    - Verify losses are finite
    """
    print("=" * 60)
    print("End-to-End Integration Test")
    print("=" * 60)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Small dataset for fast test
    num_samples = 20
    seq_length = 8
    num_energy_points = 100

    print("\n1. Generating tight-binding data...")
    seqs, comp_seqs, dos_data, trans_data, energy_grid = generate_tight_binding_data(
        num_samples=num_samples,
        seq_length=seq_length,
        num_energy_points=num_energy_points
    )
    print(f"   Generated {len(seqs)} samples")

    print("\n2. Creating dataset...")
    dataset = create_dna_dataset(
        sequences=seqs,
        dos_data=dos_data,
        transmission_data=trans_data,
        energy_grid=energy_grid,
        complementary_sequences=comp_seqs
    )
    print(f"   Dataset size: {len(dataset)}")

    print("\n3. Splitting train/val...")
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print("\n4. Creating data loaders...")
    batch_size = 4
    train_sampler = LengthBucketBatchSampler(train_dataset, batch_size, shuffle=True)
    val_sampler = LengthBucketBatchSampler(val_dataset, batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    print("\n5. Creating model...")
    model = g3nat.DNATransportHamiltonianGNN(
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        energy_grid=energy_grid,
        dropout=0.0,
        n_orb=1
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n6. Training for 2 epochs...")
    device = setup_device('cpu')  # Use CPU for test
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        learning_rate=1e-3,
        device=str(device),
        checkpoint_frequency=10
    )

    print("\n7. Verifying results...")
    print(f"   Train losses: {train_losses}")
    print(f"   Val losses: {val_losses}")

    # Verify losses are finite
    assert len(train_losses) == 2, f"Expected 2 train losses, got {len(train_losses)}"
    assert len(val_losses) == 2, f"Expected 2 val losses, got {len(val_losses)}"

    for i, loss in enumerate(train_losses):
        assert torch.isfinite(torch.tensor(loss)), f"Train loss {i} is not finite: {loss}"

    for i, loss in enumerate(val_losses):
        assert torch.isfinite(torch.tensor(loss)), f"Val loss {i} is not finite: {loss}"

    print("\n" + "=" * 60)
    print("PASSED: End-to-end integration test")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
