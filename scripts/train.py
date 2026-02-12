#!/usr/bin/env python3
"""Unified training script for G3NAT (TB and pickle data)."""

import argparse
import os
import sys

# Ensure g3nat package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

import g3nat
from g3nat.data import (generate_tight_binding_data, load_pickle_directory,
                        create_dna_dataset)
from g3nat.training import train_model, TrainingConfig, LengthBucketBatchSampler
from g3nat.training.callbacks import save_checkpoint, save_progress_file
from g3nat.utils import setup_device

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def parse_args():
    parser = argparse.ArgumentParser(description='Train DNA Transport GNN')

    parser.add_argument('--data_source', type=str, required=True,
                       choices=['tb', 'pickle'],
                       help='Data source: tb (tight-binding) or pickle')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with pickle files (for pickle source)')
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of samples (for TB source)')
    parser.add_argument('--seq_length', type=int, default=8,
                       help='Sequence length (for TB source)')
    parser.add_argument('--min_length', type=int, default=-1,
                       help='Minimum sequence length for TB source (-1 = same as seq_length)')
    parser.add_argument('--num_energy_points', type=int, default=100,
                       help='Number of energy points')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='hamiltonian',
                       choices=['standard', 'hamiltonian'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_orb', type=int, default=1)

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='auto')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"G3NAT Training (v{g3nat.__version__})")
    print(f"Data source: {args.data_source}")
    print(f"Model type: {args.model_type}")

    device = setup_device(args.device)
    print(f"Device: {device}")

    # Load data
    if args.data_source == 'tb':
        print(f"Generating {args.num_samples} TB samples...")
        seqs, comp_seqs, dos_data, trans_data, energy_grid = generate_tight_binding_data(
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            num_energy_points=args.num_energy_points,
            min_length=args.min_length
        )
    else:  # pickle
        if args.data_dir is None:
            raise ValueError("--data_dir required for pickle data source")
        print(f"Loading pickle files from {args.data_dir}...")
        seqs, comp_seqs, dos_data, trans_data, energy_grid, contact_configs = \
            load_pickle_directory(args.data_dir)

        # Extract contact configurations for pickle data
        left_contact_pos_list = [c['left_contact_pos'] for c in contact_configs]
        right_contact_pos_list = [c['right_contact_pos'] for c in contact_configs]
        left_coupling_list = [c['coupling'] for c in contact_configs]
        right_coupling_list = [c['coupling'] for c in contact_configs]

    print(f"Loaded {len(seqs)} samples")

    # Create dataset
    if args.data_source == 'pickle':
        dataset = create_dna_dataset(
            sequences=seqs,
            dos_data=dos_data,
            transmission_data=trans_data,
            energy_grid=energy_grid,
            complementary_sequences=comp_seqs,
            left_contact_positions_list=left_contact_pos_list,
            right_contact_positions_list=right_contact_pos_list,
            left_contact_coupling_list=left_coupling_list,
            right_contact_coupling_list=right_coupling_list
        )
    else:
        dataset = create_dna_dataset(
            sequences=seqs,
            dos_data=dos_data,
            transmission_data=trans_data,
            energy_grid=energy_grid,
            complementary_sequences=comp_seqs
        )

    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=0.2, random_state=42
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create loaders
    is_hamiltonian = (args.model_type == 'hamiltonian')
    if is_hamiltonian:
        train_sampler = LengthBucketBatchSampler(train_dataset, args.batch_size, shuffle=True)
        val_sampler = LengthBucketBatchSampler(val_dataset, args.batch_size, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    if args.model_type == 'standard':
        model = g3nat.DNATransportGNN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            output_dim=len(energy_grid),
            dropout=args.dropout
        )
    else:
        model = g3nat.DNATransportHamiltonianGNN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            energy_grid=energy_grid,
            dropout=args.dropout,
            n_orb=args.n_orb
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    def checkpoint_cb(model, opt, epoch, train_losses, val_losses):
        save_checkpoint(model, opt, epoch, train_losses, val_losses,
                       vars(args), energy_grid,
                       os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth'))

    def progress_cb(epoch, train_loss, val_loss):
        save_progress_file(epoch, train_loss, val_loss, args.checkpoint_dir, vars(args))

    print("Training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=str(device),
        checkpoint_frequency=10,
        checkpoint_callback=checkpoint_cb,
        progress_callback=progress_cb
    )

    # Save final model
    model_path = os.path.join(args.output_dir, f'{args.model_type}_{args.data_source}_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'energy_grid': energy_grid
    }, model_path)

    print(f"Training complete!")
    print(f"Model saved: {model_path}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")

if __name__ == '__main__':
    main()
