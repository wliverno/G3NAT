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
from g3nat.training import (train_model, TrainingConfig, LengthBucketBatchSampler,
                            set_init_seed)
from g3nat.training.callbacks import save_checkpoint, save_progress_file
from g3nat.utils import setup_device

from torch_geometric.loader import DataLoader
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
    parser.add_argument('--ldos_target', type=str, default='residue',
                       choices=['residue', 'base_only'],
                       help='which LDOS aggregation to train against')
    parser.add_argument('--loss_a', type=float, default=1.0,
                        help='weight on the transmission loss term')
    parser.add_argument('--loss_b', type=float, default=0.0,
                        help='convex mixing weight: b*LDOS + (1-b)*DOS')
    parser.add_argument('--raw_scale_loss', action='store_true',
                       help='Compare DOS/LDOS by absolute magnitude. This is now the '
                            'DEFAULT (the flag is a no-op kept for older scripts/notes '
                            'that still pass it) -- the basis-size justification for '
                            'shape comparison was wrong and has been retracted; the '
                            'measured DOS offset is a measurement of missing frontier '
                            'states in the HOMO+/-1eV window, not a basis-size artifact. '
                            'See TrainingConfig.shape_loss for the derivation. Mutually '
                            'exclusive with --shape_loss.')
    parser.add_argument('--shape_loss', action='store_true',
                       help='Opt IN to comparing DOS/LDOS by a shared offset-corrected '
                            'shape instead of absolute magnitude (pre-2026-07-30 '
                            'behaviour, since corrected: DOS and LDOS now share ONE '
                            'offset derived from DOS, so the LDOS localization signal '
                            'is not deleted -- see Trainer._compute_losses). Transmission '
                            'is never centered under either setting. Mutually exclusive '
                            'with --raw_scale_loss.')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='hamiltonian',
                       choices=['standard', 'hamiltonian'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--n_orb', type=int, default=1)
    parser.add_argument('--conv_type', type=str, default='gat',
                       choices=['gat', 'transformer'],
                       help='Graph convolution type for the hamiltonian model. Default gat '
                            'for continuity with existing runs, NOT because it is measurably '
                            'better: the old "0.547 vs 1.42" claim compared final-epoch '
                            'values under a leaking split and is retracted. On best-val with '
                            'a grouped split the two TIE (gat 0.592 +/- 0.010 over 3 seeds, '
                            'transformer 0.579 over 1). transformer does fit the synthetic TB '
                            'data better. See docs/model-results.md.')
    parser.add_argument('--use_geometry', action='store_true',
                       help='Fuse SE(3)-invariant X3DNA edge geometry (hamiltonian model). '
                            'Requires a geometry cache built via GeomCacheJob.')
    parser.add_argument('--geom_cache', type=str, default='geom_cache/geometry.pkl',
                       help='Path to the per-sequence geometry cache (used with --use_geometry).')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                       help="Optimizer. 'adam' (default) reproduces historical runs exactly. "
                            "'adamw' decouples weight decay -- Loshchilov & Hutter ICLR 2019 "
                            "show Adam's weight_decay is not true weight decay, so the "
                            "effective regularization is weaker than the nominal value.")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay. Default 1e-5 matches the historical hardcoded value.')
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Seed for the sequence-grouped train/val split. Controls WHICH '
                            'sequences are held out, and nothing else.')
    parser.add_argument('--init_seed', type=int, default=None,
                       help='Seed for model initialization, independent of --split_seed. '
                            'Default None leaves the RNGs untouched, reproducing historical '
                            'runs exactly. Set it to vary initialization at a FIXED split, '
                            'which is what a reproducibility sweep over H actually requires.')
    parser.add_argument('--structured_onsite', action='store_true',
                       help='Mix a per-base onsite baseline with the context head.')
    parser.add_argument('--alpha_granularity', choices=['global', 'per_base'], default='global')
    parser.add_argument('--alpha_mode', choices=['fixed', 'learned'], default='fixed')
    parser.add_argument('--alpha_value', type=float, default=0.0,
                       help='Fixed mixing factor in [0,1] (alpha_mode=fixed).')
    parser.add_argument('--alpha_init', type=float, default=0.9,
                       help='Initial mixing factor (alpha_mode=learned).')

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
    assert not (args.alpha_granularity == 'per_base' and args.alpha_mode == 'fixed'), \
        "per_base+fixed needs 4 alphas; use learned or global"
    assert not (args.raw_scale_loss and args.shape_loss), \
        "--raw_scale_loss and --shape_loss are mutually exclusive (absolute is now " \
        "the default; --raw_scale_loss is a no-op kept for older invocations)"

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
        ldos_data = None
    else:  # pickle
        if args.data_dir is None:
            raise ValueError("--data_dir required for pickle data source")
        print(f"Loading pickle files from {args.data_dir}...")
        seqs, comp_seqs, dos_data, trans_data, energy_grid, contact_configs, ldos_data = \
            load_pickle_directory(args.data_dir)

        # Extract contact configurations for pickle data
        left_contact_pos_list = [c['left_contact_pos'] for c in contact_configs]
        right_contact_pos_list = [c['right_contact_pos'] for c in contact_configs]
        left_coupling_list = [c['coupling'] for c in contact_configs]
        right_coupling_list = [c['coupling'] for c in contact_configs]

    print(f"Loaded {len(seqs)} samples")

    # Optional SE(3)-invariant edge geometry (hamiltonian model only)
    geom_cache = None
    geom_norm_stats = None
    if args.use_geometry:
        import pickle as _pk
        from g3nat.graph.geometry import compute_norm_stats
        print(f"Loading geometry cache from {args.geom_cache}...")
        with open(args.geom_cache, 'rb') as _f:
            geom_cache = _pk.load(_f)
        geom_norm_stats = compute_norm_stats(geom_cache)
        print(f"Geometry cache loaded: {len(geom_cache)} sequences; per-type norm stats computed")

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
            right_contact_coupling_list=right_coupling_list,
            geometry_cache=geom_cache,
            ldos_data=ldos_data,
            ldos_target=args.ldos_target
        )
    else:
        dataset = create_dna_dataset(
            sequences=seqs,
            dos_data=dos_data,
            transmission_data=trans_data,
            energy_grid=energy_grid,
            complementary_sequences=comp_seqs,
            geometry_cache=geom_cache
        )

    # Split dataset -- GROUPED by sequence so no sequence appears in both train and val.
    from g3nat.data.splits import grouped_split
    train_indices, val_indices = grouped_split(seqs, test_size=0.2, seed=args.split_seed)
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

    # Seed initialization AFTER the split (which uses its own seed) and BEFORE
    # constructing the model, so --init_seed controls weights and nothing else.
    if set_init_seed(args.init_seed):
        print(f"Initialization seeded with {args.init_seed}")
    else:
        print("Initialization NOT seeded (pass --init_seed for reproducible weights)")

    # Create model
    if args.model_type == 'standard':
        model = g3nat.DNATransportGNN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            output_dim=len(energy_grid),
            dropout=args.dropout,
            # Was omitted, so --conv_type was silently ignored here and this
            # model always used its own default ('transformer') while the
            # hamiltonian model honoured the flag. Any standard-vs-hamiltonian
            # comparison run before 2026-08-01 therefore compared two different
            # convolutions as well as two different readouts.
            conv_type=args.conv_type
        )
    else:
        model = g3nat.DNATransportHamiltonianGNN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            energy_grid=energy_grid,
            n_orb=args.n_orb,
            conv_type=args.conv_type,
            use_geometry=args.use_geometry,
            geom_norm_stats=geom_norm_stats,
            structured_onsite=args.structured_onsite,
            alpha_granularity=args.alpha_granularity,
            alpha_mode=args.alpha_mode,
            alpha_value=args.alpha_value,
            alpha_init=args.alpha_init,
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    #
    # BEST-VAL CHECKPOINTING (added 2026-07-24). The final-epoch weights are NOT the best
    # weights. Measured over six runs at identical config and identical split_seed: best val
    # is reached at epoch 549-1900 of 5000, and the model then overfits for the remaining
    # 3000-4500 epochs, ending a mean of 0.060 worse (max 0.115). That drift is also the
    # dominant source of run-to-run scatter -- final-epoch std 0.0286 vs best-val std 0.0084,
    # 3.4x tighter -- and it is capacity-dependent, so it penalises deeper models more and
    # can invert an ordering (it inverted the num_layers trend). See docs/metrics.md.
    #
    # Granularity: this fires with the checkpoint callback (every checkpoint_frequency
    # epochs), so the saved weights are within that many epochs of the true optimum, not
    # exactly at it.
    # NOTE: seeded from resume_val_losses AFTER the resume block below, which is where
    # that variable is defined. Do not move this initialisation down into the callback.
    best_val = {'value': float('inf')}

    def checkpoint_cb(model, opt, epoch, train_losses, val_losses, metric_history=None):
        save_checkpoint(model, opt, epoch, train_losses, val_losses,
                       vars(args), energy_grid,
                       os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth'),
                       metric_history=metric_history)
        # Save only when THIS epoch is the running minimum, so the stored weights actually
        # correspond to the stored value. Testing min(val_losses) instead would fire at the
        # next checkpoint after a dip and save weights from a worse epoch under the name
        # "best" (observed: best at epoch 44, weights saved from epoch 50).
        # Consequence: this is the best among CHECKPOINTED epochs, not the global best
        # epoch. With checkpoint_frequency=10 we sample every 10th epoch. The reported
        # 'best_val' below is the true global minimum of the curve and may be slightly
        # lower than the val loss of these weights; both are stored so the gap is visible.
        if val_losses and float(val_losses[-1]) <= float(min(val_losses)) + 1e-12 \
                and float(val_losses[-1]) < best_val['value'] - 1e-12:
            best_val['value'] = float(val_losses[-1])
            save_checkpoint(model, opt, epoch, train_losses, val_losses,
                           vars(args), energy_grid,
                           os.path.join(args.checkpoint_dir, 'checkpoint_best.pth'),
                           metric_history=metric_history)

    def progress_cb(epoch, train_loss, val_loss):
        save_progress_file(epoch, train_loss, val_loss, args.checkpoint_dir, vars(args))

    # Resume from checkpoint if one exists
    start_epoch = 0
    resume_train_losses = None
    resume_val_losses = None
    resume_optimizer = None
    resume_metric_history = None
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=str(device), weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        resume_train_losses = ckpt['train_losses']
        resume_val_losses = ckpt['val_losses']
        # save_checkpoint() writes 'metric_history' whenever the caller supplies
        # it (checkpoint_cb below does, via Trainer.fit()'s checkpoint_callback).
        # Older checkpoints written before this wiring existed will not carry
        # the key, so guard with .get() and start empty rather than KeyError.
        resume_metric_history = ckpt.get('metric_history')
        # Must match the optimizer the Trainer will build, or a requeue silently switches
        # optimizer mid-run and the loaded state_dict is applied to the wrong type.
        _Opt = torch.optim.AdamW if args.optimizer.lower() == 'adamw' else torch.optim.Adam
        resume_optimizer = _Opt(model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay)
        resume_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Resuming from epoch {start_epoch}")
        # Carry the running best across a requeue, or the first post-resume checkpoint
        # would overwrite a genuinely better earlier one.
        if resume_val_losses:
            best_val['value'] = float(min(resume_val_losses))
            print(f"Resuming best val: {best_val['value']:.4f}")

    print("Training...")
    metric_history = []
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        device=str(device),
        checkpoint_frequency=10,
        checkpoint_callback=checkpoint_cb,
        progress_callback=progress_cb,
        start_epoch=start_epoch,
        train_losses=resume_train_losses,
        val_losses=resume_val_losses,
        optimizer=resume_optimizer,
        loss_a=args.loss_a,
        loss_b=args.loss_b,
        ldos_target=args.ldos_target,
        shape_loss=args.shape_loss,
        metric_history=resume_metric_history,
        metric_history_out=metric_history
    )

    # Save final model
    model_path = os.path.join(args.output_dir, f'{args.model_type}_{args.data_source}_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'energy_grid': energy_grid,
        'metric_history': metric_history
    }, model_path)

    # Also publish the BEST-val weights next to the final ones. Analysis that reads the
    # model (onsite values, eta2, LDOS) should prefer these -- the final weights are
    # thousands of epochs past the optimum. Loss comparisons can use either, since
    # val_losses is stored in both.
    best_ckpt = os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
    best_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_ckpt):
        bc = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        torch.save({
            'model_state_dict': bc['model_state_dict'],
            'args': vars(args),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'energy_grid': energy_grid,
            'metric_history': metric_history,
            'best_val': float(min(val_losses)),
            'best_val_epoch': int(np.argmin(val_losses)),
            'saved_at_epoch': bc.get('epoch'),
        }, best_path)

    print(f"Training complete!")
    print(f"Model saved: {model_path}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"BEST val loss:  {min(val_losses):.4f} at epoch {int(np.argmin(val_losses))}")
    if os.path.exists(best_path):
        print(f"Best model saved: {best_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

if __name__ == '__main__':
    main()
