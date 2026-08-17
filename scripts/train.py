#!/usr/bin/env python3
"""Unified training script for G3NAT (TB and pickle data)."""

import argparse
import os
import sys
import time

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
    parser.add_argument('--loss_c', type=float, default=1.0,
                        help='weight on the whole DOS family: total = a*T + '
                             'c*(b*LDOS + (1-b)*DOS). Default 1.0 reproduces every '
                             'run on record exactly; 0.0 is transmission-only '
                             'training, the previously unreachable arm '
                             '(willll, 2026-08-10 -- see TrainingConfig.loss_c).')
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

    # NEGF-layer knobs. These previously existed ONLY as constructor defaults, so they
    # were never written into the checkpoint's `args`, and evaluation re-specified its own
    # defaults independently. For solver_type the two disagreed -- training used 'complex',
    # inference used 'frobenius' -- so every model on record was evaluated with a solver it
    # was not trained with, and log_floor became a dead knob at eval time. Defaults here
    # match DNATransportHamiltonianGNN.__init__ exactly, so behaviour is unchanged; the
    # point is that vars(args) now records them and the checkpoint carries them forward.
    # EXCEPTION (deliberate): --log_floor defaults to 1e-38 here, not the model's legacy
    # 1e-16. The floor is now a smoothing eps rather than a clamp; old checkpoints keep
    # whatever value they recorded.
    parser.add_argument('--solver_type', choices=['complex', 'frobenius'], default='complex',
                        help='NEGF solver. Must match at train and eval time.')
    parser.add_argument('--log_floor', type=float, default=1e-38,
                        help='Smoothing eps for log10 of DOS/T: log10(max(x,0)+eps). Pure '
                             'log10(0) guard -- never binds on physical values (dataset T '
                             'minimum is 6.7e-19). Recorded in args; must match at train '
                             'and eval.')
    parser.add_argument('--complex_eta', type=float, default=1e-12)
    parser.add_argument('--use_log_outputs', type=lambda s: s.lower() != 'false', default=True)
    parser.add_argument('--enforce_hermiticity', type=lambda s: s.lower() != 'false', default=True)
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
    parser.add_argument('--geom_cache', type=str, default='geom_cache/geometry_v2.pkl',
                       help='Path to the per-sequence geometry cache (used with --use_geometry). '
                            'Defaults to the v2 cache (520 sequences, matches pickle_files_v2); '
                            'the older geometry.pkl covers only 515 v1 sequences.')
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

CONFIG_DEFINING_ARGS = [
    'data_source', 'data_dir', 'model_type', 'hidden_dim', 'num_layers', 'num_heads',
    'n_orb', 'solver_type', 'log_floor', 'complex_eta', 'use_log_outputs',
    'enforce_hermiticity', 'conv_type', 'use_geometry', 'geom_cache', 'optimizer',
    'weight_decay', 'split_seed', 'init_seed', 'loss_a', 'loss_b', 'loss_c',
    'ldos_target', 'shape_loss', 'batch_size', 'learning_rate', 'num_epochs',
]


def check_resume_args(stored: dict, current: dict) -> None:
    """A checkpoint may only resume the run that wrote it. Raises on any mismatch
    of a config-defining arg, naming the offending key -- resuming under different
    args silently republishes one config's weights under another's label."""
    problems = []
    for key in CONFIG_DEFINING_ARGS:
        if key not in stored:
            problems.append(f"{key}: missing from checkpoint args")
        elif stored[key] != current.get(key):
            problems.append(f"{key}: checkpoint={stored[key]!r} vs current={current.get(key)!r}")
    if problems:
        raise ValueError(
            "checkpoint_latest.pth was written by a DIFFERENT configuration; refusing "
            "to resume. Use a fresh --checkpoint_dir per run. Mismatches: "
            + "; ".join(problems))


def main():
    args = parse_args()
    assert not (args.alpha_granularity == 'per_base' and args.alpha_mode == 'fixed'), \
        "per_base+fixed needs 4 alphas; use learned or global"
    assert not (args.raw_scale_loss and args.shape_loss), \
        "--raw_scale_loss and --shape_loss are mutually exclusive (absolute is now " \
        "the default; --raw_scale_loss is a no-op kept for older invocations)"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    from g3nat.utils.runmeta import write_run_metadata
    meta_path = write_run_metadata(args.output_dir, vars(args))
    print(f"Run metadata: {meta_path}")

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

        if args.num_energy_points != len(energy_grid):
            print(f"NOTE: --num_energy_points ({args.num_energy_points}) is ignored for "
                  f"pickle data; the grid comes from the files ({len(energy_grid)} points).")

    print(f"Loaded {len(seqs)} samples")

    # Optional SE(3)-invariant edge geometry (hamiltonian model only)
    geom_cache = None
    geom_norm_stats = None
    if args.use_geometry:
        import pickle as _pk
        print(f"Loading geometry cache from {args.geom_cache}...")
        with open(args.geom_cache, 'rb') as _f:
            geom_cache = _pk.load(_f)
        print(f"Geometry cache loaded: {len(geom_cache)} sequences")

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

    # Geometry norm stats computed from the TRAIN split only, so val/test sequences
    # do not leak into the z-score normalization.
    if args.use_geometry:
        from g3nat.graph.geometry import compute_norm_stats
        train_seqs = {seqs[i].lower() for i in train_indices}
        geom_norm_stats = compute_norm_stats(geom_cache, sequences=train_seqs)
        print(f"Geometry norm stats computed from {len(train_seqs)} training sequences")

    # Seed initialization AFTER the split (which uses its own seed) and BEFORE
    # constructing the loaders/model, so --init_seed controls weights (and, via
    # the sampler's explicit seed= below, batch composition) and nothing else.
    if set_init_seed(args.init_seed):
        print(f"Initialization seeded with {args.init_seed}")
    else:
        print("Initialization NOT seeded (pass --init_seed for reproducible weights)")

    # Create loaders
    is_hamiltonian = (args.model_type == 'hamiltonian')
    if is_hamiltonian:
        train_sampler = LengthBucketBatchSampler(train_dataset, args.batch_size,
                                                 shuffle=True, seed=args.init_seed)
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
            dropout=args.dropout,
            # Was omitted, so --conv_type was silently ignored here and this
            # model always used its own default ('transformer') while the
            # hamiltonian model honoured the flag. Any standard-vs-hamiltonian
            # comparison run before 2026-08-01 therefore compared two different
            # convolutions as well as two different readouts.
            conv_type=args.conv_type
        )
    else:
        if args.dropout != 0.0:
            print(f"WARNING: --dropout ({args.dropout}) has no effect on the hamiltonian "
                  "model (it has no dropout layers); the flag applies to --model_type "
                  "standard only.")
        model = g3nat.DNATransportHamiltonianGNN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            energy_grid=energy_grid,
            n_orb=args.n_orb,
            solver_type=args.solver_type,
            log_floor=args.log_floor,
            complex_eta=args.complex_eta,
            use_log_outputs=args.use_log_outputs,
            enforce_hermiticity=args.enforce_hermiticity,
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
    # Granularity: NO LONGER a rounding to the checkpoint cadence. As of 2026-08-16 the
    # Trainer keeps the best weights in memory, refreshed every epoch on the UNWEIGHTED
    # metric val_dos_t_unweighted, and hands them here as `best_state`. So the serialized
    # "best" weights are exactly the ones from the optimum epoch, and the selection
    # criterion no longer depends on loss_b (the weighted 'total' is scaled differently
    # in every supervision cell, which made "best" incomparable across arms).
    # NOTE: seeded from resume_val_losses AFTER the resume block below, which is where
    # that variable is defined. Do not move this initialisation down into the callback.
    best_val = {'value': float('inf')}

    def checkpoint_cb(model, opt, epoch, train_losses, val_losses, metric_history=None,
                      best_state=None):
        save_checkpoint(model, opt, epoch, train_losses, val_losses,
                       vars(args), energy_grid,
                       os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth'),
                       metric_history=metric_history)
        # Save the in-memory BEST-EPOCH weights the trainer handed us, whenever they beat
        # what has already been written to disk. best_state['value'] is the unweighted
        # metric val_dos_t_unweighted at best_state['epoch']; best_val['value'] tracks the
        # last value actually serialized here, so this single test is the whole condition.
        #
        # HISTORY (see docs/model-results.md sec. 16). Two successive defects lived here:
        # (a) until 2026-08-11 this also required `val_losses[-1] <= min(val_losses)`, so
        # once epoch-to-epoch noise exceeded the improvement across a checkpoint interval
        # it stopped firing essentially permanently -- over the 84 published runs, stored
        # weights came from median epoch 874 against a true optimum at median epoch 1730,
        # and in 14 of 84 the stored "best" was worse than the final epoch; (b) even after
        # that fix, the weights written were the LIVE model's, i.e. the checkpointed
        # epoch's, not the optimum's, and the criterion was the loss_b-weighted total.
        # Both are gone: the trainer snapshots a detached CPU copy of the weights at the
        # exact epoch that minimises the unweighted metric, and that snapshot is what is
        # serialized below. 'saved_at_epoch' is therefore the true optimum epoch.
        if best_state and best_state.get('state_dict') is not None \
                and best_state['value'] < best_val['value'] - 1e-12:
            best_val['value'] = float(best_state['value'])
            best_ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
            payload = {
                'epoch': best_state['epoch'],
                'model_state_dict': best_state['state_dict'],
                'optimizer_state_dict': opt.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': vars(args),
                'energy_grid': energy_grid,
                'metric_history': metric_history,
                'selection_metric': 'val_dos_t_unweighted',
                'selection_value': float(best_state['value']),
                'timestamp': time.time(),
            }
            # Direct torch.save rather than save_checkpoint, because the weights are the
            # in-memory best snapshot, not the live model's. Same atomic write-then-rename
            # as callbacks.save_checkpoint -- these runs are preemptible and a truncated
            # zip here costs the whole best-weights record.
            tmp_path = f"{best_ckpt_path}.tmp"
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, best_ckpt_path)
            except BaseException:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
            print(f"Checkpoint saved: {best_ckpt_path}")

    def progress_cb(epoch, train_loss, val_loss):
        save_progress_file(epoch, train_loss, val_loss, args.checkpoint_dir, vars(args))

    # Resume from checkpoint if one exists
    start_epoch = 0
    resume_train_losses = None
    resume_val_losses = None
    resume_optimizer = None
    resume_metric_history = None
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

    best_path_stale = os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
    if not os.path.exists(checkpoint_path) and os.path.exists(best_path_stale):
        # No latest checkpoint means this is a FRESH run in a reused dir: a leftover
        # best would be republished under the new args. With a latest checkpoint
        # present we are resuming, and the best is this run's own -- keep it.
        os.remove(best_path_stale)
        print(f"Removed stale checkpoint_best.pth from a previous run in {args.checkpoint_dir}")

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=str(device), weights_only=False)
        check_resume_args(ckpt.get('args', {}), vars(args))
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
        # would overwrite a genuinely better earlier one. Seed it from the SAME quantity
        # the callback now compares against -- the unweighted metric, not the weighted
        # val loss, which is a different scale entirely and would make the comparison
        # meaningless. A checkpoint written before metric_history existed carries no such
        # values, in which case the running best restarts at inf: the first post-resume
        # improvement overwrites the old best. That is the safe direction (the trainer's
        # in-memory best is empty after a restart anyway, so nothing better is lost).
        _resume_metrics = [m.get('val_dos_t_unweighted') for m in (resume_metric_history or [])]
        _resume_metrics = [float(m) for m in _resume_metrics if m is not None and m == m]
        if _resume_metrics:
            best_val['value'] = min(_resume_metrics)
            print(f"Resuming best val_dos_t_unweighted: {best_val['value']:.4f}")

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
        loss_c=args.loss_c,
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
            # NaN-safe: a single poisoned epoch in val_losses would otherwise make
            # argmin/min return that nan's index and value, publishing a nonsense
            # best-epoch. nanargmin raises only if EVERY epoch is nan, which is a
            # failure worth surfacing.
            'best_val': float(np.nanmin(val_losses)),
            'best_val_epoch': int(np.nanargmin(val_losses)),
            'saved_at_epoch': bc.get('epoch'),
            # What the published weights were actually selected on -- the unweighted
            # metric, not the loss_b-weighted 'best_val' above.
            'selection_metric': bc.get('selection_metric'),
            'selection_value': bc.get('selection_value'),
        }, best_path)

    print(f"Training complete!")
    print(f"Model saved: {model_path}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"BEST val loss:  {float(np.nanmin(val_losses)):.4f} at epoch {int(np.nanargmin(val_losses))}")
    if os.path.exists(best_path):
        print(f"Best model saved: {best_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

if __name__ == '__main__':
    main()
