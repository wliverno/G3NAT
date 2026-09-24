#!/usr/bin/env python3
"""Unified training script for G3NAT (TB and pickle data)."""

import argparse
import os
import sys
import time

# Ensure g3nat package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from g3nat.floor import LOG_FLOOR
import numpy as np

import g3nat
from g3nat.data import (generate_tight_binding_data, load_pickle_directory,
                        create_dna_dataset)
from g3nat.training import (train_model, TrainingConfig, LengthBucketBatchSampler,
                            set_init_seed)
from g3nat.training.callbacks import save_checkpoint, save_progress_file
from g3nat.training.selection import resolve_selection_metric, selection_value
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
                             '(see TrainingConfig.loss_c).')
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
                       choices=['standard', 'hamiltonian', 'standard_deep', 'hamiltonian_deep'])
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
    parser.add_argument('--allow_frobenius', action='store_true',
                        help='Escape hatch for legacy comparisons only; the Frobenius path '
                             'is silently wrong at resonances and ignores complex_eta.')
    parser.add_argument('--log_floor', type=float, default=LOG_FLOOR,
                        help='The one floor: log10(max(x, LOG_FLOOR)) on DOS, LDOS and T (g3nat/floor.py). '
                             'Never binds on training data (T minimum 6.7e-19); removes the bulk of the '
                             'reference background artifact on held-out 12- and 16-mers.')
    parser.add_argument('--floor_mode', choices=['clamp', 'smooth'], default='clamp',
                        help="Floor SEMANTICS, recorded in args. 'smooth' is "
                             'log10(max(x,0)+eps) -- gradient survives in the deep tail. '
                             "'clamp' is the pre-2026-08-15 hard clamp log10(max(x,eps)); "
                             'the model constructor defaults to it so that checkpoints '
                             'whose args predate this flag reproduce their old numbers.')
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
                            'data better. See private notes on the analysis campaign.')
    parser.add_argument('--use_geometry', action='store_true',
                       help='Fuse SE(3)-invariant X3DNA edge geometry (hamiltonian model). '
                            'Requires a geometry cache built with g3nat.graph.geometry.build_geometry_cache.')
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
    parser.add_argument('--init_seed', type=int, required=True,
                        help='Seed for model initialization AND batch composition, '
                             'independent of --split_seed. REQUIRED: an unseeded run '
                             'is not reproducible, and set_init_seed(None) silently '
                             'touches no RNG at all, so the failure is invisible. '
                             'Campaign v3 uses 1179027592, 2129768291, 3731635825.')
    parser.add_argument('--per_base_onsite', action='store_true',
                       help='Onsite = a learned per-base table (4 values shared across '
                            'all A/T/G/C sites) instead of the context head. Off in the '
                            'campaign; one post-factorial epilogue run turns it on. '
                            'Replaces the removed continuous alpha mix: this flag is the '
                            'old alpha=1, and its absence the old alpha=0.')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='auto')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--allow_arg_change', type=str, default='',
                       help='Comma-separated list of argument names that MAY differ '
                            'between this invocation and the checkpoint being resumed '
                            '(e.g. "num_epochs" when raising the epoch cap on a '
                            'requeued cell). Every exemption actually exercised is '
                            'recorded in the run\'s resolved_config.json under '
                            '"arg_exemptions", so the change stays legible in the '
                            'run artifacts rather than only in a shell history.')

    return parser.parse_args()

# INVERTED GUARD (2026-08-16, independent review finding I8). This used to be an
# ALLOWLIST, CONFIG_DEFINING_ARGS, which already omitted 15 real arguments --
# --dropout and every alpha flag among them -- so a resume could silently switch
# them, and every future flag was unguarded by default. The list is now a DENYLIST
# of the three arguments that genuinely describe the execution environment rather
# than the configuration; everything else must match. New flags are guarded the day
# they are added, with no list to remember to update.
#
# 'allow_arg_change' is always self-exempt: it is the mechanism for declaring an
# exemption, so requiring it to match would make it impossible to introduce on the
# requeue that needs it. Its value is recorded in resolved_config.json regardless.
NON_DEFINING_ARGS = {'device', 'output_dir', 'checkpoint_dir'}
_SELF_EXEMPT_ARGS = {'allow_arg_change'}


def parse_allow_arg_change(spec) -> list:
    """Parse the --allow_arg_change comma-separated spec into a list of keys."""
    if not spec:
        return []
    if isinstance(spec, (list, tuple, set)):
        return sorted(str(k).strip() for k in spec if str(k).strip())
    return [k.strip() for k in str(spec).split(',') if k.strip()]


def check_resume_args(stored: dict, current: dict, allow_changed=()) -> list:
    """A checkpoint may only resume the run that wrote it.

    Raises ValueError on any mismatch of a config-defining arg, naming every
    offending key -- resuming under different args silently republishes one
    config's weights under another's label.

    `allow_changed` names keys that may differ deliberately (from
    --allow_arg_change). Returns the list of exemptions that were actually
    exercised, so the caller can record them in the run metadata.
    """
    allowed = set(allow_changed)
    keys = (set(stored) | set(current)) - NON_DEFINING_ARGS - _SELF_EXEMPT_ARGS
    problems = []
    exemptions = []
    for key in sorted(keys):
        if key not in stored:
            desc = f"{key}: missing from checkpoint args"
        elif key not in current:
            desc = f"{key}: missing from current args"
        elif stored[key] != current[key]:
            desc = f"{key}: checkpoint={stored[key]!r} vs current={current[key]!r}"
        else:
            continue
        (exemptions if key in allowed else problems).append(desc)
    if problems:
        raise ValueError(
            "checkpoint_latest.pth was written by a DIFFERENT configuration; refusing "
            "to resume. Use a fresh --checkpoint_dir per run, or pass "
            "--allow_arg_change with the comma-separated keys you intend to change. "
            "Mismatches: " + "; ".join(problems))
    return exemptions


def maybe_clear_stale_best(checkpoint_dir: str) -> bool:
    """Delete a leftover checkpoint_best.pth ONLY when there is no checkpoint_latest.

    No latest checkpoint means this is a FRESH run in a reused dir: a leftover best
    would be republished under the new args. With a latest checkpoint present we are
    RESUMING, and the best is this run's own -- deleting it would throw away the best
    weights of a preempted run. These runs are preemptible with --requeue, so that
    second branch is exercised constantly; it is the whole reason the gate exists.

    Returns True if a stale best was removed.
    """
    latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    best = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
    if not os.path.exists(latest) and os.path.exists(best):
        os.remove(best)
        return True
    return False


#: The criterion every run used before campaign v3, kept ONLY so that a caller
#: that passes no weights reproduces the old fallback exactly (the pre-v3 tests
#: for `seed_best_value`). Production callers must pass the run's resolved
#: weights -- see the call site in main().
_LEGACY_SELECTION_WEIGHTS = {'val_dos_t_unweighted': 1.0}


def seed_best_value(checkpoint_dir: str, metric_history,
                    selection_weights=None) -> float:
    """The running best selection value to carry across a requeue.

    Prefer the on-disk checkpoint_best.pth's OWN 'selection_value'. The history
    minimum is not equivalent: checkpoint_latest.pth is written BEFORE
    checkpoint_best.pth in checkpoint_cb, so a kill between the two leaves
    metric_history (stored in latest) ahead of the weights actually on disk. Seeding
    from the history then sets a bar BETTER than the stored weights, and a later
    epoch that genuinely improves on the stored weights never gets republished --
    the run finishes carrying weights it already beat.

    Falls back to the history minimum only when the best checkpoint is absent or
    predates the 'selection_value' key. That fallback is a REAL path here: these
    runs are preemptible, so a run killed before it ever wrote a best checkpoint
    lands on it. The fallback therefore has to use the SAME criterion the trainer
    selects on (`selection_weights`, from this run's loss weights). Using the old
    fixed val_dos_t_unweighted would set a bar on a different quantity: for LDOS+T
    the run's own T+LDOS can EXCEED T+DOS, so the bar would be too strict and
    checkpoint_best.pth would never be written again for the rest of the run.

    An entry missing any contributing key is skipped rather than crashing the
    resume -- pre-v3 histories do not carry every term.
    """
    weights = selection_weights or _LEGACY_SELECTION_WEIGHTS
    best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
    if os.path.exists(best_path):
        try:
            bc = torch.load(best_path, map_location='cpu', weights_only=False)
        except Exception as exc:  # unreadable/truncated best: fall through to history
            print(f"WARNING: could not read {best_path} ({exc}); "
                  "seeding the running best from metric_history instead")
            bc = None
        if isinstance(bc, dict):
            sv = bc.get('selection_value')
            # Only trust the stored value when it is the SAME QUANTITY this run
            # minimises. A pre-v3 checkpoint_best.pth left in a reused checkpoint
            # dir stores a val_dos_t_unweighted number under this key; seeding a
            # v3 bar from it compares two different scales, which is the defect
            # fixed in the history fallback below, one branch up.
            recorded = bc.get('selection_weights')
            same_criterion = (recorded == weights if recorded is not None
                              else selection_weights is None)
            if sv is not None and float(sv) == float(sv) and same_criterion:
                return float(sv)
            if sv is not None and not same_criterion:
                print(f"WARNING: {best_path} was selected on {recorded!r}, not on this "
                      f"run's {weights!r}; seeding the running best from "
                      "metric_history instead")
    values = []
    for m in (metric_history or []):
        try:
            v = selection_value(m, weights)
        except (KeyError, TypeError, ValueError):
            continue
        if v == v and abs(v) != float('inf'):
            values.append(float(v))
    return min(values) if values else float('inf')


def best_publication_warning(best_ckpt_path: str, metric_history,
                             selection_metric_name=None):
    """Return a WARNING string when no best checkpoint exists, else None.

    Without this the script prints "Training complete!" and exits 0 after a run in
    which the selection metric was non-finite every single epoch: best_unweighted
    ['state_dict'] stays None, checkpoint_best.pth is never written, no _best.pth is
    published, and every downstream analysis silently falls back to final-epoch
    weights or skips the run. Same silent-failure class as private notes
    sec. 16.

    THE CAUSE IS COUNTED FROM THE RUN'S OWN CRITERION. `nan_selection_metric_total`
    is the trainer's cumulative count of epochs whose OWN selection metric came back
    non-finite (trainer.py, _validate_epoch). Counting non-finite
    `val_dos_t_unweighted` instead -- which this used to do -- states a WRONG CAUSE
    for exactly the arms campaign v3 introduced: an LDOS or T-only run can have a
    perfectly finite val_dos_t_unweighted in every epoch while its own criterion is
    nan throughout, and the operator would be told the metric was fine and pointed
    at the checkpoint callback, the wrong subsystem entirely.

    Falls back to counting non-finite `val_dos_t_unweighted` only for pre-v3
    histories, which do not carry the counter.
    """
    if os.path.exists(best_ckpt_path):
        return None
    history = list(metric_history or [])
    n_epochs = len(history)
    name = selection_metric_name or 'the selection metric'
    counter = history[-1].get('nan_selection_metric_total') if history else None
    if counter is not None and counter == counter:
        n_bad = int(counter)
    else:
        n_bad = sum(1 for m in history
                    if not (m.get('val_dos_t_unweighted') is not None
                            and m.get('val_dos_t_unweighted') == m.get('val_dos_t_unweighted')))
    if n_epochs == 0:
        cause = (f"no validation epoch ran, so the selection metric ({name}) "
                 "was never computed")
    elif n_bad >= n_epochs:
        cause = (f"the selection metric ({name}) was NON-FINITE in all "
                 f"{n_epochs} epochs, so no epoch could ever become the best")
    elif n_bad:
        cause = (f"the selection metric ({name}) was non-finite in "
                 f"{n_bad} of {n_epochs} epochs and no finite epoch improved on the "
                 "running best")
    else:
        cause = (f"the selection metric ({name}) was finite in all {n_epochs} epochs "
                 "but no best checkpoint was written -- the checkpoint callback may "
                 "never have fired, or the file was removed")
    return ("WARNING: NO BEST CHECKPOINT WAS PUBLISHED for this run. "
            + cause + ". The published final weights are the LAST epoch's, not the "
            "best; treat this run as FAILED rather than as a completed cell.")


def main():
    args = parse_args()
    assert not (args.raw_scale_loss and args.shape_loss), \
        "--raw_scale_loss and --shape_loss are mutually exclusive (absolute is now " \
        "the default; --raw_scale_loss is a no-op kept for older invocations)"
    if args.solver_type == 'frobenius' and not args.allow_frobenius:
        raise SystemExit(
            "--solver_type frobenius is disabled for training runs: its singular-matrix "
            "fallback is silently ~98% wrong at resonances and it ignores --complex_eta. "
            "Pass --allow_frobenius only for legacy comparisons.")

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

        # --num_samples / --seq_length / --min_length are synthetic-TB generator
        # arguments (generate_tight_binding_data above). For pickle data the
        # sample count and the strand lengths are whatever the directory holds,
        # so these flags do nothing. SAY SO: a 120-run campaign sized off
        # --num_samples would be sized off a flag with no effect.
        _ignored_tb_flags = [f'--num_samples ({args.num_samples})',
                             f'--seq_length ({args.seq_length})',
                             f'--min_length ({args.min_length})']
        print(f"NOTE: {', '.join(_ignored_tb_flags)} are ignored for pickle data; "
              f"they size the synthetic tight-binding generator only. This run uses "
              f"the {len(seqs)} sample(s) found in {args.data_dir}.")

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
    # BOTH model types use the same seeded, requeue-safe sampler. The blind branch
    # previously used DataLoader(shuffle=True) with no generator, which draws from
    # the global RNG: batch order is then unreproducible across a preemption
    # requeue, while the Hamiltonian branch (seeded, with set_epoch) is not. Two
    # families being compared head to head cannot have different reproducibility
    # guarantees -- one of them would carry extra run-to-run variance that has
    # nothing to do with the model.
    train_sampler = LengthBucketBatchSampler(train_dataset, args.batch_size,
                                             shuffle=True, seed=args.init_seed)
    # val_sampler's seed= is inert: shuffle=False means LengthBucketBatchSampler
    # never consults it (see g3nat/training/utils.py _rng), so validation order
    # is already deterministic without it. Kept anyway to match the train_sampler
    # construction, not because it does anything.
    val_sampler = LengthBucketBatchSampler(val_dataset, args.batch_size,
                                           shuffle=False, seed=args.init_seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    # Create model
    # The _deep probe variants take their base family's branch with the Deep class
    # and EXACTLY the same keyword arguments.
    _model_cls = {
        'standard': g3nat.DNATransportGNN,
        'standard_deep': g3nat.DNATransportGNNDeep,
        'hamiltonian': g3nat.DNATransportHamiltonianGNN,
        'hamiltonian_deep': g3nat.DNATransportHamiltonianGNNDeep,
    }[args.model_type]
    if args.model_type in ('standard', 'standard_deep'):
        model = _model_cls(
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
            conv_type=args.conv_type,
            # SAME OMISSION, SAME CONSTRUCTOR, FOUND 2026-08-24. --use_geometry
            # reached the hamiltonian model (below) and not this one, so every
            # standard-vs-hamiltonian comparison with geometry enabled gave the
            # hamiltonian model an input the baseline could not see -- an
            # advantage unrelated to the readout, which is the thing under test.
            #
            # THE PATTERN: this constructor takes its arguments explicitly and
            # silently keeps its own defaults for anything omitted. Adding a flag
            # to the hamiltonian branch without adding it here produces no error
            # and no warning; it produces a confounded experiment. Check both
            # branches whenever a shared flag is added.
            use_geometry=args.use_geometry,
            geom_norm_stats=geom_norm_stats,
        )
    else:
        if args.dropout != 0.0:
            print(f"WARNING: --dropout ({args.dropout}) has no effect on the hamiltonian "
                  "model (it has no dropout layers); the flag applies to --model_type "
                  "standard only.")
        model = _model_cls(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            energy_grid=energy_grid,
            n_orb=args.n_orb,
            solver_type=args.solver_type,
            log_floor=args.log_floor,
            floor_mode=args.floor_mode,
            complex_eta=args.complex_eta,
            use_log_outputs=args.use_log_outputs,
            enforce_hermiticity=args.enforce_hermiticity,
            conv_type=args.conv_type,
            use_geometry=args.use_geometry,
            geom_norm_stats=geom_norm_stats,
            per_base_onsite=args.per_base_onsite,
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
    # Trainer keeps the best weights in memory, refreshed every epoch on the run's OWN
    # validation objective (g3nat/training/selection.py; before campaign v3 this was the
    # fixed metric val_dos_t_unweighted), and hands them here as `best_state`. So the serialized
    # "best" weights are exactly the ones from the optimum epoch, and the selection
    # criterion no longer depends on loss_b (the weighted 'total' is scaled differently
    # in every supervision cell, which made "best" incomparable across arms).
    # NOTE: seeded from resume_val_losses AFTER the resume block below, which is where
    # that variable is defined. Do not move this initialisation down into the callback.
    best_val = {'value': float('inf')}

    # Resolved once, from this run's loss weights, and written into every
    # checkpoint so a consumer can verify what selection actually optimised.
    _sel_name, _sel_weights = resolve_selection_metric(
        args.loss_a, args.loss_b, args.loss_c, ldos_target=args.ldos_target)
    print(f'Selecting best checkpoint on: {_sel_name}')

    def checkpoint_cb(model, opt, epoch, train_losses, val_losses, metric_history=None,
                      best_state=None):
        save_checkpoint(model, opt, epoch, train_losses, val_losses,
                       vars(args), energy_grid,
                       os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth'),
                       metric_history=metric_history, init_seed=args.init_seed)
        # Save the in-memory BEST-EPOCH weights the trainer handed us, whenever they beat
        # what has already been written to disk. best_state['value'] is this run's own
        # selection metric (_sel_name) at best_state['epoch']; best_val['value'] tracks the
        # last value actually serialized here, so this single test is the whole condition.
        #
        # HISTORY (see private notes sec. 16). Two successive defects lived here:
        # (a) until 2026-08-11 this also required `val_losses[-1] <= min(val_losses)`, so
        # once epoch-to-epoch noise exceeded the improvement across a checkpoint interval
        # it stopped firing essentially permanently -- over the 84 published runs, stored
        # weights came from median epoch 874 against a true optimum at median epoch 1730,
        # and in 14 of 84 the stored "best" was worse than the final epoch; (b) even after
        # that fix, the weights written were the LIVE model's, i.e. the checkpointed
        # epoch's, not the optimum's, and the criterion was the loss_b-weighted total.
        # Both are gone: the trainer snapshots a detached CPU copy of the weights at the
        # exact epoch that minimises this run's own objective, and that snapshot is what is
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
                'init_seed': args.init_seed,
                'energy_grid': energy_grid,
                'metric_history': metric_history,
                'selection_metric': _sel_name,
                'selection_weights': _sel_weights,
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

    if maybe_clear_stale_best(args.checkpoint_dir):
        print(f"Removed stale checkpoint_best.pth from a previous run in {args.checkpoint_dir}")

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=str(device), weights_only=False)
        _allow_changed = parse_allow_arg_change(args.allow_arg_change)
        _exemptions = check_resume_args(ckpt.get('args', {}), vars(args),
                                        allow_changed=_allow_changed)
        if _exemptions:
            # Record every exemption actually exercised in the run's own metadata,
            # so a deliberate change (raising --num_epochs on a requeued cell) is
            # readable from the artifacts rather than only from the submit command.
            from g3nat.utils.runmeta import update_run_metadata
            update_run_metadata(args.output_dir, arg_exemptions=_exemptions)
            for _e in _exemptions:
                print(f"NOTE: resuming with an EXEMPTED argument change -- {_e}")
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
        # the callback now compares against -- this run's own selection metric
        # (_sel_weights), not the weighted val loss, which is a different scale entirely
        # and would make the comparison meaningless. Passing _sel_weights is REQUIRED:
        # seed_best_value's history fallback otherwise defaults to the pre-v3 fixed
        # criterion, which for LDOS+T sets a bar the run's own objective cannot beat.
        #
        # A checkpoint written before metric_history existed carries no such
        # values, in which case the running best restarts at inf: the first post-resume
        # improvement overwrites the old best. That is the safe direction (the trainer's
        # in-memory best is empty after a restart anyway, so nothing better is lost).
        #
        # The value comes from the on-disk best checkpoint's own 'selection_value'
        # when present AND recorded under the same criterion, NOT from the history
        # minimum -- see seed_best_value's docstring for why those differ after a
        # kill between the two writes.
        best_val['value'] = seed_best_value(args.checkpoint_dir, resume_metric_history,
                                            selection_weights=_sel_weights)
        if best_val['value'] != float('inf'):
            print(f"Resuming best {_sel_name}: {best_val['value']:.4f}")

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
        'init_seed': args.init_seed,
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
            'init_seed': args.init_seed,
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
            # What the published weights were actually selected on -- this run's own
            # validation objective, not the loss_b-weighted 'best_val' above.
            'selection_metric': bc.get('selection_metric'),
            # Carried through as well as the NAME. load_trained_model's
            # selection-validity guard prefers the recorded weights, and this
            # published file is what it is normally pointed at -- without the key
            # here the guard falls back to a name map that contains none of the
            # v3 names and silently returns, which is the exact bypass the guard
            # exists to prevent.
            'selection_weights': bc.get('selection_weights'),
            'selection_value': bc.get('selection_value'),
        }, best_path)

    # A run with no best checkpoint is a FAILED run, not a completed one. Say so
    # loudly instead of printing "Training complete!" and exiting 0.
    #
    # THE NON-ZERO EXIT IS THE POINT, not the warning text. Printing a warning and
    # exiting 0 makes SLURM record the run COMPLETED, so `sacct` shows a full
    # factorial while that cell published nothing. Nothing downstream counts the
    # runs -- hand the gates 90 of 120 checkpoints and all of them report PASS --
    # so a lost cell is invisible. The arms most exposed are exactly the ones
    # campaign v3 introduces (LDOS+T at n_orb=2, never run before; and T only),
    # which means cells would go missing NON-UNIFORMLY BY ARM: a silently wrong
    # paper number with every gate green.
    #
    # THIS DOES NOT AFFECT REQUEUE. A preempted run is killed inside the epoch loop
    # and never reaches this line; requeue is driven by SLURM's preemption signal
    # and checkpoint_latest.pth, not by this exit code. checkpoint_latest.pth is
    # deliberately NOT removed on this path either, so a failed cell can still be
    # resumed or inspected. Do NOT "fix" this back to warn-and-exit-0.
    _no_best_warning = best_publication_warning(best_ckpt, metric_history,
                                                selection_metric_name=_sel_name)
    if _no_best_warning is not None:
        print(_no_best_warning)
        print(f"Model saved: {model_path}")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
        raise SystemExit(
            "TRAINING FAILED: no best checkpoint was published, so this cell "
            "contributes nothing to the factorial. Exiting NON-ZERO so SLURM "
            "records FAILED rather than COMPLETED.")

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
