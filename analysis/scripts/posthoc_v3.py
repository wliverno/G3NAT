"""Score all 120 campaign-v3 published checkpoints -> outputs/posthoc_v3_report.json.

Weights: outputs_v3/<run>/*_model_best.pth for every cell (each run's own-objective
argmin). Any exception anywhere kills the whole process with a non-zero exit and
writes nothing: there is no skip list and no partial report.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(_ANALYSIS))
sys.path.insert(0, REPO)
sys.path.insert(0, _ANALYSIS)

from g3nat.data.pickle import load_single_pickle
from g3nat.evaluation.inference import load_trained_model

from g3nat_analysis import evaluator as EV
from g3nat_analysis.metrics import eligible_substitution_pairs, select_pairs
from verify_gates import gate_geometry_reaches_cells

EXPECTED_RUNS = os.path.join(_ANALYSIS, 'expected_runs.txt')
OUTPUTS_V3 = os.path.join(REPO, 'outputs_v3')
PICKLE_DIR = os.path.join(REPO, 'pickle_files_v2')
HELDOUT_DIRS = {12: os.path.join(REPO, 'DNADataset', 'validation_L12', 'pickles'),
               16: os.path.join(REPO, 'DNADataset', 'validation_L16', 'pickles')}

#: Training-length cache plus the held-out L=12/L=16 cache, merged. geometry_v2.pkl
#: covers only lengths 4-8; without the second file every geom=1 cell would be scored
#: with the geometry channel zeroed (the campaign-v2 defect).
GEOM_CACHES = (os.path.join(REPO, 'geom_cache', 'geometry_v2.pkl'),
              os.path.join(REPO, 'geom_cache', 'geometry_heldout_L12_L16.pkl'))

DEFAULT_OUT = os.path.join(_ANALYSIS, 'outputs', 'posthoc_v3_report.json')

HAM_WEIGHTS = 'hamiltonian_pickle_model_best.pth'
BLIND_WEIGHTS = 'standard_pickle_model_best.pth'


def load_geometry(cache_paths):
    """Merged geometry cache. Missing files are FATAL, not skipped."""
    merged = {}
    for p in cache_paths:
        if not os.path.exists(p):
            raise SystemExit(
                f"geometry cache missing: {p}\n"
                f"Build the held-out cache with "
                f"analysis/scripts/build_heldout_geometry.py. Running without it "
                f"silently disables the geometry channel for every geom=1 cell.")
        with open(p, 'rb') as f:
            merged.update(pickle.load(f))
    print(f"geometry: {len(merged)} sequences from {len(cache_paths)} caches")
    return merged


def _run_key(path):
    """'<seq>_<runN>.pkl' -> 'runN'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit('_', 1)[1]


def load_training_records():
    """(sequence.lower(), run_key) -> record. Lowercase keys are mandatory: the
    substitution lookups in metrics.matched_records are lowercase."""
    records = {}
    paths = sorted(glob.glob(os.path.join(PICKLE_DIR, '*.pkl')))
    for p in paths:
        rec = load_single_pickle(p)
        if rec is None:
            raise SystemExit(f"FATAL: unreadable/invalid training pickle: {p}")
        records[(rec['sequence'].lower(), _run_key(p))] = rec
    return records


def load_heldout(length):
    """The 16 held-out DFT records at one length. None is FATAL."""
    recs = []
    paths = sorted(glob.glob(os.path.join(HELDOUT_DIRS[length], '*.pkl')))
    for p in paths:
        rec = load_single_pickle(p)
        if rec is None:
            raise SystemExit(f"FATAL: unreadable/invalid held-out pickle: {p}")
        recs.append(rec)
    if len(recs) != 16:
        raise SystemExit(
            f"FATAL: expected 16 held-out records at L={length}, found "
            f"{len(recs)} under {HELDOUT_DIRS[length]}")
    return recs


def _check_grid(model_grid, recs, label):
    """The model's energy grid must match EVERY record's."""
    mg = np.asarray(model_grid, dtype=float)
    for i, rec in enumerate(recs):
        ref = np.asarray(rec['energy_grid'], dtype=float)
        if not np.allclose(mg, ref):
            raise SystemExit(
                f"FATAL: energy grid mismatch ({label}, record {i}): model "
                f"grid does not match the data's energy_grid")
    return mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--date', required=True,
                    help='ISO date string stamped into provenance.written')
    ap.add_argument('--onsite-all-records', action='store_true',
                    help='score onsite_near/onsite_far over EVERY pickle_files_v2 '
                         'record (all sequences, all contact variants, train and '
                         'val) instead of the seeded 80-record sample. The metric '
                         'and its averaging are unchanged; only the record set '
                         'differs. No other column is affected.')
    args = ap.parse_args()

    with open(EXPECTED_RUNS) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if len(names) != 120:
        raise SystemExit(f"FATAL: expected_runs.txt has {len(names)} names, "
                         f"expected 120")
    parsed_names = {}
    for n in names:
        p = EV.parse_run_name(n)
        if p is None:
            raise SystemExit(f"FATAL: run name failed to parse: {n}")
        parsed_names[n] = p
    print(f"expected_runs: {len(names)} names, all parsed")
    run_list = list(names)

    cache_paths = list(GEOM_CACHES)
    geometry_cache = load_geometry(cache_paths)

    print("loading training records ...")
    train_records = load_training_records()
    print(f"training records: {len(train_records)}")

    heldout = {}
    for L in sorted(HELDOUT_DIRS):
        heldout[L] = load_heldout(L)
        print(f"L={L} held-out DFT records: {len(heldout[L])} (expect 16)")

    train_sequences = sorted({s for s, _k in train_records})
    pairs = select_pairs(eligible_substitution_pairs(train_sequences))
    if len(pairs) != 33:
        raise SystemExit(f"FATAL: expected exactly 33 substitution pairs, "
                         f"got {len(pairs)}")
    print(f"substitution pairs: {len(pairs)}")

    if args.onsite_all_records:
        onsite_sample = sorted(train_records)
        onsite_provenance = ('all records: every (sequence, run_key) in '
                             f'pickle_files_v2, n={len(onsite_sample)}')
        print(f"onsite: ALL {len(onsite_sample)} (sequence, run_key) records")
    else:
        onsite_sample = EV.sample_onsite_records(train_records)
        onsite_provenance = [list(x) for x in onsite_sample]
        print(f"onsite sample: {len(onsite_sample)} (sequence, run_key) entries")

    report_runs = {}
    n = len(run_list)
    for i, run in enumerate(run_list, 1):
        parsed = parsed_names[run]
        family = parsed['family']
        weights_name = HAM_WEIGHTS if family == 'ham' else BLIND_WEIGHTS
        path = os.path.join(OUTPUTS_V3, run, weights_name)
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: missing weights for {run}: {path}")

        payload = torch.load(path, map_location='cpu', weights_only=False)
        # load_trained_model's selection guard stays live (never allow_untrained_selection).
        model, model_grid, _dev = load_trained_model(path, device='cpu')

        name_geom = bool(parsed['geometry'])
        payload_geom = bool(payload.get('args', {}).get('use_geometry', False))
        model_geom = bool(getattr(model, 'use_geometry', False))
        if not (name_geom == payload_geom == model_geom):
            raise SystemExit(
                f"FATAL: geometry flag mismatch for {run}: "
                f"name={name_geom} payload.args.use_geometry={payload_geom} "
                f"model.use_geometry={model_geom}")

        val_t = EV.transmission_at_selection(payload)

        _check_grid(model_grid, heldout[12], f'{run} L12')
        _check_grid(model_grid, heldout[16], f'{run} L16')
        _check_grid(model_grid, list(train_records.values()), f'{run} train')

        row = dict(parsed)
        row['val_transmission_at_selection'] = val_t
        row['saved_at_epoch'] = payload.get('saved_at_epoch', payload.get('epoch'))

        diag_batch = EV.gated_batch(heldout[12][0], model_grid, geometry_cache,
                                    name_geom)
        gate_detail = gate_geometry_reaches_cells(diag_batch, name_geom).detail

        row.update(EV.eval_heldout(model, heldout[12], model_grid, 'l12',
                                   geometry_cache, name_geom))
        row.update(EV.eval_heldout(model, heldout[16], model_grid, 'l16',
                                   geometry_cache, name_geom))

        sub = EV.substitution_response(model, train_records, pairs, model_grid,
                                       geometry_cache, name_geom)
        if sub['sub_n_comparisons'] == 0:
            raise SystemExit(
                f"FATAL: sub_n_comparisons == 0 for {run} -- structurally "
                f"impossible with 33 pairs")
        row.update(sub)

        if family == 'ham':
            row.update(EV.onsite_over_sample(
                model, train_records, onsite_sample, model_grid,
                geometry_cache, name_geom, parsed['n_orb']))

        report_runs[run] = row
        print(f"[{i}/{n}] {run} "
              f"l12_T={row['l12_transmission']:.4f} "
              f"l16_T={row['l16_transmission']:.4f} "
              f"sub_t={row['sub_t']:.4f} "
              f"onsite_near={row.get('onsite_near', float('nan')):.4f} "
              f"gate={gate_detail}")

    report = {
        'provenance': {
            'written': args.date,
            'weights': 'best (own-objective argmin), outputs_v3/*_model_best.pth',
            'onsite_sample': onsite_provenance,
            'substitution_pairs': [list(p) for p in pairs],
            'geometry_caches': cache_paths,
        },
        'runs': report_runs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {args.out}  ({len(report_runs)} runs)")


if __name__ == '__main__':
    main()
