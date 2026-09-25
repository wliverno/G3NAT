"""Contact-invariance sweep over the 96 campaign-v3 Hamiltonian checkpoints
-> outputs/contact_invariance_v3.json.

For every strand with all four contact variants on disk (a quartet), the learned H is
built under each variant and its drift across the four is decomposed into attachment-
mode, coupling and interaction axes (g3nat_analysis/contact_invariance.py). Any
exception anywhere -- a gate failure, a grid mismatch, a frozen-row violation, an
edge-set mismatch within a quartet -- kills the process with a non-zero exit and
writes nothing.
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
from g3nat.evaluation.physicality import onsite_block_eigs

from g3nat_analysis import evaluator as EV
from g3nat_analysis.contact_invariance import (
    SETUPS, SETUP_RUN_KEYS, strand_onsite_drift, strand_coupling_drift,
    frozen_rows)

EXPECTED_RUNS = os.path.join(REPO, 'outputs_v3', 'expected_runs.txt')
OUTPUTS_V3 = os.path.join(REPO, 'outputs_v3')
PICKLE_DIR = os.path.join(REPO, 'pickle_files_v2')
POSTHOC_REPORT = os.path.join(_ANALYSIS, 'outputs', 'posthoc_v3_report.json')

GEOM_CACHES = (os.path.join(REPO, 'geom_cache', 'geometry_v2.pkl'),
              os.path.join(REPO, 'geom_cache', 'geometry_heldout_L12_L16.pkl'))

DEFAULT_OUT = os.path.join(_ANALYSIS, 'outputs', 'contact_invariance_v3.json')

HAM_WEIGHTS = 'hamiltonian_pickle_model_best.pth'

#: Anything at or below this is machine zero, not a small nonzero drift: the
#: scatter-based assembly in construct_hamiltonian_from_graph is not bitwise
#: order-deterministic across differing edge counts. Used only in the two harness
#: assertions, never in a reported value.
MACHINE_ZERO = 1e-6

ONSITE_SCALAR_KEYS = [f'D_{cls}_{axis}'
                      for cls in ('all', 'attached', 'interior')
                      for axis in ('total', 'coupling', 'mode', 'interaction')] + ['D_onsite']
COUPLING_SCALAR_KEYS = [f'D_coupling_{label}'
                        for label in ('edges', 'coupling', 'mode', 'interaction')] + \
                       [f'D_coupling_signed_{label}'
                        for label in ('edges', 'coupling', 'mode', 'interaction')]

HEADLINE_KEY = 'D_interior_mode'


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


def load_all_records():
    """Every pickle_files_v2/*.pkl, keyed sequence.lower() -> {run_key: record}."""
    records = {}
    paths = sorted(glob.glob(os.path.join(PICKLE_DIR, '*.pkl')))
    for p in paths:
        rec = load_single_pickle(p)
        if rec is None:
            raise SystemExit(f"FATAL: unreadable/invalid training pickle: {p}")
        records.setdefault(rec['sequence'].lower(), {})[_run_key(p)] = rec
    return records


def build_quartets(records):
    """Sequences with all four SETUP_RUN_KEYS present. Each run key's recorded
    (coupling, contact_type) is checked against the SETUPS ordering it is indexed by."""
    quartets, excluded = {}, []
    for seq, by_run in sorted(records.items()):
        missing = [k for k in SETUP_RUN_KEYS if k not in by_run]
        if missing:
            excluded.append((seq, missing))
        else:
            quartets[seq] = by_run
    print(f"quartet inventory: {len(records)} sequences on disk in "
          f"{PICKLE_DIR}, {len(quartets)} complete quartets, "
          f"{len(excluded)} excluded")
    for seq, missing in excluded:
        print(f"  excluded {seq}: missing {missing}")

    expected = dict(zip(SETUP_RUN_KEYS, SETUPS))
    for seq, by_run in quartets.items():
        for k, (coup, mode) in expected.items():
            rec = by_run[k]
            if rec['contact_type'] != mode or float(rec['coupling']) != float(coup):
                raise SystemExit(
                    f"FATAL: {seq} {k}: expected (coupling={coup}, mode={mode}), "
                    f"got (coupling={rec['coupling']}, mode={rec['contact_type']})")
    return quartets, excluded


def _check_grid(model_grid, recs, label):
    mg = np.asarray(model_grid, dtype=float)
    for i, rec in enumerate(recs):
        ref = np.asarray(rec['energy_grid'], dtype=float)
        if not np.allclose(mg, ref):
            raise SystemExit(
                f"FATAL: energy grid mismatch ({label}, record {i}): model "
                f"grid does not match the data's energy_grid")
    return mg


def _dna_dna_edge_pairs(batch):
    """(m, m') H-site index pairs for DNA-DNA (backbone/hbond) edges. Nodes 0 and 1
    are the electrodes, so any edge with both endpoints >= 2 is molecular; H site
    index = node index - 2 at batch_size=1."""
    ei = batch.edge_index.cpu().numpy()
    pairs = set()
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        if a >= 2 and b >= 2:
            m, mp = a - 2, b - 2
            pairs.add((min(m, mp), max(m, mp)))
    return sorted(pairs)


def _score_strand(model, n_orb, L, batches):
    """Forward all four setup batches; return (Hs [4,n,n], levels [4,2L,n_orb])."""
    Hs, levels = [], []
    for batch in batches:
        with torch.no_grad():
            model(batch)
        H = model.H[0].detach().cpu().numpy().astype(np.float64)
        Hs.append(H)
        levels.append(onsite_block_eigs(H, n_orb).astype(np.float64))
    return np.stack(Hs), np.stack(levels)


def _build_batches(by_run, energy_grid, geometry_cache, use_geometry):
    return [EV.gated_batch(by_run[k], energy_grid, geometry_cache, use_geometry)
           for k in SETUP_RUN_KEYS]


class _StubOnsiteModel:
    """Synthetic per-base stub: H diagonal depends on base identity only, read off
    the node one-hot; couplings zero. DNA node order does not change between contact
    setups, so its D_onsite must come back exactly 0 -- a null on the harness."""

    def __init__(self, n_orb=2, seed=0):
        rng = np.random.RandomState(seed)
        self.n_orb = n_orb
        self.base_levels = {b: rng.uniform(-1.0, 1.0, size=n_orb) for b in 'ATGC'}
        self.H = None

    def eval(self):
        return self

    def __call__(self, batch):
        x = batch.x.detach().cpu().numpy()
        contact_mask = np.all(x == 0.0, axis=1)
        dna_idx = np.where(~contact_mask)[0]
        num_dna = len(dna_idx)
        bases = 'ATGC'
        H = np.zeros((num_dna * self.n_orb, num_dna * self.n_orb), dtype=np.float32)
        for local, gi in enumerate(dna_idx):
            b = bases[int(np.argmax(x[gi]))]
            levels = self.base_levels[b]
            for o in range(self.n_orb):
                idx = local * self.n_orb + o
                H[idx, idx] = levels[o]
        self.H = torch.from_numpy(H).unsqueeze(0)
        return None, None


def run_stub_null(quartets, energy_grid, geometry_cache):
    by_len = {}
    for seq in quartets:
        by_len.setdefault(len(seq), []).append(seq)
    ref_seq = sorted(by_len[8])[0] if 8 in by_len else sorted(quartets)[0]
    L = len(ref_seq)
    stub = _StubOnsiteModel(n_orb=2, seed=0).eval()
    batches = _build_batches(quartets[ref_seq], energy_grid, geometry_cache,
                             use_geometry=False)
    Hs, levels = _score_strand(stub, stub.n_orb, L, batches)
    out = strand_onsite_drift(levels, L, stub.n_orb)
    passed = out['D_onsite'] <= MACHINE_ZERO
    if not passed:
        raise SystemExit(
            f"FATAL: synthetic per-base stub null FAILED on {ref_seq} "
            f"(L={L}): D_onsite={out['D_onsite']!r}, expected <= {MACHINE_ZERO}. "
            f"The harness is reading something other than pure base identity "
            f"out of the onsite blocks -- do not trust any other number.")
    print(f"synthetic per-base stub null: PASS (reference={ref_seq}, L={L}, "
          f"D_onsite={out['D_onsite']:.3e})")
    return {'passed': True, 'reference_sequence': ref_seq, 'L': L,
            'D_onsite': out['D_onsite']}


def _score_checkpoint(run, model, n_orb, num_layers, quartets, model_grid,
                      geometry_cache, use_geometry):
    """Score every quartet under one checkpoint. Two harness checks run inline:
    identical-input null (the first strand's setup-0 batch scored twice must give a
    bitwise-identical H) and the frozen-row assertion (for every L=8 strand, rows
    outside num_layers hops of every contact must have d_total <= MACHINE_ZERO)."""
    frozen = frozen_rows(8, num_layers)
    per_strand = {}
    run1_levels_flat = []
    identical_input_null = None
    frozen_row_strands_checked = 0

    for i, seq in enumerate(sorted(quartets)):
        L = len(seq)
        by_run = quartets[seq]
        batches = _build_batches(by_run, model_grid, geometry_cache, use_geometry)

        edge_pairs = _dna_dna_edge_pairs(batches[0])
        for setup_idx, b in enumerate(batches[1:], 1):
            ep = _dna_dna_edge_pairs(b)
            if ep != edge_pairs:
                raise SystemExit(
                    f"FATAL: {run} {seq}: DNA-DNA edge set differs between "
                    f"setup 0 and setup {setup_idx} -- contacts must not "
                    f"alter molecular edges. setup0 has {len(edge_pairs)} "
                    f"edges, setup{setup_idx} has {len(ep)}.")

        Hs, levels = _score_strand(model, n_orb, L, batches)
        run1_levels_flat.append(levels[0])

        if identical_input_null is None:
            with torch.no_grad():
                model(batches[0])
            H_dup = model.H[0].detach().cpu().numpy().astype(np.float64)
            dup_diff = float(np.max(np.abs(H_dup - Hs[0])))
            if dup_diff != 0.0:
                raise SystemExit(
                    f"FATAL: identical-input null failed for {run} on "
                    f"{seq} setup0: max|H - H_dup| = {dup_diff!r}, expected "
                    f"exactly 0.0. Indexing or batching is non-deterministic.")
            identical_input_null = {'passed': True, 'reference_sequence': seq,
                                    'max_abs_diff': dup_diff}

        onsite = strand_onsite_drift(levels, L, n_orb)
        coupling = strand_coupling_drift(Hs, edge_pairs, L, n_orb)

        if L == 8 and frozen:
            frozen_row_strands_checked += 1
            d_total_rows = onsite['per_site']['d_total'][sorted(frozen)]
            worst = float(np.max(d_total_rows))
            if worst > MACHINE_ZERO:
                raise SystemExit(
                    f"FATAL: frozen-row assertion failed for {run} on {seq} "
                    f"(L=8, num_layers={num_layers}): rows {sorted(frozen)} "
                    f"must be reachability-frozen (d_total <= {MACHINE_ZERO}) "
                    f"but max d_total = {worst!r}. No other number in this "
                    f"checkpoint's report should be trusted.")

        row = {k: onsite[k] for k in ONSITE_SCALAR_KEYS}
        row.update({k: coupling[k] for k in COUPLING_SCALAR_KEYS})
        per_strand[seq] = row

    harness = {
        'identical_input_null': identical_input_null,
        'frozen_row_check': {
            'num_layers': num_layers,
            'frozen_rows': sorted(frozen),
            'fired': bool(frozen),
            'l8_strands_checked': frozen_row_strands_checked,
            'passed': True if frozen else None,
        },
    }
    return per_strand, run1_levels_flat, harness


def _worst10(per_strand, key):
    ranked = sorted(per_strand.items(), key=lambda kv: kv[1][key], reverse=True)
    return [{'sequence': seq, key: vals[key]} for seq, vals in ranked[:10]]


def _vectors(per_strand):
    keys = ONSITE_SCALAR_KEYS + COUPLING_SCALAR_KEYS
    return {k: sorted(v[k] for v in per_strand.values()) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--posthoc-report', default=POSTHOC_REPORT,
                    help='posthoc_v3_report.json, source of the companion '
                         'val_transmission_at_selection / l16_transmission')
    ap.add_argument('--date', required=True,
                    help='ISO date string stamped into provenance.written')
    args = ap.parse_args()

    with open(EXPECTED_RUNS) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    parsed_names = {}
    for n in names:
        p = EV.parse_run_name(n)
        if p is None:
            raise SystemExit(f"FATAL: run name failed to parse: {n}")
        parsed_names[n] = p
    ham_names = [n for n in names if parsed_names[n]['family'] == 'ham']
    if len(ham_names) != 96:
        raise SystemExit(
            f"FATAL: expected 96 ham runs in expected_runs.txt, got "
            f"{len(ham_names)}")
    print(f"expected_runs: {len(names)} names ({len(ham_names)} ham), all parsed")
    run_list = list(ham_names)

    geometry_cache = load_geometry(list(GEOM_CACHES))

    print("loading training records ...")
    all_records = load_all_records()
    quartets, excluded = build_quartets(all_records)

    with open(args.posthoc_report) as f:
        posthoc_runs = json.load(f)['runs']

    stub_result = run_stub_null(quartets, np.linspace(-1, 1, 201), geometry_cache)

    report_runs = {}
    n = len(run_list)
    for i, run in enumerate(run_list, 1):
        parsed = parsed_names[run]
        n_orb, num_layers = parsed['n_orb'], parsed['num_layers']
        path = os.path.join(OUTPUTS_V3, run, HAM_WEIGHTS)
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: missing weights for {run}: {path}")

        payload = torch.load(path, map_location='cpu', weights_only=False)
        model, model_grid, _dev = load_trained_model(path, device='cpu')

        name_geom = bool(parsed['geometry'])
        payload_geom = bool(payload.get('args', {}).get('use_geometry', False))
        model_geom = bool(getattr(model, 'use_geometry', False))
        if not (name_geom == payload_geom == model_geom):
            raise SystemExit(
                f"FATAL: geometry flag mismatch for {run}: "
                f"name={name_geom} payload.args.use_geometry={payload_geom} "
                f"model.use_geometry={model_geom}")

        _check_grid(model_grid,
                   [quartets[seq]['run1'] for seq in quartets], f'{run} run1')

        if run not in posthoc_runs:
            raise SystemExit(
                f"FATAL: {run} not found in {args.posthoc_report} -- companion "
                f"triple requires a matching posthoc_v3 row")
        posthoc_row = posthoc_runs[run]

        per_strand, run1_levels_flat, harness = _score_checkpoint(
            run, model, n_orb, num_layers, quartets, model_grid,
            geometry_cache, name_geom)

        all_run1_levels = np.concatenate([lv.ravel() for lv in run1_levels_flat])
        onsite_abs_mean = float(np.mean(np.abs(all_run1_levels)))
        onsite_std = float(np.std(all_run1_levels))

        row = dict(parsed)
        row['quartet_count'] = len(quartets)
        row['vectors'] = _vectors(per_strand)
        row[f'worst10_{HEADLINE_KEY}'] = _worst10(per_strand, HEADLINE_KEY)
        row['companion'] = {
            'onsite_abs_mean': onsite_abs_mean,
            'onsite_std': onsite_std,
            'val_transmission_at_selection':
                posthoc_row['val_transmission_at_selection'],
            'l16_transmission': posthoc_row['l16_transmission'],
        }
        row['harness'] = harness

        report_runs[run] = row
        median_headline = float(np.median(row['vectors'][HEADLINE_KEY]))
        print(f"[{i}/{n}] {run} median_{HEADLINE_KEY}={median_headline:.4e} "
             f"frozen_fired={harness['frozen_row_check']['fired']} "
             f"identical_null=PASS")

    report = {
        'provenance': {
            'written': args.date,
            'weights': 'outputs_v3/<run>/hamiltonian_pickle_model_best.pth',
            'geometry_caches': list(GEOM_CACHES),
            'pickle_dir': PICKLE_DIR,
            'quartet_count': len(quartets),
            'excluded_sequences': [{'sequence': s, 'missing': m}
                                   for s, m in excluded],
            'posthoc_report': POSTHOC_REPORT,
            'synthetic_per_base_stub_null': stub_result,
            'machine_zero': MACHINE_ZERO,
        },
        'runs': report_runs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {args.out}  ({len(report_runs)} runs)")


if __name__ == '__main__':
    main()
