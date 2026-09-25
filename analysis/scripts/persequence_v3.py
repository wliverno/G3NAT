"""Per-sequence head-to-head + composition split on the held-out 12- and 16-mers.

Scores one Hamiltonian cell and one direct cell (3 seeds each) against every held-out
DFT record at L=12 and L=16 (32 pickles: 4 sequences x 4 contact variants x 2 lengths),
aggregates to the SEQUENCE (mean over the 4 contact variants inside each seed, then mean
over the 3 seeds -- the duplex is the unit, its variants are not independent) and runs
the two-sided exact binomial sign test on the n=8 (sequence, length) pairs. Same
construction for log10 T and, separately, log10 DOS.

Default cells are the v3 DOE's pick on l12_transmission (doe_v3.out, HEAD-TO-HEAD):
ham_ldosonly_n2_L4_nogeom vs blind_tonly_L2_geom. The paper's other invocations:
  figure pair, geometry off:  --ham-dir-tmpl .../outputs_v3/ham_ldosonly_n2_L4_nogeom_s{seed}
                              --blind-dir-tmpl .../outputs_v3/blind_dos_L2_nogeom_s{seed}
                              --blind-geometry false --out-suffix figcells_nogeom
                              --dump-csv figures/paper/data/fig-perseq_figcells_nogeom_records.csv
  TransformerConv arm:        same with outputs_v3conv/*_tconv_s{seed}, --out-suffix figcells_nogeom_tconv
  loss probes:                outputs_v4probe/*_{huber,mse}_s{seed}, --metric {huber,mse},
                              --out-suffix v4probe_<trained>_in_<scored>

Any load failure or missing record is FATAL; there is no skip list.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(_ANALYSIS))
sys.path.insert(0, REPO)
sys.path.insert(0, _ANALYSIS)

from g3nat.data.pickle import load_single_pickle
from g3nat.evaluation.inference import load_trained_model

from g3nat_analysis import evaluator as EV
from posthoc_v3 import load_geometry, _check_grid, HELDOUT_DIRS, GEOM_CACHES

SEEDS = (1179027592, 2129768291, 3731635825)

DEFAULT_HAM_DIR_TMPL = os.path.join(
    REPO, 'outputs_v3', 'ham_ldosonly_n2_L4_nogeom_s{seed}')
DEFAULT_HAM_WEIGHTS = 'hamiltonian_pickle_model_best.pth'
DEFAULT_BLIND_DIR_TMPL = os.path.join(
    REPO, 'outputs_v3', 'blind_tonly_L2_geom_s{seed}')
DEFAULT_BLIND_WEIGHTS = 'standard_pickle_model_best.pth'


def _str2bool(v):
    if v.lower() in ('1', 'true', 'yes', 'on'):
        return True
    if v.lower() in ('0', 'false', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ham-dir-tmpl', default=DEFAULT_HAM_DIR_TMPL,
                    help="run dir template with a {seed} placeholder for the "
                         "hamiltonian cell")
    ap.add_argument('--ham-weights', default=DEFAULT_HAM_WEIGHTS)
    ap.add_argument('--ham-geometry', type=_str2bool, default=False,
                    help='use_geometry expected on the hamiltonian cell')
    ap.add_argument('--blind-dir-tmpl', default=DEFAULT_BLIND_DIR_TMPL,
                    help="run dir template with a {seed} placeholder for the "
                         "blind/direct cell")
    ap.add_argument('--blind-weights', default=DEFAULT_BLIND_WEIGHTS)
    ap.add_argument('--blind-geometry', type=_str2bool, default=True,
                    help='use_geometry expected on the blind/direct cell')
    ap.add_argument('--header-note', default=None,
                    help='free-text line inserted into the report header')
    ap.add_argument('--metric', default='huber', choices=('huber', 'mse'),
                    help='pointwise loss the per-record numbers are computed in '
                         '(post hoc, independent of what the run trained on)')
    ap.add_argument('--dump-csv', default=None,
                    help='also write every per-record loss (metric, cell, L, '
                         'sequence, contact run, seed, loss) to this CSV')
    ap.add_argument('--out-suffix', default=None,
                    help="output is outputs/persequence_v3_<suffix>.out; default "
                         "persequence_v3.out")
    return ap


def classify_composition(seq):
    """mixed / poly-G / (AT)n, by inspection of the sequence string."""
    s = seq.upper()
    if set(s) == {'G'}:
        return 'poly-G'
    if set(s) <= {'A', 'T'}:
        return '(AT)n'
    return 'mixed'


def load_heldout_indexed(length):
    """(sequence, run_key) -> record for one held-out length; count != 16 is FATAL."""
    paths = sorted(glob.glob(os.path.join(HELDOUT_DIRS[length], '*.pkl')))
    out = {}
    for p in paths:
        rec = load_single_pickle(p)
        if rec is None:
            raise SystemExit(f"FATAL: unreadable/invalid held-out pickle: {p}")
        stem = os.path.splitext(os.path.basename(p))[0]
        run_key = stem.rsplit('_', 1)[1]
        out[(rec['sequence'].upper(), run_key)] = rec
    if len(out) != 16:
        raise SystemExit(
            f"FATAL: expected 16 held-out records at L={length}, found "
            f"{len(out)} under {HELDOUT_DIRS[length]}")
    return out


def main():
    args = build_arg_parser().parse_args()

    cells = {
        'ham': {
            'use_geometry': args.ham_geometry,
            'dir_tmpl': args.ham_dir_tmpl,
            'weights': args.ham_weights,
        },
        'blind': {
            'use_geometry': args.blind_geometry,
            'dir_tmpl': args.blind_dir_tmpl,
            'weights': args.blind_weights,
        },
    }
    out_name = ('persequence_v3.out' if args.out_suffix is None
                else f'persequence_v3_{args.out_suffix}.out')
    out_path = os.path.join(_ANALYSIS, 'outputs', out_name)

    geometry_cache = load_geometry(list(GEOM_CACHES))
    loss_fn = EV.LOSS_FUNCS[args.metric]
    print(f"per-record loss: {args.metric} (post hoc; independent of the training loss)")

    heldout = {}
    for L in sorted(HELDOUT_DIRS):
        heldout[L] = load_heldout_indexed(L)
        seqs = sorted({s for s, _rk in heldout[L]})
        print(f"L={L}: {len(heldout[L])} held-out records, "
              f"{len(seqs)} sequences: {seqs}")

    # raw[metric][cell][L][(seq, run_key)][seed] = per-record loss
    raw = {m: {c: {L: {} for L in heldout} for c in cells} for m in ('t', 'dos')}

    for cell_name, cell in cells.items():
        for seed in SEEDS:
            run_dir = cell['dir_tmpl'].format(seed=seed)
            path = os.path.join(run_dir, cell['weights'])
            if not os.path.exists(path):
                raise SystemExit(f"FATAL: missing weights: {path}")

            model, model_grid, _dev = load_trained_model(path, device='cpu')
            model_geom = bool(getattr(model, 'use_geometry', False))
            if model_geom != cell['use_geometry']:
                raise SystemExit(
                    f"FATAL: geometry mismatch for {path}: expected "
                    f"use_geometry={cell['use_geometry']}, model has "
                    f"{model_geom}")

            n_scored = 0
            for L in sorted(heldout):
                recs_by_key = heldout[L]
                _check_grid(model_grid, list(recs_by_key.values()),
                            f'{cell_name} s{seed} L{L}')
                for (seq, run_key), rec in recs_by_key.items():
                    batch = EV.gated_batch(rec, model_grid, geometry_cache,
                                           cell['use_geometry'])
                    with torch.no_grad():
                        dos_p, t_p = model(batch)
                    t_p = t_p.squeeze(0).cpu().numpy()
                    dos_p = dos_p.squeeze(0).cpu().numpy()
                    t_t = np.asarray(rec['transmission'], float).ravel()
                    dos_t = np.asarray(rec['dos'], float).ravel()
                    if len(t_p) != len(t_t) or len(t_p) != len(model_grid):
                        raise SystemExit(
                            f"FATAL: transmission length mismatch for "
                            f"{cell_name} s{seed} {seq}_{run_key} L={L}: "
                            f"pred={len(t_p)} target={len(t_t)} "
                            f"grid={len(model_grid)}")
                    if len(dos_p) != len(dos_t) or len(dos_p) != len(model_grid):
                        raise SystemExit(
                            f"FATAL: DOS length mismatch for "
                            f"{cell_name} s{seed} {seq}_{run_key} L={L}: "
                            f"pred={len(dos_p)} target={len(dos_t)} "
                            f"grid={len(model_grid)}")
                    raw['t'][cell_name][L].setdefault((seq, run_key), {})[seed] = \
                        loss_fn(t_p, t_t)
                    raw['dos'][cell_name][L].setdefault((seq, run_key), {})[seed] = \
                        loss_fn(dos_p, dos_t)
                    n_scored += 1
            print(f"{cell_name} seed={seed}: scored {n_scored} records "
                  f"(expect {sum(len(v) for v in heldout.values())})")

    if args.dump_csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_csv)), exist_ok=True)
        with open(args.dump_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['metric', 'model', 'length', 'sequence', 'contact_run',
                        'seed', 'huber_loss'])
            for metric in ('t', 'dos'):
                for cell_name in cells:
                    for L in sorted(heldout):
                        for (seq, rk), by_seed in sorted(raw[metric][cell_name][L].items()):
                            for seed in SEEDS:
                                w.writerow([metric,
                                            'hamiltonian' if cell_name == 'ham' else 'direct',
                                            L, seq.upper(), rk, seed,
                                            f'{by_seed[seed]:.6f}'])
        print(f"wrote {args.dump_csv}")

    # mean over the sequence's 4 contact variants inside each seed, THEN over seeds
    final = {m: {c: {L: {} for L in heldout} for c in cells} for m in ('t', 'dos')}
    for metric in ('t', 'dos'):
        for cell_name in cells:
            for L in heldout:
                seqs = sorted({seq for (seq, _rk) in raw[metric][cell_name][L]})
                for seq in seqs:
                    seed_means = []
                    for seed in SEEDS:
                        variant_losses = [
                            raw[metric][cell_name][L][(s2, rk)][seed]
                            for (s2, rk) in raw[metric][cell_name][L] if s2 == seq]
                        if len(variant_losses) != 4:
                            raise SystemExit(
                                f"FATAL: expected 4 contact variants for "
                                f"{metric} {cell_name} L={L} seq={seq} "
                                f"seed={seed}, got {len(variant_losses)}")
                        seed_means.append(float(np.mean(variant_losses)))
                    final[metric][cell_name][L][seq] = float(np.mean(seed_means))

    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    out('=' * 100)
    out('PER-SEQUENCE HEAD-TO-HEAD (spec 5a.3) + COMPOSITION SPLIT (spec 5a.4)')
    out('=' * 100)
    if args.header_note:
        out(args.header_note)
        out('')
    out('Cells scored in this run:')
    out(f"  ham   : {args.ham_dir_tmpl} / {args.ham_weights} "
        f"(use_geometry={args.ham_geometry})")
    out(f"  blind : {args.blind_dir_tmpl} / {args.blind_weights} "
        f"(use_geometry={args.blind_geometry})")
    out('Default cells (no CLI override) are the v3 DOE pick on l12_transmission '
        '(analysis/outputs/doe_v3.out, HEAD-TO-HEAD section): '
        'ham=ldosonly_n2_L4_nogeom, blind=tonly_L2_geom.')
    out('Per-sequence values are shown at BOTH lengths (L=12 and L=16), not pooled into one row.')
    out('n=4 sequences per length. NO interval (confidence interval, error bar, etc.) is claimed '
        'at this sample size -- wins/losses and the sign-test p-value only.')
    out('Loss = full-curve Huber on log10 T (and, separately, log10 DOS), all 201 energy '
        'points, per sequence (mean over its 4 contact variants, then mean over 3 seeds). '
        'Lower is better. Both models score DOS regardless of supervision, since forward() '
        'always returns a DOS prediction; whether DOS is a trained objective for a given '
        'cell is a property of that cell, stated above/by the caller, not of this script.')
    out('')

    out('-' * 100)
    out('PER-SEQUENCE TABLE (T and DOS side by side)')
    out('-' * 100)
    header = (f"{'sequence':<20}{'length':>7}{'composition':>14}"
              f"{'ham_T':>12}{'blind_T':>12}{'winner_T':>10}"
              f"{'ham_DOS':>12}{'blind_DOS':>12}{'winner_DOS':>12}")
    out(header)
    out('-' * len(header))

    pooled_wins = {'t': {'ham': 0, 'blind': 0}, 'dos': {'ham': 0, 'blind': 0}}
    by_length_wins = {m: {12: {'ham': 0, 'blind': 0}, 16: {'ham': 0, 'blind': 0}}
                      for m in ('t', 'dos')}
    comp_rows = []  # (length, sequence, composition, winner_t, winner_dos)

    for L in sorted(heldout):
        seqs = sorted(final['t']['ham'][L])
        for metric in ('t', 'dos'):
            if sorted(final[metric]['blind'][L]) != seqs or \
               sorted(final[metric]['ham'][L]) != seqs:
                raise SystemExit(
                    f"FATAL: ham/blind sequence sets disagree at L={L} "
                    f"({metric})")
        for seq in seqs:
            h_t = final['t']['ham'][L][seq]
            b_t = final['t']['blind'][L][seq]
            w_t = 'ham' if h_t < b_t else ('blind' if b_t < h_t else 'tie')
            h_d = final['dos']['ham'][L][seq]
            b_d = final['dos']['blind'][L][seq]
            w_d = 'ham' if h_d < b_d else ('blind' if b_d < h_d else 'tie')
            comp = classify_composition(seq)
            out(f"{seq:<20}{L:>7}{comp:>14}"
                f"{h_t:>12.5f}{b_t:>12.5f}{w_t:>10}"
                f"{h_d:>12.5f}{b_d:>12.5f}{w_d:>12}")
            if w_t in pooled_wins['t']:
                pooled_wins['t'][w_t] += 1
                by_length_wins['t'][L][w_t] += 1
            if w_d in pooled_wins['dos']:
                pooled_wins['dos'][w_d] += 1
                by_length_wins['dos'][L][w_d] += 1
            comp_rows.append((L, seq, comp, w_t, w_d))
    out('')

    out('-' * 100)
    out('WINS / LOSSES')
    out('-' * 100)
    for metric, label in (('t', 'transmission'), ('dos', 'DOS')):
        for L in sorted(by_length_wins[metric]):
            out(f"[{label}] L={L}: ham wins {by_length_wins[metric][L]['ham']}, "
                f"blind wins {by_length_wins[metric][L]['blind']} (n=4)")
        out(f"[{label}] pooled (L=12 and L=16 together, n=8): "
            f"ham wins {pooled_wins[metric]['ham']}, "
            f"blind wins {pooled_wins[metric]['blind']}")
    out('')

    results = {}
    for metric, label in (('t', 'transmission'), ('dos', 'DOS')):
        n_pooled = pooled_wins[metric]['ham'] + pooled_wins[metric]['blind']
        if n_pooled != 8:
            raise SystemExit(
                f"FATAL: [{label}] pooled win+loss count is {n_pooled}, "
                f"expected 8 (a tie or a missing sequence would break the "
                f"sign test's n)")
        results[metric] = binomtest(pooled_wins[metric]['ham'], n=8, p=0.5,
                                    alternative='two-sided')
        out(f"[{label}] two-sided exact binomial sign test on the pooled n=8 "
            f"(sequence, length) pairs (scipy.stats.binomtest): "
            f"k(ham wins)={pooled_wins[metric]['ham']}, n=8, "
            f"p={results[metric].pvalue:.4f}")
    out('')

    out('-' * 100)
    out('COMPOSITION-SPLIT TABLE (winner by composition class, at each length)')
    out('-' * 100)
    header2 = (f"{'length':>7}{'composition':>14}{'sequence':<20}"
              f"{'winner_T':>10}{'winner_DOS':>12}")
    out(header2)
    out('-' * len(header2))
    for L, seq, comp, w_t, w_d in sorted(comp_rows, key=lambda r: (r[0], r[2], r[1])):
        out(f"{L:>7}{comp:>14}{seq:<20}{w_t:>10}{w_d:>12}")
    out('')
    out('Tally by composition class (pooled across both lengths):')
    class_tally = {}
    for _L, _seq, comp, w_t, w_d in comp_rows:
        class_tally.setdefault(
            comp, {'ham_t': 0, 'blind_t': 0, 'ham_dos': 0, 'blind_dos': 0, 'n': 0})
        class_tally[comp]['n'] += 1
        if w_t in ('ham', 'blind'):
            class_tally[comp][f'{w_t}_t'] += 1
        if w_d in ('ham', 'blind'):
            class_tally[comp][f'{w_d}_dos'] += 1
    for comp in sorted(class_tally):
        t = class_tally[comp]
        out(f"  {comp:<10} n={t['n']:<3} "
            f"T: ham={t['ham_t']:<3} blind={t['blind_t']:<3} "
            f"DOS: ham={t['ham_dos']:<3} blind={t['blind_dos']}")
    out('')
    out('No interval is claimed on any composition-class tally at this sample size.')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
