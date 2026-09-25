"""Capacity probe (v5): deeper readout MLP and wider GNN for the two figure cells.

Scores the 12 probe runs (outputs_v5probe) and the 6 baseline runs (outputs_v3 figure
cells) with the same evaluator as the DOE: in-distribution validation transmission at
the run's own selected epoch, pooled held-out transmission/DOS at L=12 and L=16, and
per-duplex transmission loss (mean over the 4 contact variants). Prints per-run values,
3-seed mean and seed range per arm, and the difference of each probe from its baseline
against the baseline seed spread. No interval is claimed at n=3.
"""
import csv
import glob
import os
import sys

import numpy as np
import torch

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(_ANALYSIS))
sys.path.insert(0, REPO)
sys.path.insert(0, _ANALYSIS)
sys.path.insert(0, os.path.join(_ANALYSIS, 'scripts'))

from g3nat.evaluation.inference import load_trained_model
from g3nat_analysis import evaluator as EV
from posthoc_v3 import load_geometry, _check_grid, HELDOUT_DIRS, GEOM_CACHES
from persequence_v3 import load_heldout_indexed

SEEDS = (1179027592, 2129768291, 3731635825)
ARMS = {
    # arm -> (family, dir template, weights file)
    'ham base':  ('ham',   'outputs_v3/ham_ldosonly_n2_L4_nogeom_s{seed}',            'hamiltonian_pickle_model_best.pth'),
    'ham deep':  ('ham',   'outputs_v5probe/ham_ldosonly_n2_L4_nogeom_deep_s{seed}',  'hamiltonian_deep_pickle_model_best.pth'),
    'ham h512':  ('ham',   'outputs_v5probe/ham_ldosonly_n2_L4_nogeom_h512_s{seed}',  'hamiltonian_pickle_model_best.pth'),
    'direct base': ('blind', 'outputs_v3/blind_dos_L2_nogeom_s{seed}',               'standard_pickle_model_best.pth'),
    'direct deep': ('blind', 'outputs_v5probe/blind_dos_L2_nogeom_deep_s{seed}',     'standard_deep_pickle_model_best.pth'),
    'direct h512': ('blind', 'outputs_v5probe/blind_dos_L2_nogeom_h512_s{seed}',     'standard_pickle_model_best.pth'),
}
OUT_DIR = os.path.join(_ANALYSIS, 'outputs')
OUT_TXT = os.path.join(OUT_DIR, 'probe_capacity_v5.out')
OUT_CSV = os.path.join(OUT_DIR, 'probe_capacity_v5_runs.csv')
OUT_SEQ = os.path.join(OUT_DIR, 'probe_capacity_v5_perseq.csv')


def main():
    geometry_cache = load_geometry(list(GEOM_CACHES))
    heldout = {L: load_heldout_indexed(L) for L in sorted(HELDOUT_DIRS)}
    loss_fn = EV.LOSS_FUNCS['huber']
    rows, seq_rows = [], []
    for arm, (fam, tmpl, weights) in ARMS.items():
        for seed in SEEDS:
            path = os.path.join(REPO, tmpl.format(seed=seed), weights)
            if not os.path.exists(path):
                raise SystemExit(f'FATAL: missing weights {path}')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            val_t = EV.transmission_at_selection(ck)
            n_params = sum(int(np.prod(v.shape)) for v in ck['model_state_dict'].values())
            model, grid, _ = load_trained_model(path, device='cpu')
            model.eval()
            row = {'arm': arm, 'family': fam, 'seed': seed, 'n_params': n_params,
                   'saved_at_epoch': ck.get('saved_at_epoch', ck.get('epoch')),
                   'val_transmission': val_t}
            for L, recs in heldout.items():
                _check_grid(grid, list(recs.values()), f'{arm} s{seed} L{L}')
                per_seq = {}
                t_all, d_all = [], []
                for (seq, rk), rec in recs.items():
                    batch = EV.gated_batch(rec, grid, geometry_cache, False)
                    with torch.no_grad():
                        dos_p, t_p = model(batch)
                    t_p = t_p.squeeze(0).cpu().numpy(); d_p = dos_p.squeeze(0).cpu().numpy()
                    t_t = np.asarray(rec['transmission'], float).ravel()
                    d_t = np.asarray(rec['dos'], float).ravel()
                    lt, ld = loss_fn(t_p, t_t), loss_fn(d_p, d_t)
                    t_all.append(lt); d_all.append(ld)
                    per_seq.setdefault(seq, []).append(lt)
                row[f'l{L}_transmission'] = float(np.mean(t_all))
                row[f'l{L}_dos'] = float(np.mean(d_all))
                for seq, v in per_seq.items():
                    assert len(v) == 4, (arm, seed, L, seq, len(v))
                    seq_rows.append({'arm': arm, 'seed': seed, 'length': L, 'sequence': seq,
                                     'huber_t': float(np.mean(v))})
            rows.append(row)
            print(f"{arm:12s} s{seed}  params={n_params:8d}  epoch={row['saved_at_epoch']:5}  "
                  f"val={val_t:.4f}  l12={row['l12_transmission']:.3f}  l16={row['l16_transmission']:.3f}  "
                  f"l12dos={row['l12_dos']:.3f}  l16dos={row['l16_dos']:.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    keys = list(rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(OUT_SEQ, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['arm', 'seed', 'length', 'sequence', 'huber_t'])
        w.writeheader(); w.writerows(seq_rows)

    metrics = ['val_transmission', 'l12_transmission', 'l16_transmission', 'l12_dos', 'l16_dos']
    lines = ['CAPACITY PROBE v5: 3-seed mean [min, max] per arm; delta = probe - base (means)',
             'baseline seed spread (max - min) is the bar a delta must clear at n=3', '']
    for fam, base in (('ham', 'ham base'), ('blind', 'direct base')):
        arms = [a for a in ARMS if ARMS[a][0] == fam]
        hdr = f"{'arm':12s}" + ''.join(f'{m:>34s}' for m in metrics)
        lines.append(hdr)
        stats = {}
        for arm in arms:
            vals = {m: np.array([r[m] for r in rows if r['arm'] == arm]) for m in metrics}
            stats[arm] = vals
            lines.append(f'{arm:12s}' + ''.join(
                f'{vals[m].mean():10.3f} [{vals[m].min():.3f}, {vals[m].max():.3f}]'.rjust(34) for m in metrics))
        for arm in arms:
            if arm == base:
                continue
            lines.append(f'{"delta " + arm.split()[1]:12s}' + ''.join(
                (f'{stats[arm][m].mean() - stats[base][m].mean():+10.3f}'
                 f' (base spread {np.ptp(stats[base][m]):.3f})').rjust(34) for m in metrics))
        lines.append('')
    lines.append('PER-DUPLEX transmission loss, 3-seed mean (best seed in parentheses)')
    seqs = sorted({(r['length'], r['sequence']) for r in seq_rows})
    lines.append(f"{'duplex':22s}" + ''.join(f'{a:>20s}' for a in ARMS))
    for L, seq in seqs:
        cells = []
        for arm in ARMS:
            v = np.array([r['huber_t'] for r in seq_rows if r['arm'] == arm and r['sequence'] == seq and r['length'] == L])
            cells.append(f'{v.mean():.2f} ({v.min():.2f})'.rjust(20))
        lines.append(f'{seq + " (" + str(L) + ")":22s}' + ''.join(cells))
    text = '\n'.join(lines)
    print('\n' + text)
    with open(OUT_TXT, 'w') as fh:
        fh.write(text + '\n')
    print(f'\nwrote {OUT_TXT}, {OUT_CSV}, {OUT_SEQ}')


if __name__ == '__main__':
    main()
