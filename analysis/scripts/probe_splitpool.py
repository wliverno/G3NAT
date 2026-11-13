"""Exploratory split pooling (post hoc, not a selection): the DOS+T, 2-layer, no-geometry direct
cell with mean pooling for the DOS head and sum pooling for the T head, against the same cell with
mean and with sum pooling, the selected direct model and the selected Hamiltonian model. Same
evaluator as probe_sumpool.py: validation T and DOS at the run's selected epoch, pooled held-out
T/DOS at L=12 and L=16, per-duplex losses. No interval is claimed at n=3.
"""
import csv
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
from persequence_v3 import load_heldout_indexed, classify_composition

SEEDS = (1179027592, 2129768291, 3731635825)
ARMS = {
    # arm -> (dir template, weights file, attribute, expected value)
    'direct DOS+T mean':  ('campaign/blind_dos_L2_nogeom_s{seed}', 'standard_pickle_model_best.pth',
                           'readout_pooling', 'mean'),
    'direct DOS+T sum':   ('campaign/blind_dos_L2_nogeom_sumpool_s{seed}', 'standard_pickle_model_best.pth',
                           'readout_pooling', 'sum'),
    'direct DOS+T split': ('exploratory/blind_dos_L2_nogeom_splitpool_s{seed}', 'standard_pickle_model_best.pth',
                           'readout_pooling', 'split'),
    'direct T-only sum (selected)': ('campaign/blind_tonly_L4_nogeom_sumpool_s{seed}',
                                     'standard_pickle_model_best.pth', 'readout_pooling', 'sum'),
    'ham DOS+T (selected)': ('campaign/ham_dos_n2_L4_nogeom_s{seed}',
                             'hamiltonian_pickle_model_best.pth', 'dna_only_message_passing', True),
}
OUT_DIR = os.path.join(_ANALYSIS, 'outputs')
OUT_TXT = os.path.join(OUT_DIR, 'probe_splitpool.out')
OUT_CSV = os.path.join(OUT_DIR, 'probe_splitpool_runs.csv')
OUT_SEQ = os.path.join(OUT_DIR, 'probe_splitpool_perseq.csv')


def val_at_selection(ck, key):
    epoch = ck.get('saved_at_epoch', ck.get('epoch'))
    for entry in ck.get('metric_history') or []:
        if entry.get('epoch') == epoch:
            return float(entry[key]) if key in entry else float('nan')
    raise ValueError(f'no metric_history entry for saved_at_epoch={epoch}')


def main():
    geometry_cache = load_geometry(list(GEOM_CACHES))
    heldout = {L: load_heldout_indexed(L) for L in sorted(HELDOUT_DIRS)}
    loss_fn = EV.LOSS_FUNCS['huber']
    rows, seq_rows = [], []
    for arm, (tmpl, weights, attr, expect) in ARMS.items():
        for seed in SEEDS:
            path = os.path.join(REPO, tmpl.format(seed=seed), weights)
            if not os.path.exists(path):
                raise SystemExit(f'FATAL: missing weights {path}')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            model, grid, _ = load_trained_model(path, device='cpu')
            model.eval()
            got = getattr(model, attr, None)
            if got != expect:
                raise SystemExit(f'FATAL: {path} {attr}={got!r}, expected {expect!r}')
            row = {'arm': arm, 'seed': seed,
                   'saved_at_epoch': ck.get('saved_at_epoch', ck.get('epoch')),
                   'val_transmission': EV.transmission_at_selection(ck),
                   'val_dos': val_at_selection(ck, 'val_dos')}
            for L, recs in heldout.items():
                _check_grid(grid, list(recs.values()), f'{arm} s{seed} L{L}')
                per_seq, t_all, d_all = {}, [], []
                for (seq, rk), rec in recs.items():
                    batch = EV.gated_batch(rec, grid, geometry_cache, False)
                    with torch.no_grad():
                        dos_p, t_p = model(batch)
                    lt = loss_fn(t_p.squeeze(0).cpu().numpy(), np.asarray(rec['transmission'], float).ravel())
                    ld = loss_fn(dos_p.squeeze(0).cpu().numpy(), np.asarray(rec['dos'], float).ravel())
                    t_all.append(lt); d_all.append(ld)
                    per_seq.setdefault(seq, []).append((lt, ld))
                row[f'l{L}_transmission'] = float(np.mean(t_all))
                row[f'l{L}_dos'] = float(np.mean(d_all))
                for seq, v in sorted(per_seq.items()):
                    assert len(v) == 4, (arm, seed, L, seq, len(v))
                    v = np.array(v)
                    seq_rows.append({'arm': arm, 'seed': seed, 'length': L, 'sequence': seq,
                                     'composition': classify_composition(seq),
                                     'huber_t': float(v[:, 0].mean()), 'huber_dos': float(v[:, 1].mean())})
            rows.append(row)
            print(f"{arm:16s} s{seed}  epoch={row['saved_at_epoch']:5}  val_t={row['val_transmission']:.4f}  "
                  f"val_dos={row['val_dos']:.4f}  l12={row['l12_transmission']:.3f}  "
                  f"l16={row['l16_transmission']:.3f}  l12dos={row['l12_dos']:.3f}  "
                  f"l16dos={row['l16_dos']:.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT_SEQ, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(seq_rows[0])); w.writeheader(); w.writerows(seq_rows)

    metrics = ['val_transmission', 'val_dos', 'l12_transmission', 'l16_transmission', 'l12_dos', 'l16_dos']
    lines = ['SUM-POOLING DIRECT BASELINE: per seed, then 3-seed mean [min, max]', '']
    lines.append(f"{'arm':18s}{'seed':>12s}" + ''.join(f'{m:>18s}' for m in metrics))
    for arm in ARMS:
        for r in rows:
            if r['arm'] == arm:
                lines.append(f"{arm:18s}{r['seed']:>12d}" + ''.join(f'{r[m]:18.4f}' for m in metrics))
    lines.append('')
    for arm in ARMS:
        vals = {m: np.array([r[m] for r in rows if r['arm'] == arm]) for m in metrics}
        lines.append(f'{arm:18s}' + ''.join(
            f'{vals[m].mean():.3f} [{vals[m].min():.3f}, {vals[m].max():.3f}]'.rjust(26) for m in metrics))
    lines.append('')
    seqs = sorted({(r['length'], r['composition'], r['sequence']) for r in seq_rows})
    for key, lab in (('huber_t', 'transmission'), ('huber_dos', 'DOS')):
        lines.append(f'PER-DUPLEX {lab} loss, per seed (s1179027592 / s2129768291 / s3731635825)')
        lines.append(f"{'duplex':30s}" + ''.join(f'{a:>26s}' for a in ARMS))
        for L, comp, seq in seqs:
            cells = []
            for arm in ARMS:
                v = [r[key] for s in SEEDS for r in seq_rows
                     if r['arm'] == arm and r['sequence'] == seq and r['length'] == L and r['seed'] == s]
                cells.append(' / '.join(f'{x:.2f}' for x in v).rjust(26))
            lines.append(f'{seq + " (" + comp + ")":30s}' + ''.join(cells))
        lines.append('')
    text = '\n'.join(lines)
    print('\n' + text)
    with open(OUT_TXT, 'w') as fh:
        fh.write(text + '\n')
    print(f'\nwrote {OUT_TXT}, {OUT_CSV}, {OUT_SEQ}')


if __name__ == '__main__':
    main()
