"""The four arms of probe_sumpool on the untouched 16-bp set: the 7 duplexes completed so far
(28 records, DNADataset/validation_L16/pickles_newset). Scoring the 7 before the replacement
lands was allowed by the scoring plan, written in advance. Report only; nothing here selects
a model, a cell or a seed.

Per run: pooled transmission and DOS Huber loss on log10 over all records; per duplex:
transmission and DOS loss, mean over the 4 contact variants. No interval is claimed at n=3.
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

from g3nat.data.pickle import load_single_pickle
from g3nat.evaluation.inference import load_trained_model
from g3nat_analysis import evaluator as EV
from posthoc_v3 import _check_grid
from probe_sumpool import ARMS, SEEDS

NEWSET_DIR = os.path.join(REPO, 'DNADataset', 'validation_L16', 'pickles_newset')
OUT_DIR = os.path.join(_ANALYSIS, 'outputs')
OUT_TXT = os.path.join(OUT_DIR, 'probe_newset.out')
OUT_CSV = os.path.join(OUT_DIR, 'probe_newset_runs.csv')
OUT_SEQ = os.path.join(OUT_DIR, 'probe_newset_perseq.csv')


def load_newset():
    out = {}
    for p in sorted(glob.glob(os.path.join(NEWSET_DIR, '*.pkl'))):
        rec = load_single_pickle(p)
        if rec is None:
            raise SystemExit(f'FATAL: unreadable new-set pickle: {p}')
        run_key = os.path.splitext(os.path.basename(p))[0].rsplit('_', 1)[1]
        out[(rec['sequence'].upper(), run_key)] = rec
    seqs = {s for s, _ in out}
    if len(out) != 4 * len(seqs):
        raise SystemExit(f'FATAL: {len(out)} records for {len(seqs)} duplexes')
    return out


def main():
    recs = load_newset()
    n_dup = len({s for s, _ in recs})
    loss_fn = EV.LOSS_FUNCS['huber']
    rows, seq_rows = [], []
    for arm, (tmpl, weights, attr, expect) in ARMS.items():
        for seed in SEEDS:
            path = os.path.join(REPO, tmpl.format(seed=seed), weights)
            if not os.path.exists(path):
                raise SystemExit(f'FATAL: missing weights {path}')
            model, grid, _ = load_trained_model(path, device='cpu')
            model.eval()
            if getattr(model, attr, None) != expect:
                raise SystemExit(f'FATAL: {path} {attr}={getattr(model, attr, None)!r}, expected {expect!r}')
            if getattr(model, 'use_geometry', False):
                raise SystemExit(f'FATAL: {path} uses geometry; no new-set geometry cache exists')
            _check_grid(grid, list(recs.values()), f'{arm} s{seed} newset')
            per_seq, t_all, d_all = {}, [], []
            for (seq, rk), rec in sorted(recs.items()):
                batch = EV.gated_batch(rec, grid, {}, False)
                with torch.no_grad():
                    dos_p, t_p = model(batch)
                lt = loss_fn(t_p.squeeze(0).cpu().numpy(), np.asarray(rec['transmission'], float).ravel())
                ld = loss_fn(dos_p.squeeze(0).cpu().numpy(), np.asarray(rec['dos'], float).ravel())
                t_all.append(lt); d_all.append(ld)
                per_seq.setdefault(seq, []).append((lt, ld))
            rows.append({'arm': arm, 'seed': seed, 'n_duplexes': n_dup, 'n_records': len(t_all),
                         'new_transmission': float(np.mean(t_all)), 'new_dos': float(np.mean(d_all))})
            for seq, v in sorted(per_seq.items()):
                assert len(v) == 4, (arm, seed, seq, len(v))
                v = np.array(v)
                seq_rows.append({'arm': arm, 'seed': seed, 'sequence': seq,
                                 'huber_t': float(v[:, 0].mean()), 'huber_dos': float(v[:, 1].mean())})
            print(f"{arm:16s} s{seed}  T={rows[-1]['new_transmission']:.3f}  DOS={rows[-1]['new_dos']:.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT_SEQ, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(seq_rows[0])); w.writeheader(); w.writerows(seq_rows)

    lines = [f'UNTOUCHED 16-bp SET, {n_dup} completed duplexes ({len(recs)} records); replacement pending.',
             'Per seed (s1179027592 / s2129768291 / s3731635825), then 3-seed mean.', '']
    for key, lab in (('new_transmission', 'pooled transmission'), ('new_dos', 'pooled DOS')):
        lines.append(lab)
        for arm in ARMS:
            v = [r[key] for s in SEEDS for r in rows if r['arm'] == arm and r['seed'] == s]
            lines.append(f'  {arm:18s}' + ' / '.join(f'{x:.3f}' for x in v) + f'   mean {np.mean(v):.3f}')
        lines.append('')
    for key, lab in (('huber_t', 'PER-DUPLEX transmission'), ('huber_dos', 'PER-DUPLEX DOS')):
        lines.append(lab + ' loss, per seed')
        lines.append(f"{'duplex':20s}" + ''.join(f'{a:>24s}' for a in ARMS))
        for seq in sorted({r['sequence'] for r in seq_rows}):
            cells = []
            for arm in ARMS:
                v = [r[key] for s in SEEDS for r in seq_rows
                     if r['arm'] == arm and r['sequence'] == seq and r['seed'] == s]
                cells.append(' / '.join(f'{x:.2f}' for x in v).rjust(24))
            lines.append(f'{seq:20s}' + ''.join(cells))
        lines.append('')
    text = '\n'.join(lines)
    print('\n' + text)
    with open(OUT_TXT, 'w') as fh:
        fh.write(text + '\n')
    print(f'wrote {OUT_TXT}, {OUT_CSV}, {OUT_SEQ}')


if __name__ == '__main__':
    main()
