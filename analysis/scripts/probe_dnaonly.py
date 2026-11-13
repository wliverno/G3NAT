"""Contact-free variant: the final Hamiltonian cell (LDOS+T, n_orb=2, 4 layers,
geometry off) retrained with --dna_only_message_passing, 3 seeds, against the same
cell and seeds of campaign v3 (contacts in the graph).

Per run:
  val_transmission, val_dos   at the run's own selected epoch (metric_history)
  l12/l16 transmission, dos   pooled Huber on log10 over the 4 contact variants
  per-duplex transmission     mean over the 4 contact variants
  onsite_near, onsite_far     metrics.onsite_distance, mean over EVERY training record
                              (as posthoc_v3 --onsite-all-records)
  onsite_std, onsite_abs_mean std and mean |.| of every onsite block eigenvalue,
                              run1 records of the complete quartets (as contact_invariance_v3)
  lower_min                   deepest lower onsite level over the same run1 records
  drift_median, drift_max     |change| of each site's onsite levels between the four
                              contact variants of a sequence, over every site and level
No interval is claimed at n=3.
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
from g3nat.evaluation.physicality import onsite_block_eigs
from g3nat_analysis import evaluator as EV
from g3nat_analysis.contact_invariance import SETUP_RUN_KEYS
from g3nat_analysis.metrics import onsite_distance
from posthoc_v3 import load_geometry, load_training_records, _check_grid, HELDOUT_DIRS, GEOM_CACHES
from persequence_v3 import load_heldout_indexed

SEEDS = (1179027592, 2129768291, 3731635825)
WEIGHTS = 'hamiltonian_pickle_model_best.pth'
ARMS = {
    # arm -> (dir template, expected dna_only_message_passing)
    'contacts in graph': ('outputs_v3/ham_ldosonly_n2_L4_nogeom_s{seed}', False),
    'contact-free':      ('exploratory/ham_ldosonly_n2_L4_nogeom_dnaonly_s{seed}', True),
}
OUT_DIR = os.path.join(_ANALYSIS, 'outputs')
OUT_TXT = os.path.join(OUT_DIR, 'probe_dnaonly.out')
OUT_CSV = os.path.join(OUT_DIR, 'probe_dnaonly_runs.csv')
OUT_SEQ = os.path.join(OUT_DIR, 'probe_dnaonly_perseq.csv')


def val_at_selection(ck, key):
    epoch = ck.get('saved_at_epoch', ck.get('epoch'))
    for entry in ck.get('metric_history') or []:
        if entry.get('epoch') == epoch:
            return float(entry[key]) if key in entry else float('nan')
    raise ValueError(f'no metric_history entry for saved_at_epoch={epoch}')


def main():
    geometry_cache = load_geometry(list(GEOM_CACHES))
    heldout = {L: load_heldout_indexed(L) for L in sorted(HELDOUT_DIRS)}
    train = load_training_records()
    keys_by_seq = {}
    for seq, rk in train:
        keys_by_seq.setdefault(seq, set()).add(rk)
    quartets = sorted(s for s, ks in keys_by_seq.items() if all(k in ks for k in SETUP_RUN_KEYS))
    loss_fn = EV.LOSS_FUNCS['huber']
    rows, seq_rows = [], []
    for arm, (tmpl, flag) in ARMS.items():
        for seed in SEEDS:
            path = os.path.join(REPO, tmpl.format(seed=seed), WEIGHTS)
            if not os.path.exists(path):
                raise SystemExit(f'FATAL: missing weights {path}')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            model, grid, _ = load_trained_model(path, device='cpu')
            model.eval()
            if bool(getattr(model, 'dna_only_message_passing', False)) != flag:
                raise SystemExit(f'FATAL: {path} dna_only_message_passing != {flag}')
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
                    lt = loss_fn(t_p.squeeze(0).numpy(), np.asarray(rec['transmission'], float).ravel())
                    ld = loss_fn(dos_p.squeeze(0).numpy(), np.asarray(rec['dos'], float).ravel())
                    t_all.append(lt); d_all.append(ld)
                    per_seq.setdefault(seq, []).append(lt)
                row[f'l{L}_transmission'] = float(np.mean(t_all))
                row[f'l{L}_dos'] = float(np.mean(d_all))
                for seq, v in sorted(per_seq.items()):
                    assert len(v) == 4, (arm, seed, L, seq, len(v))
                    seq_rows.append({'arm': arm, 'seed': seed, 'length': L, 'sequence': seq,
                                     'transmission': float(np.mean(v))})

            _check_grid(grid, list(train.values()), f'{arm} s{seed} train')
            near, far, levels = [], [], {}
            for (seq, rk), rec in sorted(train.items()):
                batch = EV.gated_batch(rec, grid, geometry_cache, False)
                with torch.no_grad():
                    model(batch)
                H = model.H[0].cpu().numpy().astype(np.float64)
                d = onsite_distance(H, 2, n_bases=2 * len(seq))
                near.append(d['near']); far.append(d['far'])
                levels[(seq, rk)] = onsite_block_eigs(H, 2)
            run1 = np.concatenate([levels[(s, 'run1')] for s in quartets])
            shifts = []
            for s in quartets:
                stack = np.stack([levels[(s, rk)] for rk in SETUP_RUN_KEYS])   # [4, 2L, 2]
                shifts.append((stack.max(axis=0) - stack.min(axis=0)).ravel())
            shifts = np.concatenate(shifts)
            row.update({'onsite_near': float(np.mean(near)), 'onsite_far': float(np.mean(far)),
                        'onsite_std': float(np.std(run1)), 'onsite_abs_mean': float(np.mean(np.abs(run1))),
                        'lower_min': float(run1[:, 0].min()),
                        'drift_median': float(np.median(shifts)), 'drift_max': float(shifts.max()),
                        'n_onsite_records': len(near), 'n_quartets': len(quartets)})
            rows.append(row)
            print(f'{arm} s{seed} done', flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(OUT_SEQ, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0]))
        w.writeheader(); w.writerows(seq_rows)
    cols = ['val_transmission', 'val_dos', 'l12_transmission', 'l16_transmission', 'l12_dos', 'l16_dos',
            'onsite_near', 'onsite_std', 'lower_min', 'drift_median', 'drift_max']
    lines = ['Contact-free variant vs contacts in graph, final Hamiltonian cell, per seed.',
             'seed        arm                ' + ' '.join(f'{c:>16}' for c in cols)]
    for seed in SEEDS:
        for r in rows:
            if r['seed'] == seed:
                lines.append(f'{seed:<11} {r["arm"]:<18} ' + ' '.join(f'{r[c]:>16.4f}' for c in cols))
    with open(OUT_TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
