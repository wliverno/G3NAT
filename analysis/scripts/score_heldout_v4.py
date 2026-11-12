"""Report-only held-out scoring of the retrained families (selection is already fixed by
protocol amendments 5 and 6): 96 contact-free Hamiltonian runs (outputs_v4/ham_*), 24 sum-pooled
direct runs (outputs_v4/blind_*_sumpool_*) and 24 mean-pooled direct runs (outputs_v3/blind_*).
Per run: pooled 12/16-bp transmission and DOS Huber loss on log10, and per-duplex T and DOS
(mean over the 4 contact configurations). One JSON per run.

Usage: score_heldout_v4.py <task_index> <n_tasks> <out_dir>
"""
import glob
import json
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


def runs():
    ham = sorted(glob.glob(os.path.join(REPO, 'outputs_v4', 'ham_*', 'hamiltonian_pickle_model_best.pth')))
    dsum = sorted(glob.glob(os.path.join(REPO, 'outputs_v4', 'blind_*_sumpool_s*', 'standard_pickle_model_best.pth')))
    dmean = sorted(glob.glob(os.path.join(REPO, 'outputs_v3', 'blind_*', 'standard_pickle_model_best.pth')))
    if (len(ham), len(dsum), len(dmean)) != (96, 24, 24):
        raise SystemExit(f'FATAL: run counts {len(ham)}, {len(dsum)}, {len(dmean)}')
    return ham + dsum + dmean


def main():
    task, n_tasks, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    mine = runs()[task::n_tasks]
    geometry_cache = load_geometry(list(GEOM_CACHES))
    heldout = {L: load_heldout_indexed(L) for L in sorted(HELDOUT_DIRS)}
    loss_fn = EV.LOSS_FUNCS['huber']
    os.makedirs(out_dir, exist_ok=True)
    for p in mine:
        tag = os.path.basename(os.path.dirname(p))
        ck = torch.load(p, map_location='cpu', weights_only=False)
        args = ck.get('args', {})
        model, grid, _ = load_trained_model(p, device='cpu')
        model.eval()
        geom = bool(args.get('use_geometry', False))
        if geom != bool(getattr(model, 'use_geometry', False)):
            raise SystemExit(f'FATAL: geometry mismatch {tag}')
        fam = 'ham' if tag.startswith('ham_') else 'direct'
        if fam == 'ham' and not getattr(model, 'dna_only_message_passing', False):
            raise SystemExit(f'FATAL: {tag} not contact-free')
        pool = getattr(model, 'readout_pooling', None)
        if fam == 'direct' and (pool == 'sum') != ('_sumpool_' in tag):
            raise SystemExit(f'FATAL: pooling mismatch {tag}: {pool}')
        row = {'run': tag, 'family': fam, 'pooling': pool,
               'saved_at_epoch': ck.get('saved_at_epoch', ck.get('epoch')),
               'val_transmission': EV.transmission_at_selection(ck), 'perseq': []}
        for L, recs in heldout.items():
            _check_grid(grid, list(recs.values()), f'{tag} L{L}')
            per_seq, t_all, d_all = {}, [], []
            for (seq, rk), rec in recs.items():
                batch = EV.gated_batch(rec, grid, geometry_cache, geom)
                with torch.no_grad():
                    dos_p, t_p = model(batch)
                lt = loss_fn(t_p.squeeze(0).cpu().numpy(), np.asarray(rec['transmission'], float).ravel())
                ld = loss_fn(dos_p.squeeze(0).cpu().numpy(), np.asarray(rec['dos'], float).ravel())
                t_all.append(lt); d_all.append(ld)
                per_seq.setdefault(seq, []).append((lt, ld))
            row[f'l{L}_transmission'] = float(np.mean(t_all))
            row[f'l{L}_dos'] = float(np.mean(d_all))
            for seq, v in sorted(per_seq.items()):
                assert len(v) == 4, (tag, L, seq, len(v))
                v = np.array(v)
                row['perseq'].append({'length': L, 'sequence': seq, 'composition': classify_composition(seq),
                                      'huber_t': float(v[:, 0].mean()), 'huber_dos': float(v[:, 1].mean())})
        with open(os.path.join(out_dir, tag + '.json'), 'w') as f:
            json.dump(row, f)
        print(f"{tag} l12={row['l12_transmission']:.3f} l16={row['l16_transmission']:.3f}", flush=True)


if __name__ == '__main__':
    main()
