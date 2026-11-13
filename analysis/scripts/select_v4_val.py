"""Validation-only quantities for the final-configuration rule (protocol amendment 4).

Per contact-free Hamiltonian run (campaign/ham_*): validation transmission, DOS and
per-residue LDOS loss at the run's own selected epoch (checkpoint metric history), and
onsite distance from the HOMO (metrics.onsite_distance 'near', mean over every
pickle_files_v2 record, as posthoc_v3 --onsite-all-records). No held-out data is read.

Usage: select_v4_val.py <task_index> <n_tasks> <out_dir>   (one JSON per run)
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
from g3nat_analysis.metrics import onsite_distance
from posthoc_v3 import load_training_records, _check_grid


def val_at_selection(ck, key):
    epoch = ck.get('saved_at_epoch', ck.get('epoch'))
    for entry in ck.get('metric_history') or []:
        if entry.get('epoch') == epoch:
            return float(entry[key]) if key in entry else float('nan')
    raise ValueError(f'no metric_history entry for saved_at_epoch={epoch}')


def main():
    task, n_tasks, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    paths = sorted(glob.glob(os.path.join(REPO, 'campaign', 'ham_*', 'hamiltonian_pickle_model_best.pth')))
    if len(paths) != 96:
        raise SystemExit(f'FATAL: expected 96 contact-free runs, found {len(paths)}')
    mine = paths[task::n_tasks]
    records = load_training_records()
    os.makedirs(out_dir, exist_ok=True)
    for p in mine:
        tag = os.path.basename(os.path.dirname(p))
        parsed = EV.parse_run_name(tag)
        if parsed is None or parsed['family'] != 'ham':
            raise SystemExit(f'FATAL: cannot parse {tag}')
        ck = torch.load(p, map_location='cpu', weights_only=False)
        model, grid, _ = load_trained_model(p, device='cpu')
        model.eval()
        if not getattr(model, 'dna_only_message_passing', False):
            raise SystemExit(f'FATAL: {tag} is not contact-free')
        geom = bool(parsed['geometry'])
        if geom != bool(getattr(model, 'use_geometry', False)):
            raise SystemExit(f'FATAL: geometry mismatch {tag}')
        _check_grid(grid, list(records.values()), tag)
        near, far = [], []
        # onsite levels do not depend on geometry features beyond what the model reads;
        # geometry cache is required for geom runs on training lengths (geometry_v2.pkl).
        geometry_cache = GEOM if geom else None
        for (seq, rk), rec in sorted(records.items()):
            batch = EV.gated_batch(rec, grid, geometry_cache, geom)
            with torch.no_grad():
                model(batch)
            H = model.H[0].cpu().numpy().astype(np.float64)
            d = onsite_distance(H, parsed['n_orb'], n_bases=2 * len(seq))
            near.append(d['near']); far.append(d['far'])
        row = dict(parsed, run=tag,
                   saved_at_epoch=ck.get('saved_at_epoch', ck.get('epoch')),
                   val_transmission=EV.transmission_at_selection(ck),
                   val_dos=val_at_selection(ck, 'val_dos'),
                   val_ldos_residue=val_at_selection(ck, 'val_ldos_residue'),
                   onsite_near=float(np.mean(near)), onsite_far=float(np.mean(far)),
                   onsite_n_graphs=len(near), n_onsite_records=len(near))
        with open(os.path.join(out_dir, tag + '.json'), 'w') as f:
            json.dump(row, f)
        print(f"{tag} valT={row['val_transmission']:.4f} valDOS={row['val_dos']:.4f} "
              f"valLDOS={row['val_ldos_residue']:.4f} onsite={row['onsite_near']:.4f}", flush=True)


if __name__ == '__main__':
    from posthoc_v3 import load_geometry
    GEOM = load_geometry([os.path.join(REPO, 'geom_cache', 'geometry_v2.pkl')])
    main()
