#!/usr/bin/env python3
"""Recompute every sweep table at BEST-VAL instead of final-epoch.

Why: six runs at identical config and identical split_seed differ only in init, and their
final-epoch val loss has std 0.0286 while their best-val has std 0.0084 -- 3.4x tighter.
Best val is reached at epoch 549-1900 of 5000, so every recorded number is a
post-overfitting number and the "noise" is mostly how far each run drifted past its own
optimum. `val_losses` is stored in every checkpoint, so this costs no new training.

CAVEAT this script cannot fix: the saved WEIGHTS are final-epoch, not best-val. Val-loss
comparisons are recoverable here; anything measured FROM the model (per-base baselines,
eta2, window fractions, LDOS) was computed on overfit weights and needs retraining with
best-val checkpointing to correct.

Usage (compute node):
  srun -A anantram-ckpt -p ckpt-all -c 2 --mem=8G -t 20 \
    conda run -n g3nat python scripts/collect_bestval.py
"""
import os, glob, re
import numpy as np
import torch


def curve(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    vl = np.asarray(ck.get('val_losses', []), dtype=float)
    if vl.size == 0:
        return None
    return {'final': float(vl[-1]), 'best': float(vl.min()),
            'epoch_at_best': int(vl.argmin()), 'n_epochs': int(vl.size)}


def stat(vals):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return None
    return v.mean(), (v.std(ddof=1) if v.size > 1 else 0.0), v.size


def table(title, cells, order):
    """cells: {key: [values]}. Prints final vs best side by side."""
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)
    print(f"{'cell':>8} | {'final-epoch':>22} | {'best-val':>22} | {'delta':>7}")
    print(f"{'':>8} | {'mean':>10}{'std':>7}{'n':>5} | {'mean':>10}{'std':>7}{'n':>5} |")
    print('-' * 78)
    for k in order:
        if k not in cells or not cells[k]['best']:
            continue
        fm, fs, fn = stat(cells[k]['final'])
        bm, bs, bn = stat(cells[k]['best'])
        print(f'{k:>8} | {fm:10.4f}{fs:7.4f}{fn:5d} | {bm:10.4f}{bs:7.4f}{bn:5d} | {fm-bm:+7.4f}')


# ---------------------------------------------------------------- alpha sweep
cells = {}
for a in ['0', '0.25', '0.5', '0.75', '0.9', '1.0']:
    cells[a] = {'final': [], 'best': []}
    for s in ['42', '43', '44']:
        p = f'outputs_onsite_sweep_a{a}_s{s}/hamiltonian_pickle_model.pth'
        if not os.path.exists(p):
            continue
        c = curve(p)
        if c is None:
            continue
        if c['final'] > 1.2:          # optimizer failures, excluded on record
            print(f'  [excluded] alpha={a} s={s}: final {c["final"]:.4f} (failed run)')
            continue
        cells[a]['final'].append(c['final']); cells[a]['best'].append(c['best'])
table('ALPHA SWEEP  (failed cells excluded)', cells, ['0', '0.25', '0.5', '0.75', '0.9', '1.0'])

# ---------------------------------------------------------------- layers sweep
lay = {}
for p in sorted(glob.glob('outputs_layers_*_L*_s*/hamiltonian_pickle_model.pth')):
    m = re.search(r'_L(\d+)_s(\d+)', p)
    L = m.group(1)
    c = curve(p)
    if c is None:
        continue
    if c['final'] > 1.2:
        print(f'  [excluded] {os.path.dirname(p)}: final {c["final"]:.4f} (failed run)')
        continue
    lay.setdefault(L, {'final': [], 'best': []})
    lay[L]['final'].append(c['final']); lay[L]['best'].append(c['best'])
table('LAYERS SWEEP  (layers_sweep + layers_recheck pooled, failed excluded)',
      lay, sorted(lay, key=int))

# ---------------------------------------------------------- init noise floor
print('\n' + '=' * 78)
print('INIT NOISE FLOOR  (identical config AND identical split_seed 42)')
print('=' * 78)
rep = sorted(glob.glob('outputs_baseline_r*_a0_s42/hamiltonian_pickle_model.pth')) + [
    'outputs_onsite_sweep_a0_s42/hamiltonian_pickle_model.pth',
    'outputs_layers_recheck_L4_s42/hamiltonian_pickle_model.pth']
f, b, e = [], [], []
for p in rep:
    if not os.path.exists(p):
        continue
    c = curve(p)
    if c:
        f.append(c['final']); b.append(c['best']); e.append(c['epoch_at_best'])
for name, v in [('final-epoch', f), ('best-val', b)]:
    m, s, n = stat(v)
    print(f'  {name:>12}: mean {m:.4f}  std(ddof=1) {s:.4f}  range {max(v)-min(v):.4f}  n={n}')
print(f'  best-val epoch: {sorted(e)} of 5000')
print(f'  -> USE THIS as the yardstick: a difference must beat ~2x the best-val std')
print(f'     ({2*np.std(b, ddof=1):.4f}) before it means anything.')
