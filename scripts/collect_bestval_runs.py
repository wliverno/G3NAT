#!/usr/bin/env python3
"""Collect the 2026-07-25 best-val reruns: loss tables, per-base baselines, and eta2 --
all measured on BEST-VAL weights, which is the first time any model-derived quantity in
this project has been.

Everything before these runs saved final-epoch weights only, and best val is reached around
epoch 550-1900, so the baselines/eta2/window fractions on record were read off models that
had been overfitting for thousands of epochs.
"""
import os, glob, re
import numpy as np
import torch
from torch_geometric.data import Batch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import g3nat
from g3nat.graph import sequence_to_graph
from g3nat.data.pickle import load_single_pickle

BASES = ['A', 'T', 'G', 'C']
# run-directory prefix, e.g. `bestval` (3000-epoch set) or `bv5k` (5000-epoch set)
PFX = sys.argv[1] if len(sys.argv) > 1 else 'bestval'
print(f'collecting prefix: outputs_{PFX}_*\n')


def stat(v):
    v = np.asarray(v, float)
    return (v.mean(), v.std(ddof=1) if v.size > 1 else 0.0, v.size)


def curves(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        vl = np.asarray(ck['val_losses'], float)
        out[os.path.dirname(p)] = (float(vl[-1]), float(vl.min()), int(vl.argmin()))
    return out


print('=' * 76)
print('1. LOSS TABLES from the 3000-epoch reruns  (final vs best-val)')
print('=' * 76)
for label, pat, key in [
        ('alpha', f'outputs_{PFX}_a*/hamiltonian_pickle_model.pth', r'_a([0-9.]+)_s'),
        ('layers', f'outputs_{PFX}_L*/hamiltonian_pickle_model.pth', r'_L(\d+)_s')]:
    groups = {}
    for d, (f, b, e) in curves(pat).items():
        k = re.search(key, d).group(1)
        groups.setdefault(k, {'f': [], 'b': [], 'e': []})
        groups[k]['f'].append(f); groups[k]['b'].append(b); groups[k]['e'].append(e)
    print(f'\n  {label:>6} | {"final mean":>11}{"std":>8} | {"BEST mean":>11}{"std":>8} | best epoch')
    print('  ' + '-' * 68)
    for k in sorted(groups, key=lambda x: float(x)):
        fm, fs, n = stat(groups[k]['f']); bm, bs, _ = stat(groups[k]['b'])
        print(f'  {k:>6} | {fm:11.4f}{fs:8.4f} | {bm:11.4f}{bs:8.4f} | {sorted(groups[k]["e"])}')

print('\n' + '=' * 76)
print('2. PER-BASE BASELINES from BEST-VAL weights (alpha=1.0 cells)')
print('=' * 76)
rows = {}
for p in sorted(glob.glob(f'outputs_{PFX}_a1.0_s*/hamiltonian_pickle_model_best.pth')):
    s = re.search(r'_s(\d+)', p).group(1)
    b = torch.load(p, map_location='cpu', weights_only=False)['model_state_dict'].get('onsite_baseline')
    if b is not None:
        rows[s] = b.numpy().ravel()
if rows:
    print('   seed  ' + '  '.join(f'{x:>9}' for x in BASES) + '   range')
    for s, v in sorted(rows.items()):
        print(f'   {s:>4}  ' + '  '.join(f'{x:>9.4f}' for x in v) + f'   {v.max()-v.min():.4f}')
    M = np.stack(list(rows.values())); Mg = M - M[:, [2]]      # reference to G
    print('   --- referenced to G ---')
    print('   mean  ' + '  '.join(f'{x:>9.4f}' for x in Mg.mean(0)))
    print('   std   ' + '  '.join(f'{x:>9.4f}' for x in Mg.std(0, ddof=1)))
    mg, sg = Mg.mean(0), Mg.std(0, ddof=1)
    print('\n   pairwise gap vs cross-seed scatter (RESOLVED needs gap > 2*scatter):')
    for i in range(4):
        for j in range(i + 1, 4):
            gap = abs(mg[i] - mg[j]); sc = float(np.sqrt(sg[i] ** 2 + sg[j] ** 2))
            print(f'     {BASES[i]}-{BASES[j]}: gap {gap:6.3f}  scatter {sc:6.3f}  -> '
                  f'{"RESOLVED" if sc > 0 and gap > 2 * sc else "not resolved"}')
else:
    print('   no _best.pth found for alpha=1.0')

print('\n' + '=' * 76)
print('3. eta2 on BEST-VAL weights across depth  (task 7: does depth cost base identity?)')
print('=' * 76)
print('   eta2 = SS_between / SS_total for diag(H) grouped by base. 1 = onsite fully set by')
print('   base identity; ~0 = context sets everything. Free-model reference (overfit wts) 0.028.')

files = sorted(glob.glob('pickle_files/*_run1.pkl'))[:60]
graphs = []
for f in files:
    d = load_single_pickle(f)
    if d is None:
        continue
    g = sequence_to_graph(primary_sequence=d['sequence'],
                          complementary_sequence=d['complementary_sequence'],
                          left_contact_positions=d['left_contact_pos'],
                          right_contact_positions=d['right_contact_pos'],
                          left_contact_coupling=float(d['coupling']),
                          right_contact_coupling=float(d['coupling']))
    # node order: contacts, primary IN ORDER, complementary IN ORDER (construction.py:118-131)
    graphs.append((d['sequence'] + d['complementary_sequence'], g))
print(f'   ({len(graphs)} sequences)')


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    m = g3nat.DNATransportHamiltonianGNN(
        hidden_dim=a['hidden_dim'], num_layers=a['num_layers'], num_heads=a['num_heads'],
        energy_grid=np.asarray(ck['energy_grid'], float), n_orb=a['n_orb'],
        conv_type=a.get('conv_type', 'gat'),
        structured_onsite=a.get('structured_onsite', False),
        alpha_granularity=a.get('alpha_granularity', 'global'),
        alpha_mode=a.get('alpha_mode', 'fixed'),
        alpha_value=a.get('alpha_value', 0.0), alpha_init=a.get('alpha_init', 0.9))
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


def eta2_of(model):
    per = {b: [] for b in BASES}
    for full, g in graphs:
        with torch.no_grad():
            model(Batch.from_data_list([g]))
        H = model.H[0].cpu().numpy()
        for i in range(min(len(full), H.shape[0])):
            if full[i].upper() in per:
                per[full[i].upper()].append(float(H[i, i]))
    vals = {b: np.array(v) for b, v in per.items() if len(v)}
    allv = np.concatenate(list(vals.values())); grand = allv.mean()
    ss_tot = float(np.sum((allv - grand) ** 2))
    ss_bet = float(sum(len(v) * (v.mean() - grand) ** 2 for v in vals.values()))
    return (ss_bet / ss_tot if ss_tot > 0 else 0.0), {b: v.mean() for b, v in vals.items()}


print(f'\n   {"cell":>12} {"eta2":>8}   per-base mean onsite (A/T/G/C)')
for p in sorted(glob.glob(f'outputs_{PFX}_L*/hamiltonian_pickle_model_best.pth')):
    e2, means = eta2_of(load(p))
    tag = re.search(r'(L\d+_s\d+)', p).group(1)
    print(f'   {tag:>12} {e2:8.4f}   ' + '  '.join(f'{b}:{means.get(b, float("nan")):+.3f}' for b in BASES))
