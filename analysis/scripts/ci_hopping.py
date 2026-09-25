"""Hopping-channel numbers for Results 3.4: per-run medians of the D_coupling_* vectors, dominant axis, signed/unsigned ratio, best/worst cell."""
import os
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
import json, numpy as np, collections
d = json.load(open(os.path.join(_OUT, 'contact_invariance_v3.json')))
runs = d['runs']
r0 = next(iter(runs.values()))
keys = list(r0['vectors'].keys())
print('ALL vectors keys (%d):' % len(keys)); print('  ' + ', '.join(keys))
print('other run keys:', [k for k in r0 if k not in ('vectors',)])
hop = [k for k in keys if k.startswith('D_coupling')]
q = [5, 25, 50, 75, 95]
print('\nper-run medians over the 96 runs, eV')
med = {k: np.array([np.median(r['vectors'][k]) for r in runs.values()]) for k in hop}
for k in hop:
    v = med[k]
    print('  %-28s ' % k + '  '.join('p%d=%.4f' % (p, np.percentile(v, p)) for p in q))
# axis dominance and signed-vs-unsigned
ax = [k for k in hop if not k.endswith('signed') and 'signed' not in k and k.split('_')[-1] in ('coupling', 'mode', 'interaction')]
print('\naxis keys:', ax)
if len(ax) == 3:
    M = np.stack([med[k] for k in ax], 1)
    dom = collections.Counter(ax[i].split('_')[-1] for i in M.argmax(1))
    print('dominant hopping axis over 96 runs:', dict(dom))
tot = [k for k in hop if 'signed' not in k and k.split('_')[-1] not in ('coupling', 'mode', 'interaction')]
sg = [k for k in hop if 'signed' in k and k.split('_')[-1] not in ('coupling', 'mode', 'interaction')]
print('total keys:', tot, ' signed total keys:', sg)
if tot and sg:
    a, b = med[tot[0]], med[sg[0]]
    print('signed/unsigned ratio of per-run medians: median %.3f, max %.3f; runs with signed > 1.5x unsigned: %d/96'
          % (np.median(b / a), (b / a).max(), int((b > 1.5 * a).sum())))
print('\nby arm / depth (median of per-run medians of %s)' % tot[0])
for fac in ('supervision', 'num_layers', 'n_orb'):
    for lev in sorted(set(r[fac] for r in runs.values()), key=str):
        v = [np.median(r['vectors'][tot[0]]) for r in runs.values() if r[fac] == lev]
        print('  %-11s %-9s n=%2d median=%.4f' % (fac, lev, len(v), np.median(v)))
cells = collections.defaultdict(list)
for n, r in runs.items():
    cells[(r['supervision'], r['n_orb'], r['num_layers'], r['geometry'])].append(np.median(r['vectors'][tot[0]]))
rows = sorted((np.mean(v), k) for k, v in cells.items())
print('\nbest cell: %s_n%d_L%d_geom%d = %.4f   worst: %s_n%d_L%d_geom%d = %.4f' % (*rows[0][1], rows[0][0], *rows[-1][1], rows[-1][0]))
