"""Contact-invariance numbers for the paper's "ANOVA Results" section (contact drift by supervision arm and GNN depth) and the Discussion: per-arm and per-depth distributions of the per-run median D_interior_mode, worst-10 strand frequencies, excluded sequences, stub null."""
import os
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
import json, numpy as np, collections
d = json.load(open(os.path.join(_OUT, 'contact_invariance_v3.json')))
runs = d['runs']
q = [5, 25, 50, 75, 95]
print('per-run median D_interior_mode (eV), distribution over the 24 runs of each arm')
for arm in ('dos', 'ldos', 'ldosonly', 'tonly'):
    v = np.array([np.median(r['vectors']['D_interior_mode']) for r in runs.values()
                  if r['supervision'] == arm])
    print('  %-9s n=%2d  ' % (arm, v.size) + '  '.join('p%d=%.3f' % (p, np.percentile(v, p)) for p in q)
          + '  min=%.3f max=%.3f' % (v.min(), v.max()))
print('\nsame, by num_layers')
for L in (2, 4):
    v = np.array([np.median(r['vectors']['D_interior_mode']) for r in runs.values() if r['num_layers'] == L])
    print('  L%d n=%2d  ' % (L, v.size) + '  '.join('p%d=%.3f' % (p, np.percentile(v, p)) for p in q))
print('\nworst-10 strands: how often each sequence appears across the 96 runs')
c = collections.Counter()
for r in runs.values():
    for w in r['worst10_D_interior_mode']:
        c[w['sequence']] += 1
for s, n in c.most_common(12):
    print('  %-10s L=%d  in worst-10 of %2d/96 runs' % (s, len(s), n))
print('\nexcluded sequences:', d['provenance']['excluded_sequences'])
print('stub null:', json.dumps(d['provenance']['synthetic_per_base_stub_null'])[:300])
