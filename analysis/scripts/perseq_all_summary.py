import os
import sys

import pandas as pd

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ANALYSIS, 'outputs')
IN_CSV = os.path.join(OUT_DIR, 'perseq_all_runs.csv')
OUT_TXT = os.path.join(OUT_DIR, 'perseq_all_summary.out')

_report_lines = []


def print(*args, **kwargs):
    import builtins
    text = ' '.join(str(a) for a in args)
    _report_lines.append(text)
    builtins.print(*args, **kwargs)


d = pd.read_csv(IN_CSV)
d['cell'] = d.supervision + '/n' + d.n_orb.fillna(0).astype(int).astype(str) + '/L' + d.num_layers.astype(str) + '/g' + d.geometry.astype(str)
# per (run, length, sequence): mean over 4 contact variants
r = d.groupby(['family','cell','seed','length','sequence'], as_index=False).huber_t.mean()
# per (cell, length, sequence): best seed (min) and 3-seed mean
c = r.groupby(['family','cell','length','sequence']).huber_t.agg(best='min', mean='mean').reset_index()
for stat in ('best','mean'):
    print(f'\n=== {stat} seed per cell: worst Hamiltonian cell vs best direct cell, per duplex')
    for (L, seq), g in c.groupby(['length','sequence']):
        h = g[g.family=='ham'][stat]; b = g[g.family=='blind'][stat]
        ok = h.max() < b.min()
        print(f'L{L} {seq:18s} ham {h.min():.2f}-{h.max():.2f}  direct {b.min():.2f}-{b.max():.2f}  '
              f'all32<all8: {ok}   n_ham_cells_below_best_direct: {(h < b.min()).sum()}/32')
# the figure-cell check: best checkpoint (single seed) on each duplex, ham fig cell vs direct fig cell
print('\n=== figure cells, best seed per duplex')
f = r[(r.cell.isin(['ldosonly/n2/L4/g0','dos/n0/L2/g0']))]
print(f.groupby(['cell','length','sequence']).huber_t.min().unstack(0).round(2))

with open(OUT_TXT, 'w') as fh:
    fh.write('\n'.join(_report_lines) + '\n')
