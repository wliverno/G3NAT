"""Contact-invariance numbers for Results 3.4: pooled and per-run percentiles of D_interior_mode, axis split, attached vs interior, 32-cell table, marginal means. NOTE: the identical_input_null count printed here is wrong (it tests a dict against (True, "PASS")); the JSON field itself is passed on 96/96 -- left as in the original run."""
import os
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
import json, numpy as np, itertools, collections

d = json.load(open(os.path.join(_OUT, 'contact_invariance_v3.json')))
runs = d['runs']
print('provenance keys:', list(d['provenance'].keys()))
for k, v in d['provenance'].items():
    if isinstance(v, (str, int, float, bool)):
        print('  %-28s %r' % (k, v))

# ---- harness -------------------------------------------------------------
ident = sum(1 for r in runs.values() if r['harness']['identical_input_null'] in (True, 'PASS'))
froz = [r['harness']['frozen_row_check'] for r in runs.values()]
print('\nharness: identical_input_null PASS %d/96' % ident)
print('frozen_row_check example:', json.dumps(froz[0])[:300])

# ---- per-run medians -----------------------------------------------------
CH = ['D_all_total', 'D_interior_total', 'D_attached_total',
      'D_interior_mode', 'D_interior_coupling', 'D_interior_interaction',
      'D_attached_mode', 'D_attached_coupling', 'D_attached_interaction']
rec = {}
for name, r in runs.items():
    row = dict(supervision=r['supervision'], n_orb=r['n_orb'],
               num_layers=r['num_layers'], geometry=r['geometry'],
               seed=r['seed'], nq=r['quartet_count'])
    for c in CH:
        row[c] = float(np.median(r['vectors'][c]))
    for c, v in r['companion'].items():
        row[c] = v
    rec[name] = row

print('\nquartet counts:', sorted(set(v['nq'] for v in rec.values())))

# ---- pooled distribution of the headline channel -------------------------
allv = np.concatenate([np.asarray(r['vectors']['D_interior_mode'], float)
                       for r in runs.values()])
q = [0, 5, 25, 50, 75, 95, 100]
print('\nD_interior_mode pooled over 96 runs x %d strands (n=%d), eV' % (
    runs[list(runs)[0]]['quartet_count'], allv.size))
print('  ' + '  '.join('p%-3d=%.4f' % (p, np.percentile(allv, p)) for p in q))
print('  mean=%.4f' % allv.mean())

# ---- per-run medians: the spread across the 96 checkpoints ---------------
med = np.array([rec[n]['D_interior_mode'] for n in rec])
print('\nper-run median D_interior_mode across the 96 ham checkpoints, eV')
print('  ' + '  '.join('p%-3d=%.4f' % (p, np.percentile(med, p)) for p in q))

# ---- cell means (seed-averaged) -----------------------------------------
cells = collections.defaultdict(list)
for n, r in rec.items():
    cells[(r['supervision'], r['n_orb'], r['num_layers'], r['geometry'])].append(r)
rows = []
for k, v in cells.items():
    rows.append((k, np.mean([x['D_interior_mode'] for x in v]),
                 np.mean([x['D_attached_mode'] for x in v]),
                 np.mean([x['D_interior_coupling'] for x in v]),
                 np.mean([x['onsite_abs_mean'] for x in v]),
                 np.mean([x['onsite_std'] for x in v]),
                 np.mean([x['l16_transmission'] for x in v]), len(v)))
rows.sort(key=lambda t: t[1])
print('\n32 ham cells, seed-averaged, sorted by median D_interior_mode (eV)')
print('%-28s %10s %10s %10s %10s %10s %10s %2s' % (
    'cell', 'int_mode', 'att_mode', 'int_coup', 'ons_absmn', 'ons_std', 'l16_T', 'n'))
for k, a, b, c, e, f, g, n in rows:
    print('%-28s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %2d'
          % ('%s_n%d_L%d_geom%d' % k, a, b, c, e, f, g, n))

# ---- marginal means ------------------------------------------------------
print('\nmarginal means of median D_interior_mode (eV), over the 96 runs')
for fac in ('supervision', 'n_orb', 'num_layers', 'geometry'):
    print(' ', fac)
    for lev in sorted(set(r[fac] for r in rec.values()), key=str):
        v = [r['D_interior_mode'] for r in rec.values() if r[fac] == lev]
        print('    %-10s n=%2d  mean=%.4f  median=%.4f' % (
            lev, len(v), np.mean(v), np.median(v)))

# ---- attached vs interior: the Re(Sigma) signature -----------------------
ai = np.array([[rec[n]['D_attached_total'], rec[n]['D_interior_total']] for n in rec])
print('\nattached vs interior TOTAL drift, per-run medians (eV)')
print('  attached mean=%.4f median=%.4f' % (ai[:, 0].mean(), np.median(ai[:, 0])))
print('  interior mean=%.4f median=%.4f' % (ai[:, 1].mean(), np.median(ai[:, 1])))
print('  runs where attached > interior: %d/96' % int((ai[:, 0] > ai[:, 1]).sum()))

# ---- axis dominance ------------------------------------------------------
ax = np.array([[rec[n]['D_interior_coupling'], rec[n]['D_interior_mode'],
                rec[n]['D_interior_interaction']] for n in rec])
lab = ['coupling', 'mode', 'interaction']
print('\ninterior axis medians (eV): ' + '  '.join(
    '%s=%.4f' % (l, np.median(ax[:, i])) for i, l in enumerate(lab)))
dom = collections.Counter(lab[i] for i in ax.argmax(1))
print('dominant interior axis, count over 96 runs:', dict(dom))

