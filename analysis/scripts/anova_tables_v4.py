"""LaTeX rows for the revised ANOVA tables (tab:anova-ham, tab:anova-direct) from
outputs/doe_v4.out (F, p) and outputs/posthoc_v4_report.json (choice means). Same format as the
submitted tables: every term with raw p < 0.05, grouped by response, bold if it survives BH.
Choice = lowest mean level (main effect) or lowest mean level combination (interaction).
"""
import itertools
import json
import os
import re
import statistics as st

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOE = open(os.path.join(_ANALYSIS, 'outputs', 'doe_v4.out')).read()
RUNS = json.load(open(os.path.join(_ANALYSIS, 'outputs', 'posthoc_v4_report.json')))['runs']

LEVELS = {'supervision': ('dos', 'ldos', 'ldosonly', 'tonly'), 'n_orb': (1, 2), 'num_layers': (2, 4),
          'geometry': (0, 1), 'pooling': ('mean', 'sum')}
SUP = {'dos': 'DOS+T', 'ldos': 'LDOS+DOS+T', 'ldosonly': 'LDOS+T', 'tonly': 'T-only'}
FNAME = {'supervision': 'supervision', 'n_orb': r'$n_{\mathrm{orb}}$', 'num_layers': 'layers',
         'geometry': 'geometry', 'pooling': 'pooling'}
RESP = [('val_transmission_at_selection', 'in-distribution $T$'), ('l12_transmission', '12-bp $T$'),
        ('l12_dos', '12-bp DOS'), ('l16_transmission', '16-bp $T$'), ('l16_dos', '16-bp DOS'),
        ('onsite_near', 'onsite distance'), ('val_ldos_residue', 'per-residue LDOS')]


def lvl(f, v):
    if f == 'supervision':
        return SUP[v]
    if f == 'n_orb':
        return r'$n_{\mathrm{orb}}=%d$' % v
    if f == 'num_layers':
        return '%d layers' % v
    if f == 'geometry':
        return 'geometry ' + ('on' if v else 'off')
    return v + ' pooling'


def blocks(family_marker_start, family_marker_end):
    seg = DOE.split(family_marker_start, 1)[1].split(family_marker_end, 1)[0]
    out = {}
    for b in seg.split('RESPONSE: ')[1:]:
        name = b.split('\n', 1)[0].strip()
        rows = re.findall(r'^\s{2}([a-z_:]+)\s+(\d+)\s+\S+\s+([\d.]+)\s+([\d.e+-]+)\s*$', b, re.M)
        out[name] = [(t, float(F), float(p)) for t, _df, F, p in rows]
    return out


def fmt_p(p):
    m, e = ('%.1e' % p).split('e')
    return r'$%s\times10^{%d}$' % (m, int(e))


def choice(fam_runs, factors, resp):
    L = {f: tuple(v for v in LEVELS[f] if any(r[f] == v for r in fam_runs)) for f in factors}
    if len(factors) == 1:
        f = factors[0]
        means = [st.mean(r[resp] for r in fam_runs if r[f] == v) for v in L[f]]
        best = L[f][means.index(min(means))]
        return '%s (%s)' % (lvl(f, best), ' / '.join('%.3g' % m for m in means))
    combos = []
    for vals in itertools.product(*[L[f] for f in factors]):
        rs = [r[resp] for r in fam_runs if all(r[f] == v for f, v in zip(factors, vals))]
        combos.append((st.mean(rs), vals))
    m, vals = min(combos)
    return '%s (%.3g)' % (', '.join(lvl(f, v) for f, v in zip(factors, vals)), m)


def table(family, thr, start, end):
    fam_runs = [r for r in RUNS.values() if r['family'] == family]
    bl = blocks(start, end)
    lines = []
    for key, label in RESP:
        if key not in bl:
            continue
        sig = sorted([x for x in bl[key] if x[2] < 0.05], key=lambda x: x[2])
        if not sig:
            lines.append(r'%s & \textit{none with $p<0.05$} & -- & -- & -- \\' % label)
            lines.append(r'\addlinespace')
            continue
        for i, (term, F, p) in enumerate(sig):
            factors = term.split(':')
            tname = r' $\times$ '.join(FNAME[f] for f in factors)
            cells = [label if i == 0 else '', tname, '%.1f' % F, fmt_p(p), choice(fam_runs, factors, key)]
            if p <= thr:
                cells = [r'\textbf{%s}' % c if c else '' for c in cells]
            lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\addlinespace')
    return '\n'.join(lines[:-1])


thr_h = float(re.search(r'survivors: \d+ of 105\s+threshold=([\d.]+)', DOE).group(1))
thr_d = float(re.search(r'survivors: \d+ of 75\s+threshold=([\d.]+)', DOE).group(1))
print('% HAM threshold', thr_h)
print(table('ham', thr_h, 'REVISION: contact-invariance', 'entered pooling'))
print('% DIRECT threshold', thr_d)
print(table('blind', thr_d, 'entered pooling', 'SEED-TO-SEED DISPERSION'))
