"""Untouched 16-bp set report, as fixed by the scoring plan written before its DFT results existed.

Primary: selected Hamiltonian (ham_dos_n2_L4_nogeom_s2129768291) vs selected direct model
(blind_tonly_L4_nogeom_sumpool_s3731635825), per duplex T with the margin, one-sided sign test
(Hamiltonian lower on T), n stated. DOS: the selected direct model has no DOS supervision, so DOS
is compared against the DOS+T sum-pooled direct configuration's validation-selected run
(blind_dos_L2_nogeom_sumpool_s2129768291), labelled.
Family: per duplex, fraction of (Hamiltonian cell, direct cell) pairs with the Hamiltonian lower
(cell = 3-run mean), against sum-pooled and against mean-pooled direct cells separately.
Labelled extras: LDOS+T run kept by the superseded rule; submitted pair is NOT re-scored here
(contacts-in-graph checkpoints; see probe_newset.out for the 7-duplex numbers).
Sets: primary = all 8; also the 7 original completers and strict k = 5 (first 5 completed in
manifest order).
"""
import glob
import json
import os
import statistics as st
from math import comb

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ANALYSIS, 'outputs', 'newset_v4')
OUT = os.path.join(_ANALYSIS, 'outputs', 'newset_report_v4.out')

MANIFEST = ['ttttctacgtggggga', 'aatgagcgggatccga', 'gtttatcgtctacaga', 'aaggaggcgctgcctg',
            'tattgttcagcccgga', 'accggtctatcttctc', 'ctacctcgcatatccc']   # 7 originals, manifest order
REPLACEMENT = 'tcaagaataaaggata'
SETS = {'primary (8)': MANIFEST + [REPLACEMENT], '7 original completers': MANIFEST, 'strict k = 5': MANIFEST[:5]}
HAM = 'ham_dos_n2_L4_nogeom_s2129768291'
DIR = 'blind_tonly_L4_nogeom_sumpool_s3731635825'
DIR_DOS = 'blind_dos_L2_nogeom_sumpool_s2129768291'
LDOST = 'ham_ldosonly_n2_L4_nogeom_s2129768291'


def sign_p(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    R = {}
    for f in glob.glob(os.path.join(D, '*.json')):
        r = json.load(open(f)); R[r['run']] = r
    if len(R) != 144:
        raise SystemExit(f'FATAL: {len(R)} scored runs, want 144')
    per = {k: {p['sequence'].lower(): p for p in r['perseq']} for k, r in R.items()}
    if set(per[HAM]) != set(SETS['primary (8)']):
        raise SystemExit(f'FATAL: duplex set {sorted(per[HAM])}')
    lines = []
    w = lines.append
    w('UNTOUCHED 16-bp SET (8 duplexes x 4 contact configurations), protocol + amendments 1-6.')
    w('Huber on log10, floor 1e-25, mean over the 4 contact configurations per duplex. Lower is better.')
    w('')
    w(f'Selected Hamiltonian: {HAM}   selected direct: {DIR}   DOS comparison: {DIR_DOS}')
    w('')
    w(f"{'duplex':18s} {'T ham':>7s} {'T dir':>7s} {'margin':>7s}   {'DOS ham':>7s} {'DOS dir':>7s}   {'T LDOS+T':>8s}")
    for s in SETS['primary (8)']:
        h, d, dd, l = per[HAM][s], per[DIR][s], per[DIR_DOS][s], per[LDOST][s]
        w(f"{s:18s} {h['huber_t']:7.3f} {d['huber_t']:7.3f} {d['huber_t'] - h['huber_t']:+7.3f}   "
          f"{h['huber_dos']:7.3f} {dd['huber_dos']:7.3f}   {l['huber_t']:8.3f}")
    w('margin = direct - Hamiltonian (positive = Hamiltonian lower)')
    w('')
    for name, seqs in SETS.items():
        k = sum(per[HAM][s]['huber_t'] < per[DIR][s]['huber_t'] for s in seqs)
        kd = sum(per[HAM][s]['huber_dos'] < per[DIR_DOS][s]['huber_dos'] for s in seqs)
        mt = lambda run, key: st.mean(per[run][s][key] for s in seqs)
        w(f'{name}: Hamiltonian lower on T in {k}/{len(seqs)} duplexes, one-sided sign test p = {sign_p(k, len(seqs)):.4f}; '
          f'mean T {mt(HAM, "huber_t"):.3f} vs {mt(DIR, "huber_t"):.3f}; '
          f'DOS lower in {kd}/{len(seqs)} (mean {mt(HAM, "huber_dos"):.3f} vs {mt(DIR_DOS, "huber_dos"):.3f})')
    w('')
    cells = {}
    for k, r in R.items():
        c = k.rsplit('_s', 1)[0]
        grp = 'ham' if r['family'] == 'ham' else ('sum' if r['pooling'] == 'sum' else 'mean')
        cells.setdefault(grp, {}).setdefault(c, []).append(per[k])
    for grp in ('ham', 'sum', 'mean'):
        if any(len(v) != 3 for v in cells[grp].values()):
            raise SystemExit(f'FATAL: {grp} cell without 3 runs')
    w('FAMILY: per duplex, fraction of (Hamiltonian cell, direct cell) pairs with the Hamiltonian lower on T (3-run means)')
    w(f"{'duplex':18s} {'vs sum (32x8)':>14s} {'vs mean (32x8)':>15s}  best ham cell / best sum / best mean")
    for s in SETS['primary (8)']:
        cm = {g: [st.mean(r[s]['huber_t'] for r in v) for v in cells[g].values()] for g in cells}
        fs = sum(h < d for h in cm['ham'] for d in cm['sum']) / (len(cm['ham']) * len(cm['sum']))
        fm = sum(h < d for h in cm['ham'] for d in cm['mean']) / (len(cm['ham']) * len(cm['mean']))
        w(f"{s:18s} {fs:14.2f} {fm:15.2f}  {min(cm['ham']):.3f} / {min(cm['sum']):.3f} / {min(cm['mean']):.3f}")
    w('')
    w('Per-run pooled T over the 8 duplexes, selected cells (s1179027592 / s2129768291 / s3731635825):')
    for c in ('ham_dos_n2_L4_nogeom', 'ham_ldosonly_n2_L4_nogeom', 'blind_tonly_L4_nogeom_sumpool',
              'blind_dos_L2_nogeom_sumpool', 'blind_dos_L2_nogeom'):
        v = [R[f'{c}_s{x}']['new_transmission'] for x in (1179027592, 2129768291, 3731635825)]
        dv = [R[f'{c}_s{x}']['new_dos'] for x in (1179027592, 2129768291, 3731635825)]
        w(f"  {c:32s} T {' / '.join('%.3f' % x for x in v)}  (mean {st.mean(v):.3f})   DOS {' / '.join('%.3f' % x for x in dv)}")
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
