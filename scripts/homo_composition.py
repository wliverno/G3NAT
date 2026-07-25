"""Does base composition drive the HOMO, and how wide is the common absolute window?

Two questions, one measurement (mean(Egrid) per sequence, which willll states IS the HOMO):

Q1 COMPOSITION. If the HOMO is a G-derived level in G-containing duplexes, then GC-only
   sequences should sit systematically higher (less negative) than AT-only ones, and E_HOMO
   should track GC fraction. If so, per-sequence centering removes exactly the
   base-composition signal -- and a single per-base onsite table cannot put G at 0 for one
   class and A at 0 for the other.

Q2 COMMON WINDOW. Each sequence covers E_HOMO +/- 1 eV in ABSOLUTE terms. The intersection
   of all those windows is the widest absolute axis on which every sequence has data. If
   that intersection is wide, re-referencing to a shared absolute axis is possible and
   "universal parameters are unlearnable" is a statement about this convention, not a hard
   limit. If it is narrow or empty, the limit is real.
"""
import os, glob, re, pickle
import numpy as np
from collections import defaultdict

PKL = '/mmfs1/gscratch/anantram/willll/G3NAT/pickle_files'

rows = []
for f in sorted(glob.glob(f'{PKL}/*_run1.pkl')):
    seq = re.sub(r'_run\d+\.pkl$', '', os.path.basename(f))
    try:
        E = np.asarray(pickle.load(open(f, 'rb'))['Egrid']).ravel()
    except Exception:
        continue
    if E.size == 0:
        continue
    gc = sum(c in 'gcGC' for c in seq) / len(seq)
    rows.append((seq, len(seq), gc, float(E.mean()), float(E.min()), float(E.max())))

print(f'sequences read: {len(rows)}')
E_homo = np.array([r[3] for r in rows])
gcf = np.array([r[2] for r in rows])

# ---------------------------------------------------------------- Q1
print('\n=== Q1: DOES COMPOSITION DRIVE THE HOMO? ===')
at_only = [r for r in rows if r[2] == 0.0]
gc_only = [r for r in rows if r[2] == 1.0]
mixed = [r for r in rows if 0.0 < r[2] < 1.0]
for name, grp in [('AT-only', at_only), ('mixed', mixed), ('GC-only', gc_only)]:
    if not grp:
        print(f'  {name:9}: none'); continue
    v = np.array([r[3] for r in grp])
    print(f'  {name:9}: n={len(grp):4d}   E_HOMO mean {v.mean():8.4f}  std {v.std():6.4f}'
          f'   range [{v.min():8.4f}, {v.max():8.4f}]')

if at_only and gc_only:
    a = np.array([r[3] for r in at_only]); g = np.array([r[3] for r in gc_only])
    sep = g.mean() - a.mean()
    pooled = np.sqrt((a.std()**2 + g.std()**2) / 2)
    print(f'\n  GC-only minus AT-only = {sep:+.4f} eV   pooled std {pooled:.4f}'
          f'   -> {abs(sep)/max(pooled,1e-9):.2f} sigma')
    print(f'  overlap: AT max {a.max():.4f}  vs  GC min {g.min():.4f}'
          f'   -> {"SEPARATED" if a.max() < g.min() else "overlapping"}')

if len(rows) > 2:
    r = float(np.corrcoef(gcf, E_homo)[0, 1])
    print(f'\n  correlation(GC fraction, E_HOMO) = {r:+.4f}')
    print('  positive => more GC sits higher (less negative), i.e. G sets the HOMO.')
    # per-GC-fraction means, the cleanest view
    print('\n  E_HOMO by GC fraction bucket:')
    buckets = defaultdict(list)
    for _, _, gc, e, _, _ in rows:
        buckets[round(gc, 2)].append(e)
    for k in sorted(buckets):
        v = np.array(buckets[k])
        print(f'    GC={k:4.2f}  n={len(v):4d}  E_HOMO {v.mean():8.4f} +/- {v.std():.4f}')

# ---------------------------------------------------------------- Q2
print('\n=== Q2: HOW WIDE IS THE COMMON ABSOLUTE WINDOW? ===')
lo = np.array([r[4] for r in rows]); hi = np.array([r[5] for r in rows])
print(f'  E_HOMO spread over all sequences: [{E_homo.min():.4f}, {E_homo.max():.4f}]'
      f'  = {E_homo.max()-E_homo.min():.4f} eV')
print(f'  each window is {np.median(hi-lo):.4f} eV wide')
common_lo, common_hi = lo.max(), hi.min()
print(f'  intersection of ALL windows: [{common_lo:.4f}, {common_hi:.4f}]'
      f'  = {max(common_hi-common_lo, 0):.4f} eV')
if common_hi > common_lo:
    frac = (common_hi - common_lo) / np.median(hi - lo)
    print(f'  -> a shared absolute axis IS possible, covering {frac*100:.1f}% of each')
    print('     sequence\'s window. Universal parameters would be learnable on that axis.')
else:
    print('  -> EMPTY. No absolute energy is covered by every sequence; a shared absolute')
    print('     axis is impossible and the HOMO-relative frame is forced by the data.')

# how much do we keep if we drop the worst outliers?
for keep in (0.99, 0.95, 0.90):
    k_lo = np.quantile(lo, keep); k_hi = np.quantile(hi, 1 - keep)
    print(f'  keeping {keep*100:.0f}% of sequences: [{k_lo:.4f}, {k_hi:.4f}] '
          f'= {max(k_hi-k_lo,0):.4f} eV')
