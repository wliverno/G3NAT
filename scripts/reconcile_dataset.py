"""Reconcile the dataset inventory before specifying a published format.

Numbers that currently do not add up:
  2058 pickles, 2081 DOS .mat files, 522 sequence dirs with DOS, 523 _Fock.mat.
A published dataset needs a defensible answer to "how many samples are there and
why", so find out exactly which sequences have what.
"""
import os, glob, re, pickle
from collections import defaultdict

ROOT = '/mmfs1/gscratch/anantram/asyed4/DNADataSet'
PKL = '/mmfs1/gscratch/anantram/willll/G3NAT/pickle_files'

# --- inventory from filenames (cheap; no unpickling) -------------------------
pkl_files = sorted(glob.glob(f'{PKL}/*.pkl'))
pkl_by_seq = defaultdict(set)
for f in pkl_files:
    m = re.match(r'(.+)_run(\d+)\.pkl$', os.path.basename(f))
    if m:
        pkl_by_seq[m.group(1)].add(int(m.group(2)))

seq_dirs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))

print('=== HEADLINE COUNTS ===')
print(f'pickle files                : {len(pkl_files)}')
print(f'unique seqs in pickles      : {len(pkl_by_seq)}')
print(f'dirs under DNADataSet       : {len(seq_dirs)}')

# --- what does each sequence dir actually contain? ---------------------------
have = defaultdict(dict)
for s in seq_dirs:
    d = os.path.join(ROOT, s)
    dos = glob.glob(f'{d}/run*/DOS_*.mat')
    tran = glob.glob(f'{d}/run*/Tran_*.mat')
    have[s]['dos'] = len(dos)
    have[s]['tran'] = len(tran)
    have[s]['pdb'] = os.path.exists(f'{d}/{s}.pdb')
    have[s]['gammas'] = sorted({('0.6' if '0.6' in os.path.basename(p) else '0.1') for p in dos})
    have[s]['runs'] = sorted({int(re.search(r'/run(\d+)/', p).group(1)) for p in dos
                              if re.search(r'/run(\d+)/', p)})

n_dos = sum(1 for s in have if have[s]['dos'])
n_pdb = sum(1 for s in have if have[s]['pdb'])
print(f'dirs with >=1 DOS mat       : {n_dos}')
print(f'dirs with a PDB             : {n_pdb}')
print(f'total DOS mats              : {sum(have[s]["dos"] for s in have)}')
print(f'total Tran mats             : {sum(have[s]["tran"] for s in have)}')

# --- the set differences that matter ----------------------------------------
pkl_seqs = set(pkl_by_seq)
dos_seqs = {s for s in have if have[s]['dos']}
pdb_seqs = {s for s in have if have[s]['pdb']}

print('\n=== SET DIFFERENCES ===')
print(f'in pickles but NO DOS mat   : {len(pkl_seqs - dos_seqs)}  {sorted(pkl_seqs - dos_seqs)[:8]}')
print(f'has DOS mat but NO pickle   : {len(dos_seqs - pkl_seqs)}  {sorted(dos_seqs - pkl_seqs)[:8]}')
print(f'in pickles but NO PDB       : {len(pkl_seqs - pdb_seqs)}  {sorted(pkl_seqs - pdb_seqs)[:8]}')
print(f'usable for LDOS training    : {len(pkl_seqs & dos_seqs & pdb_seqs)} sequences')

# --- per-sequence completeness profile --------------------------------------
print('\n=== COMPLETENESS PROFILE (how many runs per sequence) ===')
prof = defaultdict(int)
for s in dos_seqs:
    prof[(len(have[s]['runs']), tuple(have[s]['gammas']))] += 1
for k in sorted(prof, key=lambda x: -prof[x]):
    print(f'  {prof[k]:4d} sequences: {k[0]} run dirs, gammas {list(k[1])}')

prof2 = defaultdict(int)
for s, runs in pkl_by_seq.items():
    prof2[len(runs)] += 1
print('\n  pickle variants per sequence:')
for k in sorted(prof2):
    print(f'    {prof2[k]:4d} sequences have {k} pickle(s)')

# --- length distribution (for the dataset card) ------------------------------
print('\n=== SEQUENCE LENGTHS ===')
lens = defaultdict(int)
for s in pkl_seqs:
    lens[len(s)] += 1
for k in sorted(lens):
    print(f'  length {k}: {lens[k]} sequences')

# --- known-bad records -------------------------------------------------------
print('\n=== KNOWN-BAD ===')
bad = []
for f in pkl_files:
    if os.path.getsize(f) < 1000:
        bad.append((os.path.basename(f), os.path.getsize(f)))
print(f'suspiciously small pickles: {bad if bad else "none by size"}')
try:
    pickle.load(open(f'{PKL}/gaaac_run2.pkl', 'rb'))
    print('gaaac_run2.pkl: loads fine')
except Exception as e:
    print(f'gaaac_run2.pkl: {type(e).__name__}: {e}  (size {os.path.getsize(f"{PKL}/gaaac_run2.pkl")} bytes)')

# --- does gjf_text geometry match the PDB? ----------------------------------
print('\n=== gjf_text vs PDB geometry (spot check, aaac) ===')
d = pickle.load(open(f'{PKL}/aaac_run1.pkl', 'rb'))
gjf = d.get('gjf_text', '')
coords = []
for line in gjf.splitlines():
    p = line.split()
    if len(p) == 4 and re.fullmatch(r'[A-Z][a-z]?', p[0]):
        try:
            coords.append((p[0], float(p[1]), float(p[2]), float(p[3])))
        except ValueError:
            pass
pdbc = []
for line in open(f'{ROOT}/aaac/aaac.pdb'):
    if line.startswith(('ATOM', 'HETATM')):
        pdbc.append((line[76:78].strip() or line[12:16].strip()[0],
                     float(line[30:38]), float(line[38:46]), float(line[46:54])))
print(f'gjf atoms {len(coords)}   pdb atoms {len(pdbc)}')
if coords and len(coords) == len(pdbc):
    dmax = max(abs(a[i] - b[i]) for a, b in zip(coords, pdbc) for i in (1, 2, 3))
    print(f'max coordinate difference: {dmax:.4f} A  -> {"SAME geometry" if dmax < 1e-2 else "DIFFERENT"}')
else:
    print('atom counts differ -> gjf and pdb are not a row-for-row match')

# --- verify the energy reference IS the HOMO --------------------------------
print('\n=== ENERGY REFERENCE: is mean(Egrid) the HOMO? ===')
from scipy.io import loadmat
import numpy as np
for s in ['aaac', 'aaaa', 'gggg', 'acgt']:
    pf = f'{PKL}/{s}_run1.pkl'
    if not os.path.exists(pf):
        print(f'  {s}: no pickle'); continue
    E = np.asarray(pickle.load(open(pf, 'rb'))['Egrid']).ravel()
    ef, oc = f'{ROOT}/{s}/{s}_eigen.mat', f'{ROOT}/{s}/{s}_occ.mat'
    if not (os.path.exists(ef) and os.path.exists(oc)):
        print(f'  {s}: mean(Egrid)={E.mean():.4f}  (no eigen/occ to check against)'); continue
    try:
        ev = np.asarray(loadmat(ef)[[k for k in loadmat(ef) if not k.startswith("__")][0]]).ravel()
        oo = np.asarray(loadmat(oc)[[k for k in loadmat(oc) if not k.startswith("__")][0]]).ravel()
        occupied = ev[oo > 0.5] if oo.size == ev.size else ev[:int(oo.sum()//2)]
        homo = occupied.max() if occupied.size else float('nan')
        print(f'  {s}: mean(Egrid)={E.mean():9.4f}   HOMO={homo:9.4f}   '
              f'diff={abs(E.mean()-homo):.4f}  range=[{E.min():.3f},{E.max():.3f}]')
    except Exception as e:
        print(f'  {s}: mean(Egrid)={E.mean():.4f}  (check failed: {type(e).__name__})')
print('  -> diff ~0 confirms "Egrid is centered on the HOMO"; the reference is PER-SEQUENCE.')
