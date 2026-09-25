"""Build the geometry cache for the 8 held-out strands (L=12 and L=16)
-> G3NAT/geom_cache/geometry_heldout_L12_L16.pkl (run from the G3NAT repo root).

geometry_v2.pkl covers only the training sequences (lengths 4-8) and stays
byte-identical; this separate cache is merged with it at evaluation time. Without it
graph construction zeroes the geometry edge features and their mask for the held-out
strands, so every geometry-trained cell would be scored with the channel off.

build_geometry_cache warns and skips on any per-sequence error (a missing DSSR binary
yields an EMPTY cache and exit code 0), so the expected count and the per-sequence
shapes are asserted here and the script exits non-zero if anything is missing.
"""
import os
import pickle
import sys

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get('G3NAT_ROOT', os.path.dirname(_ANALYSIS))
sys.path.insert(0, REPO)
sys.path.insert(0, _ANALYSIS)

os.environ.setdefault('X3DNA_DSSR', 'x3dna-dssr')

from g3nat.graph.geometry import build_geometry_cache

DIRS = {12: 'DNADataset/validation_L12', 16: 'DNADataset/validation_L16'}
OUT = 'geom_cache/geometry_heldout_L12_L16.pkl'

if not os.path.exists(os.environ['X3DNA_DSSR']):
    sys.exit(f"DSSR binary not found at {os.environ['X3DNA_DSSR']}")
print(f"DSSR: {os.environ['X3DNA_DSSR']}")

merged = {}
for L, d in DIRS.items():
    seqs = sorted(s for s in os.listdir(d)
                  if os.path.isdir(os.path.join(d, s))
                  and os.path.exists(os.path.join(d, s, f'{s}.pdb')))
    print(f"\nL={L}: {len(seqs)} sequences with a PDB -> {seqs}")
    # per-length temp file, so a failure in one length leaves no half-written cache
    tmp = f'geom_cache/.heldout_L{L}.tmp.pkl'
    cache = build_geometry_cache(d, tmp, sequences=seqs)
    for s in seqs:
        if s not in cache:
            print(f"  MISSING after build: {s}")
    merged.update(cache)
    os.remove(tmp)

print(f"\n{'sequence':20s} {'len':>4s} {'bp_pars':>12s} {'step_pars':>12s} "
      f"{'primary':>10s} {'comp':>10s}")
for s in sorted(merged):
    e = merged[s]
    print(f"{s:20s} {len(s):4d} {str(e['bp_pars'].shape):>12s} "
          f"{str(e['step_pars'].shape):>12s} "
          f"{str(e['primary_centroids'].shape):>10s} "
          f"{str(e['comp_centroids'].shape):>10s}")

expected = sum(len([s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))
                    and os.path.exists(os.path.join(d, s, f'{s}.pdb'))])
               for d in DIRS.values())
if len(merged) != expected:
    sys.exit(f"FAILED: built {len(merged)} of {expected} sequences")

# L base pairs and L-1 steps per strand; a partial DSSR structure would be quietly wrong
bad = [s for s in merged
       if merged[s]['bp_pars'].shape[0] != len(s)
       or merged[s]['step_pars'].shape[0] != len(s) - 1]
if bad:
    sys.exit(f"FAILED: shape mismatch (expected L pairs, L-1 steps) for {bad}")

with open(OUT, 'wb') as f:
    pickle.dump(merged, f)
print(f"\nwrote {OUT}  ({len(merged)} sequences, all shapes consistent)")
