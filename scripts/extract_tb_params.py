#!/usr/bin/env python3
"""Extract the learned per-base onsite (baseline) values.

!! READ BEFORE TRUSTING THE OUTPUT (2026-07-24) !!

Two caveats make the literature comparison below far weaker than it looks, and this script
will happily print a confident "ordering matches Roche" verdict in cases where that verdict
is guaranteed by construction:

1. THE G COLUMN IS SET BY THE ENERGY CONVENTION, NOT BY THE FIT. The energy grid is centred
   per sequence on that sequence's HOMO, and the HOMO is a G-derived level whenever any G is
   present -- which is 495 of 515 sequences. So G is pinned near 0 by construction, and the
   gauge-shift-to-G=0 step below does NOT neutralise this; it bakes it in. A "G highest,
   ordering matches Roche" result is close to guaranteed regardless of what the model learned.
   Measured support: AT-only sequences sit 0.813 eV below GC-only ones at 13.6 sigma with zero
   overlap (docs/dataset.md).

2. THE TABLE IS ONLY MEANINGFUL AT alpha=1.0. For any alpha < 1 the mixing is a vacuous
   reparametrization of the free model -- the baseline collapses to a near-constant and the
   context head does the fitting -- so `onsite_baseline` from those checkpoints is not "the
   learned parameters" in any useful sense (docs/model-results.md, "CORRECTION").

The project has retired the goal of recovering universal ABSOLUTE per-base parameters; these
values are HOMO-referenced and per-sequence. Compare ORDERING and SPACING only, never
magnitudes, and do not present G's position as a recovered physical quantity.

Original description follows.

Compare the learned per-base onsite (baseline) TB parameters to literature.
Comparisons are GAUGE-CORRECTED (each set shifted so G=0) -- absolute onsite is gauge-dependent.
Usage: conda run -n g3nat python scripts/extract_tb_params.py <checkpoint.pth>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from g3nat.evaluation.physicality import baseline_distinctness

BASES = ['A', 'T', 'G', 'C']
ROCHE = {'A': -0.49, 'T': -1.39, 'G': 0.00, 'C': -1.12}   # g3nat/utils/physics.py

def _gauge(d):  # shift so G = 0
    return {b: d[b] - d['G'] for b in BASES}

def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    base = sd['onsite_baseline'].reshape(4, -1).mean(1).numpy()   # A,T,G,C order (BASE_TO_IDX)
    learned = {b: float(base[i]) for i, b in enumerate(BASES)}
    print("base  learned  learned(G=0)  Roche(G=0)")
    lg, rg = _gauge(learned), _gauge(ROCHE)
    for b in BASES:
        print(f"{b:>4} {learned[b]:>+8.3f} {lg[b]:>+12.3f} {rg[b]:>+11.3f}")
    order_learned = sorted(BASES, key=lambda b: learned[b])
    order_roche = sorted(BASES, key=lambda b: ROCHE[b])
    print(f"\nordering  learned: {order_learned}   Roche: {order_roche}   match={order_learned==order_roche}")
    print(f"distinctness: {baseline_distinctness(base.reshape(4,1))}")

if __name__ == '__main__':
    main()
