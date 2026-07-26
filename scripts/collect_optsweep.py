#!/usr/bin/env python3
"""Optimizer/weight-decay sweep: did regularization actually close the overfitting gap?

The headline is NOT val loss -- it is the TRAIN-VAL GAP and whether best-val stops being
reached early. An arm that lowers val loss without closing the gap has not regularized
anything, it just landed better.

adam@1e-5 is the control (every historical run). adamw@1e-5 isolates the DECOUPLING alone
(Loshchilov & Hutter ICLR 2019) with decay held fixed; the 1e-3/1e-2/1e-1 arms are the
magnitude sweep. Comparing only the magnitudes would confound the two.
"""
import glob, os, re
import numpy as np, torch

ARMS = [('adam', '1e-5'), ('adamw', '1e-5'), ('adamw', '1e-3'),
        ('adamw', '1e-2'), ('adamw', '1e-1')]
print(f"{'arm':>16} {'best-val':>16} {'final':>16} {'train tail':>11} {'gap':>8} {'best epoch':>18}")
print('-' * 92)
rows = {}
for opt, wd in ARMS:
    b, f, tr, e = [], [], [], []
    for p in sorted(glob.glob(f'outputs_optsweep_{opt}_wd{wd}_s*/hamiltonian_pickle_model.pth')):
        ck = torch.load(p, map_location='cpu', weights_only=False)
        vl = np.asarray(ck['val_losses'], float); tl = np.asarray(ck['train_losses'], float)
        b.append(vl.min()); f.append(vl[-1]); tr.append(tl[-50:].mean()); e.append(int(vl.argmin()))
    if not b:
        continue
    b, f, tr = map(np.array, (b, f, tr))
    rows[(opt, wd)] = (b.mean(), b.std(ddof=1), tr.mean(), f.mean())
    gap = f.mean() - tr.mean()
    print(f"{opt+'@'+wd:>16} {b.mean():8.4f}+/-{b.std(ddof=1):5.4f} "
          f"{f.mean():8.4f}+/-{f.std(ddof=1):5.4f} {tr.mean():11.4f} {gap:8.4f}   {sorted(e)}")

ctrl = rows.get(('adam', '1e-5'))
if ctrl:
    print(f"\ncontrol adam@1e-5: best-val {ctrl[0]:.4f}, train-val gap {ctrl[3]-ctrl[2]:.4f}")
    print("Read: gap SHRINKING = regularization working. best-val epoch moving LATER = the")
    print("model is no longer racing past its optimum. Val loss alone can improve without either.")
