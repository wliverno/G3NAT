# g3nat/evaluation/physicality.py
"""Physicality diagnostics for a learned Hamiltonian. A 'win' requires onsite AND
eigenvalues to move into the window together -- else the model just relocated the
unphysical states into the couplings ('shifted, not fixed')."""
import numpy as np


def onsite_metrics(H_diag, window=(-1.0, 1.0)):
    d = np.asarray(H_diag).ravel()
    lo, hi = window
    return {'frac_in_window': float(np.mean((d >= lo) & (d <= hi))),
            'min': float(d.min()), 'max': float(d.max()), 'range': float(d.max() - d.min())}


def eig_metrics(H, window=(-1.0, 1.0)):
    w = np.linalg.eigvalsh(np.asarray(H))
    lo, hi = window
    return {'frac_eig_in_window': float(np.mean((w >= lo) & (w <= hi)))}


def coupling_bandwidth(H):
    H = np.asarray(H)
    off = H - np.diag(np.diag(H))
    return float(np.abs(off).max())


def baseline_distinctness(baseline):
    # Only the 4 baseline values are available here, so distinctness = pairwise spread.
    # (True eta^2 needs per-SITE onsite over the val set; that lives in
    # scripts/probe_onsite_dilution.py::variance_decomposition, which has the per-site data.)
    b = np.asarray(baseline).reshape(len(baseline), -1).mean(axis=1)  # 1 scalar per base
    pw = [abs(b[i] - b[j]) for i in range(len(b)) for j in range(i + 1, len(b))]
    return {'min_pairwise': float(min(pw)) if pw else 0.0, 'spread': float(b.std())}


def is_physical_win(before, after, eps=1e-6):
    return (after['frac_in_window'] > before['frac_in_window'] + eps and
            after['frac_eig_in_window'] > before['frac_eig_in_window'] + eps)
