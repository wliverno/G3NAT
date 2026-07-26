# g3nat/evaluation/physicality.py
"""Window-membership metrics for a learned Hamiltonian.

READ THIS BEFORE USING THE NUMBERS. The module name says "physicality"; that is a
historical misnomer and the project has retracted the interpretation behind it.

What these functions actually measure: what fraction of some set of numbers (onsite
entries, eigenvalues) falls inside a fixed interval, by default [-1, 1].

What that interval IS: the SUPERVISION RANGE. The energy grid is 201 points centred
per sequence on that sequence's HOMO, so the window is HOMO +/- 1 eV, and there is no
DFT data outside it at all. DOS and transmission are supervised only there.

What it is therefore NOT: a physicality criterion. Eigenvalues outside the window are
unconstrained by the loss, not wrong -- real DNA has states outside any window we pick.
A model can be entirely sensible and score low here. Read `frac_in_window` as a coarse
sanity check against runaway values (onsite at -33 eV is pathological and this catches
it), never as a success criterion, and never as evidence that one model is "more
physical" than another.

See `docs/model-results.md` ("The [-1,1] window is the supervision range, not a
physicality criterion") and `docs/dataset.md` (HOMO centring).
"""
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


def both_window_fracs_increased(before, after, eps=1e-6):
    """Did BOTH the onsite and eigenvalue in-window fractions rise between two states?

    The co-gate is still worth keeping: onsite alone can be pushed into the window while
    the badness relocates into the couplings, so requiring both to move together catches
    "shifted, not fixed".

    But this is NOT a verdict that a model became more physical -- it only says more
    numbers landed inside the supervision range. See the module docstring.
    """
    return (after['frac_in_window'] > before['frac_in_window'] + eps and
            after['frac_eig_in_window'] > before['frac_eig_in_window'] + eps)


# Deprecated alias. The old name asserts a conclusion the project has retracted; kept so
# existing callers keep working. Prefer both_window_fracs_increased.
is_physical_win = both_window_fracs_increased
