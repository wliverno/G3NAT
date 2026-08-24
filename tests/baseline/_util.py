"""Compare-or-capture helper for the baseline regression fixtures.

These files used to `pickle.dump` unconditionally on every run, which meant they
could never fail (zero assertions) and dirtied the git tree on every `pytest`.
They now COMPARE against the stored fixture and only capture when it is missing,
or when regeneration is asked for explicitly:

    G3NAT_REGEN_BASELINES=1 python -m pytest tests/baseline/

Regenerating is a deliberate act: it should be a separate commit whose message
says what behaviour changed and why the new numbers are correct.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import torch

BASELINE_DIR = Path(__file__).parent / "outputs"
BASELINE_DIR.mkdir(exist_ok=True)

REGEN = os.environ.get("G3NAT_REGEN_BASELINES") == "1"

# Tolerances: these fixtures are regression guards, not bit-exactness guards.
#
# ATOL WAS 1e-6 AND THAT WAS TOO TIGHT. MEASURED, 2026-08-24, not assumed:
# the same checkpoint, code, torch 2.7.1, numpy 2.3.2 and thread count, run on
# two node types, on ATATATAT through hamiltonian_DFT_gat_baseaware.pth --
#
#   compute-bigmem (n3066): max |delta| vs fixture  0.0        (bit-exact)
#   cpu-g2         (n3477): max |delta| vs fixture  3.67e-05   (transmission)
#                                                  1.20e-05   (DOS)
#
# So cross-hardware drift on the near-degenerate solves in this fixture set is
# ~4e-5, NOT "the last bits". A different CPU generation takes a different BLAS
# kernel path, which changes summation order in the eigensolve; ATATATAT is the
# most near-degenerate sequence here and therefore the most sensitive. The
# fixtures were captured on compute-bigmem-class hardware, so they failed for a
# day purely because the test suite was being run on cpu-g2.
#
# THE OLD COMMENT CLAIMED cross-BLAS drift sits "far below" these tolerances.
# It does not -- 3.67e-5 is about twice the effective tolerance for log10 values
# near -1.7 (atol + rtol*|v|). That sentence is what sends the next person
# hunting for a code bug that is not there. Replaced with the measurement.
#
# HEADROOM FOR THE THING THESE FIXTURES ACTUALLY GUARD: mutation testing
# (documented in test_baseline_legacy_checkpoints.py) shows a real default
# change moves these outputs by DECADES -- the solver_type defect that shipped
# once moved near-resonance log10 T by up to 3.8. So 1e-3 sits two orders above
# measured hardware drift and three below the smallest real regression. A
# fixture that cries wolf gets regenerated to silence it, which destroys the
# guard; that failure mode is worse than a slightly loose bound.
ATOL = 1e-3
RTOL = 1e-5


def _diff(new, old, path):
    """Recursively compare, returning a list of human-readable mismatches."""
    if type(new) is not type(old):
        return [f"{path}: type {type(new).__name__} != stored {type(old).__name__}"]

    if isinstance(new, dict):
        out = []
        missing = set(old) - set(new)
        added = set(new) - set(old)
        if missing:
            out.append(f"{path}: keys missing vs stored: {sorted(missing)}")
        if added:
            out.append(f"{path}: keys added vs stored: {sorted(added)}")
        for k in sorted(set(new) & set(old)):
            out += _diff(new[k], old[k], f"{path}.{k}")
        return out

    if isinstance(new, torch.Tensor):
        if new.shape != old.shape:
            return [f"{path}: shape {tuple(new.shape)} != stored {tuple(old.shape)}"]
        if not torch.allclose(new.float(), old.float(), atol=ATOL, rtol=RTOL):
            d = (new.float() - old.float()).abs().max().item()
            return [f"{path}: values differ, max abs delta {d:.3e}"]
        return []

    if isinstance(new, np.ndarray):
        if new.shape != old.shape:
            return [f"{path}: shape {new.shape} != stored {old.shape}"]
        if not np.allclose(new, old, atol=ATOL, rtol=RTOL):
            d = float(np.abs(new - old).max())
            return [f"{path}: values differ, max abs delta {d:.3e}"]
        return []

    if new != old:
        return [f"{path}: {new!r} != stored {old!r}"]
    return []


def check_or_capture(filename, baseline):
    """Assert `baseline` matches the stored fixture; capture it if there is none."""
    path = BASELINE_DIR / filename

    if REGEN or not path.exists():
        with open(path, "wb") as f:
            pickle.dump(baseline, f)
        reason = "regenerating (G3NAT_REGEN_BASELINES=1)" if REGEN else "no stored baseline"
        print(f"CAPTURED {filename}: {reason}")
        return

    with open(path, "rb") as f:
        stored = pickle.load(f)

    mismatches = _diff(baseline, stored, filename)
    assert not mismatches, (
        f"Baseline regression in {filename}:\n  "
        + "\n  ".join(mismatches)
        + "\n\nIf this change is intended, regenerate deliberately:\n"
        "  G3NAT_REGEN_BASELINES=1 python -m pytest tests/baseline/\n"
        "and commit the new fixtures separately, explaining what behaviour changed."
    )
