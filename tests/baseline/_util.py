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
# Floating point results can differ in the last bits across BLAS/hardware, but a
# real behaviour change moves them far more than this.
ATOL = 1e-6
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
