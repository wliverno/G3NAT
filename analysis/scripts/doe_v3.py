"""Campaign-v3 DOE analysis -> outputs/doe_v3.out (the factorial ANOVA behind Results 3.3-3.5).

Two independent saturated factorials: the Hamiltonian family (supervision x n_orb x
num_layers x geometry, 32 cells x 3 seeds = 96 runs) and the blind/direct family
(supervision x num_layers x geometry, 8 cells x 3 seeds = 24 runs), each enumerated
from its own factor list. Deviation (sum-to-zero) coding, drop-one Type III F-tests,
block-orthogonality asserted (balanced design), Benjamini-Hochberg over one family per
model family (all terms x all tested responses).

Inputs, read fresh every run and never written: outputs/posthoc_v3_report.json (the 120
per-run rows), outputs/contact_invariance_v3.json (ci_interior_mode / ci_interior_total,
merged onto the ham rows as the per-run MEDIAN over strands) and
outputs/best_epoch_cache.json (the selected epoch of every run, from
best_epoch_extract.py). The full 120-row structure is asserted before any fit.

Beside the location analysis: a DISPERSION channel (y = max - min over the 3 seeds, one
observation per cell, own BH family) and a TRAINING-TIME channel (the selected epoch and
its log10, own BH family; descriptive, direction carries no judgement).

Every length response is a full-curve Huber loss at a fixed length; no rate, slope or
threshold appears anywhere.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

SEEDS: Tuple[int, ...] = (1179027592, 2129768291, 3731635825)

HAM_FACTORS: Tuple[str, ...] = ('supervision', 'n_orb', 'num_layers', 'geometry')
HAM_LEVELS: Dict[str, Tuple] = {
    'supervision': ('dos', 'ldos', 'ldosonly', 'tonly'),
    'n_orb': (1, 2),
    'num_layers': (2, 4),
    'geometry': (0, 1),
}

BLIND_FACTORS: Tuple[str, ...] = ('supervision', 'num_layers', 'geometry')
BLIND_LEVELS: Dict[str, Tuple] = {
    'supervision': ('dos', 'tonly'),
    'num_layers': (2, 4),
    'geometry': (0, 1),
}

N_HAM_CELLS = 4 * 2 * 2 * 2       # 32
N_BLIND_CELLS = 2 * 2 * 2         # 8
N_HAM_RUNS = N_HAM_CELLS * len(SEEDS)     # 96
N_BLIND_RUNS = N_BLIND_CELLS * len(SEEDS)  # 24
N_TOTAL_RUNS = N_HAM_RUNS + N_BLIND_RUNS  # 120

#: Every response is a loss / distance: LOWER IS BETTER uniformly, so "best cell" is
#: the minimum seed-averaged value everywhere. sub_t is absent by ruling (2026-08-26):
#: it correlates at r=+0.95 with in-distribution fit. Reports carrying it are read
#: tolerantly; the value is not analyzed.
RESPONSES_COMMON: Tuple[str, ...] = (
    'val_transmission_at_selection', 'l12_transmission', 'l12_dos',
    'l16_transmission', 'l16_dos',
)
HAM_EXTRA_RESPONSE = 'onsite_near'

#: Contact-invariance responses (ham-only). Each is the per-run MEDIAN OVER STRANDS of
#: the raw vector in the CI report. Only the MODE (attachment-axis) response is
#: F-tested; ci_interior_total is the mode axis plus two small components in
#: quadrature and is descriptive only (ruling 2026-09-01), so it does not pad the BH
#: family.
CI_INTERIOR_MODE = 'ci_interior_mode'
CI_INTERIOR_TOTAL = 'ci_interior_total'
CI_TESTED_RESPONSES: Tuple[str, ...] = (CI_INTERIOR_MODE,)
CI_DESCRIPTIVE_ONLY: Tuple[str, ...] = (CI_INTERIOR_TOTAL,)

CI_FIELD_FOR_RESPONSE: Dict[str, str] = {
    CI_INTERIOR_MODE: 'D_interior_mode',
    CI_INTERIOR_TOTAL: 'D_interior_total',
}

#: Companion columns printed beside the ci best/worst cells ("never a lone drift
#: number"), pulled VERBATIM from each ham run's own CI entry, not re-derived.
CI_COMPANION_FIELDS: Tuple[str, ...] = (
    'onsite_abs_mean', 'onsite_std', 'val_transmission_at_selection',
    'l16_transmission')

#: onsite_far is reported descriptively only -- no F-test.
ONSITE_DESCRIPTIVE_ONLY = ('onsite_far',)

_OUTPUTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
DEFAULT_REPORT_PATH = os.path.join(_OUTPUTS, 'posthoc_v3_report.json')
DEFAULT_CI_PATH = os.path.join(_OUTPUTS, 'contact_invariance_v3.json')
DEFAULT_OUT_PATH = os.path.join(_OUTPUTS, 'doe_v3.out')
DEFAULT_BEST_EPOCH_CACHE = os.path.join(_OUTPUTS, 'best_epoch_cache.json')

BEST_EPOCH_RESPONSES: Tuple[str, ...] = ('best_epoch', 'log10_best_epoch')

BEST_EPOCH_DESCRIPTIVE_NOTE = (
    "DESCRIPTIVE RESPONSE -- DIRECTION CARRIES NO QUALITY JUDGEMENT. Peaking "
    "earlier is not better and peaking later is not better "
    "(analysis/outputs/best_epoch_doe.md). print_family labels the minimum-mean "
    "cell BEST and the maximum-mean cell WORST because every response it was "
    "written for is a loss; here read those two words as EARLIEST and LATEST.")

SUB_T_REMOVAL_NOTE = (
    "RULING (2026-08-26): sub_t (substitution response) REMOVED from "
    "the tested/printed response set -- r=+0.95 with in-distribution "
    "transmission fit across the 32 v3 Hamiltonian cells, cannot be separated "
    "from fitting. Old reports carrying a stray 'sub_t' key are read "
    "tolerantly (not an abort); the value is simply not analyzed. "
    "(ruling 1, 2026-08-26: r=+0.95 with in-distribution fit, not separable from fitting)")

#: Analysis-time RNG seed for the matched-selection-pressure resampling (not a
#: training seed), recorded so a rerun reproduces the same draws.
RESAMPLE_SEED = 20260825
N_RESAMPLES = 10_000


# ---------------------------------------------------------------------------
# Structure validation -- runs BEFORE any fit, aborts loudly on violation.
# ---------------------------------------------------------------------------

class StructureError(SystemExit):
    """The report does not match the campaign's exact 96/24 factor-level structure.
    A silently-missing or duplicated cell would otherwise produce a design with real
    but wrong degrees of freedom and a p-value that looks legitimate."""


def _expected_keys(factors, levels, family):
    cells = list(itertools.product(*(levels[f] for f in factors)))
    return {(family,) + cell + (s,) for cell in cells for s in SEEDS}


def validate_structure(runs: Dict[str, dict]) -> None:
    """Exactly 120 rows, 96 ham / 24 blind, every (cell, seed) exactly once, every
    response present and finite, the onsite triple on every ham row, and
    sub_n_comparisons identical across all 120 rows."""
    if len(runs) != N_TOTAL_RUNS:
        raise StructureError(
            f"FATAL: expected {N_TOTAL_RUNS} runs in the report, found "
            f"{len(runs)}. Aborting before any fit -- a partial report is not a "
            f"malformed convenience, it changes the degrees of freedom.")

    ham_keys, blind_keys = set(), set()
    families_seen = {'ham': 0, 'blind': 0}
    sub_n = set()

    for name, row in runs.items():
        family = row.get('family')
        if family not in ('ham', 'blind'):
            raise StructureError(
                f"FATAL: run {name!r} has family={family!r}, expected 'ham' or "
                f"'blind'")
        families_seen[family] += 1

        for resp in RESPONSES_COMMON:
            if resp not in row:
                raise StructureError(
                    f"FATAL: run {name!r} (family={family}) is missing response "
                    f"{resp!r}")
            if not np.isfinite(row[resp]):
                raise StructureError(
                    f"FATAL: run {name!r} (family={family}) has non-finite "
                    f"response {resp!r}={row[resp]!r}. Aborting before any fit -- "
                    f"a NaN/inf silently propagates through the ANOVA rather than "
                    f"raising.")

        if 'sub_n_comparisons' not in row:
            raise StructureError(
                f"FATAL: run {name!r} is missing sub_n_comparisons")
        sub_n.add(row['sub_n_comparisons'])

        if family == 'ham':
            for k in ('onsite_near', 'onsite_far', 'onsite_n_graphs'):
                if k not in row:
                    raise StructureError(
                        f"FATAL: ham run {name!r} is missing {k!r} -- every ham "
                        f"row must carry the onsite triple")
                if not np.isfinite(row[k]):
                    raise StructureError(
                        f"FATAL: ham run {name!r} has non-finite {k!r}="
                        f"{row[k]!r}")
            try:
                key = ('ham',) + tuple(row[f] for f in HAM_FACTORS) + (row['seed'],)
            except KeyError as e:
                raise StructureError(
                    f"FATAL: ham run {name!r} is missing factor {e}")
            ham_keys.add(key)
        else:
            try:
                key = ('blind',) + tuple(row[f] for f in BLIND_FACTORS) + (row['seed'],)
            except KeyError as e:
                raise StructureError(
                    f"FATAL: blind run {name!r} is missing factor {e}")
            blind_keys.add(key)

    if families_seen['ham'] != N_HAM_RUNS or families_seen['blind'] != N_BLIND_RUNS:
        raise StructureError(
            f"FATAL: expected {N_HAM_RUNS} ham / {N_BLIND_RUNS} blind runs, found "
            f"{families_seen['ham']} ham / {families_seen['blind']} blind")

    exp_ham = _expected_keys(HAM_FACTORS, HAM_LEVELS, 'ham')
    exp_blind = _expected_keys(BLIND_FACTORS, BLIND_LEVELS, 'blind')

    if ham_keys != exp_ham:
        missing = exp_ham - ham_keys
        extra = ham_keys - exp_ham
        raise StructureError(
            f"FATAL: ham factor-level structure does not match the design. "
            f"missing={sorted(missing)[:5]}{'...' if len(missing) > 5 else ''} "
            f"extra={sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    if blind_keys != exp_blind:
        missing = exp_blind - blind_keys
        extra = blind_keys - exp_blind
        raise StructureError(
            f"FATAL: blind factor-level structure does not match the design. "
            f"missing={sorted(missing)[:5]}{'...' if len(missing) > 5 else ''} "
            f"extra={sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    if len(sub_n) != 1:
        raise StructureError(
            f"FATAL: sub_n_comparisons is not identical across all 120 rows: "
            f"observed values {sorted(sub_n)}. The substitution pair set must be "
            f"the same for every run.")


def merge_ci(runs: Dict[str, dict], ci_report: dict) -> Dict[str, dict]:
    """Attach ci_interior_mode / ci_interior_total (per-run median over strands of the
    CI vectors) and the `companion` object onto every ham row. Missing entry, empty or
    non-finite vector, mismatched vector lengths or a missing companion field are all
    fatal. Blind rows pass through unchanged. Returns a NEW dict."""
    ci_runs = ci_report.get('runs', {})
    out: Dict[str, dict] = {}
    for name, row in runs.items():
        if row.get('family') != 'ham':
            out[name] = row
            continue

        entry = ci_runs.get(name)
        if entry is None:
            raise StructureError(
                f"FATAL: ham run {name!r} has no matching entry in the CI "
                f"report -- every ham run must have a contact-invariance "
                f"measurement before the DOE fit can run.")

        vectors = entry.get('vectors', {})
        medians: Dict[str, float] = {}
        lengths = set()
        for resp, field in CI_FIELD_FOR_RESPONSE.items():
            vec = vectors.get(field)
            if not vec:
                raise StructureError(
                    f"FATAL: ham run {name!r} CI entry has an empty or "
                    f"missing vectors[{field!r}]")
            arr = np.asarray(vec, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise StructureError(
                    f"FATAL: ham run {name!r} CI vectors[{field!r}] contains "
                    f"non-finite values. Aborting before any fit -- a NaN/inf "
                    f"silently propagates through the ANOVA rather than "
                    f"raising.")
            lengths.add(len(arr))
            medians[resp] = float(np.median(arr))

        if len(lengths) != 1:
            raise StructureError(
                f"FATAL: ham run {name!r} CI vectors have mismatched lengths "
                f"across fields (expected equal-length per-strand vectors): "
                f"{sorted(lengths)}")

        companion_raw = entry.get('companion')
        if companion_raw is None:
            raise StructureError(
                f"FATAL: ham run {name!r} CI entry has no 'companion' object "
                f"-- spec section 3.3 requires {list(CI_COMPANION_FIELDS)} "
                f"beside every reported drift number.")
        companion: Dict[str, float] = {}
        for field in CI_COMPANION_FIELDS:
            if field not in companion_raw:
                raise StructureError(
                    f"FATAL: ham run {name!r} CI entry's companion object is "
                    f"missing {field!r} (spec section 3.3 requires all of "
                    f"{list(CI_COMPANION_FIELDS)})")
            val = companion_raw[field]
            if not np.isfinite(val):
                raise StructureError(
                    f"FATAL: ham run {name!r} CI companion[{field!r}]={val!r} "
                    f"is non-finite. Aborting before any fit -- a NaN/inf "
                    f"silently propagates rather than raising.")
            companion[field] = float(val)

        new_row = dict(row)
        new_row.update(medians)
        new_row['ci_companion'] = companion
        out[name] = new_row
    return out


def _cell_companion_means(family_runs: Dict[str, dict], factors, cell: dict
                          ) -> Optional[Dict[str, float]]:
    """Mean over the cell's seeds of each ci_companion field."""
    collected: Dict[str, List[float]] = {f: [] for f in CI_COMPANION_FIELDS}
    any_found = False
    for row in family_runs.values():
        if any(row.get(f) != cell[f] for f in factors):
            continue
        comp = row.get('ci_companion')
        if comp is None:
            continue
        any_found = True
        for f in CI_COMPANION_FIELDS:
            collected[f].append(comp[f])
    if not any_found:
        return None
    return {f: float(np.mean(vals)) for f, vals in collected.items()}


# ---------------------------------------------------------------------------
# Saturated-model Type III ANOVA, parametric over (factors, levels).
# ---------------------------------------------------------------------------

def deviation_columns(n_levels: int) -> np.ndarray:
    """Sum-to-zero contrast columns, shape (n_levels, n_levels - 1)."""
    k = n_levels
    C = np.zeros((k, k - 1))
    for i in range(k - 1):
        C[i, i] = 1.0
    C[k - 1, :] = -1.0
    return C


def _term_columns(cells, term_factors, level_index, levels):
    per_factor = []
    for f in term_factors:
        C = deviation_columns(len(levels[f]))
        idx = np.array([level_index[f][c[f]] for c in cells])
        per_factor.append(C[idx])
    out = per_factor[0]
    for nxt in per_factor[1:]:
        out = np.einsum('ni,nj->nij', out, nxt).reshape(len(cells), -1)
    return out


def build_design(cells, factors, levels):
    """Model matrix for the saturated crossing of `factors`, deviation coded.
    terms = [(name, column_slice), ...]: all mains, then all two-ways, ..., up to
    the full interaction."""
    level_index = {f: {lv: i for i, lv in enumerate(levels[f])} for f in factors}
    cols = [np.ones((len(cells), 1))]
    terms = []
    pos = 1
    for k in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, k):
            block = _term_columns(cells, combo, level_index, levels)
            cols.append(block)
            terms.append((':'.join(combo), slice(pos, pos + block.shape[1])))
            pos += block.shape[1]
    return np.hstack(cols), terms


def assert_block_orthogonal(X, terms, tol=1e-8):
    """Different terms' column blocks must be mutually orthogonal (and to the
    intercept). Holds exactly on a balanced, complete design and is what makes
    Type I = Type II = Type III here."""
    blocks = [('intercept', slice(0, 1))] + list(terms)
    scale = max(np.abs(X.T @ X).max(), 1.0)
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            (ni, si), (nj, sj) = blocks[i], blocks[j]
            cross = X[:, si].T @ X[:, sj]
            if np.abs(cross).max() > tol * scale:
                raise AssertionError(
                    f"terms '{ni}' and '{nj}' are not orthogonal (max cross "
                    f"product {np.abs(cross).max():.6g}). The design is "
                    f"unbalanced.")


class ConstantResponseError(ValueError):
    """No residual variance is left to test against, so no F-test is defined.
    Two distinct situations raise it and the caller must tell them apart: a
    genuinely constant response (every observation identical) and zero REPLICATE
    variance (the seeds agree exactly within every cell while the cell means
    still differ)."""


@dataclass
class TermResult:
    term: str
    df: int
    ss: float
    F: float
    p: float


def fit_type3(X, terms, y):
    """Drop-one (Type III) F-tests on a deviation-coded, balanced design: each term's
    SS is SSE(model without that term's columns) - SSE(full model)."""
    n = len(y)
    assert_block_orthogonal(X, terms)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    df_res = n - X.shape[1]
    if df_res <= 0:
        raise ValueError("no residual df: the model saturates the observations")
    ms_res = ss_res / df_res

    # Scale-free zero-residual-variance guard. lstsq leaves rounding residue
    # PROPORTIONAL to the response's own magnitude, so ss_res is compared against
    # the response's own corrected total SS, never against an absolute number:
    # the F statistics are invariant under y -> c*y and so is this test. (The
    # previous form, max(mean(y**2), 1.0) * 1e-20, was absolute for any response
    # whose values are below 1, and declared small-magnitude responses constant.)
    ss_total = float(((y - y.mean()) ** 2).sum())
    if ss_total == 0.0:
        raise ConstantResponseError(
            f"all {n} observations are identical ({y[0]:.6g}): the response is "
            f"constant and no F-test is defined.")
    if ss_res <= 1e-24 * ss_total:
        raise ConstantResponseError(
            f"residual SS is ~0 ({ss_res:.3g} against a total SS of "
            f"{ss_total:.3g}): the model reproduces every observation exactly, "
            f"so there is no residual variance to test against. No F-test is "
            f"defined. This is ZERO REPLICATE VARIANCE, not a constant "
            f"response: the response may still vary across cells.")

    out = []
    for name, sl in terms:
        keep = np.ones(X.shape[1], dtype=bool)
        keep[sl] = False
        Xr = X[:, keep]
        br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        rr = y - Xr @ br
        ss_term = float(rr @ rr) - ss_res
        df_term = sl.stop - sl.start
        F = (ss_term / df_term) / ms_res
        out.append(TermResult(name, df_term, ss_term, float(F),
                              float(sps.f.sf(F, df_term, df_res))))
    return out, ss_res, df_res, ms_res


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05):
    """Return (k_star, threshold_at_k_star, survivor_mask) over ONE family of tests."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return 0, 0.0, np.zeros(0, dtype=bool)
    order = np.argsort(p)
    k_star = 0
    for i, idx in enumerate(order, start=1):
        if p[idx] <= q * i / m:
            k_star = i
    mask = np.zeros(m, dtype=bool)
    if k_star:
        mask[order[:k_star]] = True
    thr = q * k_star / m if k_star else 0.0
    return k_star, thr, mask


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_cells(factors, levels):
    """All cells, itertools.product order (last factor fastest)."""
    return [dict(zip(factors, combo))
            for combo in itertools.product(*(levels[f] for f in factors))]


def collect_family_runs(runs: Dict[str, dict], family: str) -> Dict[str, dict]:
    return {name: row for name, row in runs.items() if row.get('family') == family}


def response_array(family_runs: Dict[str, dict], factors, levels, response: str):
    """(cells, Y) with Y of shape (n_cells, n_seeds); validate_structure has already
    asserted every cell x seed combination is present exactly once."""
    cells = build_cells(factors, levels)
    level_index = {f: {lv: i for i, lv in enumerate(levels[f])} for f in factors}
    shape = tuple(len(levels[f]) for f in factors)
    Y = np.full(shape + (len(SEEDS),), np.nan)
    for row in family_runs.values():
        idx = tuple(level_index[f][row[f]] for f in factors)
        Y[idx + (SEEDS.index(row['seed']),)] = row[response]
    Yflat = Y.reshape(len(cells), len(SEEDS))
    return cells, Yflat


def cell_name(factors, cell: dict) -> str:
    return '_'.join(f"{f}={cell[f]}" for f in factors)


def descriptive_cell_means(family_runs, factors, levels, response):
    """Best/worst cell means for a response reported descriptively only (no F-test)."""
    cells, Y = response_array(family_runs, factors, levels, response)
    means = Y.mean(axis=1)
    best_i = int(np.argmin(means))
    worst_i = int(np.argmax(means))
    return (cell_name(factors, cells[best_i]), float(means[best_i]),
            cell_name(factors, cells[worst_i]), float(means[worst_i]))


# ---------------------------------------------------------------------------
# Per-family analysis
# ---------------------------------------------------------------------------

@dataclass
class ResponseResult:
    response: str
    status: str  # 'ok' | 'constant' | 'zero_replicate_variance'
    terms: List[TermResult]
    ms_res: Optional[float]
    residual_sd: Optional[float]
    best_cell: Optional[dict]
    best_cell_name: Optional[str]
    best_mean: Optional[float]
    best_seed_values: Optional[List[float]]
    worst_cell: Optional[dict]
    worst_cell_name: Optional[str]
    worst_mean: Optional[float]
    worst_seed_values: Optional[List[float]]
    best_worst_ratio: Optional[float]
    constant_value: Optional[float] = None
    best_companion: Optional[Dict[str, float]] = None
    worst_companion: Optional[Dict[str, float]] = None


def analyze_response(family_runs, factors, levels, response,
                     use_ci_companion: bool = False) -> ResponseResult:
    cells, Y = response_array(family_runs, factors, levels, response)
    n_cells = len(cells)
    y = Y.reshape(-1)
    obs = [c for c in cells for _s in SEEDS]
    X, terms = build_design(obs, factors, levels)

    means = Y.mean(axis=1)
    best_i = int(np.argmin(means))
    worst_i = int(np.argmax(means))
    best_cell, worst_cell = cells[best_i], cells[worst_i]
    best_mean, worst_mean = float(means[best_i]), float(means[worst_i])
    ratio = (worst_mean / best_mean) if best_mean != 0 else float('nan')

    best_companion = worst_companion = None
    if use_ci_companion:
        best_companion = _cell_companion_means(family_runs, factors, best_cell)
        worst_companion = _cell_companion_means(family_runs, factors, worst_cell)

    try:
        term_res, ss_res, df_res, ms_res = fit_type3(X, terms, y)
    except ConstantResponseError:
        # Zero residual MS means zero REPLICATE variance, which is only the same
        # thing as a constant response when every observation is identical.
        # Otherwise the cell means still differ and best_mean/worst_mean below
        # bracket them; constant_value is meaningful only in the constant case.
        truly_constant = len(set(y.tolist())) == 1
        return ResponseResult(
            response=response,
            status='constant' if truly_constant else 'zero_replicate_variance',
            terms=[], ms_res=0.0,
            residual_sd=0.0, best_cell=best_cell,
            best_cell_name=cell_name(factors, best_cell), best_mean=best_mean,
            best_seed_values=list(Y[best_i]), worst_cell=worst_cell,
            worst_cell_name=cell_name(factors, worst_cell), worst_mean=worst_mean,
            worst_seed_values=list(Y[worst_i]), best_worst_ratio=ratio,
            constant_value=float(y[0]) if truly_constant else None,
            best_companion=best_companion, worst_companion=worst_companion)

    return ResponseResult(
        response=response, status='ok', terms=term_res, ms_res=ms_res,
        residual_sd=float(np.sqrt(ms_res)), best_cell=best_cell,
        best_cell_name=cell_name(factors, best_cell), best_mean=best_mean,
        best_seed_values=list(Y[best_i]), worst_cell=worst_cell,
        worst_cell_name=cell_name(factors, worst_cell), worst_mean=worst_mean,
        worst_seed_values=list(Y[worst_i]), best_worst_ratio=ratio,
        best_companion=best_companion, worst_companion=worst_companion)


DISPERSION_CAVEAT = (
    "CAVEAT: runs are seeded but GPU reductions are nondeterministic, so seed "
    "spread includes batch-order and hardware noise, not init variance alone.")

#: Dispersion responses: every response the location analysis F-tests for the ham
#: family (ruling 2026-09-11). The three transmission responses stay first, in their
#: original order, so the earlier blocks print in the same order.
DISPERSION_RESPONSES: Tuple[str, ...] = (
    'val_transmission_at_selection', 'l12_transmission', 'l16_transmission',
    'l12_dos', 'l16_dos', HAM_EXTRA_RESPONSE, CI_INTERIOR_MODE)

BLIND_DISPERSION_RESPONSES: Tuple[str, ...] = tuple(
    r for r in DISPERSION_RESPONSES if r in RESPONSES_COMMON)

BEST_EPOCH_DISPERSION_RESPONSES: Tuple[str, ...] = ('best_epoch',)


def dispersion_responses_for(family: str, family_runs: Dict[str, dict],
                             extra: Sequence[str] = ()) -> List[str]:
    """This family's dispersion response list, filtered to the responses present on
    EVERY row of the family; the entered list is printed by the caller."""
    base = list(DISPERSION_RESPONSES if family == 'ham'
                else BLIND_DISPERSION_RESPONSES) + list(extra)
    if not family_runs:
        return []
    present = set.intersection(*(set(row) for row in family_runs.values()))
    return [r for r in base if r in present]

#: Reduced-model order per family for the dispersion channel, fixed in advance. With
#: one observation per cell the saturated model leaves 0 residual df, so ham uses
#: mains + two-way (10 terms, df_res=13 on 32 cells) and blind mains only (3 terms,
#: df_res=4 on 8 cells).
DISPERSION_MAX_ORDER: Dict[str, int] = {'ham': 2, 'blind': 1}

DISPERSION_ZERO_DF_REASON = (
    "saturated dispersion model on 8 cells leaves 0 residual df; main "
    "effects only")


def cell_spreads(Y: np.ndarray) -> np.ndarray:
    """y = max - min across the seed axis, one value per cell. No floor: an
    exact-zero spread is data."""
    return Y.max(axis=1) - Y.min(axis=1)


def _arm_means(cells, levels, values: np.ndarray
               ) -> Tuple[Dict[str, float], int]:
    """Mean of `values` over the cells of each supervision arm, plus the per-arm
    cell count (balanced: 8 ham cells per arm, 4 blind)."""
    means: Dict[str, float] = {}
    counts = set()
    for lv in levels['supervision']:
        sel = [i for i, c in enumerate(cells) if c['supervision'] == lv]
        counts.add(len(sel))
        means[str(lv)] = float(np.mean(values[sel]))
    return means, (counts.pop() if len(counts) == 1 else -1)


def _arm_mean_and_range(cells, levels, values: np.ndarray
                        ) -> Tuple[Dict[str, Tuple[float, float, float]], int]:
    """As _arm_means, with the (min, max) of the cell values behind each arm mean."""
    out: Dict[str, Tuple[float, float, float]] = {}
    counts = set()
    for lv in levels['supervision']:
        sel = [i for i, c in enumerate(cells) if c['supervision'] == lv]
        counts.add(len(sel))
        v = values[sel]
        out[str(lv)] = (float(v.mean()), float(v.min()), float(v.max()))
    return out, (counts.pop() if len(counts) == 1 else -1)


def build_design_limited(cells, factors, levels, max_order: int):
    """As build_design but only through combinations of size <= max_order."""
    level_index = {f: {lv: i for i, lv in enumerate(levels[f])} for f in factors}
    cols = [np.ones((len(cells), 1))]
    terms = []
    pos = 1
    for k in range(1, max_order + 1):
        for combo in itertools.combinations(factors, k):
            block = _term_columns(cells, combo, level_index, levels)
            cols.append(block)
            terms.append((':'.join(combo), slice(pos, pos + block.shape[1])))
            pos += block.shape[1]
    return np.hstack(cols), terms


@dataclass
class DispersionTermResult:
    term: str
    df: int
    p: float


@dataclass
class DispersionResponseResult:
    response: str
    status: str  # 'ok' | 'constant' | 'zero_residual_variance'
    n_cells: int
    df_res: int
    n_terms: int
    terms: List[DispersionTermResult]
    most_stable_cell: str
    most_stable_spread: float
    least_stable_cell: str
    least_stable_spread: float
    constant_value: Optional[float] = None
    arm_mean_spread: Optional[Dict[str, float]] = None
    arm_n_cells: int = 0


def analyze_dispersion_response(family_runs, factors, levels, response,
                                max_order: int) -> DispersionResponseResult:
    """One observation per cell (y = max - min over the 3 seeds), reduced model of
    order max_order, raw scale only."""
    cells, Y = response_array(family_runs, factors, levels, response)
    n_cells = len(cells)
    spreads = cell_spreads(Y)

    arm_mean_spread, arm_n_cells = _arm_means(cells, levels, spreads)

    order = np.argsort(spreads)
    most_i, least_i = int(order[0]), int(order[-1])
    most_stable_cell = cell_name(factors, cells[most_i])
    most_stable_spread = float(spreads[most_i])
    least_stable_cell = cell_name(factors, cells[least_i])
    least_stable_spread = float(spreads[least_i])

    X, terms = build_design_limited(cells, factors, levels, max_order)
    n_params = X.shape[1]
    if n_params >= n_cells:
        raise StructureError(
            f"FATAL: dispersion model for {response!r} has {n_params} "
            f"parameters on {n_cells} cells -- non-positive residual df. "
            f"DISPERSION_MAX_ORDER for this family is too high for its own "
            f"factor set; fix the fixed-in-advance model order before "
            f"trusting any dispersion p-value.")
    df_res = n_cells - n_params

    try:
        term_res, _ss_res, _df_res, _ms_res = fit_type3(X, terms, spreads)
    except ConstantResponseError:
        # As in analyze_response: zero residual variance of the REDUCED model is
        # only the same thing as a constant spread surface when every cell spread
        # is identical. An exactly additive spread surface also lands here, with
        # 32 different spreads; most/least_stable_spread bracket them.
        truly_constant = len(set(spreads.tolist())) == 1
        return DispersionResponseResult(
            response=response,
            status='constant' if truly_constant else 'zero_residual_variance',
            n_cells=n_cells,
            df_res=df_res, n_terms=len(terms), terms=[],
            most_stable_cell=most_stable_cell,
            most_stable_spread=most_stable_spread,
            least_stable_cell=least_stable_cell,
            least_stable_spread=least_stable_spread,
            constant_value=float(spreads[0]) if truly_constant else None,
            arm_mean_spread=arm_mean_spread, arm_n_cells=arm_n_cells)

    dterms = [DispersionTermResult(t.term, t.df, t.p) for t in term_res]
    return DispersionResponseResult(
        response=response, status='ok', n_cells=n_cells, df_res=df_res,
        n_terms=len(terms), terms=dterms, most_stable_cell=most_stable_cell,
        most_stable_spread=most_stable_spread,
        least_stable_cell=least_stable_cell,
        least_stable_spread=least_stable_spread,
        arm_mean_spread=arm_mean_spread, arm_n_cells=arm_n_cells)


@dataclass
class FamilyDispersionAnalysis:
    family: str
    max_order: int
    n_cells: int
    df_res: int
    n_terms: int
    responses: Dict[str, DispersionResponseResult]
    bh_m: int
    bh_k_star: int
    bh_threshold: float
    bh_survivors: List[Tuple[str, str, float]]  # (response, term, p)
    all_tested: List[Tuple[str, str, float]]


def analyze_family_dispersion(runs: Dict[str, dict], family: str, factors,
                              levels, responses: Sequence[str]
                              ) -> FamilyDispersionAnalysis:
    family_runs = collect_family_runs(runs, family)
    max_order = DISPERSION_MAX_ORDER[family]

    per_response: Dict[str, DispersionResponseResult] = {}
    all_tested: List[Tuple[str, str, float]] = []
    n_cells = df_res = n_terms = 0
    for r in responses:
        dr = analyze_dispersion_response(family_runs, factors, levels, r,
                                         max_order)
        per_response[r] = dr
        n_cells, df_res, n_terms = dr.n_cells, dr.df_res, dr.n_terms
        if dr.status == 'ok':
            for t in dr.terms:
                all_tested.append((r, t.term, t.p))

    if all_tested:
        pvals = [p for _r, _t, p in all_tested]
        k_star, thr, mask = benjamini_hochberg(pvals, q=0.05)
        survivors = [all_tested[i] for i in range(len(all_tested)) if mask[i]]
    else:
        k_star, thr, survivors = 0, 0.0, []

    return FamilyDispersionAnalysis(
        family=family, max_order=max_order, n_cells=n_cells, df_res=df_res,
        n_terms=n_terms, responses=per_response, bh_m=len(all_tested),
        bh_k_star=k_star, bh_threshold=thr, bh_survivors=survivors,
        all_tested=all_tested)


def print_family_dispersion(out, fd: FamilyDispersionAnalysis):
    out("=" * 100)
    out(f"DISPERSION -- FAMILY: {fd.family.upper()}   n={fd.n_cells} cells "
       f"(one observation per cell = max-min over 3 seeds)   "
       f"df_res={fd.df_res}   terms={fd.n_terms}   max_order={fd.max_order}")
    out("=" * 100)
    out("NOTE: dispersion p-values are NOT comparable to location p-values "
       "above -- different n, different power (n/df printed on every row).")
    if fd.family == 'blind':
        out(f"WHY main-effects-only: {DISPERSION_ZERO_DF_REASON}")
    out("")

    for r, dr in fd.responses.items():
        out("-" * 100)
        out(f"DISPERSION RESPONSE: {r}   n={dr.n_cells}   df_res={dr.df_res}")
        if dr.status == 'constant':
            out(f"  CONSTANT spread across all cells at {dr.constant_value:.6g}. "
               f"No F-test is defined; this response's terms are EXCLUDED from "
               f"the dispersion BH family.")
        elif dr.status == 'zero_residual_variance':
            out(f"  ZERO RESIDUAL VARIANCE: the reduced dispersion model "
               f"reproduces every cell spread exactly; no F-test defined; cell "
               f"spreads range {dr.most_stable_spread:.6g} to "
               f"{dr.least_stable_spread:.6g}; this response's terms are "
               f"EXCLUDED from the dispersion BH family.")
        else:
            out(f"  {'term':30s} {'df':>3s} {'p':>12s}")
            for t in sorted(dr.terms, key=lambda x: x.p):
                out(f"  {t.term:30s} {t.df:3d} {t.p:12.4g}")
        out(f"  MOST STABLE  cell: {dr.most_stable_cell:60s} "
           f"spread={dr.most_stable_spread:.6g}")
        out(f"  LEAST STABLE cell: {dr.least_stable_cell:60s} "
           f"spread={dr.least_stable_spread:.6g}")
        if dr.arm_mean_spread:
            arm = "  ".join(f"{k}={v:.6g}"
                            for k, v in dr.arm_mean_spread.items())
            out(f"  MEAN SPREAD BY SUPERVISION ARM (mean over the "
               f"{dr.arm_n_cells} cells of each arm): {arm}")
        out("")

    out(f"DISPERSION BENJAMINI-HOCHBERG (q=0.05) over {fd.family} dispersion "
       f"family (own family, separate from the location BH family above): "
       f"m={fd.bh_m}")
    out(f"  survivors: {fd.bh_k_star} of {fd.bh_m}   "
       f"threshold={fd.bh_threshold:.5g}")
    for r, t, p in sorted(fd.bh_survivors, key=lambda x: x[2]):
        out(f"    SURVIVOR  {r:28s} {t:30s} p={p:.4g}")
    out("")


@dataclass
class FamilyAnalysis:
    family: str
    factors: Tuple[str, ...]
    levels: Dict[str, Tuple]
    entered_levels: Dict[str, list]
    responses: Dict[str, ResponseResult]
    bh_m: int
    bh_k_star: int
    bh_threshold: float
    bh_survivors: List[Tuple[str, str, float]]  # (response, term, p)
    all_tested: List[Tuple[str, str, float]]


def analyze_family(runs: Dict[str, dict], family: str, factors, levels,
                    responses: Sequence[str],
                    ci_companion_responses: Sequence[str] = ()
                    ) -> FamilyAnalysis:
    family_runs = collect_family_runs(runs, family)
    entered_levels = {
        f: sorted({row[f] for row in family_runs.values()}, key=str)
        for f in factors
    }

    per_response: Dict[str, ResponseResult] = {}
    all_tested: List[Tuple[str, str, float]] = []
    for r in responses:
        rr = analyze_response(family_runs, factors, levels, r,
                              use_ci_companion=(r in ci_companion_responses))
        per_response[r] = rr
        if rr.status == 'ok':
            for t in rr.terms:
                all_tested.append((r, t.term, t.p))

    if all_tested:
        pvals = [p for _r, _t, p in all_tested]
        k_star, thr, mask = benjamini_hochberg(pvals, q=0.05)
        survivors = [all_tested[i] for i in range(len(all_tested)) if mask[i]]
    else:
        k_star, thr, survivors = 0, 0.0, []

    return FamilyAnalysis(
        family=family, factors=factors, levels=levels,
        entered_levels=entered_levels, responses=per_response,
        bh_m=len(all_tested), bh_k_star=k_star, bh_threshold=thr,
        bh_survivors=survivors, all_tested=all_tested)


# ---------------------------------------------------------------------------
# Training time: best-checkpoint epoch (own BH family)
# ---------------------------------------------------------------------------

def merge_best_epoch(runs: Dict[str, dict], epochs: dict) -> Dict[str, dict]:
    """Attach best_epoch and log10_best_epoch onto every row. The cache's run-name
    set must equal the report's exactly and every shared factor must agree, else the
    cache was written against a different campaign (fatal). Returns a NEW dict."""
    only_report = sorted(set(runs) - set(epochs))
    only_cache = sorted(set(epochs) - set(runs))
    if only_report or only_cache:
        raise StructureError(
            f"FATAL: best-epoch cache does not match the report's run set. "
            f"only-in-report={only_report[:5]} only-in-cache={only_cache[:5]}. "
            f"The cache is stale with respect to this report; regenerate it "
            f"with analysis/scripts/best_epoch_extract.py before trusting any "
            f"training-time number.")

    out: Dict[str, dict] = {}
    for name, row in runs.items():
        e = epochs[name]
        shared = ['family', 'supervision', 'num_layers', 'geometry', 'seed']
        if row.get('family') == 'ham':
            shared.append('n_orb')
        for f in shared:
            if row.get(f) != e.get(f):
                raise StructureError(
                    f"FATAL: best-epoch cache disagrees with the report on run "
                    f"{name!r}: {f}={row.get(f)!r} in the report, "
                    f"{e.get(f)!r} in the cache.")
        ep = e.get('best_epoch')
        if ep is None or not np.isfinite(ep) or ep <= 0:
            raise StructureError(
                f"FATAL: run {name!r} has best_epoch={ep!r} in the cache; a "
                f"selected epoch must be present and strictly positive.")
        new_row = dict(row)
        new_row['best_epoch'] = float(ep)
        new_row['log10_best_epoch'] = float(np.log10(float(ep)))
        out[name] = new_row
    return out


def best_epoch_arm_table(runs: Dict[str, dict], family: str, factors, levels
                         ) -> Tuple[Dict[str, Tuple[float, float, float]], int]:
    """Per-arm mean of the CELL-MEAN selected epoch, with the range of the cell means."""
    family_runs = collect_family_runs(runs, family)
    cells, Y = response_array(family_runs, factors, levels, 'best_epoch')
    return _arm_mean_and_range(cells, levels, Y.mean(axis=1))


def print_best_epoch_section(out, runs: Dict[str, dict]) -> dict:
    # Both families are fitted first so the "Model:" line below can state the
    # family sizes that were actually used rather than a hardcoded arithmetic
    # string. Printing order is unchanged.
    analyses = {}
    for family, factors, levels in (('ham', HAM_FACTORS, HAM_LEVELS),
                                    ('blind', BLIND_FACTORS, BLIND_LEVELS)):
        analyses[family] = analyze_family(runs, family, factors, levels,
                                          list(BEST_EPOCH_RESPONSES))
    ht, hr, hm = family_size_parts(analyses['ham'])
    bt, br, bm = family_size_parts(analyses['blind'])

    out("=" * 100)
    out("TRAINING TIME: BEST-CHECKPOINT EPOCH (own BH family)")
    out("=" * 100)
    out("Response: saved_at_epoch of each run's PUBLISHED best checkpoint "
       "(outputs_v3/<run>/*_model_best.pth), read from the cache written by "
       "analysis/scripts/best_epoch_extract.py, cross-checked factor-by-factor "
       "against this report's own rows before any fit.")
    out(BEST_EPOCH_DESCRIPTIVE_NOTE)
    out("MULTIPLE TESTING: OWN BH FAMILY per model family, separate from the "
       "location families and from the dispersion families above. The "
       "published thresholds are untouched; m is printed on each family's own "
       "BENJAMINI-HOCHBERG line below.")
    out(f"Model: the same saturated model, sum-to-zero deviation coding and "
       f"Type III drop-one F used for every other location response "
       f"(analyze_family) -- {ht} terms x {hr} responses = {hm} for ham, "
       f"{bt} x {br} = {bm} for blind.")
    out("")

    for family, factors, levels in (('ham', HAM_FACTORS, HAM_LEVELS),
                                    ('blind', BLIND_FACTORS, BLIND_LEVELS)):
        fa = analyses[family]
        print_family(out, fa)

        arm, n_arm_cells = best_epoch_arm_table(runs, family, factors, levels)
        out(f"MEAN SELECTED EPOCH BY SUPERVISION ARM -- {family} (mean over "
           f"the {n_arm_cells} cells of each arm; range = min-max of those "
           f"cell means). Descriptive: earlier is not better.")
        out(f"  {'arm':12s} {'mean epoch':>12s} {'cell-mean range':>24s}")
        for lv, (mean, lo, hi) in arm.items():
            out(f"  {lv:12s} {mean:12.1f} {lo:11.1f} - {hi:10.1f}")
        out("")

    return analyses


# ---------------------------------------------------------------------------
# Head-to-head: select on L=12, report on L=16
# ---------------------------------------------------------------------------

@dataclass
class HeadToHead:
    ham_cell_means_l12: Dict[str, float]
    ham_cell_means_l16: Dict[str, float]
    blind_cell_means_l12: Dict[str, float]
    blind_cell_means_l16: Dict[str, float]
    best_ham_cell_on_l12: str
    best_ham_l16: float
    best_blind_cell_on_l12: str
    best_blind_l16: float
    resample_l16_distribution: np.ndarray
    resample_seed: int
    n_resamples: int


def head_to_head(runs: Dict[str, dict]) -> HeadToHead:
    """SELECT on l12_transmission, REPORT l16_transmission. Lower is better."""
    ham_runs = collect_family_runs(runs, 'ham')
    blind_runs = collect_family_runs(runs, 'blind')

    ham_cells = build_cells(HAM_FACTORS, HAM_LEVELS)
    blind_cells = build_cells(BLIND_FACTORS, BLIND_LEVELS)

    _c, ham_l12 = response_array(ham_runs, HAM_FACTORS, HAM_LEVELS, 'l12_transmission')
    _c, ham_l16 = response_array(ham_runs, HAM_FACTORS, HAM_LEVELS, 'l16_transmission')
    _c, blind_l12 = response_array(blind_runs, BLIND_FACTORS, BLIND_LEVELS,
                                   'l12_transmission')
    _c, blind_l16 = response_array(blind_runs, BLIND_FACTORS, BLIND_LEVELS,
                                   'l16_transmission')

    ham_l12_mean = ham_l12.mean(axis=1)
    ham_l16_mean = ham_l16.mean(axis=1)
    blind_l12_mean = blind_l12.mean(axis=1)
    blind_l16_mean = blind_l16.mean(axis=1)

    ham_names = [cell_name(HAM_FACTORS, c) for c in ham_cells]
    blind_names = [cell_name(BLIND_FACTORS, c) for c in blind_cells]

    best_ham_i = int(np.argmin(ham_l12_mean))
    best_blind_i = int(np.argmin(blind_l12_mean))

    # Matched selection pressure: subsample the 32 ham cells down to 8, 10000
    # seeded draws; each draw selects the best-on-l12 cell OF THAT DRAW and
    # reports its l16. The draw is 8 of 32 without replacement, so the globally
    # best cell is present in about a quarter of the draws and is selected
    # whenever it is: the draw is not forced to the global best, it is simply
    # not given the whole family to choose from.
    rng = np.random.default_rng(RESAMPLE_SEED)
    n = len(ham_cells)
    dist = np.empty(N_RESAMPLES)
    for i in range(N_RESAMPLES):
        idx = rng.choice(n, size=N_BLIND_CELLS, replace=False)
        local_best = idx[np.argmin(ham_l12_mean[idx])]
        dist[i] = ham_l16_mean[local_best]

    return HeadToHead(
        ham_cell_means_l12=dict(zip(ham_names, ham_l12_mean.tolist())),
        ham_cell_means_l16=dict(zip(ham_names, ham_l16_mean.tolist())),
        blind_cell_means_l12=dict(zip(blind_names, blind_l12_mean.tolist())),
        blind_cell_means_l16=dict(zip(blind_names, blind_l16_mean.tolist())),
        best_ham_cell_on_l12=ham_names[best_ham_i],
        best_ham_l16=float(ham_l16_mean[best_ham_i]),
        best_blind_cell_on_l12=blind_names[best_blind_i],
        best_blind_l16=float(blind_l16_mean[best_blind_i]),
        resample_l16_distribution=dist,
        resample_seed=RESAMPLE_SEED, n_resamples=N_RESAMPLES)


# ---------------------------------------------------------------------------
# Reporting (printing)
# ---------------------------------------------------------------------------

def family_size_parts(fa: FamilyAnalysis) -> Tuple[int, int, int]:
    """(terms per fitted response, number of fitted responses, m) read BACK OFF
    the family that was actually fitted. Every printed family size is derived
    from this, so an announced size can never disagree with the m handed to
    benjamini_hochberg -- a response that drops out (no residual variance, no
    F-test) shrinks both numbers together."""
    responses: List[str] = []
    for r, _t, _p in fa.all_tested:
        if r not in responses:
            responses.append(r)
    n_resp = len(responses)
    return (fa.bh_m // n_resp if n_resp else 0), n_resp, fa.bh_m


def print_family(out, fa: FamilyAnalysis):
    out("=" * 100)
    out(f"FAMILY: {fa.family.upper()}")
    out("=" * 100)
    out("ALWAYS CHECK THIS LINE -- levels that entered the fit:")
    for f in fa.factors:
        out(f"  entered {f:12s}: {fa.entered_levels[f]}  (design: {list(fa.levels[f])})")
    out(f"  seeds (fixed)       : {list(SEEDS)}")
    out("")

    for r, rr in fa.responses.items():
        out("-" * 100)
        out(f"RESPONSE: {r}")
        if rr.status == 'constant':
            out(f"  CONSTANT across all runs at {rr.constant_value:.6g}. No F-test "
               f"is defined; this response's terms are EXCLUDED from the BH family.")
        elif rr.status == 'zero_replicate_variance':
            out(f"  ZERO REPLICATE VARIANCE: seeds agree exactly within every "
               f"cell; no F-test defined; cell means range {rr.best_mean:.6g} to "
               f"{rr.worst_mean:.6g}; this response's terms are EXCLUDED from "
               f"the BH family.")
        else:
            out(f"  residual SD (seed noise floor) = {rr.residual_sd:.6g}")
            out(f"  {'term':44s} {'df':>3s} {'SS':>12s} {'F':>10s} {'p':>12s}")
            for t in sorted(rr.terms, key=lambda x: x.p):
                out(f"  {t.term:44s} {t.df:3d} {t.ss:12.5g} {t.F:10.3f} {t.p:12.4g}")
        out(f"  BEST : {rr.best_cell_name:60s} mean={rr.best_mean:.6g}  "
           f"seeds={np.round(np.asarray(rr.best_seed_values), 6).tolist()}")
        out(f"  WORST: {rr.worst_cell_name:60s} mean={rr.worst_mean:.6g}  "
           f"seeds={np.round(np.asarray(rr.worst_seed_values), 6).tolist()}")
        if rr.best_companion is not None:
            comp_best = "  ".join(f"{k}={v:.6g}" for k, v in rr.best_companion.items())
            comp_worst = "  ".join(f"{k}={v:.6g}" for k, v in rr.worst_companion.items())
            out(f"    COMPANION @ BEST  cell ({rr.best_cell_name}): {comp_best}")
            out(f"    COMPANION @ WORST cell ({rr.worst_cell_name}): {comp_worst}")
            out("    (never a lone drift number -- spec section 3.3)")
        out(f"  best/worst ratio = {rr.best_worst_ratio:.4g}  "
           f"(reported even when no term is significant)")
        out("")

    out(f"BENJAMINI-HOCHBERG (q=0.05) over this family's tests: m={fa.bh_m}")
    out("  Kept as insurance, not load-bearing: BH controls the false-discovery "
       "fraction across the pooled family, and is applied whether or not any "
       "term sits near the boundary.")
    n_clear_nothing = fa.bh_m - fa.bh_k_star
    out(f"  survivors: {fa.bh_k_star} of {fa.bh_m}   threshold={fa.bh_threshold:.5g}")
    # (response x term) pairs, not distinct terms: one term is counted once per
    # response it was tested on.
    out(f"  tests clearing nothing (not a BH survivor): {n_clear_nothing}")
    for r, t, p in sorted(fa.bh_survivors, key=lambda x: x[2]):
        out(f"    SURVIVOR  {r:28s} {t:40s} p={p:.4g}")
    out("")


def print_head_to_head(out, h2h: HeadToHead):
    out("=" * 100)
    out("HEAD-TO-HEAD (spec section 5a): SELECT ON L=12, REPORT ON L=16. NEVER BOTH.")
    out("=" * 100)
    out("PRIMARY NUMBER -- the family distribution at l16_transmission "
       "(lower is better):")
    out(f"  ham cells   (n={len(h2h.ham_cell_means_l16)}):")
    for name, v in sorted(h2h.ham_cell_means_l16.items(), key=lambda kv: kv[1]):
        out(f"    {name:60s} {v:.6g}")
    out(f"  blind cells (n={len(h2h.blind_cell_means_l16)}):")
    for name, v in sorted(h2h.blind_cell_means_l16.items(), key=lambda kv: kv[1]):
        out(f"    {name:60s} {v:.6g}")
    out("")
    out("SECONDARY, and labelled as such -- best-cell vs best-cell:")
    out(f"  best ham cell on l12_transmission   : {h2h.best_ham_cell_on_l12}  "
       f"-> l16_transmission = {h2h.best_ham_l16:.6g}")
    out(f"  best blind cell on l12_transmission : {h2h.best_blind_cell_on_l12}  "
       f"-> l16_transmission = {h2h.best_blind_l16:.6g}")
    out(f"  margin (blind_best - ham_best on l16) = "
       f"{h2h.best_blind_l16 - h2h.best_ham_l16:+.6g}")
    out("")
    d = h2h.resample_l16_distribution
    out(f"MATCHED SELECTION PRESSURE ({h2h.n_resamples} seeded draws, "
       f"analysis RNG seed={h2h.resample_seed}):")
    out("  each draw subsamples the ham family to 8 cells (matching the blind "
       "family's size), selects THAT draw's own best-on-l12 cell, and reports "
       "its l16_transmission.")
    out(f"  distribution: mean={d.mean():.6g} median={np.median(d):.6g} "
       f"sd={d.std(ddof=1):.6g} min={d.min():.6g} max={d.max():.6g}")
    out(f"  blind best (single value, blind has exactly 8 cells -- no "
       f"resampling needed): {h2h.best_blind_l16:.6g}")
    frac_blind_wins = float(np.mean(d > h2h.best_blind_l16))
    out(f"  fraction of matched-pressure draws where the blind cell's l16 "
       f"value is still better (lower) than the resampled ham draw: "
       f"{frac_blind_wins:.4g}")
    out("")
    out("NOTE: the per-sequence sign test (spec 5a.3) requires per-record data "
       "not present in this report and is run as a separate targeted pass "
       "(controller ruling).")
    out("")


CAVEAT_LINE = (
    "CAVEAT: each arm's val_transmission_at_selection is read at that arm's OWN "
    "selection optimum -- selection metrics differ by arm by design (spec section "
    "2). The response itself is the uniformly-computed validation transmission "
    "loss, compared at each run's own best epoch; that is the designed "
    "comparison, not a confound.")


def run_analysis(report: dict, ci_report: dict, best_epoch_cache: dict,
                 out=print) -> dict:
    """Run the full v3 DOE analysis; `out` is called once per output line."""
    runs = report.get('runs', {})
    runs = merge_ci(runs, ci_report)
    runs = merge_best_epoch(runs, best_epoch_cache)
    validate_structure(runs)

    out("=" * 100)
    out("CAMPAIGN-V3 DOE")
    out("=" * 100)
    out(CAVEAT_LINE)
    out("NO RATE, SLOPE, OR THRESHOLD METRIC APPEARS ANYWHERE IN THIS ANALYSIS. "
       "Every length response is a full-curve Huber loss at a fixed length.")
    out("")
    out(SUB_T_REMOVAL_NOTE)
    out("")

    ham_responses = list(RESPONSES_COMMON) + [HAM_EXTRA_RESPONSE] + list(CI_TESTED_RESPONSES)
    ci_companion_responses: Sequence[str] = CI_TESTED_RESPONSES

    # Fitted BEFORE the banner is printed so the announced family size is the m
    # that benjamini_hochberg was actually given, not a count of what was
    # intended to be fitted. print_family below still prints in the same order.
    ham = analyze_family(runs, 'ham', HAM_FACTORS, HAM_LEVELS, ham_responses,
                         ci_companion_responses=ci_companion_responses)
    n_terms, n_ham_resp, ham_m = family_size_parts(ham)
    out(f"CI RESPONSE ADDED: {list(CI_TESTED_RESPONSES)} (ham-only, "
       f"ruling 2, 2026-08-26). ham BH family size is now {n_terms} terms x "
       f"{n_ham_resp} responses = {ham_m} "
       f"(dropped sub_t, added ci_interior_mode). "
       f"{list(CI_DESCRIPTIVE_ONLY)} is merged and exported but "
       f"DESCRIPTIVE ONLY, no F-test (2026-09-01: it is the mode "
       f"axis plus two small components in quadrature -- a near-copy of "
       f"the headline that would only pad the family). "
       f"Companion columns ({list(CI_COMPANION_FIELDS)}) are printed "
       f"beside their best/worst cells per spec section 3.3.")
    out("")

    print_family(out, ham)

    ham_runs = collect_family_runs(runs, 'ham')
    descriptive = list(ONSITE_DESCRIPTIVE_ONLY) + list(CI_DESCRIPTIVE_ONLY)
    for of_name in descriptive:
        best_name, best_mean, worst_name, worst_mean = descriptive_cell_means(
            ham_runs, HAM_FACTORS, HAM_LEVELS, of_name)
        why = ("spec section 4: not a spec-defined response"
               if of_name in ONSITE_DESCRIPTIVE_ONLY else
               "near-copy of ci_interior_mode, ruling 2026-09-01")
        out(f"DESCRIPTIVE {of_name} ({why}, no F-test) -- best cell "
           f"{best_name} mean={best_mean:.6g}; worst cell {worst_name} "
           f"mean={worst_mean:.6g}")
        out("")

    blind = analyze_family(runs, 'blind', BLIND_FACTORS, BLIND_LEVELS,
                           list(RESPONSES_COMMON))
    print_family(out, blind)

    out("=" * 100)
    out("SEED-TO-SEED DISPERSION ANALYSIS (spread = max-min over 3 seeds per "
       "cell)")
    out("=" * 100)
    out(DISPERSION_CAVEAT)
    out("Covers EVERY response this family's location analysis F-tests "
       "(owner ruling 2026-09-11, extending the 2026-08-26 transmission-only "
       "ruling), plus best_epoch when the epoch cache is available. "
       "ci_interior_total is descriptive-only in the location analysis and is "
       "therefore not a dispersion response either. Raw scale only -- pooled "
       "into its own BH family per model family, separate from the location "
       "BH families above (different n, different power: not comparable "
       "across analyses).")

    extra_disp = BEST_EPOCH_DISPERSION_RESPONSES
    ham_disp_responses = dispersion_responses_for(
        'ham', collect_family_runs(runs, 'ham'), extra_disp)
    blind_disp_responses = dispersion_responses_for(
        'blind', collect_family_runs(runs, 'blind'), extra_disp)
    out(f"ALWAYS CHECK THIS LINE -- dispersion responses that entered: "
       f"ham {ham_disp_responses}; blind {blind_disp_responses}")
    out("")

    ham_dispersion = analyze_family_dispersion(
        runs, 'ham', HAM_FACTORS, HAM_LEVELS, ham_disp_responses)
    print_family_dispersion(out, ham_dispersion)

    blind_dispersion = analyze_family_dispersion(
        runs, 'blind', BLIND_FACTORS, BLIND_LEVELS, blind_disp_responses)
    print_family_dispersion(out, blind_dispersion)

    best_epoch = print_best_epoch_section(out, runs)

    h2h = head_to_head(runs)
    print_head_to_head(out, h2h)

    return {'ham': ham, 'blind': blind, 'ham_dispersion': ham_dispersion,
           'blind_dispersion': blind_dispersion, 'best_epoch': best_epoch,
           'head_to_head': h2h}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default=DEFAULT_REPORT_PATH,
                    help='path to posthoc_v3_report.json (read fresh; never cached)')
    ap.add_argument('--ci', default=DEFAULT_CI_PATH,
                    help='path to contact_invariance_v3.json (read fresh; never cached)')
    ap.add_argument('--best-epoch-cache', default=DEFAULT_BEST_EPOCH_CACHE,
                    help='path to the best-checkpoint-epoch cache written by '
                         'best_epoch_extract.py (read fresh; never cached)')
    ap.add_argument('--out', default=DEFAULT_OUT_PATH,
                    help='output file, OVERWRITTEN every run, never read back')
    args = ap.parse_args(argv)

    # The --out file is claimed BEFORE anything can fail. Writing it only at the
    # end would leave the previous run's report sitting there, undated and
    # indistinguishable from a fresh one, whenever this run aborted -- and every
    # structure check in this script is designed to abort.
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def fail(reason: str) -> None:
        with open(out_path, 'w') as f:
            f.write(f"FAILED: {reason}\n")

    fail('analysis started and has not completed')

    lines = []

    def out(line=''):
        print(line)
        lines.append(line)

    try:
        with open(args.report) as f:
            report = json.load(f)
        with open(args.ci) as f:
            ci_report = json.load(f)
        with open(args.best_epoch_cache) as f:
            best_epoch_cache = json.load(f)
        run_analysis(report, ci_report, best_epoch_cache, out=out)
    except BaseException as exc:
        fail(f"{type(exc).__name__}: {exc}")
        raise

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
