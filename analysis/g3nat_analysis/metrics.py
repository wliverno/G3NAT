"""Metric implementations for campaign v3. Every metric uses the FULL quantity it is
defined over: all 201 energy points, all 2L sites, no thresholds, no subsets."""
import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

from g3nat.evaluation.physicality import onsite_block_eigs


def onsite_distance(H, n_orb: int, n_bases: int) -> Dict[str, float]:
    """Distance from the HOMO reference (zero) of the learned onsite levels. Per site,
    'near' is the block eigenvalue nearest zero and 'far' the one furthest; each is the
    mean over all 2L sites. `n_bases` must be 2*len(sequence), never derived from H:
    the check catches an H spanning only one strand."""
    eigs = onsite_block_eigs(H, n_orb)          # [n_sites, n_orb], ascending
    n_sites = eigs.shape[0]
    if n_sites != n_bases:
        raise ValueError(
            f'site count mismatch: H has {n_sites} sites but {n_bases} bases were '
            f'given. H spans BOTH strands (2L sites for a length-L strand); '
            f'truncating to the shorter of the two silently discards the '
            f'complementary strand, which is a defect this check exists to '
            f'prevent.')
    lv = np.abs(np.asarray(eigs, float))
    return {'near': float(lv.min(axis=1).mean()),
            'far': float(lv.max(axis=1).mean()),
            'n_sites': int(n_sites)}


Pair = Tuple[str, str, int]


def eligible_substitution_pairs(sequences: Sequence[str]) -> List[Pair]:
    """All (a, b, position) of equal length differing at exactly one INTERIOR base
    (positions 1..L-2; the ends carry the contacts)."""
    by_len = {}
    for s in sorted(set(sequences)):
        by_len.setdefault(len(s), []).append(s)
    out: List[Pair] = []
    for L, group in sorted(by_len.items()):
        for a, b in itertools.combinations(group, 2):
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            if len(diff) == 1 and 0 < diff[0] < L - 1:
                out.append((a, b, diff[0]))
    return out


#: L=8 is excluded: it has exactly one eligible pair.
SUBSTITUTION_LENGTHS = (4, 5, 6, 7)


def select_pairs(pairs: Sequence[Pair], per_length_cap: int = 10,
                 seed: int = 20260824,
                 lengths: Sequence[int] = SUBSTITUTION_LENGTHS) -> List[Pair]:
    """At most `per_length_cap` pairs per strand length, chosen reproducibly."""
    rng = np.random.default_rng(seed)
    keep = set(lengths)
    by_len = {}
    for p in pairs:
        if len(p[0]) in keep:
            by_len.setdefault(len(p[0]), []).append(p)
    out: List[Pair] = []
    for L in sorted(by_len):
        group = sorted(by_len[L])           # sort first: input order must not matter
        if len(group) <= per_length_cap:
            out.extend(group)
        else:
            idx = rng.choice(len(group), size=per_length_cap, replace=False)
            out.extend(group[i] for i in sorted(idx))
    return out


def matched_records(records, pair: Pair):
    """Yield (contact_key, record_a, record_b) for every contact configuration BOTH
    sequences have. `records` maps (sequence_lowercase, run_key) -> record."""
    a, b, _pos = pair
    keys_a = {k for (s, k) in records if s == a.lower()}
    keys_b = {k for (s, k) in records if s == b.lower()}
    for key in sorted(keys_a & keys_b):
        yield key, records[(a.lower(), key)], records[(b.lower(), key)]


def _huber(residual, delta: float = 1.0):
    r = np.abs(np.asarray(residual, float))
    return np.where(r <= delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))


def substitution_loss(pred_a, pred_b, true_a, true_b, delta: float = 1.0) -> float:
    """Huber between the predicted and true CHANGE, over every grid point."""
    d_true = np.asarray(true_b, float) - np.asarray(true_a, float)
    d_pred = np.asarray(pred_b, float) - np.asarray(pred_a, float)
    if d_true.shape != d_pred.shape:
        raise ValueError(f'shape mismatch: predicted {d_pred.shape} vs '
                         f'reference {d_true.shape}')
    return float(_huber(d_pred - d_true, delta).mean())
