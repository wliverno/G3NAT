"""Resolve which validation quantity a run should be checkpointed on.

THE RULE (campaign v3 spec, section 2): the best checkpoint is the argmin of the
run's OWN validation objective. Previously every arm was selected on the fixed
metric `val_dos_t_unweighted`, which is correct only where DOS is trained.

The training loss is

    L = a*T + c*[ b*LDOS + (1-b)*DOS ]

so the validation counterpart weights `val_transmission` by a, `val_ldos_residue`
by c*b, and `val_dos` by c*(1-b). Terms whose weight is zero are DROPPED rather
than kept at zero weight: a metric that contributes nothing can still be nan, and
0.0 * nan is nan, which would silently disable selection for the whole run.
"""
from typing import Dict, Tuple

TRANSMISSION = 'val_transmission'
DOS = 'val_dos'
LDOS = 'val_ldos_residue'

_EPS = 1e-12


def resolve_selection_metric(loss_a: float, loss_b: float,
                             loss_c: float) -> Tuple[str, Dict[str, float]]:
    """Return (human-readable name, {validation metric key: weight}).

    Raises ValueError if the run trains nothing, which is not a valid config and
    must not silently produce an empty selection metric.
    """
    weights: Dict[str, float] = {}
    if abs(loss_a) > _EPS:
        weights[TRANSMISSION] = float(loss_a)
    if abs(loss_c) > _EPS:
        w_ldos = float(loss_c) * float(loss_b)
        w_dos = float(loss_c) * (1.0 - float(loss_b))
        if abs(w_ldos) > _EPS:
            weights[LDOS] = w_ldos
        if abs(w_dos) > _EPS:
            weights[DOS] = w_dos
    if not weights:
        raise ValueError(
            f'no trained term for loss_a={loss_a}, loss_b={loss_b}, '
            f'loss_c={loss_c}: this run optimises nothing and has no valid '
            f'selection metric')
    short = {TRANSMISSION: 'transmission', DOS: 'dos', LDOS: 'ldos'}
    name = '+'.join(f'{short[k]}*{v:g}' for k, v in sorted(weights.items()))
    return name, weights


def selection_value(entry: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted sum of one metric_history entry. nan in any contributing term
    propagates, which is correct -- selection genuinely cannot happen that epoch."""
    return sum(w * float(entry[k]) for k, w in weights.items())
