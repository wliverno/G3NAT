"""Resolve which validation quantity a run should be checkpointed on.

THE RULE (campaign v3 spec, section 2): the best checkpoint is the argmin of the
run's OWN validation objective. Previously every arm was selected on the fixed
metric `val_dos_t_unweighted`, which is correct only where DOS is trained.

The training loss is

    L = a*T + c*[ b*LDOS + (1-b)*DOS ]

so the validation counterpart weights `val_transmission` by a, the LDOS key by
c*b, and `val_dos` by c*(1-b). Terms whose weight is zero are DROPPED rather
than kept at zero weight: a metric that contributes nothing can still be nan, and
0.0 * nan is nan, which would silently disable selection for the whole run.

WHICH LDOS KEY: `Trainer._validate_epoch` records the measured LDOS agreement under
`val_ldos_{config.ldos_target}` and pins the OTHER variant to nan, so the selection
metric has to follow `ldos_target` too -- select on what was actually measured.
Hardcoding one variant leaves a LATENT TRAP for the other: a `--ldos_target
base_only` run selected on `val_ldos_residue` would have a nan criterion every single
epoch, never update `best_unweighted`, and publish no weights at all, silently.

This closes that trap; it does not fix anything the campaign currently does.
`base_only` is a SUPPORTED BUT UNEXERCISED option -- `aggregate_by_residue(...,
base_only=True)` in `g3nat/data/ldos.py` implements it (nucleobase atoms only, sugar
and phosphate excluded), all 84 published campaign-v2 checkpoints recorded
`ldos_target='residue'`, and `ldos_target` is not a factor in either campaign-v3
factorial, so every v3 run takes the 'residue' default.

`ldos_target` is an OPTIONAL keyword, so three-argument callers keep exactly the
behaviour they had.
"""
from typing import Dict, Tuple

TRANSMISSION = 'val_transmission'
DOS = 'val_dos'
LDOS = 'val_ldos_residue'

#: Mirrors TrainingConfig.ldos_target's allowed values, and hence which
#: `val_ldos_*` key Trainer._validate_epoch actually fills in.
LDOS_TARGETS = ('residue', 'base_only')

_EPS = 1e-12


def resolve_selection_metric(loss_a: float, loss_b: float, loss_c: float,
                             ldos_target: str = 'residue'
                             ) -> Tuple[str, Dict[str, float]]:
    """Return (human-readable name, {validation metric key: weight}).

    Raises ValueError if the run trains nothing, which is not a valid config and
    must not silently produce an empty selection metric, or if `ldos_target` is not
    one of LDOS_TARGETS -- a typo would otherwise build a key that never appears in
    a metric_history entry, and the resulting KeyError would surface thousands of
    epochs later inside the epoch loop rather than here at construction.
    """
    if ldos_target not in LDOS_TARGETS:
        raise ValueError(
            f'unknown ldos_target {ldos_target!r}; expected one of '
            f'{list(LDOS_TARGETS)}. The selection metric keys off '
            f'val_ldos_<ldos_target>, which must be a key '
            f'Trainer._validate_epoch actually writes.')
    ldos_key = f'val_ldos_{ldos_target}'
    weights: Dict[str, float] = {}
    if abs(loss_a) > _EPS:
        weights[TRANSMISSION] = float(loss_a)
    if abs(loss_c) > _EPS:
        w_ldos = float(loss_c) * float(loss_b)
        w_dos = float(loss_c) * (1.0 - float(loss_b))
        if abs(w_ldos) > _EPS:
            weights[ldos_key] = w_ldos
        if abs(w_dos) > _EPS:
            weights[DOS] = w_dos
    if not weights:
        raise ValueError(
            f'no trained term for loss_a={loss_a}, loss_b={loss_b}, '
            f'loss_c={loss_c}: this run optimises nothing and has no valid '
            f'selection metric')
    # The NAME says 'ldos' for either target -- the WEIGHTS dict carries the exact
    # key, and that is what the checkpoint records and what the guard checks.
    short = {TRANSMISSION: 'transmission', DOS: 'dos', ldos_key: 'ldos'}
    name = '+'.join(f'{short[k]}*{v:g}' for k, v in sorted(weights.items()))
    return name, weights


def selection_value(entry: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted sum of one metric_history entry. nan in any contributing term
    propagates, which is correct -- selection genuinely cannot happen that epoch."""
    return sum(w * float(entry[k]) for k, w in weights.items())
