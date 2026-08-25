"""The guard from docs/metrics.md sec. 1b: refuse weights selected on a metric
containing a term the run never trained.

Fixtures are built from the real v2 loss encoding
(the campaign runner, private notes, lines 99-101), where loss_a is transmission and
loss_c switches the DOS family:

    dos   : loss_a=1.0 loss_b=0.0 loss_c=1.0
    ldos  : loss_a=1.0 loss_b=0.5 loss_c=1.0
    tonly : loss_a=1.0 loss_b=0.0 loss_c=0.0   <- DOS never trained

The dos/ldos cases must PASS while the tonly case FAILS; a guard that rejected
everything would be useless and is what these paired cases rule out.
"""
import pytest

from g3nat.evaluation.inference import check_selection_metric_trained

SEL = 'val_dos_t_unweighted'


def _args(loss_a=1.0, loss_b=0.0, loss_c=1.0):
    return {'loss_a': loss_a, 'loss_b': loss_b, 'loss_c': loss_c, 'n_orb': 2}


def test_tonly_selected_on_a_dos_metric_is_refused():
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(_args(loss_c=0.0), SEL, source='tonly_x.pth')
    msg = str(e.value)
    assert 'loss_c=0' in msg
    assert 'tonly_x.pth' in msg
    # It must name metric_history as the alternative, or the message tells the
    # reader what is wrong without telling them what to do instead.
    assert 'metric_history' in msg


def test_dos_arm_is_accepted():
    check_selection_metric_trained(_args(loss_c=1.0), SEL, source='dos_x.pth')


def test_ldos_arm_is_accepted():
    """ldos OMITS a trained term rather than including an untrained one.

    That costs ~6.6% on LDOS (sec. 18d) but the weights still sit near the
    optimum of what the metric does measure, so this guard must not fire --
    otherwise it would reject 24 usable cells.
    """
    check_selection_metric_trained(_args(loss_b=0.5, loss_c=1.0), SEL,
                                   source='ldos_x.pth')


def test_transmission_term_with_zero_loss_a_is_also_caught():
    """The rule is general, not DOS-specific."""
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(_args(loss_a=0.0), SEL, source='x.pth')
    assert 'loss_a=0' in str(e.value)


def test_v1_checkpoints_without_a_selection_metric_pass_through():
    """v1 files carry no selection_metric; the guard has nothing to check and
    must not block them (sec. 16e)."""
    check_selection_metric_trained(_args(loss_c=0.0), None, source='v1.pth')


def test_missing_loss_weights_do_not_raise():
    """An arg dict with no loss weights is unknown, not known-bad."""
    check_selection_metric_trained({'n_orb': 2}, SEL, source='x.pth')


def test_an_unrecognized_metric_is_not_treated_as_known_bad():
    """Unknown metric name -> no claim either way. Silently permissive by design;
    the map in inference.py is the place a new metric must be registered."""
    check_selection_metric_trained(_args(loss_c=0.0), 'val_something_new',
                                   source='x.pth')


def test_ldos_metric_with_untrained_ldos_is_refused():
    """The rule is not DOS-specific: selecting a dos-arm run (loss_b=0, so LDOS
    never trained) on an LDOS metric is the same error as the tonly case.

    This also pins the substring hazard that broke the first implementation:
    'dos' is a substring of 'val_ldos_residue', so a name-matching guard
    reported the DOS term here. The correct term for this metric is LDOS, and
    the weight that trains it is loss_b -- not loss_c.
    """
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(_args(loss_b=0.0, loss_c=1.0),
                                       'val_ldos_residue', source='x.pth')
    msg = str(e.value)
    assert 'loss_b=0' in msg
    assert 'loss_c' not in msg, "must not attribute an LDOS metric to the DOS weight"


def test_dos_metric_is_not_attributed_to_the_ldos_weight():
    """Converse of the above: val_dos must key off loss_c, and must not fire
    merely because loss_b=0 (which is the normal dos-arm setting)."""
    check_selection_metric_trained(_args(loss_b=0.0, loss_c=1.0), 'val_dos',
                                   source='x.pth')
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(_args(loss_b=0.0, loss_c=0.0), 'val_dos',
                                       source='x.pth')
    assert 'loss_c=0' in str(e.value)


# ------------------------------------------ campaign v3: weights, not a metric name

def test_v3_style_weights_still_trigger_the_guard():
    """The v3 selection metric names are not in _METRIC_TERM_WEIGHTS. Before the
    weights-first lookup they made this guard a no-op for every campaign run."""
    args = {'loss_a': 1.0, 'loss_b': 0.0, 'loss_c': 0.0, 'n_orb': 1,
            'selection_weights': {'val_transmission': 1.0, 'val_dos': 1.0}}
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(args, 'dos*1+transmission*1', source='x.pth')
    assert 'loss_c=0' in str(e.value)


def test_v3_weights_that_match_the_loss_are_accepted():
    args = {'loss_a': 1.0, 'loss_b': 0.0, 'loss_c': 0.0, 'n_orb': 1,
            'selection_weights': {'val_transmission': 1.0}}
    check_selection_metric_trained(args, 'transmission*1', source='x.pth')


def test_every_arm_the_campaign_runs_passes_its_own_resolved_weights():
    """End to end against the real resolver: the four supervision arms must each
    be accepted when selected on what resolve_selection_metric gives them."""
    from g3nat.training.selection import resolve_selection_metric

    for loss_a, loss_b, loss_c in [(1.0, 0.0, 1.0),    # DOS+T
                                   (1.0, 0.5, 1.0),    # LDOS+DOS+T
                                   (1.0, 1.0, 1.0),    # LDOS+T
                                   (1.0, 0.0, 0.0)]:   # T only
        name, weights = resolve_selection_metric(loss_a, loss_b, loss_c)
        args = {'loss_a': loss_a, 'loss_b': loss_b, 'loss_c': loss_c, 'n_orb': 2,
                'selection_weights': weights}
        check_selection_metric_trained(args, name, source='x.pth')


def test_an_ldos_arm_selected_on_dos_is_refused_by_the_weights_path():
    """THE LDOS+T ARM: loss_b=1 AND loss_c=1.

    DOS is trained with weight c*(1-b), which is 0 here, so a recorded val_dos
    term is the defect. But loss_c is 1, so a guard that judges val_dos on
    loss_c ALONE sails straight past it -- and LDOS+T at n_orb=2 is the arm
    campaign v3 newly promotes and has never run before.

    This fixture used to set loss_c=0.0 while its docstring named loss_b=1 as the
    mechanism, so it passed on the T-only mechanism instead and left the arm it
    is named for entirely uncovered.
    """
    args = {'loss_a': 1.0, 'loss_b': 1.0, 'loss_c': 1.0, 'n_orb': 2,
            'selection_weights': {'val_transmission': 1.0, 'val_dos': 1.0,
                                  'val_ldos_residue': 1.0}}
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(args, 'dos*1+ldos*1+transmission*1',
                                       source='x.pth')
    assert 'DOS' in str(e.value)
    assert 'loss_b' in str(e.value), \
        'the message must name loss_b=1 as what zeroes the DOS half'


def test_an_ldos_term_under_loss_c_zero_is_refused_by_the_weights_path():
    """The mirror hole. LDOS is trained with weight c*b, so at c=0 it is untrained
    no matter what b is -- but loss_b alone is 1 here, so a guard judging
    val_ldos_* on loss_b ALONE would pass it. Both terms are the PRODUCT."""
    args = {'loss_a': 1.0, 'loss_b': 1.0, 'loss_c': 0.0, 'n_orb': 2,
            'selection_weights': {'val_transmission': 1.0,
                                  'val_ldos_residue': 1.0}}
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(args, 'ldos*1+transmission*1',
                                       source='x.pth')
    assert 'LDOS' in str(e.value)
    assert 'loss_c=0' in str(e.value)


def test_recorded_weights_override_a_stale_metric_name():
    """A checkpoint whose NAME is the old fixed metric but whose recorded weights
    say transmission-only must be judged on the weights."""
    args = {'loss_a': 1.0, 'loss_b': 0.0, 'loss_c': 0.0, 'n_orb': 1,
            'selection_weights': {'val_transmission': 1.0}}
    check_selection_metric_trained(args, SEL, source='x.pth')


# --------------------------------------- legacy name-map path: same hole, one level over


def test_legacy_dos_metric_with_dos_untrained_via_loss_b_is_refused():
    """A pre-v3 checkpoint (no `selection_weights`) at the LDOS+T arm
    (loss_b=1, loss_c=1) has DOS untrained via c*(1-b)=0, but is selected on
    'val_dos_t_unweighted'. The legacy name map used to judge DOS on loss_c
    alone (sees 1.0, does not raise) -- the same hole the recorded-weights path
    closed in _KEY_TO_FACTORS, left open here one level over. There are 12 real
    campaign-v2 checkpoints at exactly this (loss_a, loss_b, loss_c)."""
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(_args(loss_b=1.0, loss_c=1.0), SEL,
                                       source='v2_ldos_t.pth')
    msg = str(e.value)
    assert 'DOS' in msg
    assert 'loss_b' in msg, 'message must identify loss_b as what zeroes the DOS half'


def test_legacy_ldos_metric_with_ldos_untrained_via_loss_c_is_refused():
    """Mirror of the above: a pre-v3 checkpoint with loss_b=1, loss_c=0 has LDOS
    untrained (c*b=0 whatever b is), selected on 'val_ldos_residue'. The legacy
    map used to judge LDOS on loss_b alone (sees 1.0, does not raise)."""
    with pytest.raises(ValueError) as e:
        check_selection_metric_trained(_args(loss_b=1.0, loss_c=0.0),
                                       'val_ldos_residue', source='v2_ldos.pth')
    msg = str(e.value)
    assert 'LDOS' in msg
    assert 'loss_c=0' in msg
