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
