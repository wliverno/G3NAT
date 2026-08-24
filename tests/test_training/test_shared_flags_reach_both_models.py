"""Flags that apply to BOTH model types must be passed to BOTH constructors.

WHY THIS TEST EXISTS. `scripts/train.py` builds one of two models in an if/else, and
each constructor takes its arguments explicitly, silently keeping its own defaults for
anything omitted. Adding a shared flag to one branch and not the other raises no error
and prints no warning -- it produces a CONFOUNDED EXPERIMENT, which is worse than a
crash because the run completes and the numbers look fine.

It has happened twice:

  --conv_type    omitted from the standard branch, so the baseline always used
                 'transformer' while the hamiltonian model honoured the flag. Every
                 standard-vs-hamiltonian comparison before 2026-08-01 compared two
                 different convolutions as well as two different readouts.
  --use_geometry omitted from the standard branch until 2026-08-24, so a
                 geometry-enabled comparison gave the hamiltonian model an input the
                 baseline could not see -- an advantage unrelated to the readout,
                 which is the thing the whole comparison is testing.

This reads the source rather than running training, so it costs milliseconds and
cannot be skipped for being slow.
"""
import ast
import pathlib

import pytest

TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'train.py'

#: Flags meaningful to both models. A flag genuinely specific to one model
#: (n_orb, solver_type, log_floor, energy_grid ...) does NOT belong here.
SHARED_FLAGS = ('conv_type', 'use_geometry', 'geom_norm_stats',
                'hidden_dim', 'num_layers', 'num_heads')

STANDARD = 'DNATransportGNN'
HAMILTONIAN = 'DNATransportHamiltonianGNN'


def _kwargs_of_call(tree, func_name):
    """Keyword names passed to the first call whose callee ends in func_name.

    Matches on the attribute/name tail so `g3nat.DNATransportGNN(...)` and a bare
    `DNATransportGNN(...)` both resolve, and so that DNATransportHamiltonianGNN is
    not mistaken for DNATransportGNN.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
        if name == func_name:
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    return None


@pytest.fixture(scope='module')
def calls():
    tree = ast.parse(TRAIN_PY.read_text())
    std = _kwargs_of_call(tree, STANDARD)
    ham = _kwargs_of_call(tree, HAMILTONIAN)
    assert std is not None, f"no {STANDARD}(...) call found in {TRAIN_PY}"
    assert ham is not None, f"no {HAMILTONIAN}(...) call found in {TRAIN_PY}"
    return std, ham


@pytest.mark.parametrize('flag', SHARED_FLAGS)
def test_shared_flag_reaches_both_model_constructors(calls, flag):
    std, ham = calls
    pairs = ((STANDARD, std), (HAMILTONIAN, ham))
    missing = [n for n, kw in pairs if flag not in kw]
    present = [n for n, kw in pairs if flag in kw]
    assert not missing, (
        f"'{flag}' is passed to {present} but NOT to {missing}. "
        f"The omitted constructor silently keeps its own default, so the two "
        f"models differ by more than their readout and the comparison is "
        f"confounded. Add it to both branches.")


def test_the_two_models_are_not_accidentally_the_same_call(calls):
    """Guards the fixture itself: if the matcher collapsed both names onto one
    call, every test above would pass vacuously."""
    std, ham = calls
    assert 'energy_grid' in ham and 'energy_grid' not in std
    assert 'output_dim' in std and 'output_dim' not in ham
