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

#: train.py selects the class through this local (base or _deep probe variant of
#: the same family) so the two families keep ONE constructor call each and the
#: deep variants inherit its keyword arguments by construction.
MODEL_CLS = '_model_cls'


def _name_of(expr):
    return expr.attr if isinstance(expr, ast.Attribute) else getattr(expr, 'id', None)


def _callee_name(node):
    return _name_of(node.func)


def _first_call(nodes, callee):
    for stmt in nodes:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and _callee_name(node) == callee:
                return {kw.arg for kw in node.keywords if kw.arg is not None}
    return None


def _family_calls(tree):
    """(standard kwargs, hamiltonian kwargs) of the model-construction if/else.

    The branch is the `if` whose test mentions 'standard': its body builds the
    standard family, its orelse the hamiltonian family. Each branch calls
    MODEL_CLS(...), which is resolved from args.model_type just above it.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        consts = {c.value for c in ast.walk(node.test) if isinstance(c, ast.Constant)}
        if 'standard' not in consts:
            continue
        std = _first_call(node.body, MODEL_CLS)
        ham = _first_call(node.orelse, MODEL_CLS)
        if std is not None and ham is not None:
            return std, ham
    return None, None


def _model_cls_mapping(tree):
    """{model_type: class name} from the `_model_cls = {...}[args.model_type]` line."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == MODEL_CLS for t in node.targets):
            continue
        d = node.value.value if isinstance(node.value, ast.Subscript) else node.value
        assert isinstance(d, ast.Dict)
        return {k.value: _name_of(v) for k, v in zip(d.keys, d.values)}
    return None


def _model_type_choices(tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _callee_name(node) == 'add_argument'):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == '--model_type'):
            continue
        for kw in node.keywords:
            if kw.arg == 'choices':
                return {c.value for c in kw.value.elts}
    return None


@pytest.fixture(scope='module')
def tree():
    return ast.parse(TRAIN_PY.read_text())


@pytest.fixture(scope='module')
def calls(tree):
    std, ham = _family_calls(tree)
    assert std is not None, f"no standard-family {MODEL_CLS}(...) call found in {TRAIN_PY}"
    assert ham is not None, f"no hamiltonian-family {MODEL_CLS}(...) call found in {TRAIN_PY}"
    return std, ham


def test_every_model_type_choice_maps_to_the_right_family_class(tree):
    """Each --model_type must resolve to a class, and the _deep probe variants
    must resolve to the Deep subclass of their own family so they take that
    family's branch with its exact keyword arguments."""
    mapping = _model_cls_mapping(tree)
    assert mapping is not None, f"no `{MODEL_CLS} = {{...}}` assignment in {TRAIN_PY}"
    assert set(mapping) == _model_type_choices(tree)
    assert mapping == {
        'standard': STANDARD,
        'standard_deep': STANDARD + 'Deep',
        'hamiltonian': HAMILTONIAN,
        'hamiltonian_deep': HAMILTONIAN + 'Deep',
    }


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
