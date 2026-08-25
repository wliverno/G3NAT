"""--init_seed is mandatory, and lands in the checkpoint.

An unseeded run is not reproducible and set_init_seed(None) silently touches no
RNG (g3nat/training/utils.py:33), so the failure is invisible at runtime. The
campaign spec fixes three seeds; a run that did not record which one it used
cannot be matched to them afterwards.
"""
import ast
import pathlib

import pytest

TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'train.py'


def _argparse_kwargs(flag):
    """Keyword args of the add_argument call declaring `flag`."""
    tree = ast.parse(TRAIN_PY.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
        if name != 'add_argument':
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and node.args[0].value == flag:
            return {kw.arg: kw.value for kw in node.keywords}
    return None


def test_init_seed_is_required():
    kw = _argparse_kwargs('--init_seed')
    assert kw is not None, '--init_seed not declared in train.py'
    assert 'required' in kw, '--init_seed must declare required=True'
    assert isinstance(kw['required'], ast.Constant) and kw['required'].value is True
    assert 'default' not in kw, (
        'a required argument must not also carry a default -- a default is how '
        'the unseeded runs happened')


def test_init_seed_is_written_into_the_checkpoint():
    src = TRAIN_PY.read_text()
    assert "'init_seed': args.init_seed" in src, (
        "checkpoint payload must record init_seed so a run can be matched to "
        "the campaign's three fixed seeds after the fact")
