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
CALLBACKS_PY = pathlib.Path(__file__).resolve().parents[2] / 'g3nat' / 'training' / 'callbacks.py'


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


def _dict_literals_with_key(source_path, key):
    """Every ast.Dict literal anywhere in `source_path` that has `key` among
    its (string-constant) keys."""
    tree = ast.parse(source_path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if key in keys:
            found.append(node)
    return found


def _dict_has_key(dict_node, key):
    return any(isinstance(k, ast.Constant) and k.value == key for k in dict_node.keys)


def test_every_energy_grid_checkpoint_in_train_py_carries_init_seed():
    """Every checkpoint payload in train.py -- not just the one substring
    match above -- must carry init_seed. Catches a regression that adds a
    new payload, or drops the key from one of the existing three, which the
    plain substring test above cannot see (it only proves the string occurs
    at least once, anywhere, including in a comment)."""
    literals = _dict_literals_with_key(TRAIN_PY, 'energy_grid')
    assert len(literals) >= 3, (
        f"expected at least 3 checkpoint payload dict literals containing "
        f"'energy_grid' in train.py, found {len(literals)} -- the checkpoint "
        f"inventory this test protects may have shrunk")
    missing = [i for i, d in enumerate(literals) if not _dict_has_key(d, 'init_seed')]
    assert not missing, (
        f"checkpoint payload(s) at index {missing} of {len(literals)} found "
        f"in train.py contain 'energy_grid' but not 'init_seed'")


def test_callbacks_save_checkpoint_carries_init_seed():
    """save_checkpoint() in g3nat/training/callbacks.py writes
    checkpoint_latest.pth every epoch -- it is a checkpoint type distinct
    from the three in train.py and must carry init_seed too."""
    literals = _dict_literals_with_key(CALLBACKS_PY, 'energy_grid')
    assert len(literals) >= 1, (
        "expected at least 1 checkpoint payload dict literal containing "
        "'energy_grid' in callbacks.py (save_checkpoint's payload), found none")
    missing = [i for i, d in enumerate(literals) if not _dict_has_key(d, 'init_seed')]
    assert not missing, (
        f"checkpoint payload(s) at index {missing} of {len(literals)} found "
        f"in callbacks.py contain 'energy_grid' but not 'init_seed'")
