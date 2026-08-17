import importlib.util
import os
import sys
import pytest

_spec = importlib.util.spec_from_file_location(
    "train_script",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py"))
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def test_matching_args_pass():
    cur = {k: 1 for k in train_script.CONFIG_DEFINING_ARGS}
    train_script.check_resume_args(dict(cur), cur)  # no raise


def test_mismatched_arg_raises_with_the_key_named():
    cur = {k: 1 for k in train_script.CONFIG_DEFINING_ARGS}
    stored = dict(cur)
    stored['n_orb'] = 2
    with pytest.raises(ValueError, match='n_orb'):
        train_script.check_resume_args(stored, cur)


def test_missing_stored_key_raises():
    cur = {k: 1 for k in train_script.CONFIG_DEFINING_ARGS}
    stored = dict(cur)
    del stored['solver_type']
    with pytest.raises(ValueError, match='solver_type'):
        train_script.check_resume_args(stored, cur)
