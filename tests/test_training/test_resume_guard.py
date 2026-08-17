import importlib.util
import os
import sys
import pytest

_spec = importlib.util.spec_from_file_location(
    "train_script",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py"))
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


# A stand-in for vars(args): a handful of guarded keys plus the three
# environment-describing keys the guard must ignore.
def _args(**overrides):
    base = {
        'data_source': 'pickle', 'data_dir': 'pickle_files_v2', 'model_type': 'hamiltonian',
        'hidden_dim': 128, 'num_layers': 4, 'n_orb': 1, 'solver_type': 'complex',
        'dropout': 0.0, 'per_base_onsite': False, 'num_epochs': 5000,
        'device': 'auto', 'output_dir': './outputs', 'checkpoint_dir': './ckpt',
        'allow_arg_change': '',
    }
    base.update(overrides)
    return base


def test_matching_args_pass():
    cur = _args()
    assert train_script.check_resume_args(dict(cur), cur) == []


def test_mismatched_arg_raises_with_the_key_named():
    cur = _args()
    stored = _args(n_orb=2)
    with pytest.raises(ValueError, match='n_orb'):
        train_script.check_resume_args(stored, cur)


def test_missing_stored_key_raises():
    cur = _args()
    stored = _args()
    del stored['solver_type']
    with pytest.raises(ValueError, match='solver_type'):
        train_script.check_resume_args(stored, cur)


def test_non_defining_args_may_differ():
    """device/output_dir/checkpoint_dir describe the environment, not the config,
    and change legitimately on every requeue (a different node, a moved dir)."""
    cur = _args(device='cuda', output_dir='/scratch/out', checkpoint_dir='/scratch/ck')
    stored = _args(device='cpu', output_dir='/other/out', checkpoint_dir='/other/ck')
    assert train_script.check_resume_args(stored, cur) == []


@pytest.mark.parametrize('key,stored_value,current_value', [
    ('dropout', 0.0, 0.3),
    ('per_base_onsite', False, True),
])
def test_args_omitted_by_the_old_allowlist_are_now_guarded(key, stored_value, current_value):
    """I8: CONFIG_DEFINING_ARGS was an allowlist that already omitted 15 args --
    dropout and every onsite-head flag among them -- so a resume could silently
    switch them. The guard is now a denylist, so these are caught, and flags added
    later (per_base_onsite, which replaced the alpha flags) are guarded on day one
    with no list to remember to update."""
    cur = _args(**{key: current_value})
    stored = _args(**{key: stored_value})
    with pytest.raises(ValueError, match=key):
        train_script.check_resume_args(stored, cur)


def test_allow_arg_change_exempts_a_named_key_and_reports_it():
    """Raising the epoch cap on a requeued cell is routine; it must be possible,
    and it must leave a trace."""
    cur = _args(num_epochs=8000)
    stored = _args(num_epochs=5000)
    with pytest.raises(ValueError, match='num_epochs'):
        train_script.check_resume_args(stored, cur)
    exemptions = train_script.check_resume_args(stored, cur, allow_changed=['num_epochs'])
    assert len(exemptions) == 1 and 'num_epochs' in exemptions[0]


def test_allow_arg_change_does_not_exempt_other_keys():
    cur = _args(num_epochs=8000, n_orb=2)
    stored = _args(num_epochs=5000, n_orb=1)
    with pytest.raises(ValueError, match='n_orb'):
        train_script.check_resume_args(stored, cur, allow_changed=['num_epochs'])


def test_allow_arg_change_itself_may_differ():
    """The flag is the mechanism for declaring an exemption, so it cannot be
    required to match the checkpoint that predates its use."""
    cur = _args(allow_arg_change='num_epochs')
    stored = _args(allow_arg_change='')
    assert train_script.check_resume_args(stored, cur) == []


def test_parse_allow_arg_change():
    assert train_script.parse_allow_arg_change('') == []
    assert train_script.parse_allow_arg_change('num_epochs') == ['num_epochs']
    assert train_script.parse_allow_arg_change(' num_epochs , dropout ') == \
        ['num_epochs', 'dropout']
