"""Real pre-2026-08 checkpoints from trained_models/ must still load.

Every other load test in the suite saves a checkpoint with the CURRENT code and
reads it straight back, so it can only catch a change that breaks the code
against itself. The files in `trained_models/` are v1-era (n_orb=1, no geometry)
and their `args` predate `solver_type`, `log_floor`, `floor_mode` and the whole
onsite-alpha surface -- exactly the checkpoints the campaign's baseline
comparisons depend on. Nothing tested that they still load.

The three axes exercised here, each of which a plausible edit would break:
  * the `energy_grid_t` buffer is `persistent=False`, so it is ABSENT from these
    state dicts by design; making it persistent turns every one of these files
    into a strict-load failure;
  * `floor_mode` defaults to 'clamp' (with log_floor 1e-16) for args-less
    checkpoints, which is what reproduces their recorded numbers;
  * the alpha -> `per_base_onsite` mapping, plus the state-dict cross-check, must
    resolve these alpha-free checkpoints to the context head without raising.
"""
import os

import numpy as np
import pytest
import torch

from g3nat.evaluation.inference import (
    load_trained_model,
    per_base_onsite_from_args,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(REPO_ROOT, 'trained_models')

# Tracked in git, so these are present in any clone -- not skipped-if-missing.
HAMILTONIAN_CKPTS = [
    'hamiltonian_DFT_gat_baseaware.pth',            # gat, 201-point DFT grid
    'hamiltonian_2000x_4to10BP_5000epoch.pth',      # transformer, tb grid
    'hamiltonian_2000x_4to10BP_5000epoch_baseblind.pth',  # legacy proj layout
]


def _path(name):
    p = os.path.join(MODEL_DIR, name)
    assert os.path.exists(p), f"tracked checkpoint missing from the clone: {p}"
    return p


def _raw(name):
    return torch.load(_path(name), map_location='cpu', weights_only=False)


@pytest.mark.parametrize('name', HAMILTONIAN_CKPTS)
def test_real_legacy_checkpoint_loads(name):
    """Constructs and load_state_dict succeeds, with the weights actually restored."""
    raw = _raw(name)
    model, energy_grid, device = load_trained_model(_path(name), device='cpu')

    assert len(energy_grid) == len(raw['energy_grid'])
    # Weights restored, not merely a fresh model of the right shape.
    ref_key = 'node_proj.weight'
    assert torch.allclose(model.node_proj.weight.detach().cpu(),
                          raw['model_state_dict'][ref_key].cpu())


@pytest.mark.parametrize('name', HAMILTONIAN_CKPTS)
def test_energy_grid_buffer_is_absent_from_legacy_state_dicts(name):
    """persistent=False is what keeps these files strict-loadable."""
    sd = _raw(name)['model_state_dict']
    assert 'energy_grid_t' not in sd, (
        "this legacy checkpoint predates the buffer; if it appears here the "
        "fixture was regenerated and the test no longer proves anything")

    model, energy_grid, _ = load_trained_model(_path(name), device='cpu')
    # A persistent buffer would make this a missing key under strict loading.
    result = model.load_state_dict(sd, strict=True)
    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []
    # The buffer exists on the model and carries the checkpoint's own grid.
    assert hasattr(model, 'energy_grid_t')
    assert np.allclose(model.energy_grid_t.cpu().numpy(),
                       np.asarray(energy_grid, dtype=np.float64))


@pytest.mark.parametrize('name', HAMILTONIAN_CKPTS)
def test_argsless_checkpoint_gets_legacy_floor_semantics(name):
    args = _raw(name)['args']
    assert 'floor_mode' not in args and 'log_floor' not in args, (
        "fixture no longer predates the floor arguments")

    model, _, _ = load_trained_model(_path(name), device='cpu')
    assert model.floor_mode == 'clamp', (
        "an args-less legacy checkpoint must keep the clamp semantics it was "
        "trained under; 'smooth' would silently change its deep-tail numbers")
    assert model.log_floor == pytest.approx(1e-16)


@pytest.mark.parametrize('name', HAMILTONIAN_CKPTS)
def test_alpha_mapping_resolves_legacy_checkpoints_to_the_context_head(name):
    raw = _raw(name)
    args, sd = raw['args'], raw['model_state_dict']
    assert 'per_base_onsite' not in args and 'structured_onsite' not in args, (
        "fixture no longer predates the alpha surface")
    assert 'onsite_baseline' not in sd and 'onsite_alpha_fixed' not in sd

    # The cross-check runs and agrees with the args.
    assert per_base_onsite_from_args(args, name, sd) is False
    model, _, _ = load_trained_model(_path(name), device='cpu')
    assert model.per_base_onsite is False
    assert not hasattr(model, 'onsite_baseline')


def test_cross_check_catches_a_per_base_table_under_legacy_args():
    """The cross-check is live, not decorative.

    Same real args as a legacy checkpoint (which resolve to per_base_onsite
    False), but a state dict carrying the per-base table. Resolving on args
    alone would silently drop the table and score a model nobody trained.
    """
    raw = _raw(HAMILTONIAN_CKPTS[0])
    args = dict(raw['args'])
    sd = dict(raw['model_state_dict'])
    sd['onsite_baseline'] = torch.zeros(4, 1)

    with pytest.raises(ValueError, match='onsite_baseline'):
        per_base_onsite_from_args(args, '<mutated>', sd)
