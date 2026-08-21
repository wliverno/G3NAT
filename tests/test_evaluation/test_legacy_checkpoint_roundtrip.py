"""Real pre-2026-08 checkpoints from trained_models/ must still load.

Every other load test in the suite saves a checkpoint with the CURRENT code and
reads it straight back, so it can only catch a change that breaks the code
against itself. The files in `trained_models/` are v1-era (n_orb=1, no geometry)
and their `args` predate `solver_type`, `log_floor`, `floor_mode` and the whole
onsite-alpha surface -- exactly the checkpoints the campaign's baseline
comparisons depend on. Nothing tested that they still load.

The axes exercised here, each of which a plausible edit would break:
  * the `energy_grid_t` buffer is `persistent=False`, so it is ABSENT from these
    state dicts by design; making it persistent turns every one of these files
    into a strict-load failure;
  * `floor_mode` defaults to 'clamp' (with log_floor 1e-16) for args-less
    checkpoints, which is what reproduces their recorded numbers;
  * the alpha -> `per_base_onsite` mapping, plus the state-dict cross-check, must
    resolve these alpha-free checkpoints to the context head without raising;
  * EVERY OTHER args-less fallback, one assertion each. This file previously
    claimed to exercise "every args-less fallback at once" while asserting only
    the floor pair and the alpha mapping. `solver_type` is the one that matters
    most: it ONCE SHIPPED WRONG (the loader said 'frobenius', the constructor
    and therefore training said 'complex'), which silently invalidated the
    length curves behind private analysis notes 12a, and a reintroduction raises
    nothing -- the load simply succeeds through the wrong solver, moving
    near-resonance log10 T by up to 3.8 decades.
  * the DIRECT/standard baseline checkpoint, whose args carry no `conv_type` at
    all (it rides the standard branch's 'transformer' fallback) and whose
    readout width comes from the stored energy grid, not `num_energy_points` --
    reading that arg is a breakage this repo has already had once.
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
    # LOADS but is NOT RUNNABLE: its base-blind coupling head is 1*hidden wide while
    # the current model feeds 3*hidden. Everything in this file is a LOAD assertion,
    # which it passes honestly; see
    # tests/baseline/test_baseline_legacy_checkpoints.py::test_baseblind_checkpoint_loads_but_cannot_run
    # for the forward-pass consequence. Do not read "loads" as "reproduces its numbers".
    'hamiltonian_2000x_4to10BP_5000epoch_baseblind.pth',  # legacy proj layout
]

# The direct/blind baseline. Different constructor branch entirely, so it cannot
# join the Hamiltonian-only assertions -- but it is tracked, the campaign's blind
# control, and nothing covered it.
STANDARD_CKPT = 'standard_2000x_4to10BP_5000epoch.pth'
ALL_CKPTS = HAMILTONIAN_CKPTS + [STANDARD_CKPT]

# conv_type IS recorded in these three checkpoints' args, unlike everything else
# asserted below; pinning the value per file keeps the assertion from degenerating
# into `args.get(...) == args.get(...)`.
EXPECTED_CONV_TYPE = {
    'hamiltonian_DFT_gat_baseaware.pth': 'gat',
    'hamiltonian_2000x_4to10BP_5000epoch.pth': 'transformer',
    'hamiltonian_2000x_4to10BP_5000epoch_baseblind.pth': 'transformer',
}

# The layer whose weights prove restoration for each file. For the baseblind
# checkpoint that MUST be the legacy-proj rebuild (inference.py ~270-290
# constructs a FRESH random onsite_proj/coupling_proj); `node_proj` is never
# rebuilt and so cannot witness that path at all.
RESTORATION_KEYS = {
    'hamiltonian_DFT_gat_baseaware.pth': ['node_proj.weight', 'onsite_proj.2.weight'],
    'hamiltonian_2000x_4to10BP_5000epoch.pth': ['node_proj.weight', 'onsite_proj.2.weight'],
    'hamiltonian_2000x_4to10BP_5000epoch_baseblind.pth': [
        'node_proj.weight', 'onsite_proj.3.weight', 'coupling_proj.3.weight'],
    # The standard branch rebuilds nothing, but its readout width is decided by
    # `output_dim=len(energy_grid)`; dos_proj.3/transmission_proj.3 are exactly the
    # layers that go size-mismatched if that ever reverts to num_energy_points.
    STANDARD_CKPT: ['node_proj.weight', 'dos_proj.3.weight', 'transmission_proj.3.weight'],
}


def _get_param(model, dotted):
    obj = model
    for part in dotted.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _path(name):
    p = os.path.join(MODEL_DIR, name)
    assert os.path.exists(p), f"tracked checkpoint missing from the clone: {p}"
    return p


def _raw(name):
    return torch.load(_path(name), map_location='cpu', weights_only=False)


@pytest.mark.parametrize('name', ALL_CKPTS)
def test_real_legacy_checkpoint_loads(name):
    """Constructs and load_state_dict succeeds, with the weights actually restored."""
    raw = _raw(name)
    model, energy_grid, device = load_trained_model(_path(name), device='cpu')

    assert len(energy_grid) == len(raw['energy_grid'])
    # Weights restored, not merely a fresh model of the right shape -- and on the
    # layers each branch actually REBUILDS, not only on ones it never touches.
    for ref_key in RESTORATION_KEYS[name]:
        stored = raw['model_state_dict'][ref_key].cpu()
        live = _get_param(model, ref_key).detach().cpu()
        assert live.shape == stored.shape, f"{name}: {ref_key} shape drifted"
        assert torch.allclose(live, stored), (
            f"{name}: {ref_key} holds fresh random values, not the trained ones")


@pytest.mark.parametrize('name', HAMILTONIAN_CKPTS)
def test_argsless_defaults_match_what_these_checkpoints_were_trained_with(name):
    """Every remaining args-less fallback, asserted one by one.

    None of these raise when they regress: the load succeeds either way and the
    numbers move silently. solver_type is the demonstrated case -- see the module
    docstring and inference.py's comment at the solver_type= line.
    """
    args = _raw(name)['args']
    for absent in ('solver_type', 'use_geometry', 'enforce_hermiticity',
                   'complex_eta', 'use_log_outputs'):
        assert absent not in args, (
            f"fixture no longer predates {absent!r}; this test asserts a FALLBACK "
            "and would stop testing one if the arg were recorded")

    model, _, _ = load_trained_model(_path(name), device='cpu')

    assert model.solver_type == 'complex', (
        "the loader must fall back to the CONSTRUCTOR default, which is what "
        "training used; 'frobenius' here is the shipped defect that evaluated "
        "every recorded model through a solver it was not trained with")
    assert model.use_geometry is False
    assert model.enforce_hermiticity is True
    assert model.complex_eta == pytest.approx(1e-12)
    assert model.use_log_outputs is True
    # Recorded in args for these three, so this pins the arg is READ, not defaulted.
    assert model.conv_type == EXPECTED_CONV_TYPE[name] == str(args['conv_type']).lower()


def test_direct_baseline_checkpoint_rides_the_transformer_fallback():
    """The standard branch's own args-less fallback, plus the grid-width trap.

    `standard_2000x_4to10BP_5000epoch.pth` records no conv_type at all, so it
    depends on the standard branch defaulting to 'transformer' -- a 'gat' default
    would rebuild a different convolution and fail the strict load.

    HONEST LIMIT: this file is a tight-binding run whose stored grid (100) HAPPENS
    to equal its args['num_energy_points'] (100), so it cannot by itself witness
    the num_energy_points regression of inference.py ~258-263. No direct-model DFT
    checkpoint is tracked, so that case is covered synthetically by the test below.
    """
    raw = _raw(STANDARD_CKPT)
    args = raw['args']
    assert 'conv_type' not in args, "fixture no longer predates conv_type"

    model, energy_grid, _ = load_trained_model(_path(STANDARD_CKPT), device='cpu')
    assert model.conv_type == 'transformer'
    assert len(energy_grid) == len(raw['energy_grid'])
    assert model.dos_proj[3].weight.shape[0] == len(energy_grid)
    assert model.transmission_proj[3].weight.shape[0] == len(energy_grid)


def test_direct_model_readout_width_follows_the_stored_grid(tmp_path):
    """The documented regression, on the geometry that actually triggers it.

    A direct model trained on the 201-point DFT grid keeps args['num_energy_points']
    at its default 100 (that arg only drives the synthetic tight-binding source).
    Sizing the readout from the arg made every such checkpoint fail to load with a
    dos_proj/transmission_proj size mismatch.
    """
    from g3nat.models import DNATransportGNN

    grid = np.linspace(-1.0, 1.0, 201)
    model = DNATransportGNN(hidden_dim=16, num_layers=2, num_heads=2,
                            output_dim=len(grid), dropout=0.0,
                            conv_type='transformer')
    ckpt = tmp_path / 'direct_dft.pth'
    torch.save({'args': {'model_type': 'standard', 'hidden_dim': 16,
                         'num_layers': 2, 'num_heads': 2, 'dropout': 0.0,
                         'num_energy_points': 100},
                'energy_grid': grid,
                'model_state_dict': model.state_dict()}, ckpt)

    loaded, energy_grid, _ = load_trained_model(str(ckpt), device='cpu')
    assert len(energy_grid) == 201
    assert loaded.dos_proj[3].weight.shape[0] == 201
    assert torch.allclose(loaded.dos_proj[3].weight.detach(),
                          model.dos_proj[3].weight.detach())


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
