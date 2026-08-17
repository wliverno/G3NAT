"""Optional modules must not perturb the shared core's init RNG stream.

Toggling an optional feature flag (geometry, per-base onsite) must not change a
single always-present parameter at a fixed seed, and must not change the draws of
optional modules constructed before it. This is what makes a factorial
across those flags comparable: differences between arms are the feature, not a
reshuffled initialization.

The test must FAIL on a wrong construction order; that was verified by temporarily
moving the geometry-encoder construction above the conv layers (mutation check).
"""
import numpy as np
import torch

from g3nat.models.hamiltonian import DNATransportHamiltonianGNN

OPTIONAL_PREFIXES = ('geom_encoder', 'geom_mean', 'geom_std', 'onsite_baseline')


def _core_weights(model):
    return {k: v.clone() for k, v in model.state_dict().items()
            if not k.startswith(OPTIONAL_PREFIXES)}


def _build(**kw):
    torch.manual_seed(1234)
    grid = np.linspace(-1, 1, 8)
    stats = {'backbone': {'mean': np.zeros(7), 'std': np.ones(7)},
             'hbond': {'mean': np.zeros(7), 'std': np.ones(7)}}
    if kw.get('use_geometry'):
        kw['geom_norm_stats'] = stats
    return DNATransportHamiltonianGNN(hidden_dim=16, num_layers=1, num_heads=2,
                                      energy_grid=grid, n_orb=1, **kw)


def test_geometry_flag_does_not_shift_core_init():
    """Plain case: no other optional module present."""
    off = _core_weights(_build(use_geometry=False))
    on = _core_weights(_build(use_geometry=True))
    assert off.keys() == on.keys()
    for k in off:
        assert torch.equal(off[k], on[k]), f"core param {k} differs when geometry toggles"


def test_geometry_flag_does_not_shift_core_init_with_per_base_onsite():
    """Discriminating case: the per-base onsite head is an RNG consumer that is
    NOT the geometry encoder, so a bad ordering can move it."""
    m_off = _build(use_geometry=False, per_base_onsite=True)
    m_on = _build(use_geometry=True, per_base_onsite=True)

    # The optional modules really exist in both models (otherwise this test is vacuous).
    assert hasattr(m_off, 'onsite_baseline') and hasattr(m_on, 'onsite_baseline')
    assert not hasattr(m_off, 'geom_encoder')
    assert hasattr(m_on, 'geom_encoder')

    off, on = _core_weights(m_off), _core_weights(m_on)
    assert off.keys() == on.keys()
    assert len(off) > 0
    for k in off:
        assert torch.equal(off[k], on[k]), f"core param {k} differs when geometry toggles"

    # Optional modules constructed BEFORE geometry must also be untouched by the
    # geometry flag: structured onsite is built first, geometry last.
    assert torch.equal(m_off.onsite_baseline, m_on.onsite_baseline), \
        "onsite_baseline draw shifted when the geometry flag toggled"
