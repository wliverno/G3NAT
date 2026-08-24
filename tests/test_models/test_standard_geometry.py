"""The physics-blind baseline's optional SE(3)-invariant geometry channel.

WHY THIS EXISTS. The paper's comparison is between two models that share an encoder
and differ ONLY in their readout: the Hamiltonian model builds a matrix and solves
NEGF, the baseline pools node embeddings and emits spectra directly. That claim is
only true if every other input is available to both. Until now the baseline had no
geometry channel at all, so a geometry-on Hamiltonian model was being compared against
a baseline that could not see geometry -- an input advantage unrelated to the readout.

THE EQUIVALENCE TEST IS THE POINT. `test_fusion_matches_the_hamiltonian_model`
verifies the two implementations produce identical fused edge embeddings from identical
inputs and identical weights. The alternative -- extracting a shared module -- would
rename state-dict keys (`geom_encoder.0.weight` becomes nested) and break loading for
every existing geometry-trained checkpoint. A behavioural test is the stronger
guarantee anyway: shared code assumes equivalence, this checks it.
"""
import numpy as np
import pytest
import torch

from g3nat.models.standard import DNATransportGNN
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN

GEOM_DIM = 7
HIDDEN = 32
EDGE_FEATURES = 5          # 3 one-hot + directionality + coupling (standard.py:23)

#: Deliberately not round numbers, and deliberately not centred on zero: a
#: normalization bug that divides by the wrong row, or forgets the mean, stays
#: invisible against symmetric unit-scale fixtures.
NORM_STATS = {
    'backbone': {'mean': [0.5, -1.2, 3.4, 0.03, -2.8, 35.9, 0.1],
                 'std':  [0.2, 0.4, 0.05, 0.01, 0.3, 1.5, 0.05]},
    'hbond':    {'mean': [-0.1, -0.17, 0.0, -0.03, -1.25, -1.64, 0.2],
                 'std':  [0.15, 0.06, 0.02, 0.02, 0.1, 0.5, 0.08]},
}


def _edges(n_backbone=3, n_hbond=2, n_masked=2):
    """Edge fixtures with a known type split and a masked group.

    Returns (edge_attr_initial, edge_geom, mask). Column 0 of edge_attr flags
    backbone, column 1 flags H-bond (hamiltonian.py:_fuse_geometry). Masked edges
    carry deliberately WILD geometry -- if the mask is ever dropped, the resulting
    contribution is enormous and unmissable rather than a subtle shift.

    GEOMETRY IS DRAWN IN DISTRIBUTION, i.e. near each edge type's own mean and on
    the scale of its own std. An earlier version of this fixture used bare randn,
    which -- against stats whose means run to 3.4 with stds of 0.05 -- fed the
    encoder inputs tens of sigma out and made the near-zero-init test fail on a
    correct implementation. Real geometry sits near these means; the near-zero
    guarantee is a statement about in-distribution input and the fixture has to
    reflect that or it tests something else.
    """
    n = n_backbone + n_hbond + n_masked
    ea = torch.zeros(n, EDGE_FEATURES)
    ea[:n_backbone, 0] = 1.0
    ea[n_backbone:n_backbone + n_hbond, 1] = 1.0

    def draw(kind, count):
        m = torch.tensor(NORM_STATS[kind]['mean'])
        s = torch.tensor(NORM_STATS[kind]['std'])
        return (m + s * torch.randn(count, GEOM_DIM)).float()

    geom = torch.zeros(n, GEOM_DIM)
    if n_backbone:
        geom[:n_backbone] = draw('backbone', n_backbone)
    if n_hbond:
        geom[n_backbone:n_backbone + n_hbond] = draw('hbond', n_hbond)
    geom[n_backbone + n_hbond:] = 1e4
    mask = torch.ones(n, 1)
    mask[n_backbone + n_hbond:] = 0.0
    return ea, geom, mask


class _Data:
    """Minimal stand-in for the fields _fuse_geometry reads off a PyG Data."""

    def __init__(self, edge_geom, edge_geom_mask):
        self.edge_geom = edge_geom
        self.edge_geom_mask = edge_geom_mask


def _standard(use_geometry, **kw):
    return DNATransportGNN(hidden_dim=HIDDEN, num_layers=1, num_heads=4,
                           output_dim=11, use_geometry=use_geometry, **kw)


def test_default_off_adds_no_parameters_or_buffers():
    """Default must stay byte-identical so existing baseline checkpoints load.

    This is the guard that lets the feature ship without invalidating the six
    already-trained baseline models.
    """
    off = DNATransportGNN(hidden_dim=HIDDEN, num_layers=1, output_dim=11)
    on = _standard(True, geom_norm_stats=NORM_STATS)
    extra = set(on.state_dict()) - set(off.state_dict())
    assert not set(off.state_dict()) - set(on.state_dict()), (
        'turning geometry on must not REMOVE any key')
    assert extra == {'geom_mean', 'geom_std',
                     'geom_encoder.0.weight', 'geom_encoder.0.bias',
                     'geom_encoder.2.weight', 'geom_encoder.2.bias'}, extra
    assert not hasattr(off, 'geom_encoder')


def test_an_existing_geometry_free_checkpoint_still_loads():
    a = DNATransportGNN(hidden_dim=HIDDEN, num_layers=1, output_dim=11)
    b = DNATransportGNN(hidden_dim=HIDDEN, num_layers=1, output_dim=11)
    b.load_state_dict(a.state_dict())          # must not raise


def test_masked_edges_contribute_exactly_zero():
    """Contact edges, and any strand absent from the geometry cache, must be
    untouched. The fixture gives masked edges geometry of 1e4 so a dropped mask
    cannot hide inside numerical noise."""
    torch.manual_seed(0)
    m = _standard(True, geom_norm_stats=NORM_STATS)
    ea, geom, mask = _edges()
    proj = torch.zeros(ea.shape[0], HIDDEN)
    out = m._fuse_geometry(proj, ea, _Data(geom, mask))
    assert torch.allclose(out[-2:], torch.zeros(2, HIDDEN), atol=0.0)
    assert not torch.allclose(out[:3], torch.zeros(3, HIDDEN))


def test_backbone_and_hbond_edges_use_different_normalization_rows():
    """A bug that reads row 0 for every edge type is silent on symmetric fixtures.
    Feeding each edge type its OWN mean must therefore give exactly zero input to
    the encoder, and the two rows differ, so a single-row bug breaks one of them."""
    torch.manual_seed(0)
    m = _standard(True, geom_norm_stats=NORM_STATS)
    ea, _geom, mask = _edges(n_backbone=1, n_hbond=1, n_masked=0)
    geom = torch.stack([torch.tensor(NORM_STATS['backbone']['mean']),
                        torch.tensor(NORM_STATS['hbond']['mean'])]).float()
    proj = torch.zeros(2, HIDDEN)
    out = m._fuse_geometry(proj, ea, _Data(geom, mask))
    zero_in = m.geom_encoder(torch.zeros(1, GEOM_DIM))
    assert torch.allclose(out[0:1], zero_in, atol=1e-6)
    assert torch.allclose(out[1:2], zero_in, atol=1e-6)


def test_zero_std_does_not_produce_nan():
    """A geometry column that is constant across the dataset has std 0. The
    Hamiltonian implementation guards this; the baseline must too, or a single
    degenerate column poisons every edge embedding with nan."""
    stats = {k: {'mean': list(v['mean']), 'std': list(v['std'])}
             for k, v in NORM_STATS.items()}
    stats['backbone']['std'][2] = 0.0
    m = _standard(True, geom_norm_stats=stats)
    ea, geom, mask = _edges(n_backbone=2, n_hbond=0, n_masked=0)
    out = m._fuse_geometry(torch.zeros(2, HIDDEN), ea, _Data(geom, mask))
    assert torch.isfinite(out).all()


def test_geometry_contributes_near_zero_at_initialization():
    """Output-layer init is near-zero in the Hamiltonian model so that switching
    geometry on does not perturb a fresh model. Same requirement here, otherwise
    geom=on and geom=off runs start from materially different points and the
    factor is confounded with initialization."""
    torch.manual_seed(0)
    m = _standard(True, geom_norm_stats=NORM_STATS)
    ea, geom, mask = _edges(n_backbone=3, n_hbond=2, n_masked=0)
    proj = torch.zeros(5, HIDDEN)
    out = m._fuse_geometry(proj, ea, _Data(geom, mask))
    assert out.abs().max() < 0.1, out.abs().max()


def test_fusion_matches_the_hamiltonian_model():
    """THE SCIENTIFIC REQUIREMENT. Identical inputs and identical weights must give
    identical fused embeddings, so the two models differ only in their readout."""
    torch.manual_seed(0)
    std = _standard(True, geom_norm_stats=NORM_STATS)
    # The Hamiltonian model takes an energy_grid, not an output_dim.
    ham = DNATransportHamiltonianGNN(hidden_dim=HIDDEN, num_layers=1, num_heads=4,
                                     energy_grid=np.linspace(-1, 1, 11),
                                     use_geometry=True,
                                     geom_norm_stats=NORM_STATS)
    ham.geom_encoder.load_state_dict(std.geom_encoder.state_dict())
    assert torch.equal(ham.geom_mean, std.geom_mean)
    assert torch.equal(ham.geom_std, std.geom_std)

    ea, geom, mask = _edges()
    proj = torch.randn(ea.shape[0], HIDDEN)
    d = _Data(geom, mask)
    assert torch.allclose(std._fuse_geometry(proj, ea, d),
                          ham._fuse_geometry(proj, ea, d), atol=1e-6)


def test_forward_runs_with_geometry_and_changes_the_output():
    """End to end: geometry must reach the prediction. A channel that is wired up
    but never consumed would pass every test above."""
    torch.manual_seed(0)
    m = _standard(True, geom_norm_stats=NORM_STATS)
    with torch.no_grad():                      # break the near-zero init
        m.geom_encoder[-1].weight.normal_(std=0.5)
    m.eval()
    n_nodes = 4
    ea, geom, mask = _edges(n_backbone=3, n_hbond=0, n_masked=0)

    class D:
        pass
    d = D()
    d.x = torch.eye(4)[:n_nodes]
    d.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    d.edge_attr = ea
    d.batch = torch.zeros(n_nodes, dtype=torch.long)
    d.edge_geom = geom
    d.edge_geom_mask = mask
    dos_a, t_a = m(d)

    d.edge_geom = geom + 5.0
    dos_b, t_b = m(d)
    assert not torch.allclose(t_a, t_b), 'geometry did not reach the output'
    assert torch.isfinite(t_a).all() and torch.isfinite(dos_a).all()


def test_geometry_off_ignores_geometry_fields_entirely():
    """With the channel off, geometry present on the Data must change nothing."""
    torch.manual_seed(0)
    m = _standard(False)
    m.eval()
    ea, geom, mask = _edges(n_backbone=3, n_hbond=0, n_masked=0)

    class D:
        pass
    d = D()
    d.x = torch.eye(4)
    d.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    d.edge_attr = ea
    d.batch = torch.zeros(4, dtype=torch.long)
    d.edge_geom = geom
    d.edge_geom_mask = mask
    _, t_a = m(d)
    d.edge_geom = geom + 100.0
    _, t_b = m(d)
    assert torch.allclose(t_a, t_b)
