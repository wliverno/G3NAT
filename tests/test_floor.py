"""The one log10 floor (1e-25) applied to every DOS, LDOS and transmission,
everywhere: reference pipeline, pickle loader, both models, checkpoint loading.

Each test names the production change that makes it pass; each was watched to
fail first (2026-09-16).
"""
import os
import sys
import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELDOUT_G16 = os.path.join(REPO, 'DNADataset', 'validation_L16', 'pickles', 'gggggggggggggggg_run2.pkl')
TRAIN_8MER = os.path.join(REPO, 'pickle_files_v2', 'aaacgacg_run2.pkl')
CKPT_SMOOTH38 = os.path.join(REPO, 'outputs_v3', 'ham_ldosonly_n2_L4_nogeom_s3731635825',
                             'hamiltonian_pickle_model_best.pth')
needs_data = pytest.mark.skipif(not os.path.exists(HELDOUT_G16), reason='held-out pickles not present')
needs_ckpt = pytest.mark.skipif(not os.path.exists(CKPT_SMOOTH38), reason='campaign-v3 checkpoint not present')


def test_floor_constant_is_1e_minus_20():
    """g3nat/floor.py exists and defines the one number."""
    from g3nat.floor import LOG_FLOOR, LOG10_FLOOR
    assert LOG_FLOOR == 1e-25
    assert LOG10_FLOOR == -25.0


def test_log10_floored_clamp_reads_exactly_minus_25_below_the_floor():
    """hamiltonian.log10_floored with the one floor: hard clamp, exact -20."""
    from g3nat.floor import LOG_FLOOR
    from g3nat.models.hamiltonian import log10_floored
    x = torch.tensor([1e-30, 1e-25, 1e-10, 0.0])
    out = log10_floored(x, LOG_FLOOR, 'clamp')
    assert torch.allclose(out, torch.tensor([-25.0, -25.0, -10.0, -25.0]))


def test_hamiltonian_model_defaults_to_the_one_floor():
    """DNATransportHamiltonianGNN constructor defaults: log_floor=1e-25, floor_mode='clamp'."""
    from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
    m = DNATransportHamiltonianGNN(hidden_dim=8, num_layers=1, num_heads=1,
                                   energy_grid=np.linspace(-1, 1, 5))
    assert m.log_floor == 1e-25
    assert m.floor_mode == 'clamp'


@needs_data
def test_loader_floors_heldout_transmission_at_minus_25_exactly():
    """pickle loader: log10(max(T, 1e-25)); the G16 reference plateau (-25.5) reads -25."""
    from g3nat.data.pickle import load_single_pickle
    r = load_single_pickle(HELDOUT_G16)
    t = np.asarray(r['transmission'])
    assert t.min() == -25.0
    assert (t == -25.0).sum() > 20          # the part of the plateau below the floor


@needs_data
def test_loader_leaves_training_data_untouched():
    """No training value is below 1e-25, so the floor must not change it."""
    from g3nat.data.pickle import load_single_pickle
    r = load_single_pickle(TRAIN_8MER)
    t = np.asarray(r['transmission'])
    assert t.min() > -25.0
    assert abs(t.min() - (-17.2229)) < 1e-3   # unchanged raw log10 minimum of this record


@needs_data
def test_loader_floors_dos_and_ldos_with_the_same_number():
    """Same floor on DOS and per-residue LDOS (it never binds on real data, but it is applied)."""
    from g3nat.data.pickle import load_single_pickle
    r = load_single_pickle(HELDOUT_G16)
    assert np.asarray(r['dos']).min() >= -25.0
    assert np.asarray(r['ldos_residue']).min() >= -25.0
    # and the loader applies it rather than relying on the data: patch a tiny DOS in
    import pickle, tempfile
    d = pickle.load(open(HELDOUT_G16, 'rb'))
    d['DOS'] = np.full_like(np.asarray(d['DOS']), 1e-30)
    with tempfile.NamedTemporaryFile('wb', suffix='_run2.pkl', delete=False) as fh:
        pickle.dump(d, fh); tmp = fh.name
    try:
        r2 = load_single_pickle(tmp)
        assert np.all(np.asarray(r2['dos']) == -25.0)
    finally:
        os.remove(tmp)


@needs_ckpt
def test_loaded_checkpoint_evaluates_under_the_one_floor_not_its_recorded_one():
    """load_trained_model overrides the checkpoint's recorded (1e-38, 'smooth') with (1e-25, 'clamp')."""
    from g3nat.evaluation.inference import load_trained_model
    model, _, _ = load_trained_model(CKPT_SMOOTH38)
    assert model.log_floor == 1e-25
    assert model.floor_mode == 'clamp'


def test_direct_model_output_never_below_minus_25():
    """standard.DNATransportGNN clamps its log10 outputs at -20."""
    from g3nat.models.standard import DNATransportGNN
    from g3nat.graph.construction import sequence_to_graph
    from torch_geometric.data import Batch
    m = DNATransportGNN(hidden_dim=8, num_layers=1, num_heads=1, output_dim=5)
    # force the heads far below the floor
    for head in (m.dos_proj, m.transmission_proj):
        last = [mod for mod in head.modules() if isinstance(mod, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.bias.fill_(-50.0); last.weight.zero_()
    batch = Batch.from_data_list([sequence_to_graph('ACGT')])
    m.eval()
    with torch.no_grad():
        dos, t = m(batch)
    assert torch.all(dos >= -25.0) and torch.all(t >= -25.0)
    assert torch.all(dos == -25.0) and torch.all(t == -25.0)


def _tiny_params():
    # two atoms, one orbital each; left contact on atom 1, right on atom 2
    return {'Orbitals': np.array([1, 1]), 'Lsite': np.array([1]), 'Rsite': np.array([2]),
            'gammaL': 0.1, 'gammaR': 0.1, 'Energy': np.array([0.0, 500.0])}


def test_reference_transmission_is_floored_at_generation():
    """DNADataset/transmission.compute_transmission returns max(T, 1e-25)."""
    sys.path.insert(0, os.path.join(REPO, 'DNADataset'))
    from negf_common import LOG_FLOOR
    from transmission import compute_transmission
    assert LOG_FLOOR == 1e-25
    H0 = np.array([[0.0, 1e-8], [1e-8, 0.0]])   # T ~ 1.6e-13 at E=0 (kept), ~1.6e-25 at E=50: too close to 1e-25, use E=500 -> ~1.6e-29
    _, T = compute_transmission(_tiny_params(), H0, eta=0.0)
    assert T.min() == 1e-25
    assert T[0] > 1e-25                                    # the resonant point is untouched


def test_reference_dos_is_floored_and_dosatom_is_not():
    """DNADataset/dos_calc.compute_dos floors total DOS; the per-atom decomposition stays raw."""
    sys.path.insert(0, os.path.join(REPO, 'DNADataset'))
    from dos_calc import compute_dos
    H0 = np.array([[0.0, 1e-8], [1e-8, 0.0]])
    p = _tiny_params(); p['Energy'] = np.array([1e13])   # E=1e13: DOS ~ 3e-28, below the 1e-25 floor
    _, DOS, DOSAtom = compute_dos(p, H0, eta=0.0)
    assert DOS.min() == 1e-25
    assert DOSAtom.min() < 1e-25                            # raw, unfloored


def test_converter_applies_the_floor_to_T_and_DOS():
    """convert_to_pickle floors T and DOS on conversion (module-level LOG_FLOOR imported from negf_common)."""
    sys.path.insert(0, os.path.join(REPO, 'DNADataset'))
    import convert_to_pickle as c
    assert c.LOG_FLOOR == 1e-25
    import inspect
    src = inspect.getsource(c.build_record)
    assert 'np.maximum(np.asarray(T_vals' in src and 'np.maximum(np.asarray(DOS_vals' in src
