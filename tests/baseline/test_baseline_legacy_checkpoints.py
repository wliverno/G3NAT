# tests/baseline/test_baseline_legacy_checkpoints.py
"""Golden DOS/T from the REAL tracked checkpoints, end to end.

The flag-by-flag assertions in
`tests/test_evaluation/test_legacy_checkpoint_roundtrip.py` each guard one
args-less fallback. This file guards the composition of all of them NUMERICALLY:
fixed sequences go through `load_trained_model` + `predict_sequence` and the
outputs are compared against stored fixtures. Any silent change of a loader
default -- solver_type, floor_mode, log_floor, complex_eta, conv_type, the
legacy-proj rebuild -- moves these numbers, whether or not anyone remembered to
add a flag assertion for it. (The solver_type defect that once shipped moved
near-resonance log10 T by up to 3.8 decades; nothing raised.)

The fixtures were generated on a compute node, never by hand:
    G3NAT_REGEN_BASELINES=1 python -m pytest tests/baseline/ -q
Regenerating is a deliberate act -- see `_util.py`. If these fail, the question
is what changed about legacy-checkpoint semantics, not what tolerance to use.

MEASURED mutation coverage of THIS file (job 38618836), because "catches any silent
default change" turned out to be too strong a claim:
  * `use_log_outputs` True -> False: CAUGHT (both Hamiltonian fixtures move).
  * legacy-proj weights left random: CAUGHT.
  * `solver_type` 'complex' -> 'frobenius': NOT caught here. On these short,
    off-resonance sequences the two solvers agree inside _util's tolerance --
    consistent with the documented median 3.2e-5 log10-T gap; the heavy tail lives
    at near-resonance energies these fixtures do not sample.
  * `enforce_hermiticity` True -> False: NOT caught, and cannot be: every tracked
    checkpoint is n_orb=1, where symmetrizing a 1x1 block is an exact no-op.
So the per-flag assertions in
`tests/test_evaluation/test_legacy_checkpoint_roundtrip.py` are NOT redundant with
this file -- they carry solver_type and enforce_hermiticity on their own.

Adding this file immediately paid for itself: the baseblind checkpoint turns out to
LOAD and then RAISE on the forward pass (see the last test here). Every load-only
test passed on it.

Tolerance caveat: comparison is at _util's ATOL/RTOL (1e-6/1e-5) on log10-scale
outputs. Cross-BLAS last-bit drift is far below that; a real default change is far
above it.
"""
import os

import pytest

from g3nat.evaluation.inference import load_trained_model, predict_sequence
from ._util import check_or_capture

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(REPO_ROOT, 'trained_models')

CKPTS = [
    'hamiltonian_DFT_gat_baseaware.pth',
    'hamiltonian_2000x_4to10BP_5000epoch.pth',
    'standard_2000x_4to10BP_5000epoch.pth',
]

# NOT in CKPTS, and the reason is a finding, not an omission -- see
# test_baseblind_checkpoint_loads_but_cannot_run below.
BASEBLIND_CKPT = 'hamiltonian_2000x_4to10BP_5000epoch_baseblind.pth'

# Fixed, inside the 4-10 bp range these checkpoints were trained on. Chosen to
# span composition: mixed, GC-only (the conducting extreme), AT-only.
SEQUENCES = ['ACGT', 'GCGCGCGC', 'ATATATAT']

_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def _revcomp(seq):
    return ''.join(_COMPLEMENT[b] for b in reversed(seq))


@pytest.mark.parametrize('ckpt', CKPTS)
def test_legacy_checkpoint_golden_forward_pass(ckpt):
    path = os.path.join(MODEL_DIR, ckpt)
    assert os.path.exists(path), f"tracked checkpoint missing from the clone: {path}"

    model, energy_grid, _ = load_trained_model(path, device='cpu')

    baseline = {'n_energy': len(energy_grid)}
    for seq in SEQUENCES:
        dos, trans = predict_sequence(
            model, seq, _revcomp(seq),
            left_contact_positions=0,
            right_contact_positions=len(seq) - 1,
            left_contact_coupling=0.1,
            right_contact_coupling=0.1,
        )
        baseline[f'{seq}_dos'] = dos
        baseline[f'{seq}_transmission'] = trans

    check_or_capture(f"legacy_ckpt_{ckpt.replace('.pth', '')}.pkl", baseline)


def test_baseblind_checkpoint_loads_but_cannot_run():
    """FINDING (2026-08-17, found by adding this file): the baseblind checkpoint
    LOADS into current code and then RAISES on the first forward pass.

    `load_trained_model` succeeds -- strict `load_state_dict` and all the
    args-less fallbacks are fine -- so every load-only test passes on it. But the
    current model builds `coupling_proj` as `Linear(3 * hidden_dim, ...)` because
    coupling is now BASE-AWARE (concat of both endpoint embeddings with the edge
    feature, hamiltonian.py ~552). This checkpoint is from the base-BLIND era: its
    stored `coupling_proj.0.weight` is [128, 256], one hidden_dim wide, and
    inference.py's legacy-proj rebuild faithfully reconstructs that 256-wide layer.
    So the forward pass feeds 768 features into a 256-wide layer and dies with
    `mat1 and mat2 shapes cannot be multiplied (Nx768 and 256x128)`.

    This is NOT fixable by widening the rebuild: the trained weights simply do not
    contain the 512 endpoint-embedding columns the current architecture needs.
    The checkpoint records a DIFFERENT model, not an older serialization of this
    one. Nothing in `docs/`, `scripts/` or `g3nat/` references this file (grep,
    2026-08-17), so no recorded result depends on it -- but "it still loads" must
    not be read as "it still reproduces its numbers". It cannot produce any.

    This test pins the current, honest state. If someone adds a genuine
    base-blind compatibility path, this test SHOULD start failing.
    """
    path = os.path.join(MODEL_DIR, BASEBLIND_CKPT)
    model, _, _ = load_trained_model(path, device='cpu')  # loading is fine
    with pytest.raises(RuntimeError, match='shapes cannot be multiplied'):
        predict_sequence(model, 'ACGT', _revcomp('ACGT'),
                         left_contact_positions=0, right_contact_positions=3,
                         left_contact_coupling=0.1, right_contact_coupling=0.1)
