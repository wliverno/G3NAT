import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from g3nat.data.datasets import create_dna_dataset
from g3nat.models.hamiltonian import DNATransportHamiltonianGNN
from g3nat.training.config import TrainingConfig
from g3nat.training.trainer import Trainer, _center


def _dataset(with_ldos):
    # UPPERCASE is required: BASE_FEATURES is keyed 'A'/'T'/'G'/'C' and
    # g3nat/graph/construction.py does no case normalization. The real pipeline
    # only works because load_single_pickle calls .upper() (pickle.py:37).
    seqs = ['AAAC', 'AAAG']
    n_e = 11
    egrid = np.linspace(-1, 1, n_e)
    ldos_data = None
    if with_ldos:
        arrays = [np.full((8, n_e), -1.0) for _ in seqs]
        ldos_data = {'residue': arrays, 'base_only': arrays}
    # dos and transmission targets must be distinct (not both zero/identical),
    # otherwise a bug that swaps dos_target and transmission_target inside
    # _compute_losses would be invisible to every test in this file.
    return create_dna_dataset(
        sequences=seqs,
        dos_data=np.full((2, n_e), -1.0),
        transmission_data=np.full((2, n_e), -3.0),
        energy_grid=egrid,
        complementary_sequences=['GTTT', 'CTTT'],
        ldos_data=ldos_data,
    )


def _trainer(loss_a=1.0, loss_b=0.0, shape_loss=False):
    # shape_loss defaults to False HERE (not TrainingConfig's own True default)
    # so every pre-existing test below, which calls this helper without
    # mentioning shape_loss, keeps exercising the raw-magnitude formula it was
    # written against. Tests that care about shape_loss pass it explicitly.
    model = DNATransportHamiltonianGNN(
        energy_grid=np.linspace(-1, 1, 11), hidden_dim=16, num_layers=1, n_orb=1
    )
    config = TrainingConfig(num_epochs=1, batch_size=2, device='cpu',
                            loss_a=loss_a, loss_b=loss_b, shape_loss=shape_loss)
    return Trainer(model, config)


def test_config_defaults_reproduce_todays_weights():
    config = TrainingConfig()
    assert config.loss_a == 1.0
    assert config.loss_b == 0.0
    # shape_loss=False (absolute comparison) is the default, reverted 2026-07-30:
    # the basis-size justification for shape_loss=True was quantitatively wrong
    # (predicted 2.55 decades vs a measured 0.2199, which instead matches
    # level-counting in the HOMO+/-1eV window at 0.2005) -- see
    # TrainingConfig.shape_loss's docstring. shape_loss=True stays reachable.
    assert config.shape_loss is False


def test_b_zero_total_equals_dos_plus_transmission():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=1.0, loss_b=0.0)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    expected = losses['dos'] + losses['transmission']
    torch.testing.assert_close(losses['total'], expected)


def test_b_zero_does_not_compute_an_ldos_term():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.0)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert losses['ldos'] is None


def test_b_zero_never_enters_the_ldos_path(monkeypatch):
    # `losses['ldos'] is None` alone cannot distinguish "skipped by branch"
    # from "computed then discarded" -- both yield None. Detonate the LDOS
    # path so that entering it at all is a test failure, then confirm b=0
    # still composes a loss. The dataset HAS a target here, so only loss_b
    # can be what gates the path.
    # Patch target moved 2026-08-16: the trainer now calls the model method
    # site_ldos_log10_recorded (so LDOS floor semantics and floor diagnostics
    # come from the model), not the module-level site_ldos_log10.
    from g3nat.models.hamiltonian import DNATransportHamiltonianGNN

    def _boom(*args, **kwargs):
        raise AssertionError(
            "the LDOS path must not be entered when loss_b == 0")

    monkeypatch.setattr(DNATransportHamiltonianGNN, "site_ldos_log10_recorded",
                        _boom)

    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.0)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert losses['ldos'] is None
    assert torch.isfinite(losses['total'])


def test_b_positive_composes_all_three_terms():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=2.0, loss_b=0.25)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert losses['ldos'] is not None
    expected = (2.0 * losses['transmission']
                + 0.25 * losses['ldos']
                + 0.75 * losses['dos'])
    torch.testing.assert_close(losses['total'], expected)


def test_unweighted_diagnostic_ignores_the_weights():
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=3.0, loss_b=0.5)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    torch.testing.assert_close(
        losses['dos_t_unweighted'], losses['dos'] + losses['transmission']
    )


def test_b_positive_without_a_target_raises_a_named_error():
    dataset = _dataset(with_ldos=False)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.5)

    dos_pred, trans_pred = trainer.model(batch)
    with pytest.raises(ValueError, match="no LDOS target"):
        trainer._compute_losses(batch, dos_pred, trans_pred)


def test_b_positive_with_a_model_lacking_ldos_raises():
    # Only the Hamiltonian model exposes self.ldos as a forward side effect.
    # A model without it must fail loudly rather than silently degrading to a
    # two-term loss. Simulate that by removing the attribute after forward.
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.5)

    dos_pred, trans_pred = trainer.model(batch)
    del trainer.model.ldos

    # Distinctive substring so this cannot be satisfied by the batch-has-no-
    # target error above: that message uses uppercase "LDOS target" and never
    # contains "'ldos' attribute".
    with pytest.raises(ValueError, match="exposes no 'ldos' attribute"):
        trainer._compute_losses(batch, dos_pred, trans_pred)


def test_b_zero_trains_on_data_without_any_ldos_target():
    dataset = _dataset(with_ldos=False)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.0)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert torch.isfinite(losses['total'])


# --- shape_loss: opt-in shared-offset shape comparison; absolute is the default ---
#
# Reverted 2026-07-30: the basis-size justification for shape_loss=True (DOS/LDOS
# magnitude "set by basis size") was quantitatively wrong -- the training window is
# HOMO+/-1eV, which holds only frontier levels, and the measured DOS offset matches
# level-counting there, not basis size. Absolute comparison is the default; see
# TrainingConfig.shape_loss for the derivation. shape_loss=True remains available
# and, as of this revert, shares ONE offset between DOS and LDOS (see
# Trainer._compute_losses) instead of centering them independently, so it no longer
# deletes the LDOS localization signal. Transmission is a dimensionless,
# basis-size-independent observable and must match absolutely, so it is never
# centered under either setting.


def test_shape_loss_ignores_constant_dos_offset():
    # Centering removes a constant offset: adding a constant to the prediction
    # must leave the shape loss unchanged but change the raw loss.
    dataset = _dataset(with_ldos=False)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_b=0.0, shape_loss=True)

    dos_pred, trans_pred = trainer.model(batch)
    losses_orig = trainer._compute_losses(batch, dos_pred, trans_pred)
    losses_offset = trainer._compute_losses(batch, dos_pred + 5.0, trans_pred)

    torch.testing.assert_close(losses_offset['dos_shape'], losses_orig['dos_shape'])
    assert abs(losses_offset['dos'].item() - losses_orig['dos'].item()) > 1e-6


def test_ldos_shape_centers_per_sequence_not_per_site(monkeypatch):
    # LDOS centering must be per SEQUENCE (dims=(1, 2), site AND energy
    # jointly), never per SITE (dims=2 alone). Construct a target whose sites
    # differ by known constant offsets (site i holds value i, constant across
    # energy) and a prediction that flattens all site-to-site structure to the
    # site average. Joint per-sequence centering must still penalise this --
    # that is precisely the quantity the LDOS term exists to constrain. A
    # per-site centering would remove each site's own mean independently and
    # score the flattened prediction as PERFECT, which is the regression this
    # test exists to catch -- so the second assertion below confirms that
    # failure mode is real, not a strawman.
    n_sites, n_energy = 4, 5
    site_offsets = torch.tensor([0.0, 1.0, 2.0, 3.0])
    # Batch size must match the real batch (2 sequences) or _compute_losses'
    # per-sequence offset, shape [2, 1], broadcasts against a [1, ...] tensor and
    # torch warns that the result is not what either side intends.
    n_batch = 2
    target = site_offsets.view(1, n_sites, 1).expand(n_batch, n_sites, n_energy).clone()
    flattened_pred = torch.full((n_batch, n_sites, n_energy), site_offsets.mean().item())

    # Integration check: force the trainer's real LDOS loss path onto this
    # pred/target pair, so the assertion exercises the actual dims=(1, 2) call
    # inside _compute_losses, not just the _center helper in isolation.
    import g3nat.training.trainer as trainer_mod
    monkeypatch.setattr(
        trainer_mod.Trainer, "_ldos_pred_and_target",
        lambda self, batch, batch_size: (flattened_pred, target)
    )

    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=1.0, loss_b=1.0, shape_loss=True)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    assert losses['ldos_shape'].item() > 1e-6, (
        "joint per-sequence centering (dims=(1, 2)) must still penalise a "
        "flattened site profile"
    )

    # Sanity check on the test itself: per-site centering (dims=2 only) really
    # would erase the site-to-site profile and score this as a perfect match.
    per_site_loss = torch.nn.HuberLoss()(_center(flattened_pred, 2), _center(target, 2))
    assert per_site_loss.item() < 1e-9, (
        "per-site centering was expected to score the flattened prediction as "
        "perfect -- if this fails, the test no longer exercises the regression "
        "it is named for"
    )


def test_transmission_never_centered():
    # Transmission is the absolute observable and must never be centered,
    # under either shape_loss setting: adding a constant to the transmission
    # prediction must change the loss both when shape_loss=True and False.
    dataset = _dataset(with_ldos=False)
    batch = Batch.from_data_list([dataset[0], dataset[1]])

    for shape_flag in (True, False):
        trainer = _trainer(loss_b=0.0, shape_loss=shape_flag)
        dos_pred, trans_pred = trainer.model(batch)

        losses_orig = trainer._compute_losses(batch, dos_pred, trans_pred)
        losses_offset = trainer._compute_losses(batch, dos_pred, trans_pred + 5.0)

        assert abs(losses_offset['transmission'].item()
                   - losses_orig['transmission'].item()) > 1e-6, shape_flag
        assert abs(losses_offset['total'].item()
                   - losses_orig['total'].item()) > 1e-6, shape_flag


def test_shape_loss_false_reproduces_pre_shape_total():
    # shape_loss=False (the default, both now and pre-2026-07-30) must compose
    # total = a*T + b*LDOS + (1-b)*DOS using the RAW (uncentered) dos/ldos
    # terms, byte-identical to before shape_loss existed.
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=1.0, loss_b=0.25, shape_loss=False)

    dos_pred, trans_pred = trainer.model(batch)
    losses = trainer._compute_losses(batch, dos_pred, trans_pred)

    expected = (1.0 * losses['transmission']
                + 0.25 * losses['ldos']
                + 0.75 * losses['dos'])
    torch.testing.assert_close(losses['total'], expected)


def test_shape_loss_shares_offset_between_dos_and_ldos():
    # Regression test for the coupled-offset fix (2026-07-30 revert). Under the
    # OLD, now-retracted implementation, DOS and LDOS shape terms were each
    # centered by their OWN independent mean, so shifting only the DOS
    # prediction left ldos_shape completely unchanged -- that independence is
    # exactly the defect that deletes <J_pred - J_target>, the LDOS
    # localization signal (sum_i LDOS_i = DOS implies mean_i log10 LDOS_i =
    # log10(DOS) - log10(n_sites) - J; see TrainingConfig.shape_loss and
    # Trainer._compute_losses).
    #
    # Under the fix, DOS and LDOS share ONE offset (the median DOS residual),
    # so shifting DOS also shifts the offset subtracted from LDOS, and
    # ldos_shape must change even though ldos_pred/ldos_target themselves
    # never moved. This test FAILS under the old independent-offset
    # implementation, where ldos_shape would come out unchanged.
    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0], dataset[1]])
    trainer = _trainer(loss_a=1.0, loss_b=0.5, shape_loss=True)

    dos_pred, trans_pred = trainer.model(batch)
    losses_orig = trainer._compute_losses(batch, dos_pred, trans_pred)
    losses_offset = trainer._compute_losses(batch, dos_pred + 5.0, trans_pred)

    # DOS shape is unaffected: the shared offset is a median of the DOS
    # residual, which shifts by the exact same added constant, so it cancels.
    torch.testing.assert_close(losses_offset['dos_shape'], losses_orig['dos_shape'])
    # LDOS shape MUST change: ldos_pred/ldos_target never moved, but the
    # shared offset moved by +5.0 (because it is now derived from DOS), so the
    # effective LDOS residual shifted too.
    assert abs(losses_offset['ldos_shape'].item()
               - losses_orig['ldos_shape'].item()) > 1e-6


def test_val_ldos_localization_gap_positive_when_prediction_more_localized(monkeypatch):
    # val_ldos_localization_gap = <log10 dos_pred - log10 dos_target> -
    # <log10 ldos_pred - log10 ldos_target> (Trainer._ldos_agreement). Hold DOS
    # identical on both sides (residual exactly 0) and make the LDOS
    # prediction concentrate weight onto one site -- most sites near a very
    # negative log10 value, one site large -- against a uniform target, i.e. a
    # prediction that is MORE log-localized than the target. The gap must
    # come out positive.
    n_sites, n_energy = 4, 1
    ldos_target = torch.zeros(1, n_sites, n_energy)
    ldos_pred = torch.tensor([2.0, -3.0, -3.0, -3.0]).view(1, n_sites, n_energy)

    dataset = _dataset(with_ldos=True)
    batch = Batch.from_data_list([dataset[0]])
    trainer = _trainer(loss_b=0.0, shape_loss=False)
    # A real forward pass populates self.model.ldos, which _ldos_agreement's
    # hasattr guard requires; the monkeypatch below replaces the value it
    # feeds into the loss, so the forward pass's own LDOS content is unused.
    trainer.model(batch)

    import g3nat.training.trainer as trainer_mod
    monkeypatch.setattr(
        trainer_mod.Trainer, "_ldos_pred_and_target",
        lambda self, batch, batch_size: (ldos_pred, ldos_target)
    )

    # dos_pred == batch.dos exactly (both -1.0, see _dataset's dos_data), so
    # the DOS term of the gap is exactly zero and the sign is decided purely
    # by the LDOS asymmetry constructed above.
    dos_pred = torch.full((1, 11), -1.0)

    _, _, gap = trainer._ldos_agreement(batch, dos_pred)
    assert gap > 0
