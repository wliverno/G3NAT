import torch
from g3nat.data import generate_tight_binding_data, create_dna_dataset
from g3nat.training import LengthBucketBatchSampler


def _tiny_dataset():
    seqs, comp, dos, trans, grid = generate_tight_binding_data(
        num_samples=24, seq_length=4, num_energy_points=8)
    return create_dna_dataset(sequences=seqs, dos_data=dos, transmission_data=trans,
                              energy_grid=grid, complementary_sequences=comp)


def _epoch_batches(sampler, epoch):
    sampler.set_epoch(epoch)
    return [list(b) for b in sampler]


def test_same_seed_same_epoch_identical_batches():
    ds = _tiny_dataset()
    s1 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    s2 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    assert _epoch_batches(s1, 0) == _epoch_batches(s2, 0)
    assert _epoch_batches(s1, 7) == _epoch_batches(s2, 7)


def test_different_epochs_differ_and_different_seeds_differ():
    ds = _tiny_dataset()
    s = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    assert _epoch_batches(s, 0) != _epoch_batches(s, 1)
    s_other = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=124)
    assert _epoch_batches(s, 0) != _epoch_batches(s_other, 0)


def test_seed_none_and_shuffle_false_paths_still_work():
    ds = _tiny_dataset()
    unseeded = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=None)
    assert len(list(iter(unseeded))) == len(unseeded)
    ordered = LengthBucketBatchSampler(ds, batch_size=4, shuffle=False)
    assert _epoch_batches(ordered, 0) == _epoch_batches(ordered, 5)


def test_set_epoch_survives_resume_semantics():
    # Epoch N's batches must depend only on (seed, N), not on iteration history --
    # that is what makes a requeue at epoch N reproduce the original epoch N.
    ds = _tiny_dataset()
    s1 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=99)
    for e in range(5):
        _epoch_batches(s1, e)
    fresh = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=99)
    assert _epoch_batches(s1, 5) == _epoch_batches(fresh, 5)
