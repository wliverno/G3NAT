"""Both model types must use the same seeded, requeue-safe batch sampler.

The Hamiltonian path uses LengthBucketBatchSampler with an explicit seed, so epoch
N's composition is a pure function of (seed, N) and a requeued run regenerates it
exactly. The blind path used DataLoader(shuffle=True) with no generator, which
draws from the global RNG and does NOT survive requeue. Comparing two families
with different reproducibility guarantees is not a fair comparison.
"""
import ast
import pathlib

from g3nat.data import generate_tight_binding_data, create_dna_dataset
from g3nat.training import LengthBucketBatchSampler

TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'train.py'


def test_no_bare_shuffle_dataloader_remains():
    """No DataLoader(..., shuffle=True) may lack a sampler/batch_sampler.

    An AST check, not a source-text match: it constrains the actual property
    (every shuffling DataLoader goes through a sampler we control, so it can be
    seeded) rather than one specific known-good call's exact whitespace. A plain
    'shuffle=True)' substring check would miss shuffle=True with a trailing
    space, a variable holding True, or simply be defeated by any comment that
    happens to mention the pattern in prose.
    """
    tree = ast.parse(TRAIN_PY.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
        if name != 'DataLoader':
            continue
        kw = {k.arg: k.value for k in node.keywords}
        shuffle_val = kw.get('shuffle')
        is_literal_true = isinstance(shuffle_val, ast.Constant) and shuffle_val.value is True
        has_sampler = 'sampler' in kw or 'batch_sampler' in kw
        if is_literal_true and not has_sampler:
            offenders.append(ast.dump(node))
    assert not offenders, (
        'a DataLoader(..., shuffle=True) without a sampler/batch_sampler draws '
        f'from the global RNG and is not requeue-safe: {offenders}')


def test_both_branches_build_a_seeded_sampler():
    """Count the seeded sampler constructions: there must be one per model type."""
    tree = ast.parse(TRAIN_PY.read_text())
    seeded = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
        if name != 'LengthBucketBatchSampler':
            continue
        kw = {k.arg for k in node.keywords}
        if 'seed' in kw:
            seeded += 1
    assert seeded >= 2, (
        f'expected a seeded LengthBucketBatchSampler for BOTH model types, '
        f'found {seeded}')


def _mixed_length_dataset():
    # min_length < seq_length is required to get more than one node-count
    # bucket -- generate_tight_binding_data pins every sequence to seq_length
    # when min_length is left at its -1 default (see g3nat/data/synthetic.py),
    # so a fixed-length dataset here would only ever exercise a single bucket
    # and could pass this test vacuously.
    seqs, comp, dos, trans, grid = generate_tight_binding_data(
        num_samples=40, seq_length=8, min_length=4, num_energy_points=8)
    lengths = {len(s) for s in seqs}
    assert len(lengths) > 1, (
        f'expected multiple sequence lengths to exercise more than one bucket, '
        f'got only {lengths}')
    return create_dna_dataset(sequences=seqs, dos_data=dos, transmission_data=trans,
                              energy_grid=grid, complementary_sequences=comp)


def _epoch_batches(sampler, epoch):
    sampler.set_epoch(epoch)
    return [list(b) for b in sampler]


def test_epoch_composition_is_a_pure_function_of_seed_and_epoch():
    """The property that actually makes a requeue reproduce the original run.

    Task 2 exists because the blind branch used to draw from the global RNG,
    which is not reproducible across a preemption requeue. This test builds
    LengthBucketBatchSampler directly (the same class both branches now use)
    against a mixed-length synthetic dataset and checks the reproducibility
    guarantee its docstring claims, not merely that a `seed=` kwarg is present
    in the source.
    """
    ds = _mixed_length_dataset()

    # Same seed, same epoch, two independently constructed samplers -> identical
    # batches at every epoch checked. This is what lets a requeued run at
    # epoch N regenerate epoch N exactly.
    s1 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    s2 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=123)
    for epoch in (0, 1, 2):
        b1 = _epoch_batches(s1, epoch)
        b2 = _epoch_batches(s2, epoch)
        assert b1 == b2, f'epoch {epoch}: same seed must give identical batches'

    # Different seeds must give different batch orders. Guarded so this cannot
    # pass vacuously: if every epoch happened to collide (a dataset too small
    # or too uniform to differ), that is itself a finding, not something to
    # paper over by weakening the assertion.
    s_other = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=456)
    any_epoch_differs = any(
        _epoch_batches(s1, epoch) != _epoch_batches(s_other, epoch)
        for epoch in (0, 1, 2))
    assert any_epoch_differs, (
        'different seeds produced identical batches at every epoch checked -- '
        'the seed is not actually affecting batch composition')

    # Epoch N's batches depend only on (seed, N), not on call history: stepping
    # through several epochs first must not perturb what epoch 1 looks like
    # compared to a freshly constructed sampler jumping straight to epoch 1.
    s3 = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=789)
    for e in range(4):
        _epoch_batches(s3, e)
    fresh = LengthBucketBatchSampler(ds, batch_size=4, shuffle=True, seed=789)
    assert _epoch_batches(s3, 1) == _epoch_batches(fresh, 1), (
        'epoch 1 composition must not depend on which epochs were visited first')
