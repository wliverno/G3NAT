"""Both model types must use the same seeded, requeue-safe batch sampler.

The Hamiltonian path uses LengthBucketBatchSampler with an explicit seed, so epoch
N's composition is a pure function of (seed, N) and a requeued run regenerates it
exactly. The blind path used DataLoader(shuffle=True) with no generator, which
draws from the global RNG and does NOT survive requeue. Comparing two families
with different reproducibility guarantees is not a fair comparison.
"""
import ast
import pathlib

TRAIN_PY = pathlib.Path(__file__).resolve().parents[2] / 'scripts' / 'train.py'


def test_no_bare_shuffle_dataloader_remains():
    src = TRAIN_PY.read_text()
    assert 'shuffle=True)' not in src.replace(
        'LengthBucketBatchSampler(train_dataset, args.batch_size,\n'
        '                                             shuffle=True,', ''), (
        'a DataLoader(..., shuffle=True) without a generator draws from the '
        'global RNG and is not requeue-safe')


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
