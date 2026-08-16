"""Training utilities: length-bucketing batch sampler and init seeding.

The flat-index `split_dataset` that used to live here (sklearn train_test_split
over raw indices) was REMOVED on 2026-08-09: the DFT dataset has ~4 contact
variants per sequence, so a flat split leaks sequences across train/val. Use
`g3nat.data.splits.grouped_split`, which is what scripts/train.py uses.
"""

import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


def set_init_seed(seed: Optional[int]) -> bool:
    """Seed every RNG that affects model initialization. Returns True if seeded.

    This is deliberately SEPARATE from the train/val split seed. Before
    2026-07-31 `--split_seed` was the only seed in the codebase, so a sweep over
    "seeds" varied the held-out set while initialization was left to torch's
    global RNG. Every cross-seed spread measured that way mixes split variance
    with init variance and is not reproducible.

    Asking whether a loss term makes the recovered Hamiltonian more reproducible
    requires the opposite: hold the split fixed and vary init. That is only
    possible once the two are independent.

    `seed=None` touches no RNG, reproducing historical behaviour exactly, so
    existing runs and byte-identical comparisons are unaffected.
    """
    if seed is None:
        return False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return True


class LengthBucketBatchSampler(Sampler[List[int]]):
    """BatchSampler grouping indices by DNA-node count into uniform-size batches.

    seed=None reproduces the historical unseeded behavior (fresh OS entropy every
    epoch -- NOT reproducible; see determinism finding, 2026-08-13). With a seed,
    epoch N's batch composition is a pure function of (seed, N) via set_epoch, so a
    requeued run that calls set_epoch(N) regenerates the original epoch N exactly.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        buckets = {}
        for idx in range(len(dataset)):
            data = dataset[idx]
            num_dna = int(getattr(data, 'num_dna_nodes', data.x.size(0) - 2))
            buckets.setdefault(num_dna, []).append(idx)
        self.buckets = buckets
        self._batches = self._build_batches()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self):
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng((self.seed, self._epoch))

    def _build_batches(self):
        rng = self._rng() if self.shuffle else None
        batches = []
        for _, indices in sorted(self.buckets.items()):
            indices = list(indices)
            if rng is not None:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        if rng is not None:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        if self.shuffle:
            self._batches = self._build_batches()
        for b in self._batches:
            yield b

    def __len__(self):
        return len(self._batches)


