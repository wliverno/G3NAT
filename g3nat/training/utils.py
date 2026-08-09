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
    """BatchSampler that groups indices by number of DNA nodes to create uniform-size batches."""
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        # Build buckets: num_dna_nodes -> list of indices
        buckets = {}
        for idx in range(len(dataset)):
            data = dataset[idx]
            # For Subset, dataset[idx] yields underlying Data object
            num_dna = int(getattr(data, 'num_dna_nodes', data.x.size(0) - 2))
            buckets.setdefault(num_dna, []).append(idx)
        self.buckets = buckets
        # Precompute batches
        self._batches = self._build_batches()

    def _build_batches(self):
        batches = []
        for _, indices in self.buckets.items():
            if self.shuffle:
                rng = np.random.default_rng()
                rng.shuffle(indices)
            # chunk into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                batches.append(batch)
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        # Rebuild each epoch if shuffling
        if self.shuffle:
            self._batches = self._build_batches()
        for b in self._batches:
            yield b

    def __len__(self):
        return len(self._batches)


