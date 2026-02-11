"""Training utilities including batch samplers and dataset splitting."""

import numpy as np
from typing import List
from torch.utils.data import Sampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


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


def split_dataset(dataset, train_split: float = 0.8):
    """Split dataset into training and validation sets."""
    dataset_size = len(dataset)
    train_indices, val_indices = train_test_split(
        range(dataset_size),
        test_size=1-train_split,
        random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    return train_dataset, val_dataset
