"""Sequence-grouped train/val splitting.

The DFT dataset has ~4 contact-variant samples per unique sequence. A flat-index
split leaks sequence identity into val and biases comparisons toward high-capacity
heads. Always split by sequence group.
"""
from typing import List, Tuple
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(groups: List, test_size: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split indices [0..len(groups)) so no group label appears on both sides.

    groups[i] is the group key (e.g. the sequence string) for dataset item i.
    Returns (train_indices, val_indices).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    indices = list(range(len(groups)))
    train_idx, val_idx = next(gss.split(indices, groups=groups))
    return train_idx.tolist(), val_idx.tolist()
