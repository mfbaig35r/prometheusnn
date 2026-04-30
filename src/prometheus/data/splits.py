"""Data splits and batch iteration for N-level training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DataSplit:
    """A split of data for training/validation/calibration."""

    embeddings: np.ndarray  # (N, embed_dim)
    labels_by_depth: dict[int, np.ndarray]  # {0: (N,), 1: (N,), ...}
    texts: list[str] | None = None

    def __len__(self) -> int:
        return len(self.embeddings)

    @property
    def num_levels(self) -> int:
        return len(self.labels_by_depth)

    def get_labels(self, depth: int) -> np.ndarray:
        return self.labels_by_depth[depth]

    @property
    def leaf_labels(self) -> np.ndarray:
        max_depth = max(self.labels_by_depth.keys())
        return self.labels_by_depth[max_depth]


class BatchIterator:
    """Yields batches of (embeddings, label dicts) for training."""

    def __init__(
        self,
        data: DataSplit,
        batch_size: int = 64,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_state)

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch = {"embeddings": self.data.embeddings[batch_idx]}
            for depth, labels in self.data.labels_by_depth.items():
                batch[depth] = labels[batch_idx]
            yield batch
