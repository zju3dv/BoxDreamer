"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 19:52:10
Description: Concatenated dataset implementation for BoxDreamer.
"""

import bisect
from typing import List, Tuple, Optional
from torch.utils.data import ConcatDataset
from .base import BoxDreamerBaseDataset


class BoxDreamerDynamicConcatDataset(ConcatDataset):
    """Concatenated dataset for BoxDreamer training data that supports dynamic
    length."""

    def __init__(self, datasets: List[BoxDreamerBaseDataset]):
        """Initialize the concatenated dataset.

        Args:
            datasets: List of BoxDreamerBaseDataset instances
        """
        super().__init__(datasets)
        self.datasets = datasets

        # Determine length constraints across all datasets
        self.max_length = -1
        self.min_length = 1e9

        for dataset in datasets:
            if hasattr(dataset, "max_length"):
                self.max_length = max(self.max_length, dataset.max_length)
                self.min_length = min(self.min_length, dataset.min_length)
            else:
                self.max_length = max(self.max_length, dataset.length)
                self.min_length = min(self.min_length, dataset.length)

        # Check if dynamic length is supported
        self._dynamic_length = False
        for dataset in datasets:
            if getattr(dataset, "dynamic_length", False):
                self._dynamic_length = True
                break

        if self._dynamic_length:
            # Ensure all datasets support dynamic length
            for dataset in datasets:
                assert getattr(
                    dataset, "dynamic_length", False
                ), "All datasets must support dynamic length when one does"

    @property
    def dynamic_length(self) -> bool:
        """Return whether the dataset supports dynamic length."""
        return self._dynamic_length

    def __getitem__(self, idx_len: Tuple[int, Optional[int]]):
        """Get an item from the appropriate dataset.

        Supports both standard indexing and dynamic length indexing.
        """
        if isinstance(idx_len, tuple):
            idx, batch_len = idx_len
        else:
            idx, batch_len = idx_len, None

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Handle dynamic length if supported
        if batch_len is not None:
            if getattr(self.datasets[dataset_idx], "dynamic_length", False):
                return self.datasets[dataset_idx].__getitem__((sample_idx, batch_len))
            else:
                # If this specific dataset doesn't support dynamic length,
                # use its default length (requires drop_last=True in batch sampler)
                return self.datasets[dataset_idx].__getitem__(sample_idx)
        else:
            return self.datasets[dataset_idx].__getitem__(sample_idx)
