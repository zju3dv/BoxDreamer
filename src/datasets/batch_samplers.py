"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 19:52:24
Description: Batch samplers for BoxDreamer datasets.
"""

import math
import random
from torch.utils.data import BatchSampler, Sampler
from .base import BoxDreamerBaseDataset


class DynamicBatchSampler(BatchSampler):
    """Batch sampler that handles variable-length sequences.

    Note: This sampler requires careful integration with PyTorch Lightning.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool,
        dataset: BoxDreamerBaseDataset,
    ):
        """Initialize the dynamic batch sampler.

        Args:
            sampler: Base sampler
            batch_size: Number of samples per batch (for longest sequences)
            drop_last: Whether to drop the last incomplete batch
            dataset: The dataset to sample from
        """
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        # For dynamic length training, batch_size means the batch size at max length
        self.drop_last = drop_last
        self.dataset = dataset

    def __iter__(self):
        """Yield batches with dynamic sequence lengths."""
        batch = []
        length = None
        local_batch_size = self.batch_size

        for idx in self.sampler:
            # Dynamically determine sequence length and adjust batch size
            if getattr(self.dataset, "dynamic_length", False) and length is None:
                min_len, max_len = self.dataset.min_length, self.dataset.max_length
                full_batch_size = (
                    self.batch_size * max_len
                )  # GPU memory usage at max length
                length = random.randint(min_len, max_len)
                # Adjust batch size to maintain similar memory usage
                local_batch_size = full_batch_size // length

            # Add sample with sequence length
            batch.append((idx, length))

            # Yield batch when full
            if len(batch) == local_batch_size:
                yield batch
                batch = []
                local_batch_size = self.batch_size
                length = None

        # Handle last batch if not dropping it
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        """Return the number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)
