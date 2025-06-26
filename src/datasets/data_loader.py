"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 19:46:14
Description: Data loader factory for BoxDreamer datasets.
"""

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from omegaconf import DictConfig, OmegaConf
from .batch_samplers import DynamicBatchSampler


def make_dataloader(dataset: Dataset, cfg: DictConfig) -> DataLoader:
    """Create a DataLoader with the specified configuration.

    Args:
        dataset: The dataset to load
        cfg: Configuration for the DataLoader

    Returns:
        Configured DataLoader instance
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Create appropriate sampler based on shuffle setting
    if cfg.shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    # Use DynamicBatchSampler if dynamic batching is required
    if getattr(dataset, "dynamic_length", False):
        batch_sampler = DynamicBatchSampler(
            sampler, cfg.batch_size, cfg.drop_last, dataset
        )
        # Since PyTorch Lightning 2.5, the shuffle setting will be correctly handled
        # For older versions: https://github.com/Lightning-AI/pytorch-lightning/issues/20326

        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
        )

    return dataloader
