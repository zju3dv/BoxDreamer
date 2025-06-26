"""
Author: Yuanhong Yu
Date: 2025-03-17
Description: Lightning data module for BoxDreamer handling multiple datasets.
"""

from loguru import logger
from pytorch_lightning import LightningDataModule
from typing import List, Dict, Any, Optional, Union, Callable
import os.path as osp

from src.datasets.linemod import LINEMOD_BoxDreamer
from src.datasets import make_dataloader, BoxDreamerDynamicConcatDataset
from src.datasets.co3d import Co3DV2Dataset
from src.datasets.onepose import OnePoseDataset
from src.datasets.objaverse import ObjaverseBoxDreamerDataset
from src.datasets.linemodo import LINEMOD_Occulusion
from src.datasets.ycbv import YCBV_BoxDreamer
from src.datasets.moped import MOPED_BoxDreamer


class BoxDreamerDataModule(LightningDataModule):
    """Data module that manages multiple BoxDreamer datasets for training,
    validation and testing.

    This module supports various dataset types including:
    - CO3D
    - LINEMOD
    - LINEMOD-Occlusion
    - OnePose
    - Objaverse
    - YCBV
    - MOPED
    """

    DATASET_REGISTRY = {
        "CO3D": Co3DV2Dataset,
        "LINEMOD": LINEMOD_BoxDreamer,
        "LINEMODO": LINEMOD_Occulusion,
        "OnePose": OnePoseDataset,
        "Objaverse": ObjaverseBoxDreamerDataset,
        "YCBV": YCBV_BoxDreamer,
        "MOPED": MOPED_BoxDreamer,
    }

    def __init__(self, *args, **kwargs):
        """Initialize the data module with dataset configurations.

        Args:
            dataset_name: List of dataset names for [train, val, test]
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster data transfer to GPU
            Other dataset-specific configurations
        """
        super().__init__()

        self.dataset_name = kwargs["dataset_name"]

        # Get dataset configurations for each stage
        self.training_configs = [kwargs[key] for key in self.dataset_name[0]]
        self.val_configs = [kwargs[key] for key in self.dataset_name[1]]
        self.test_configs = [kwargs[key] for key in self.dataset_name[2]]

        # Data loader parameters
        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.pin_memory = kwargs["pin_memory"]

        # Configure loader parameters for different stages
        self._configure_loader_params()

        # Initialize dataset containers
        self.train_data = None
        self.data_vals = None
        self.data_tests = None

    def _configure_loader_params(self):
        """Set up data loader parameters for train, validation and test
        stages."""
        common_params = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": True,
        }

        self.train_loader_params = {**common_params, "shuffle": True, "drop_last": True}

        self.val_loader_params = {**common_params, "shuffle": True, "drop_last": True}

        self.test_loader_params = {
            **common_params,
            "shuffle": False,
            "drop_last": False,
        }

    def prepare_data(self):
        """Prepare data before setup (e.g., download, preprocessing).

        This method is called once for all GPUs.
        """
        pass

    def _create_dataset(self, dataset_config: Dict[str, Any], stage: str) -> Any:
        """Create a dataset instance based on configuration.

        Args:
            dataset_config: Dataset configuration dictionary with 'name' and 'config' keys
            stage: Dataset stage ('train', 'val', 'test')

        Returns:
            Dataset instance
        """
        dataset_name = dataset_config["name"]
        dataset_config = dataset_config["config"]

        if dataset_name not in self.DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_class = self.DATASET_REGISTRY[dataset_name]

        # Handle different constructor signatures
        if dataset_name == "CO3D":
            return dataset_class(dataset_config, stage)
        elif dataset_name == "LINEMOD":
            return dataset_class(split=stage, config=dataset_config)
        else:
            return dataset_class(dataset_config, split=stage)

    def _setup_datasets(self, configs: List[Dict[str, Any]], stage: str) -> List[Any]:
        """Set up datasets from a list of configurations.

        Args:
            configs: List of dataset configurations
            stage: Dataset stage ('train', 'val', 'test')

        Returns:
            List of dataset instances
        """
        datasets = []

        for config in configs:
            result = self._create_dataset(config, stage)

            # Handle the case where a dataset setup returns multiple datasets
            if isinstance(result, list):
                datasets.extend(result)
            else:
                datasets.append(result)

        return datasets

    def _log_dataset_sizes(self, datasets, stage: str):
        """Log the size of datasets.

        Args:
            datasets: Dataset or list of datasets
            stage: Stage name for logging
        """
        if isinstance(datasets, list):
            for i, dataset in enumerate(datasets):
                logger.info(f"{stage} dataset {i}: {len(dataset)} samples")
        else:
            logger.info(f"{stage} dataset: {len(datasets)} samples")

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for the given stage.

        Args:
            stage: 'fit' for training/validation or 'test' for testing
        """
        if stage == "fit":
            # Set up training datasets
            train_datasets = self._setup_datasets(self.training_configs, "train")

            # Create concatenated dataset if there are multiple training datasets
            if len(train_datasets) > 1:
                self.train_data = BoxDreamerDynamicConcatDataset(train_datasets)
            else:
                self.train_data = train_datasets[0]

            # Set up validation datasets
            self.data_vals = self._setup_datasets(self.val_configs, "val")

            # Log dataset sizes
            self._log_dataset_sizes(self.train_data, "Train")
            self._log_dataset_sizes(self.data_vals, "Validation")

        elif stage == "test" or stage is None:
            # Set up test datasets
            self.data_tests = self._setup_datasets(self.test_configs, "test")

            # Log dataset sizes
            self._log_dataset_sizes(self.data_tests, "Test")

        logger.info(
            f"DataModule setup completed for '{stage}' stage. Datasets: {self.dataset_name}"
        )

    def train_dataloader(self):
        """Create data loader for training."""
        return make_dataloader(self.train_data, self.train_loader_params)

    def val_dataloader(self):
        """Create data loaders for validation."""
        return [
            make_dataloader(data, self.val_loader_params) for data in self.data_vals
        ]

    def test_dataloader(self):
        """Create data loaders for testing."""
        return [
            make_dataloader(data, self.test_loader_params) for data in self.data_tests
        ]
