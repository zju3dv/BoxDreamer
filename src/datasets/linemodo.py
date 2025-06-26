"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 14:52:08
Description: This module provides a dataset class for the LINEMOD-Occulusion dataset
. It is designed to be compatible with the BoxDreamerBaseDataset for seamless data loading and processing.
Note: This dataset is only used for test

"""


import os
import numpy as np
from src.lightning.utils.vis.vis_utils import *
from src.utils.log import *
from .base import BoxDreamerBaseDataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from omegaconf import DictConfig


def get_files(path: str, extension: str) -> list:
    """Retrieve and sort all files with a given extension in a directory.

    Args:
        path (str): Directory path to search for files.
        extension (str): File extension to filter by (e.g., '.png').

    Returns:
        list: Sorted list of file paths matching the extension.
    """
    return sorted(
        [
            entry.path
            for entry in os.scandir(path)
            if entry.is_file() and entry.name.endswith(extension)
        ]
    )


class LINEMOD_Occulusion(BoxDreamerBaseDataset):
    """LINEMOD_Occulusion Dataset Class.

    This class loads the LINEMOD dataset preprocessed by OnePose++. It provides
    functionalities to load images, bounding boxes, poses, and camera intrinsics,
    ensuring compatibility with the BoxDreamerBaseDataset processing pipeline.

    Attributes:
        dataset_name (str): Name of the dataset ('linemod').
        root (str): Root directory of the dataset, containing training, testing, and model directories.
        model_root (str): Directory containing 3D model files.
        train_root (str): Directory containing training data.
        test_root (str): Directory containing testing data.
        split (str): Dataset split ('train', 'val', 'test').
        config (DictConfig): Configuration object.
        default_intrinsics (list): Default camera intrinsic parameters [fx, fy, cx, cy].
        dataset (str): Name of the dataset ('linemod').
    """

    def __init__(self, config: DictConfig, split: str):
        """Initialize the LINEMOD_BoxDreamer dataset.

        Args:
            config (DictConfig): Configuration object containing dataset paths and parameters.
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        super().__init__(config.base, split)
        self.dataset_name = "linemod-o"

        assert split in [
            "test"
        ], f"split {split} not supported, beacuse LINEMOD-O dataset only has test split"

        self.root = config.root  # Dataset root, includes real_train, real_test & models
        self.ref_root = os.path.join(
            self.root,
            "real_train"
            + (
                config.get("reference_suffix", "")
                if config.get("reference_suffix", "") is not None
                else ""
            ),
        )
        self.model_root = os.path.join(
            self.root,
            "models"
            + (
                config.get("model_suffix", "")
                if config.get("model_suffix", "") is not None
                else ""
            ),
        )
        self.test_root = os.path.join(self.root, "test-preprocessed")
        self.cat_filter = config.get("cat_filter", None)

        self.split = split
        self.config = config

        self.default_intrinsics = [
            572.4114,
            573.57043,
            325.2611,
            242.04899,
        ]  # 480 * 640 resolution
        self.dataset = "linemod-o"
        self.load_data()

    def load_data(self):
        """Load dataset images, bounding boxes, poses, and intrinsics.

        Always loads the training data. Additionally loads validation or
        testing data if the current split is 'val' or 'test'.
        """
        self._load_data("ref")  # Load training data regardless of the current split
        self._load_data("test")

    def _load_data(self, split: str):
        """Load data for a specific split.

        Args:
            split (str): Dataset split to load ('ref', 'test').
        """
        data_root = self.ref_root if split == "ref" else self.test_root
        self._load_data_from_dir(data_root, split)

    def _load_data_from_dir(self, root: str, split: str, max_workers: int = 8):
        """Load data from a specified directory and split with concurrent
        processing.

        Args:
            root (str): Directory containing the split data.
            split (str): Dataset split ('train', 'val', 'test').
            max_workers (int): Maximum number of threads for concurrent processing.
        """
        if split not in self.images:
            self.images[split] = {}
            self.boxes[split] = {}
            self.poses[split] = {}
            self.intrinsics[split] = {}
            self.cat_len[split] = {}

        categories = [
            cat for cat in os.listdir(root) if os.path.isdir(os.path.join(root, cat))
        ]
        if self.cat_filter:
            categories = [cat for cat in categories if cat in self.cat_filter]
        # Mutex for thread-safe updates to shared dictionaries
        lock = threading.Lock()

        def process_category(cat):
            cat_dir = os.path.join(root, cat)

            self.cat_len[split][cat] = 0

            # Define file patterns
            rgb_pattern = "-color.png"
            bbox_pattern = "-box.txt"
            pose_pattern = "-pose.txt"
            intrinsic_pattern = "-intrisic.txt"

            # Load file paths
            image_files = get_files(cat_dir, rgb_pattern)
            box_files = get_files(cat_dir, bbox_pattern)
            pose_files = get_files(cat_dir, pose_pattern)
            intrinsic_files = get_files(cat_dir, intrinsic_pattern)

            # Handle camera intrinsics
            if not intrinsic_files:
                # If intrinsic files do not exist, create them using default intrinsics
                def save_intrinsic(image_path):
                    base_name = os.path.basename(image_path).split("-")[0]
                    intrinsic_file = os.path.join(cat_dir, f"{base_name}-intrisic.txt")
                    fx, fy, cx, cy = self.default_intrinsics
                    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    np.savetxt(intrinsic_file, intrinsic_matrix)

                with ThreadPoolExecutor(max_workers=max_workers) as inner_executor:
                    list(inner_executor.map(save_intrinsic, image_files))

                # Reload intrinsic files after creation
                intrinsic_files = get_files(cat_dir, intrinsic_pattern)

            # Ensure all lists are sorted and have the same length
            image_files_sorted = sorted(image_files)
            box_files_sorted = sorted(box_files)
            pose_files_sorted = sorted(pose_files)
            intrinsic_files_sorted = sorted(intrinsic_files)

            assert (
                len(image_files_sorted)
                == len(box_files_sorted)
                == len(pose_files_sorted)
                == len(intrinsic_files_sorted)
            ), f"Number of images, boxes, poses, and intrinsics must match for category {cat}"

            # Prepare the result
            result = {
                "images": image_files_sorted,
                "boxes": box_files_sorted,
                "poses": pose_files_sorted,
                "intrinsics": intrinsic_files_sorted,
                "length": len(image_files_sorted),
            }

            return cat, result

        # Use ThreadPoolExecutor to process categories concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all category processing tasks
            futures = {
                executor.submit(process_category, cat): cat for cat in categories
            }

            # Use tqdm to display the progress
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Loading Linemod {split} data",
            ):
                cat, data = future.result()
                with lock:
                    self.images[split][cat] = data["images"]
                    self.boxes[split][cat] = data["boxes"]
                    self.poses[split][cat] = data["poses"]
                    self.intrinsics[split][cat] = data["intrinsics"]
                    self.cat_len[split][cat] = data["length"]

        # Load model paths (only once)
        if split == "ref":
            models = [
                model
                for model in os.listdir(self.model_root)
                if os.path.isdir(os.path.join(self.model_root, model))
            ]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._load_model_path, model): model
                    for model in models
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Loading model paths",
                ):
                    model, path = future.result()
                    if path:
                        self.model_paths[model] = path
                    else:
                        WARNING(f"Model file for {model} does not exist.")

    def _load_model_path(self, model_name):
        """Helper function to load a single model path.

        Args:
            model_name (str): Name of the model.

        Returns:
            tuple: (model_name, model_path or None)
        """
        model_dir = os.path.join(self.model_root, model_name)
        model_path = os.path.join(model_dir, f"{model_name}.ply")
        if os.path.exists(model_path):
            return model_name, model_path
        else:
            return model_name, None

    def read_boxes(self, box_paths: list, split: str) -> list:
        """Override the parent class method to read bounding boxes from text
        files.

        Args:
            box_paths (list): List of bounding box file paths.
            split (str): Dataset split ('train', 'val', 'test').

        Returns:
            list: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
        """
        boxes = []
        for path in box_paths:
            box = np.loadtxt(path)
            box_converted = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            boxes.append(box_converted)
        return boxes
