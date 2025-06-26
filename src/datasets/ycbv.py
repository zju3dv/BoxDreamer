"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 14:50:06
Description: This module provides a dataset class for the YCB-V-BoxDreamer dataset,
which is preprocessed by BoxDreamer scripts. It is designed to be compatible with
the BoxDreamerBaseDataset for seamless data loading and processing.

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


class YCBV_BoxDreamer(BoxDreamerBaseDataset):
    """YCBV_BoxDreamer Dataset Class.

    This class loads the YCB-Video dataset preprocessed by BoxDreamer. It provides
    functionalities to load images, bounding boxes, poses, and camera intrinsics,
    ensuring compatibility with the BoxDreamerBaseDataset processing pipeline.

    Attributes:
        dataset_name (str): Name of the dataset ('ycbv').
        root (str): Root directory of the dataset, containing training, testing, and model directories.
        model_root (str): Directory containing 3D model files.
        train_root (str): Directory containing training data.
        test_root (str): Directory containing testing data.
        split (str): Dataset split ('train', 'val', 'test').
        config (DictConfig): Configuration object.
        default_intrinsics (list): Default camera intrinsic parameters [fx, fy, cx, cy].
        dataset (str): Name of the dataset ('ycbv').
    """

    def __init__(self, config: DictConfig, split: str):
        """Initialize the LINEMOD_BoxDreamer dataset.

        Args:
            config (DictConfig): Configuration object containing dataset paths and parameters.
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        super().__init__(config.base, split)
        self.dataset_name = "ycbv"

        self.root = config.root  # Dataset root, includes real_train, real_test & models
        self.model_root = os.path.join(
            self.root,
            "models"
            + (
                config.get("model_suffix", "")
                if config.get("model_suffix", "") is not None
                else ""
            ),
        )
        self.train_root = os.path.join(
            self.root,
            "train"
            + (
                config.get("reference_suffix", "")
                if config.get("reference_suffix", "") is not None
                else ""
            ),
        )
        self.test_root = os.path.join(self.root, "test")
        self.val_root = os.path.join(self.root, "val")
        self.ref_mode = config.get("ref_mode", "random_seq")  # random_seq or same_seq
        self.cat_filter = config.get("cat_filter", None)

        self.split = split
        assert split in ["train", "val", "test"], f"split {split} not supported"
        self.config = config

        self.dataset = "ycbv"
        self.load_data()

        # all sequences pose gt data is under the same coordinate system

    def load_data(self):
        """Load dataset images, bounding boxes, poses, and intrinsics.

        Always loads the training data. Additionally loads validation or
        testing data if the current split is 'val' or 'test'.
        """
        self._load_data("train")  # Load training data regardless of the current split
        if self.split in ["val", "test"]:
            self._load_data(self.split)

        if self.ref_mode == "random_seq":
            self.images["ref"] = self.images["train"].copy()
            self.boxes["ref"] = self.boxes["train"].copy()
            self.poses["ref"] = self.poses["train"].copy()
            self.intrinsics["ref"] = self.intrinsics["train"].copy()
            self.cat_len["ref"] = self.cat_len["train"].copy()

            # shuffle the reference data
            for cat in self.images["ref"]:
                print(f"Shuffling {cat} data")
                idx = np.random.permutation(len(self.images["ref"][cat]))
                self.images["ref"][cat] = [self.images["ref"][cat][i] for i in idx]
                self.boxes["ref"][cat] = [self.boxes["ref"][cat][i] for i in idx]
                self.poses["ref"][cat] = [self.poses["ref"][cat][i] for i in idx]
                self.intrinsics["ref"][cat] = [
                    self.intrinsics["ref"][cat][i] for i in idx
                ]
                self.cat_len["ref"][cat] = len(idx)

        else:
            self.images["ref"] = self.images[self.split].copy()
            self.boxes["ref"] = self.boxes[self.split].copy()
            self.poses["ref"] = self.poses[self.split].copy()
            self.intrinsics["ref"] = self.intrinsics[self.split].copy()
            self.cat_len["ref"] = self.cat_len[self.split].copy()

    def _load_data(self, split: str):
        """Load data for a specific split.

        Args:
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        data_root = self.train_root if split == "train" else self.test_root
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

        subdirs = os.listdir(root)
        # check whether the data is stored in subdirectories
        have_seq = False
        for subdir in subdirs:
            if os.path.isdir(os.path.join(root, subdir)):
                have_seq = True
                break

        if not have_seq:
            categories = [
                cat
                for cat in os.listdir(root)
                if os.path.isdir(os.path.join(root, cat))
            ]
        else:
            categories = []
            objs = os.listdir(root)
            for obj in objs:
                seqs = os.listdir(os.path.join(root, obj))
                for seq in seqs:
                    categories.append(obj + "/" + seq)

        if self.cat_filter:
            categories = [
                cat
                for cat in categories
                if cat in self.cat_filter or cat.split("/")[0] in self.cat_filter
            ]
        # Mutex for thread-safe updates to shared dictionaries
        lock = threading.Lock()

        def process_category(cat):
            cat_dir = os.path.join(root, cat)

            self.cat_len[split][cat] = 0

            # Define file patterns
            rgb_pattern = "-color.png"
            bbox_pattern = "-box.txt"
            pose_pattern = "-pose.txt"
            intrinsic_pattern = "-intrinsics.txt"

            # Load file paths
            image_files = get_files(cat_dir, rgb_pattern)
            box_files = get_files(cat_dir, bbox_pattern)
            pose_files = get_files(cat_dir, pose_pattern)
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

            return cat.split("/")[0], result

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
                desc=f"Loading YCBV {split} data",
            ):
                cat, data = future.result()
                with lock:
                    self.images[split][cat] = (
                        data["images"]
                        if cat not in self.images[split]
                        else self.images[split][cat] + data["images"]
                    )
                    self.boxes[split][cat] = (
                        data["boxes"]
                        if cat not in self.boxes[split]
                        else self.boxes[split][cat] + data["boxes"]
                    )
                    self.poses[split][cat] = (
                        data["poses"]
                        if cat not in self.poses[split]
                        else self.poses[split][cat] + data["poses"]
                    )
                    self.intrinsics[split][cat] = (
                        data["intrinsics"]
                        if cat not in self.intrinsics[split]
                        else self.intrinsics[split][cat] + data["intrinsics"]
                    )
                    self.cat_len[split][cat] = (
                        data["length"]
                        if cat not in self.cat_len[split]
                        else self.cat_len[split][cat] + data["length"]
                    )

        # Load model paths (only once)
        if split == "train":
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
        model_path = os.path.join(model_dir, f"points.xyz")
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
            # if split in ['ref', 'train']:
            #     # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            #     box_converted = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            # elif split in ['val', 'test']:
            #     # Use [x_min, y_min, x_max, y_max] directly
            #     box_converted = np.array([box[0], box[1], box[2], box[3]])
            # else:
            #     raise ValueError(f"split {split} not supported")
            # box_converted = np.array([box[0], box[1], box[2], box[3]])
            box_converted = np.array([box[0], box[1], box[2], box[3]])
            boxes.append(box_converted)
        return boxes
