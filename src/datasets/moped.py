"""MOPED-BoxDreamer dataset Module.

This module provides a dataset class for MOPED-BoxDreamer dataset.

Author: Yuanhong Yu
Date: 2025-01-12
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
import PIL


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


class MOPED_BoxDreamer(BoxDreamerBaseDataset):
    """MOPED Dataset Class.

    This class loads the MOPED dataset and is optimized for efficient data loading.
    It provides functionalities to load images, bounding boxes, poses, and camera intrinsics,
    ensuring compatibility with the BoxDreamerBaseDataset processing pipeline.

    Attributes:
        split (str): Dataset split ('train', 'val', 'test').
        root (str): Root directory of the dataset.
        train_root (str): Directory containing training data.
        val_root (str): Directory containing validation data.
        test_root (str): Directory containing testing data.
        config (DictConfig): Configuration object.
        dataset (str): Name of the dataset ('onepose').
    """

    def __init__(self, config: DictConfig, split: str):
        """Initialize the MOPED_BoxDreamer dataset.

        Args:
            config (DictConfig): Configuration object containing dataset paths and parameters.
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        super().__init__(config.base, split)
        self.dataset_name = "moped"

        self.root = config.root  # Dataset root, includes real_train, real_test & models
        self.model_root = os.path.join(self.root, "models")
        self.train_root = os.path.join(self.root, "reference")
        self.test_root = os.path.join(self.root, "test")
        self.val_root = os.path.join(self.root, "test")

        self.cat_filter = config.get("cat_filter", None)

        self.ref_mode = config.get("ref_mode", "same_seq")  # 'same_seq' or 'random_seq'
        # default same_seq, because moped's data seq have unique obj point cloud

        self.split = split
        assert split in ["train", "val", "test"], f"split {split} not supported"
        self.config = config

        self.dataset = "moped"

        self.load_data()

    def load_data(self):
        """Load dataset images, bounding boxes, poses, and intrinsics.

        Always loads the training data. Additionally loads validation or
        testing data if the current split is 'val' or 'test'.
        """
        self._load_data("train")  # Load training data regardless of the current split
        if self.split in ["val", "test"]:
            self._load_data(self.split)

        # Assign training data as reference data
        self.images["ref"] = self.images[self.split].copy()
        self.boxes["ref"] = self.boxes[self.split].copy()
        self.poses["ref"] = self.poses[self.split].copy()
        self.intrinsics["ref"] = self.intrinsics[self.split].copy()
        self.cat_len["ref"] = self.cat_len[self.split].copy()
        self.model_paths["ref"] = self.model_paths[self.split].copy()

        # cat format : reference-obj-seq or test-obj-seq
        # replace reference- with test- to get the test data
        if (
            self.split == "test" or self.split == "val"
        ) and self.ref_mode == "random_seq":
            for cat in self.images[self.split].keys():
                prefix, obj, seq = cat.split("-")
                # random select one reference data seq
                ref_keys = list(self.images["ref"].keys())
                # eliminate the test data
                ref_keys = [
                    key
                    for key in ref_keys
                    if key.split("-")[0] == "reference" and key.split("-")[1] == obj
                ]

                # random select one
                ref_key = np.random.choice(ref_keys)
                self.images["ref"][cat] = self.images["ref"][ref_key].copy()
                self.boxes["ref"][cat] = self.boxes["ref"][ref_key].copy()
                self.poses["ref"][cat] = self.poses["ref"][ref_key].copy()
                self.intrinsics["ref"][cat] = self.intrinsics["ref"][ref_key].copy()
                self.cat_len["ref"][cat] = self.cat_len["ref"][ref_key]
                self.model_paths["ref"][cat] = self.model_paths["ref"][ref_key].copy()

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
            self.model_paths[split] = {}

        categories = []
        objs = os.listdir(root)
        for obj in objs:
            seqs = os.listdir(os.path.join(root, obj))
            for seq in seqs:
                if split == "train":
                    categories.append("reference-" + obj + "-" + seq)
                else:
                    categories.append("test-" + obj + "-" + seq)

        if self.cat_filter and split != "train":
            categories = [cat for cat in categories if cat in self.cat_filter]
        # Mutex for thread-safe updates to shared dictionaries
        lock = threading.Lock()

        def process_category(cat):
            # cat name format: reference-obj-seq or test-obj-seq
            # cat -> dir: obj/seq
            obj, seq = cat.split("-")[1:]
            cat_dir = os.path.join(root, obj, seq)
            self.cat_len[split][cat] = 0

            color_dir = os.path.join(cat_dir, "color")
            box_dir = os.path.join(cat_dir, "mask")
            pose_dir = os.path.join(cat_dir, "pose")
            intrinsic_dir = os.path.join(cat_dir, "intrinsics")

            image_files = get_files(color_dir, ".jpg")
            box_files = get_files(box_dir, ".png")
            pose_files = get_files(pose_dir, ".txt")
            intrinsic_files = get_files(intrinsic_dir, ".txt")

            # Ensure all lists are sorted and have the same length
            image_files_sorted = sorted(image_files)
            box_files_sorted = sorted(box_files)
            pose_files_sorted = sorted(pose_files)
            intrinsic_files_sorted = sorted(intrinsic_files)

            # moped's mask maybe less than the image, in this case, use mask to filter out the image
            if len(image_files_sorted) > len(box_files_sorted):
                image_files_sorted = [
                    image
                    for image in image_files_sorted
                    if image.split("/")[-1].replace(".jpg", ".png") in box_files_sorted
                ]
                pose_files_sorted = [
                    pose
                    for pose in pose_files_sorted
                    if pose.split("/")[-1].replace(".txt", ".png") in box_files_sorted
                ]
                intrinsic_files_sorted = [
                    intrinsic
                    for intrinsic in intrinsic_files_sorted
                    if intrinsic.split("/")[-1].replace(".txt", ".png")
                    in box_files_sorted
                ]

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
                desc=f"Loading MOPED {split} data",
            ):
                cat, data = future.result()
                with lock:
                    if data:
                        self.images[split][cat] = data["images"]
                        self.boxes[split][cat] = data["boxes"]
                        self.poses[split][cat] = data["poses"]
                        self.intrinsics[split][cat] = data["intrinsics"]
                        self.cat_len[split][cat] = data["length"]

        # Load model paths
        for cat in categories:
            path = os.path.join(self.model_root, cat + ".ply")
            self.model_paths[split][cat] = [path] * self.cat_len[split][cat]

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
            # moped box is a mask image, we need to convert it to bbox
            try:
                mask = PIL.Image.open(path)
                mask = np.array(mask)

                # get bbox
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                boxes.append([cmin, rmin, cmax, rmax])
            except Exception as e:
                ERROR(f"Failed to read box from {path}: {e}")
                boxes.append([0, 0, 0, 0])

        return boxes
