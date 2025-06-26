"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 14:50:48
Description: This module provides a dataset class for the OnePose dataset, optimized for efficient data loading.
It inherits from BoxDreamerBaseDataset and includes functionalities to load images, bounding boxes,
poses, and camera intrinsics. The class ensures that all preloaded file paths are sorted and
correctly aligned to maintain data integrity during training and evaluation.

"""

import os
import os.path as osp
from typing import Optional


import logging
from tqdm import tqdm

from .base import BoxDreamerBaseDataset
from .utils.data_io import *
from .utils.data_utils import *
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import lmdb


class OnePoseDataset(BoxDreamerBaseDataset):
    """OnePose Dataset Class.

    This class loads the OnePose dataset and is optimized for efficient data loading.
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
        """Initialize the OnePoseDataset.

        Args:
            config (DictConfig): Configuration object containing dataset paths and parameters.
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        super().__init__(config.base, split)
        self.split = split
        self.root = config.root
        self.cat_filter = config.get("cat_filter", None)

        self.train_root = osp.join(self.root, "train_data")
        self.val_root = osp.join(self.root, "val_data")
        self.test_root = osp.join(
            self.root,
            "test_data"
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

        self.ref_mode = config.get("ref_mode", "first_seq")  # 'same_seq'

        self.config = config
        self.dataset = "onepose"
        self.load_data(self.split)

        # Assign training or validation/testing data as reference based on the current split
        self.images["ref"] = self.images[self.split].copy()
        self.poses["ref"] = self.poses[self.split].copy()
        self.intrinsics["ref"] = self.intrinsics[self.split].copy()
        self.boxes["ref"] = self.boxes[self.split].copy()
        self.reproj_box["ref"] = self.reproj_box[self.split].copy()
        self.bbox_3d["ref"] = self.bbox_3d[self.split].copy()
        self.cat_len["ref"] = self.cat_len[self.split].copy()

        # handle the difference ref cases
        # if same_seq, no need to edit
        # else, set ref to the first seq

        all_objs = []
        for cat_key in self.images["ref"]:
            obj, seq = cat_key.split("_")
            if obj not in all_objs:
                all_objs.append(obj)

        if self.ref_mode == "first_seq":
            fks = []
            # init reference obj's
            for obj in all_objs:
                self.images["ref"][obj] = []
                self.poses["ref"][obj] = []
                self.intrinsics["ref"][obj] = []
                self.boxes["ref"][obj] = []
                self.reproj_box["ref"][obj] = []
                self.bbox_3d["ref"][obj] = ""
                self.cat_len["ref"][obj] = 0
                self.images[split][obj] = []
                self.poses[split][obj] = []
                self.intrinsics[split][obj] = []
                self.boxes[split][obj] = []
                self.reproj_box[split][obj] = []
                self.bbox_3d[split][obj] = ""
                self.cat_len[split][obj] = 0
                self.model_paths[obj] = ""

            for cat_key in self.images["ref"]:
                try:
                    obj, seq = cat_key.split("_")
                except:
                    continue
                first_cat_key = f"{obj}_1"
                if first_cat_key not in fks:
                    fks.append(first_cat_key)
                self.images["ref"][obj] = self.images["ref"][first_cat_key]
                self.poses["ref"][obj] = self.poses["ref"][first_cat_key]
                self.intrinsics["ref"][obj] = self.intrinsics["ref"][first_cat_key]
                self.boxes["ref"][obj] = self.boxes["ref"][first_cat_key]
                self.reproj_box["ref"][obj] = self.reproj_box["ref"][first_cat_key]
                self.bbox_3d["ref"][obj] = self.bbox_3d["ref"][first_cat_key]
                self.cat_len["ref"][obj] = self.cat_len["ref"][first_cat_key]
                self.model_paths[obj] = self.model_paths[first_cat_key]

            # remove the first seq from split
            for fk in fks:
                self.images[split].pop(fk)
                self.poses[split].pop(fk)
                self.intrinsics[split].pop(fk)
                self.boxes[split].pop(fk)
                self.reproj_box[split].pop(fk)
                self.bbox_3d[split].pop(fk)
                self.cat_len[split].pop(fk)
                self.model_paths.pop(fk)

                self.images["ref"].pop(fk)
                self.poses["ref"].pop(fk)
                self.intrinsics["ref"].pop(fk)
                self.boxes["ref"].pop(fk)
                self.reproj_box["ref"].pop(fk)
                self.bbox_3d["ref"].pop(fk)
                self.cat_len["ref"].pop(fk)

            # concat obj's seqs
            all_objs = []
            sks = []
            for cat_key in self.images[split]:
                try:
                    obj, seq = cat_key.split("_")
                except:
                    continue
                sks.append(cat_key)
                if obj not in all_objs:
                    all_objs.append(obj)

                self.images[split][obj].extend(self.images[split][cat_key])
                self.poses[split][obj].extend(self.poses[split][cat_key])
                self.intrinsics[split][obj].extend(self.intrinsics[split][cat_key])
                self.boxes[split][obj].extend(self.boxes[split][cat_key])
                self.reproj_box[split][obj].extend(self.reproj_box[split][cat_key])
                self.bbox_3d[split][obj] = self.bbox_3d[split][cat_key]
                self.cat_len[split][obj] += self.cat_len[split][cat_key]

            # remove the seqs
            for sk in sks:
                self.images[split].pop(sk)
                self.poses[split].pop(sk)
                self.intrinsics[split].pop(sk)
                self.boxes[split].pop(sk)
                self.reproj_box[split].pop(sk)
                self.bbox_3d[split].pop(sk)
                self.cat_len[split].pop(sk)
                self.model_paths.pop(sk)

                self.images["ref"].pop(sk)
                self.poses["ref"].pop(sk)
                self.intrinsics["ref"].pop(sk)
                self.boxes["ref"].pop(sk)
                self.reproj_box["ref"].pop(sk)
                self.bbox_3d["ref"].pop(sk)
                self.cat_len["ref"].pop(sk)

    def load_data(self, split: str = "train", max_workers: int = 8):
        """Load dataset data for the specified split with concurrent
        processing.

        Args:
            split (str, optional): Dataset split to load ('train', 'val', 'test'). Defaults to "train".
            max_workers (int, optional): Maximum number of threads for concurrent processing. Defaults to 8.
        """
        # Determine the root directory based on the split
        root = (
            self.train_root
            if split == "train"
            else (self.val_root if split == "val" else self.test_root)
        )
        # List all objects in the root directory
        objs = os.listdir(root)

        lmdb_path = osp.join(root, f"data.lmdb")
        if osp.exists(lmdb_path):
            logging.info(f"Found existing LMDB file at {lmdb_path}")
            # open the lmdb file
            try:
                env = lmdb.open(
                    lmdb_path, map_size=1099511627776, readonly=True, lock=False
                )
                txn = env.begin(write=False)
                self.lmdb = txn
                logging.info(f"Opened LMDB file at {lmdb_path}")
            except Exception as e:
                logging.error(f"Error opening LMDB file: {e}")
                self.lmdb = None

        # Initialize dictionaries for the split if not already present
        if split not in self.images:
            self.images[split] = {}
            self.poses[split] = {}
            self.intrinsics[split] = {}
            self.boxes[split] = {}
            self.cat_len[split] = {}
            self.reproj_box[split] = {}
            self.bbox_3d[split] = {}

        def get_files(path: str, extension: str) -> list:
            """Retrieve and sort all files with a given extension in a
            directory.

            Args:
                path (str): Directory path to search for files.
                extension (str): File extension to filter by (e.g., '.png').

            Returns:
                list: Sorted list of file paths matching the extension.
            """

            def extract_number(filename):
                """Extract the numeric part from a filename.

                If no number is found, return a very large number to
                sort it at the end.
                """
                match = re.search(r"(\d+)", filename)
                return int(match.group(1)) if match else float("inf")

            # Get all files with the specified extension
            files = [
                entry.path
                for entry in os.scandir(path)
                if entry.is_file() and entry.name.endswith(extension)
            ]

            # Sort files by the numeric part of their **file name** (not the full path)
            return sorted(files, key=lambda x: extract_number(os.path.basename(x)))

        # Mutex for thread-safe updates to shared dictionaries
        lock = threading.Lock()

        def process_sequence(obj: str, seq: str):
            """Process a single sequence to load its data.

            Args:
                obj (str): Object name.
                seq (str): Sequence name.

            Returns:
                dict: A dictionary containing all necessary data for the sequence.
            """
            seq_dir = osp.join(root, obj, seq)
            color_dir = osp.join(seq_dir, "color")

            # Check if sequence directory exists and contains the 'color' subdirectory
            if not osp.isdir(seq_dir) or not osp.isdir(color_dir):
                return None  # Skip invalid sequences

            # Create a unique key for the category based on object and sequence names
            cat_key = f"{obj.split('-')[0]}_{seq.split('-')[-1]}"

            model_dir = osp.join(self.model_root, cat_key.split("_")[0])

            if self.cat_filter is not None and obj.split("-")[0] not in self.cat_filter:
                return None

            # Define paths for different file types
            if (
                os.path.exists(osp.join(seq_dir, "color_full"))
                and os.path.exists(osp.join(seq_dir, "intrinsics_full"))
                and len(os.listdir(osp.join(seq_dir, "color_full")))
                == len(os.listdir(osp.join(seq_dir, "intrinsics_full")))
            ):
                # image_path = osp.join(seq_dir, 'color_full')
                # intri_path = osp.join(seq_dir, 'intrinsics_full')
                image_path = osp.join(seq_dir, "color")
                intri_path = osp.join(seq_dir, "intrin_ba")
                # for debug
            else:
                image_path = osp.join(seq_dir, "color")
                intri_path = osp.join(seq_dir, "intrin_ba")
            pose_path = osp.join(seq_dir, "poses_ba")
            reproj_box_path = osp.join(seq_dir, "reproj_box")
            bbox_3d_path = osp.join(root, obj, "box3d_corners.txt")

            # Retrieve and sort files
            images = get_files(image_path, ".png")
            poses = get_files(pose_path, ".txt")
            intrinsics = get_files(intri_path, ".txt")
            reproj_boxes = get_files(reproj_box_path, ".txt")

            # Assign box paths to 'boxes' (currently using reproj_box as boxes)
            boxes = reproj_boxes

            # Assign the number of images
            cat_length = len(images)

            # Verification: Ensure that the number of files match across categories
            num_images = len(images)
            num_boxes = len(boxes)
            num_poses = len(poses)
            num_intrinsics = len(intrinsics)

            if not (num_images == num_boxes == num_poses == num_intrinsics):
                # logging.warning(
                #     f"File count mismatch in category '{cat_key}': "
                #     f"Images: {num_images}, Boxes: {num_boxes}, "
                #     f"Poses: {num_poses}, Intrinsics: {num_intrinsics}"
                # )
                return None  # Skip sequences with mismatched file counts

            # Optional: Further verification to ensure base names match
            min_count = min(num_images, num_boxes, num_poses, num_intrinsics)
            for i in range(min_count):
                image_base = osp.basename(images[i]).split(".")[0]
                box_base = osp.basename(boxes[i]).split(".")[0]
                pose_base = osp.basename(poses[i]).split(".")[0]
                intrin_base = osp.basename(intrinsics[i]).split(".")[0]
                reproj_base = osp.basename(reproj_boxes[i]).split(".")[0]

                if not (
                    image_base == box_base == pose_base == intrin_base == reproj_base
                ):
                    # logging.warning(
                    #     f"File name mismatch in category '{cat_key}' at index {i}: "
                    #     f"Image: {image_base}, Box: {box_base}, Pose: {pose_base}, "
                    #     f"Intrinsic: {intrin_base}, Reproj Box: {reproj_base}"
                    # )
                    return None  # Skip sequences with mismatched file names

            # Prepare the result dictionary
            result = {
                "cat_key": cat_key,
                "images": images,
                "boxes": boxes,
                "poses": poses,
                "intrinsics": intrinsics,
                "reproj_box": reproj_boxes,
                "bbox_3d": bbox_3d_path,
                "length": cat_length,
            }
            if osp.exists(model_dir):
                # check whether the model exists
                model_files = osp.join(model_dir, "model.ply")
                if osp.exists(model_files):
                    result["model_path"] = model_files
                else:
                    result["model_path"] = "none"
            else:
                result["model_path"] = "none"

            return result

        # Gather all (obj, seq) pairs
        obj_seq_pairs = []
        for obj in objs:
            obj_dir = osp.join(root, obj)
            if not osp.isdir(obj_dir):
                continue
            seqs = os.listdir(obj_dir)
            for seq in seqs:
                obj_seq_pairs.append((obj, seq))

        # Use ThreadPoolExecutor to process sequences concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all sequence processing tasks
            futures = {
                executor.submit(process_sequence, obj, seq): (obj, seq)
                for obj, seq in obj_seq_pairs
            }

            # Use tqdm to display the progress
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Loading OnePose {split} data",
            ):
                data = future.result()
                if data is None:
                    continue  # Skip sequences that failed processing

                cat_key = data["cat_key"]
                with lock:
                    self.images[split][cat_key] = data["images"]
                    self.boxes[split][cat_key] = data["boxes"]
                    self.poses[split][cat_key] = data["poses"]
                    self.intrinsics[split][cat_key] = data["intrinsics"]
                    self.reproj_box[split][cat_key] = data["reproj_box"]
                    self.bbox_3d[split][cat_key] = data["bbox_3d"]
                    self.cat_len[split][cat_key] = data["length"]
                    self.model_paths[cat_key] = data["model_path"]

    def read_boxes(self, box_paths: list, split: Optional[str] = None) -> list:
        """Read bounding boxes from file paths.

        Args:
            box_paths (list): List of bounding box file paths.
            split (Optional[str], optional): Dataset split ('train', 'val', 'test'). Defaults to None.

        Returns:
            list: List of bounding boxes. Currently returns None placeholders.
        """
        boxes = []
        for path in box_paths:
            # path file contains the 2d projected box [8, 2]
            # mask a 2d bbox from this [8, 2] by using the max and min of the 8 points
            # bbox type: [x1, y1, x2, y2]
            if self.lmdb is not None:
                try:
                    points = self.lmdb.get(path.encode())
                    points = np.frombuffer(points, dtype=np.float64).reshape(-1, 2)
                except Exception as e:
                    logging.error(f"Error reading from LMDB: {e}")
                    points = np.loadtxt(path)
            else:
                points = np.loadtxt(path)

            x1 = np.min(points[:, 0])
            x2 = np.max(points[:, 0])
            y1 = np.min(points[:, 1])
            y2 = np.max(points[:, 1])

            boxes.append(None)

        return boxes
