"""CO3D V2 Dataset Module.

This module provides a dataset class for the CO3D V2 dataset, optimized for efficient data loading.
It inherits from BoxDreamerBaseDataset and includes functionalities to load images, bounding boxes,
poses, and camera intrinsics. The class ensures that all preloaded file paths are sorted and
correctly aligned to maintain data integrity during training and evaluation.

Author: Yuanhong Yu
Date: 2024-11-01
"""

import os.path as osp
import json

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from .base import BoxDreamerBaseDataset
from .utils.data_io import *
from .utils.data_utils import *
from src.utils.log import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class Co3DV2Dataset(BoxDreamerBaseDataset):
    """CO3D V2 Dataset Class.

    This class loads the CO3D V2 dataset preprocessed by the Dust3R script. It provides
    functionalities to load images, bounding boxes, poses, and camera intrinsics,
    ensuring compatibility with the BoxDreamerBaseDataset processing pipeline.

    Attributes:
        split (str): Dataset split ('train', 'val', 'test').
        root (str): Root directory of the dataset.
        train_root (str): Directory containing training data.
        val_root (str): Directory containing validation data (unseen scenes).
        test_root (str): Directory containing testing data (unseen scenes).
        train_root_raw (str): Directory containing raw training data.
        test_root_raw (str): Directory containing raw testing data.
        config (DictConfig): Configuration object.
        dataset (str): Name of the dataset ('co3d').
    """

    def __init__(self, config: DictConfig, split: str):
        """Initialize the Co3DV2Dataset.

        Args:
            config (DictConfig): Configuration object containing dataset paths and parameters.
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        super().__init__(config.base, split)
        self.split = split
        self.root = config.root

        # Define paths for different splits
        self.train_root = osp.join(self.root, "co3d_train")
        self.val_root = osp.join(self.root, "co3d_test")  # Unseen scenes
        self.test_root = osp.join(self.root, "co3d_test")  # Unseen scenes

        self.train_root_raw = osp.join(self.root, "co3d_train_raw")
        self.test_root_raw = osp.join(self.root, "co3d_test_raw")

        self.config = config
        self.dataset = "co3d"
        self.load_data(self.split)

        # Assign current split data as reference data
        self.cat_len["ref"] = self.cat_len[self.split]
        self.images["ref"] = self.images[self.split]
        self.poses["ref"] = self.poses[self.split]
        self.intrinsics["ref"] = self.intrinsics[self.split]
        self.boxes["ref"] = self.boxes[self.split]

    def load_data(self, split: str = "train", max_workers: int = 8):
        """Load dataset data for the specified split with concurrent
        processing.

        Args:
            split (str, optional): Dataset split to load ('train', 'val', 'test'). Defaults to "train".
            max_workers (int, optional): Maximum number of threads for concurrent processing. Defaults to 8.
        """
        # Determine the appropriate root directory based on the split
        if split == "train" or split == "val":
            root = self.train_root
            raw_root = self.train_root_raw
            meta_file = (
                osp.join(root, "selected_seqs_train.json")
                if split == "train"
                else osp.join(root, "selected_seqs_test.json")
            )
        else:  # 'test'
            root = self.val_root
            raw_root = self.test_root_raw
            meta_file = osp.join(root, "selected_seqs_test.json")

        # Load metadata from the JSON file
        try:
            with open(meta_file, "r") as f:
                meta_data = json.load(f)
        except Exception as e:
            ERROR(f"Failed to load metadata file {meta_file}: {e}")
            return

        # Initialize dictionaries for the split if not already present
        if split not in self.images:
            self.images[split] = {}
            self.poses[split] = {}
            self.intrinsics[split] = {}
            self.boxes[split] = {}
            self.cat_len[split] = {}
            self.reproj_box[split] = {}
            self.bbox_3d[split] = {}

        # Mutex for thread-safe updates to shared dictionaries
        lock = threading.Lock()

        def process_camera(cam_path: str):
            """Process a single camera npz file, converting it to pose and
            intrinsic txt files if necessary.

            Args:
                cam_path (str): Path to the camera npz file.

            Returns:
                tuple: (pose_file, intrinsic_file) paths or (None, None) if processing failed.
            """
            pose_file = cam_path.replace(".npz", "_pose.txt")
            intrinsic_file = cam_path.replace(".npz", "_intrinsic.txt")

            if not osp.exists(pose_file) or not osp.exists(intrinsic_file):
                try:
                    cam_data = np.load(cam_path)
                    pose = cam_data.get("camera_pose")
                    intrinsic = cam_data.get("camera_intrinsics")

                    if pose is None or intrinsic is None:
                        WARNING(
                            f"Missing 'camera_pose' or 'camera_intrinsics' in {cam_path}"
                        )
                        return (None, None)

                    np.savetxt(pose_file, pose)
                    np.savetxt(intrinsic_file, intrinsic)
                except Exception as e:
                    ERROR(f"Error processing camera file {cam_path}: {e}")
                    return (None, None)

            if osp.exists(pose_file) and osp.exists(intrinsic_file):
                return (pose_file, intrinsic_file)
            else:
                return (None, None)

        def process_sequence(cat: str, seq: str):
            """Process a single sequence to load its data.

            Args:
                cat (str): Category name.
                seq (str): Sequence name.

            Returns:
                dict: A dictionary containing all necessary data for the sequence.
            """
            cat_key = f"{cat}_{seq}"
            seq_path = osp.join(root, cat, seq)

            # Retrieve selected frames from metadata
            selected_frames = meta_data[cat][seq]  # List of frame indices

            # Generate file paths for images, cameras, and masks
            selected_images = [
                osp.join(seq_path, "images", f"frame{frame:06d}.jpg")
                for frame in selected_frames
            ]
            selected_cameras = [
                osp.join(seq_path, "images", f"frame{frame:06d}.npz")
                for frame in selected_frames
            ]
            selected_masks = [
                osp.join(seq_path, "masks", f"frame{frame:06d}.png")
                for frame in selected_frames
            ]

            # Process camera files concurrently
            poses = []
            intrinsics = []
            with ThreadPoolExecutor(max_workers=max_workers) as cam_executor:
                cam_futures = {
                    cam_executor.submit(process_camera, cam_path): cam_path
                    for cam_path in selected_cameras
                }
                for cam_future in as_completed(cam_futures):
                    pose_file, intrinsic_file = cam_future.result()
                    poses.append(pose_file)
                    intrinsics.append(intrinsic_file)

            # Assign 3D bounding box path
            bbox_3d_path = osp.join(raw_root, cat, seq, "pointcloud.ply")
            bbox_3d = bbox_3d_path if osp.exists(bbox_3d_path) else None

            # Assign model path
            model_path = bbox_3d_path if osp.exists(bbox_3d_path) else None

            # sort all paths
            selected_images = sorted(selected_images)
            selected_masks = sorted(selected_masks)
            poses = sorted(poses)
            intrinsics = sorted(intrinsics)

            # Prepare the result dictionary
            result = {
                "cat_key": cat_key,
                "images": selected_images,
                "boxes": selected_masks,  # Assuming boxes are same as masks
                "poses": poses,
                "intrinsics": intrinsics,
                "reproj_box": selected_masks,  # Assuming reproj_box is same as masks
                "bbox_3d": bbox_3d,
                "model_path": model_path,
                "length": len(selected_images),
            }

            return result

        # Gather all (cat, seq) pairs from metadata
        obj_seq_pairs = []
        for cat, seqs in meta_data.items():
            for seq in seqs:
                obj_seq_pairs.append((cat, seq))

        # Use ThreadPoolExecutor to process sequences concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all sequence processing tasks
            future_to_seq = {
                executor.submit(process_sequence, cat, seq): (cat, seq)
                for cat, seq in obj_seq_pairs
            }

            # Use tqdm to display the progress
            for future in tqdm(
                as_completed(future_to_seq),
                total=len(future_to_seq),
                desc=f"Loading data for {split}",
            ):
                try:
                    data = future.result()
                except Exception as exc:
                    cat, seq = future_to_seq[future]
                    ERROR(f"Sequence {cat}_{seq} generated an exception: {exc}")
                    continue  # Skip failed sequences

                if data is None:
                    continue  # Skip if no data returned

                cat_key = data["cat_key"]
                with lock:
                    self.images[split][cat_key] = data["images"]
                    self.boxes[split][cat_key] = data["boxes"]
                    self.poses[split][cat_key] = data["poses"]
                    self.intrinsics[split][cat_key] = data["intrinsics"]
                    self.reproj_box[split][cat_key] = data["reproj_box"]
                    self.bbox_3d[split][cat_key] = data["bbox_3d"]
                    self.model_paths[cat_key] = data["model_path"]
                    self.cat_len[split][cat_key] = data["length"]

        INFO(f"Data loading completed for split: {split}")
