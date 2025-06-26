"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 14:53:05
Description: For real-life usage.

"""


import os
import numpy as np
from src.lightning.utils.vis.vis_utils import *
from src.utils.log import *
from .base import BoxDreamerBaseDataset
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


class CustomDataset(BoxDreamerBaseDataset):
    def __init__(self, config: DictConfig, split: str):
        """Initialize the LINEMOD_BoxDreamer dataset.

        Args:
            config (DictConfig): Configuration object containing dataset paths and parameters.
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        super().__init__(config.base, split="demo")
        self.dataset_name = "Custom"

        self.model_path = None
        self.ref_root = None
        self.test_root = None

        self.split = "demo"
        self.config = config

        self.default_intrinsics = None
        self.dataset = "Custom"

    def set_intrinsic(self, intrinsic):
        self.default_intrinsics = intrinsic

    def set_model_path(self, model_path):
        self.model_path = model_path
        self.model_paths = {"real": model_path}

    def set_ref_root(self, ref_root):
        self.ref_root = ref_root

    def set_test_root(self, test_root):
        self.test_root = test_root

    def load_data(self):
        # check whether all needed parameters are set
        if self.model_path is None:
            raise ValueError("Model path is not set.")
        if self.ref_root is None:
            raise ValueError("Reference data root is not set.")
        if self.test_root is None:
            raise ValueError("Test data root is not set.")
        if self.default_intrinsics is None:
            raise ValueError("Default intrinsics are not set.")

        # for real life usage, we only have one cat called 'real'
        # init all the data structures
        self.images = {"demo": {}, "ref": {}}
        self.poses = {"demo": {}, "ref": {}}
        self.boxes = {"demo": {}, "ref": {}}
        self.cat_len = {"demo": {}, "ref": {}}
        self.intrinsics = {"demo": {}, "ref": {}}

        # load all images
        # support extensions: .png, .jpg, .jpeg
        image_paths = (
            get_files(self.test_root, "-color.png")
            + get_files(self.test_root, "-color.jpg")
            + get_files(self.test_root, "-color.jpeg")
        )
        self.images["demo"]["real"] = image_paths
        # sort the images
        self.images["demo"]["real"].sort()
        self.cat_len["demo"]["real"] = len(self.images["demo"]["real"])

        # load reference images
        ref_image_paths = (
            get_files(self.ref_root, "-color.png")
            + get_files(self.ref_root, "-color.jpg")
            + get_files(self.ref_root, "-color.jpeg")
        )
        self.images["ref"]["real"] = ref_image_paths
        self.images["ref"]["real"].sort()
        self.cat_len["ref"]["real"] = len(self.images["ref"]["real"])

        # load reference poses
        ref_pose_paths = get_files(self.ref_root, "-pose.txt")
        ref_pose_paths.sort()
        self.poses["ref"]["real"] = ref_pose_paths

        # load reference poses with none
        self.poses["demo"]["real"] = ["none"] * len(image_paths)

        # check whether have -box.txt files, if not, replace with None
        ref_box_paths = get_files(self.ref_root, "-mask.png")
        ref_box_paths.sort()
        if len(ref_box_paths) == 0:
            ref_box_paths = [None] * len(ref_pose_paths)
        self.boxes["ref"]["real"] = ref_box_paths

        # load test box if exists
        test_box_paths = get_files(self.test_root, "-mask.png")
        test_box_paths.sort()
        if len(test_box_paths) == 0:
            test_box_paths = [None] * len(image_paths)
        self.boxes["demo"]["real"] = test_box_paths

        # load reference intrinsics path
        ref_intri_paths = get_files(self.ref_root, "-intrinsics.txt")
        ref_intri_paths.sort()
        self.intrinsics["ref"]["real"] = ref_intri_paths
        # load test intrinsics
        self.intrinsics["demo"]["real"] = ["none"] * len(image_paths)

    def read_poses(self, pose_paths: list, idx: int) -> list:
        if pose_paths[0] == "none":
            return [np.eye(4)] * len(pose_paths)
        else:
            return super().read_poses(pose_paths, idx)

    def read_intrinsics(self, intri_paths: list, idx: int) -> list:
        if intri_paths[0] == "none":
            return [self.default_intrinsics] * len(intri_paths)
        else:
            return super().read_intrinsics(intri_paths, idx)

    def read_boxes(
        self, box_paths: List[str], split: Optional[str] = None
    ) -> List[List[int]]:
        """Read bounding boxes from mask image file paths.

        Args:
            box_paths (list): List of mask image file paths.
            split (Optional[str], optional): Dataset split ('train', 'val', 'test'). Defaults to None.

        Returns:
            list: List of bounding boxes in format [x0, y0, x1, y1].
        """
        boxes = []
        for path in box_paths:
            if path is None:
                # Add a None placeholder if path is None
                boxes.append(None)
                continue

            try:
                # Use PIL to read the mask image (RGB)
                mask = Image.open(path)
                mask = np.array(mask)

                # If mask has 3 channels (RGB), average the last dimension to get grayscale
                if len(mask.shape) == 3 and mask.shape[2] == 3:
                    mask = mask.mean(axis=-1)

                # Threshold the mask to get binary values
                binary_mask = mask > 0

                # Find coordinates of non-zero elements (pixels belonging to the object)
                if np.any(binary_mask):
                    # Get indices of non-zero elements
                    y_indices, x_indices = np.where(binary_mask)

                    # Get bounding box coordinates
                    x0 = int(np.min(x_indices))
                    y0 = int(np.min(y_indices))
                    x1 = int(np.max(x_indices))
                    y1 = int(np.max(y_indices))

                    # pre-padding for occulusion stability
                    padding = 0.3
                    x0 = max(0, x0 - padding * (x1 - x0))
                    y0 = max(0, y0 - padding * (y1 - y0))
                    x1 = min(mask.shape[1], x1 + padding * (x1 - x0))
                    y1 = min(mask.shape[0], y1 + padding * (y1 - y0))

                    # Add bounding box to the list
                    boxes.append([x0, y0, x1, y1])
                else:
                    # If mask is empty, add None
                    boxes.append(None)

            except Exception as e:
                print(f"Error processing mask {path}: {e}")
                import traceback

                traceback.print_exc()
                boxes.append(None)

        return boxes
