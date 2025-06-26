"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 14:50:56
Description: This module provides a dataset class for the Objaverse, which
rendered by blender. It inherits from BoxDreamerBaseDataset and includes
functionalities to load images, bounding boxes, poses, and camera intrinsics.

"""


import os
import numpy as np
from src.lightning.utils.vis.vis_utils import *
from src.utils.log import *
from .base import BoxDreamerBaseDataset
from tqdm import tqdm
import json
from typing import Optional, Dict, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from src.utils.customize.sample_points_on_cad import get_all_points_on_model
import torch.distributed as dist
from PIL import Image
import copy


class ObjaverseBoxDreamerDataset(BoxDreamerBaseDataset):
    """Objaverse-BoxDreamer Dataset Class.

    This class loads the Objaverse-BoxDreamer dataset and is optimized for efficient data loading.
    It provides functionalities to load images, bounding boxes, poses, and camera intrinsics,
    ensuring compatibility with the BoxDreamerBaseDataset processing pipeline.

    Attributes:
        split (str): Dataset split ('train', 'val', 'test').
        root (str): Root directory of the dataset.
        train_root (str): Directory containing training data.
        val_root (str): Directory containing validation data.
        test_root (str): Directory containing testing data.
        config (DictConfig): Configuration object.
        dataset (str): Name of the dataset ('objaverse').
    """

    def __init__(self, config, split):
        """Initialize the ObjaverseBoxDreamerDataset.

        Args:
            config (DictConfig): Configuration object containing dataset paths and parameters.
            split (str): Dataset split to load ('train', 'val', 'test').
        """
        super().__init__(config.base, split)
        self.dataset = "objaverse"
        self.config = config
        self.split = split
        self.root = config.root
        self.subdir_num = config.subdir_num
        self.model_root = osp.join(self.root, "Objaverse/Objaverse_glbs")
        self.train_root = osp.join(self.root, "objaverse_render/_v0")
        self.val_root = osp.join(self.root, "objaverse_render/_v0")

        self.bg_root = osp.join(self.root, "SUN2012pascalformat")

        # there are no test splits for objaverse
        assert self.split in ["train", "val", "test"], f"Invalid split: {self.split}"

        self.val_root = None
        self.test_root = None

        self.bg_images = {}
        self.load_data(split=split)

        # Assign training or validation/testing data as reference based on the current split
        self.images["ref"] = self.images[self.split]
        self.poses["ref"] = self.poses[self.split]
        self.intrinsics["ref"] = self.intrinsics[self.split]
        self.boxes["ref"] = self.boxes[self.split]
        self.cat_len["ref"] = self.cat_len[self.split]
        self.bbox_3d["ref"] = self.bbox_3d[self.split]

        self.occlusion_objs = copy.deepcopy(self.images["ref"])
        # flatten the occlusion_objs into a list with image path
        self.occlusion_objs = [
            item for sublist in self.occlusion_objs.values() for item in sublist
        ]

    def load_data(self, split="train", max_workers: int = 32):
        """
        Core Strategy:
            1. Retrieve all objects in the Objaverse dataset (only .glb format available).
            2. For each object, map the model filename to its path.
            3. Based on the objects, gather corresponding images and pose metadata.
            4. Handle cases where certain directories or files might be missing.
        """
        # Initialize dictionaries for the specified split if not already present
        if split not in self.images:
            self.images[split] = {}
            self.poses[split] = {}
            self.intrinsics[split] = {}
            self.boxes[split] = {}
            self.cat_len[split] = {}
            self.bbox_3d[split] = {}
            self.bg_images[split] = {}

        def get_files(path: str, extension: str) -> list:
            """Retrieve and sort all files with a given extension in a
            directory.

            Args:
                path (str): Directory path to search for files.
                extension (str): File extension to filter by (e.g., '.png').

            Returns:
                list: Sorted list of file paths matching the extension.
            """
            try:
                return sorted(
                    [
                        entry.path
                        for entry in os.scandir(path)
                        if entry.is_file() and entry.name.endswith(extension)
                    ]
                )
            except FileNotFoundError:
                # WARNING(f"Directory not found: {path}")
                return []

        # Step 1: Retrieve all .glb files for each sub_obj concurrently
        objs = os.listdir(self.model_root)
        all_models = {}
        all_bbox_3d = {}
        lock = (
            threading.Lock()
        )  # To ensure thread-safe updates to shared data structures

        def process_sub_obj(sub_obj: str):
            """Process a single sub-object by retrieving its .glb files.

            Args:
                sub_obj (str): Sub-object name.

            Returns:
                tuple: (sub_obj, list of .glb file paths)
            """
            sub_obj_path = osp.join(self.model_root, sub_obj)
            if not osp.isdir(sub_obj_path):
                # WARNING(f"Sub-object path is not a directory: {sub_obj_path}. Skipping.")
                return sub_obj, []
            glb_files = get_files(sub_obj_path, ".glb")
            # check whether the glb files can be loaded successfully
            for glb_file in glb_files:
                try:
                    pts = get_all_points_on_model(glb_file)
                    bbox_3d = get_3d_bbox_from_pts(pts)
                    with lock:
                        all_bbox_3d[glb_file] = bbox_3d
                except Exception as e:
                    # WARNING(f"Error loading .glb file: {glb_file}. Skipping.")
                    glb_files.remove(glb_file)

            return sub_obj, glb_files

        # if the ok_glb_files.json exists, load the file
        if osp.exists(osp.join(self.root, "ok_glb_files.json")):
            with open(osp.join(self.root, "ok_glb_files.json"), "r") as f:
                all_models = json.load(f)
            # check whether the bbox_3d is loaded successfully
            bbox_3d_path = osp.join(self.root, "bbox_3d")

            def check_sub_obj(sub_obj: str, glb_files: list):
                """Check if the 3D bounding box file exists for each object.

                Args:
                    sub_obj (str): Sub-object name.
                    glb_files (list): List of .glb file paths.
                """
                for glb_file in glb_files:
                    filename = osp.splitext(osp.basename(glb_file))[0]
                    bbox_3d_path = osp.join(self.root, "bbox_3d", f"{filename}.txt")
                    not_exist_files = []
                    if osp.exists(bbox_3d_path):
                        continue
                    else:
                        # load the bbox_3d from the glb file
                        # try:
                        #     pts = get_all_points_on_model(glb_file)
                        #     bbox_3d = get_3d_bbox_from_pts(pts)
                        #     all_bbox_3d[glb_file] = bbox_3d
                        # except Exception as e:
                        #     # WARNING(f"Error loading .glb file: {glb_file}. Skipping.")
                        #     with lock:
                        #         all_models[sub_obj].remove(glb_file)

                        # if no bbox file, remove the glb file (pessimistic strategy for speed up the process)
                        not_exist_files.append(glb_file)

                return sub_obj, not_exist_files

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(check_sub_obj, sub_obj, glb_files): (
                        sub_obj,
                        glb_files,
                    )
                    for sub_obj, glb_files in all_models.items()
                }
                not_exist_num = 0
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Checking 3D Bounding Boxes",
                ):
                    sub_obj, not_exist_files = future.result()
                    # remove the not exist files
                    not_exist_num += len(not_exist_files)
                    for glb_file in not_exist_files:
                        all_models[sub_obj].remove(glb_file)

                INFO(f"Total {not_exist_num} 3D bounding box files are not found.")

        else:
            # Use ThreadPoolExecutor to concurrently process sub-objects
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all sub-object processing tasks
                futures = {
                    executor.submit(process_sub_obj, sub_obj): sub_obj
                    for sub_obj in objs
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Loading .glb files"
                ):
                    sub_obj, glb_files = future.result()
                    all_models[sub_obj] = glb_files

        # Step 2: Map model filenames to their paths concurrently

        # dump all ok glb files into a json file under the root
        # check whether the file exists
        if not osp.exists(osp.join(self.root, "ok_glb_files.json")):
            with open(osp.join(self.root, "ok_glb_files.json"), "w") as f:
                json.dump(all_models, f, indent=4)

        bbox_3d_path = osp.join(self.root, "bbox_3d")
        if not osp.exists(bbox_3d_path):
            os.makedirs(bbox_3d_path)

        # maybe have parrallel writing issue? for multi-threading training?
        # only dump the bbox_3d files in the rank 0 process(main process)
        if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
            for glb_file, bbox_3d in tqdm(
                all_bbox_3d.items(), desc="Dumping 3D Bounding Boxes"
            ):
                # dump into a txt file
                filename = osp.splitext(osp.basename(glb_file))[0]
                if osp.exists(osp.join(bbox_3d_path, f"{filename}.txt")):
                    continue
                with open(osp.join(bbox_3d_path, f"{filename}.txt"), "w") as f:
                    for i in range(8):
                        f.write(f"{bbox_3d[i][0]} {bbox_3d[i][1]} {bbox_3d[i][2]}\n")

        # other processes need to wait for the rank 0 process to finish the writing
        if dist.is_initialized():
            dist.barrier()

        def map_model_paths(sub_obj: str, glb_files: list):
            """Map each model filename to its full path.

            Args:
                sub_obj (str): Sub-object name.
                glb_files (list): List of .glb file paths.
            """
            for obj_path in glb_files:
                filename = osp.splitext(osp.basename(obj_path))[0]
                render_types = ["uniform", "uniform_z", "random", "random_fix"]
                with lock:
                    for render_type in render_types:
                        obj_name = f"{filename}_{render_type}"
                        self.model_paths[obj_name] = obj_path

        # Use ThreadPoolExecutor to concurrently map model paths
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(map_model_paths, sub_obj, glb_files)
                for sub_obj, glb_files in all_models.items()
            ]
            # Wait for all mapping tasks to complete
            for _ in tqdm(
                as_completed(futures), total=len(futures), desc="Mapping model paths"
            ):
                pass  # No action needed; just ensuring all tasks complete

        # Step 3: Process each object to gather images and poses concurrently

        def process_obj(sub_obj: str, obj_path: str):
            """Process a single object by retrieving its images and pose
            metadata for multiple render types.

            Args:
                sub_obj (str): Sub-object name.
                obj_path (str): Path to the .glb file.

            Returns:
                dict or None: Dictionary containing data for the object or None if processing fails.
            """
            obj = osp.splitext(osp.basename(obj_path))[0]
            render_types = ["uniform", "uniform_z", "random", "random_fix"]
            # Initialize a dictionary to store data for all supported render types
            render_data = {}

            for render_type in render_types:
                images_dir = osp.join(
                    self.train_root, sub_obj, obj, render_type, "renderings"
                )
                meta_dir = osp.join(self.train_root, sub_obj, obj, render_type)

                # Check if the images directory exists
                if not osp.exists(images_dir):
                    # Optional: Log a warning if needed
                    # WARNING(f"Images directory does not exist for render type '{render_type}': {images_dir}. Skipping.")
                    continue  # Skip to the next render_type

                # Retrieve all .png image files
                images = get_files(images_dir, ".png")
                if not images:
                    # Optional: Log a warning if needed
                    # WARNING(f"No images found in: {images_dir} for render type '{render_type}'. Skipping.")
                    continue  # Skip to the next render_type

                # Retrieve the first .json meta file
                meta_files = get_files(meta_dir, ".json")
                if not meta_files:
                    # Optional: Log a warning if needed
                    # WARNING(f"No meta JSON files found in: {meta_dir} for render type '{render_type}'. Skipping.")
                    continue  # Skip to the next render_type
                meta = meta_files[0]  # Assuming only one meta JSON file exists

                # load from the bbox_3d file
                bbox_3d_path = osp.join(self.root, "bbox_3d", f"{obj}.txt")
                if not osp.exists(bbox_3d_path):
                    # WARNING(f"3D bounding box file does not exist: {bbox_3d_path}. Skipping.")
                    continue

                # Add the processed data for the current render_type to render_data
                render_data[render_type] = {
                    "obj": obj,
                    "images": images,
                    "meta": meta,
                    "poses": [meta]
                    * len(images),  # TODO: Adjust if pose data differs per image
                    "intrinsics": [meta]
                    * len(images),  # TODO: Adjust based on actual intrinsics data
                    "boxes": images,  # [None] * len(images),  # Adjust based on business logic if needed
                    "reproj_box": [None]
                    * len(images),  # Adjust based on business logic if needed
                    "bbox_3d": bbox_3d_path,
                    "render_type": render_type,
                }

            # If no render types were successfully processed, return None
            if not render_data:
                return None

            data_list = list(render_data.values())
            # Return the first render type data for now
            return data_list

        def populate_data(data: Union[Dict, List[Dict]]):
            """Populate the shared data structures with the processed data.

            Args:
                data (dict): Dictionary containing data for the object.
            """
            if isinstance(data, list):
                for d in data:
                    populate_data(d)
                return
            else:
                obj = data["obj"]
                render_type = data["render_type"]
                obj_name = f"{obj}_{render_type}"
                with lock:
                    self.images[split][obj_name] = data["images"]
                    self.poses[split][obj_name] = data["poses"]
                    self.intrinsics[split][obj_name] = data["intrinsics"]
                    self.boxes[split][obj_name] = data["boxes"]
                    # self.reproj_box[split][obj] = data['reproj_box']
                    self.cat_len[split][obj_name] = len(data["images"])
                    # If bbox_3d is applicable, assign it here
                    self.bbox_3d[split][obj_name] = data["bbox_3d"]

        # Gather all (sub_obj, obj_path) pairs
        obj_pairs = []
        sum_of_objs = 0
        for sub_obj, glb_files in all_models.items():
            # Control the number of subdirectories loaded based on split
            if split == "train" and self.subdir_num is not None:
                if len(obj_pairs) >= self.subdir_num * len(glb_files):
                    break
            sum_of_objs += len(glb_files)
            for obj_path in glb_files:
                obj_pairs.append((sub_obj, obj_path))

            if split == "val" or split == "test":
                # for validation, we only need to load the first subdir_num subdirectories
                break

        # Use ThreadPoolExecutor to concurrently process each object
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all object processing tasks
            future_to_obj = {
                executor.submit(process_obj, sub_obj, obj_path): (sub_obj, obj_path)
                for sub_obj, obj_path in obj_pairs
            }

            # Iterate over completed tasks with a progress bar
            for future in tqdm(
                as_completed(future_to_obj),
                total=len(future_to_obj),
                desc=f"Processing objects for {split}",
            ):
                sub_obj, obj_path = future_to_obj[future]
                try:
                    data = future.result()
                except Exception as exc:
                    ERROR(f"Error processing {sub_obj}/{obj_path}: {exc}")
                    continue  # Skip this object and continue with others

                if data is not None:
                    # Populate shared data structures
                    populate_data(data)

        INFO(f"Data loading completed for split: {split}")
        # log how many objects are loaded
        INFO(f"Total {sum_of_objs} objects are loaded.")

        # load the background images
        if split == "train":
            bg_index_path = osp.join(self.bg_root, "ImageSets/Main/train.txt")
        else:
            bg_index_path = osp.join(self.bg_root, "ImageSets/Main/test.txt")

        with open(bg_index_path, "r") as f:
            bg_index = f.readlines()
            bg_index = [x.strip() for x in bg_index]

        bg_images_root = osp.join(self.bg_root, "JPEGImages")
        # files names are index.jpg
        bg_images = [osp.join(bg_images_root, f"{x}.jpg") for x in bg_index]
        self.bg_images[split] = bg_images

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
            # the path is the image path, load the rgba image and get aplha mask,
            # then get the bounding box from the mask
            # load the image
            img = Image.open(path)
            img = np.array(img)

            # get the alpha mask
            alpha = img[:, :, 3]
            # get the bounding box
            mask = alpha > 0
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # get the bounding box x0, y0, x1, y1
            boxes.append([cmin, rmin, cmax, rmax])

        return boxes

    def read_poses(self, pose_paths: list, idx: Optional[list] = None) -> list:
        """Read camera poses from file paths.

        Args:
            pose_paths (list): List of camera pose file paths.
            idx (Optional[int], or list, optional): Index of the camera poses to read. Defaults to None.

        Returns:
            list: List of camera poses.
        """

        # load the json file
        json_path = pose_paths[0]
        ret = []
        if idx is None:
            raise ValueError("Index must be provided")
            idx = [i for i in range(len(pose_paths))]
        with open(json_path, "r") as f:
            data = json.load(f)
            ret = [np.array(data["frames"][i]["w2c"]) for i in idx]

        return ret

    def read_intrinsics(self, intri_paths: list, idx: Optional[list] = None) -> list:
        """Read camera intrinsics from file paths.

        Args:
            intri_paths (list): List of camera intrinsics file paths.
            idx (Optional[int], optional): Index of the camera intrinsics to read. Defaults to None.

        Returns:
            list: List of camera intrinsics.
        """

        # load the json file
        json_path = intri_paths[0]
        fx, fy, cx, cy = [], [], [], []
        if idx is None:
            raise ValueError("Index must be provided")
            idx = [i for i in range(len(intri_paths))]
        with open(json_path, "r") as f:
            data = json.load(f)
            # meta data provide fx,fy,cx,cy
            # make intrinsics matrix manually
            fx = [data["frames"][i]["fx"] for i in idx]
            fy = [data["frames"][i]["fy"] for i in idx]
            cx = [data["frames"][i]["cx"] for i in idx]
            cy = [data["frames"][i]["cy"] for i in idx]

        # make intrinsics matrix
        intrinsics = []
        for i in range(len(fx)):
            # this is non-ndc intrinsics
            intrinsics.append(
                np.array([[fx[i], 0, cx[i]], [0, fy[i], cy[i]], [0, 0, 1]])
            )

        return intrinsics
