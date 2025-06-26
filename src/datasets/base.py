"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 21:35:00
Description: This module provides the base dataset class and utilities for loading and processing
pose estimation data. It includes functionalities for image augmentation, camera
intrinsic and extrinsic adjustments, and bounding box operations.

"""
import copy
import random
from typing import Optional, Tuple
import cv2
import numpy as np
import omegaconf
import torch
from PIL import Image
from torch.utils.data import Dataset

from pytorch3d.utils.camera_conversions import (
    cameras_from_opencv_projection,
    opencv_from_cameras_projection,
)

from ..utils.camera_transform import *
from .utils.data_io import *
from .utils.data_utils import *
from .utils.aug import AugmentationProcessor

from omegaconf import DictConfig

from .utils.preprocess import *
import traceback

from .utils.base.sampling_utils import select_ref_indices
from .utils.base.camera_utils import make_proj_bbox
from .utils.base.bbox_utils import (
    extract_bboxes,
    make_mask_by_bbox,
    adjust_bbox_by_proj,
    prepare_bbox3d,
    make_bbox_features,
)
from src.lightning.utils.vis.vis_utils import draw_bbox_heatmap


class BoxDreamerBaseDataset(Dataset):
    """Base class for implementing datasets.

    Provides basic functionalities for data loading and processing.

    BoxDreamer training data is a batch sequence of images and camera
    poses.
    """

    def __init__(self, config: DictConfig, split: str = "train"):
        """Initialize the dataset.

        Args:
            config (DictConfig): Configuration object.
            split (str, optional): Dataset split ('train', 'test', 'val'). Defaults to 'train'.
        """
        self.config = config
        self._validate_split(split)
        self._setup_image_size()
        self._setup_sequence_params()
        self._setup_augmentation_flags()

        # Initialize processors and flags
        self.augmentor_processor = AugmentationProcessor(config=config.augmentation)
        self.compute_optical = config.compute_optical
        self.max_norm = config.max_norm
        self.precision = config.precision
        self.coordinate = config.coordinate

        # Set feature flags with defaults
        self.use_bbox = config.get("use_bbox", True)
        self.use_mask = config.get("use_mask", False)
        self.mask_bg = config.get("mask_bg", True)
        self.ref_aug = config.get("ref_aug", False)
        self.reconstruction_stage = config.get("reconstruction_stage", False)
        self.fps_sampling = config.get("fps_sampling", True)

        # Set representation formats
        self.pose_represenation = config.pose_representation
        self.bbox_representation = config.bbox_representation

        # Initialize data storage containers
        self._init_data_containers()

    def _validate_split(self, split):
        """Validate the dataset split."""
        valid_splits = ["train", "test", "val", "demo"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split {split}. Must be one of {valid_splits}")
        self.split = split

    def _setup_image_size(self):
        """Setup the image size configuration."""
        self.image_size = self.config.image_size  # H, W
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]

        if not isinstance(self.image_size, (list, omegaconf.ListConfig)):
            raise ValueError("Invalid image size format")

    def _setup_sequence_params(self):
        """Setup sequence length and stride parameters."""
        # Length setup
        self.length = self.config.length
        self.dynamic_length = self.config.dynamic_length
        if self.dynamic_length:
            if not isinstance(self.length, (omegaconf.ListConfig, list)):
                raise ValueError("Dynamic length requires a list of [min, max]")
            self.min_length, self.max_length = self.length[0], self.length[1]

        # Stride setup
        self.stride = self.config.stride
        self.dynamic_stride = self.config.dynamic_stride
        if self.dynamic_stride:
            if not isinstance(self.stride, (omegaconf.ListConfig, list)):
                raise ValueError("Dynamic stride requires a list of [min, max]")
            self.min_stride, self.max_stride = self.stride[0], self.stride[1]
        else:
            self.min_stride = self.max_stride = self.stride

        # Sampling flags
        self.random_stride = self.config.random_stride
        self.uniform_sampling = self.config.uniform_sampling

    def _setup_augmentation_flags(self):
        """Setup augmentation flags."""
        self.pose_augmentation = self.config.pose_augmentation
        self.intri_augmentation = self.config.intri_augmentation
        self.mask_augmentation = self.config.get("mask_augmentation", False)

    def _init_data_containers(self):
        """Initialize data storage containers."""
        # Data storage as category-wise paths
        self.images = {}
        self.boxes = {}
        self.poses = {}
        self.intrinsics = {}
        self.model_paths = {}  # Object coordinate settings
        self.cat_len = {}
        self.reproj_box = {}  # OnePose provides reprojection box
        self.bbox_3d = {}  # OnePose and Objaverse provides 3D bounding box
        self.bg_images = None  # Background augmentation (especially for Objaverse)
        self.occlusion_objs = None  # Occlusion augmentation (especially for Objaverse)
        self.dataset = None
        self.lmdb = None

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        if self.reconstruction_stage:
            # return reference len
            return sum(len(self.images["ref"][key]) for key in self.images["ref"])
        else:
            return sum(
                len(self.images[self.split][key]) for key in self.images[self.split]
            )

    def read_images(self, image_paths: list) -> np.ndarray:
        """Read images from paths and apply augmentations if needed.

        Args:
            image_paths (list): List of image file paths.

        Returns:
            np.ndarray: Array of processed images.
        """
        images = []
        for path in image_paths:
            if self.lmdb is not None:
                try:
                    # path is the key of lmdb
                    image = self.lmdb.get(path.encode())
                    # decode the image
                    image = cv2.imdecode(
                        np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR
                    )
                except:
                    traceback.print_exc()
                    image = cv2.imread(path, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.augmentor is not None:
                image = self.augmentor(image)
            images.append(image)

        # Resize images to the smallest shape among them
        min_shape = np.min([image.shape for image in images], axis=0)
        images = [cv2.resize(image, (min_shape[1], min_shape[0])) for image in images]
        images = np.array(images).astype(np.float32) / 255.0
        return images

    def read_images_pil(self, image_paths: list) -> list:
        """Read images from paths using PIL and apply augmentations if needed.

        Args:
            image_paths (list): List of image file paths.

        Returns:
            list: List of processed PIL images.
        """
        images = []
        for path in image_paths:
            if self.lmdb is not None:
                try:
                    # path is the key of lmdb
                    image = self.lmdb.get(path.encode())
                    # decode the image
                    image = np.frombuffer(image, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    # bgr to rgb
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # convert to PIL
                    image = Image.fromarray(image)
                except:
                    traceback.print_exc()
                    image = Image.open(path).convert("RGB")
            else:
                image = Image.open(path).convert("RGB")
            if self.split == "train":
                image = self.augmentor_processor.apply_rgb_augmentation(image)
            images.append(image)
        return images

    def read_boxes(self, box_paths: list, split: Optional[str] = None) -> list:
        """Read bounding boxes from paths. Supports text files or mask images.

        Args:
            box_paths (list): List of bounding box file paths.
            split (Optional[str], optional): Dataset split. Defaults to None.

        Returns:
            list: List of bounding boxes.
        """
        boxes = []
        for path in box_paths:
            if path.endswith((".png", ".jpg", ".jpeg")):
                box = extract_bboxes(path, self.lmdb)
            else:
                if self.lmdb is not None:
                    try:
                        box = self.lmdb.get(path)
                        box = np.frombuffer(box, dtype=np.float32)
                    except:
                        box = np.loadtxt(path)
                else:
                    box = np.loadtxt(path)
            boxes.append(box)
        return boxes

    def read_poses(self, pose_paths: list, idx: Optional[int] = None) -> list:
        """Read poses from file paths. Supports 3x4 or 4x4 matrices.

        Args:
            pose_paths (list): List of pose file paths.

        Returns:
            list: List of homogeneous pose matrices.
        """
        poses = []
        for path in pose_paths:
            pose = np.eye(4)
            if self.lmdb is not None:
                try:
                    pose = self.lmdb.get(path.encode())
                    pose = np.frombuffer(pose, dtype=np.float64)
                    pose = pose.reshape(4, 4)
                except:
                    with open(path, "r") as f:
                        lines = f.readlines()
                        matrix_values = [
                            list(map(float, line.strip().split())) for line in lines
                        ]
                        assert (
                            len(matrix_values) == 3 or len(matrix_values) == 4
                        ) and all(
                            len(row) == 4 for row in matrix_values
                        ), "Pose file must contain a 3x4 or 4x4 matrix."

                        if len(matrix_values) == 4:
                            matrix_3x4 = np.array(matrix_values)[:3, :]
                        else:
                            matrix_3x4 = np.array(matrix_values)

                        R_mat = matrix_3x4[:, :3]
                        T_vec = matrix_3x4[:, 3]

                    pose[:3, :3] = R_mat
                    pose[:3, 3] = T_vec
            else:
                with open(path, "r") as f:
                    lines = f.readlines()
                    matrix_values = [
                        list(map(float, line.strip().split())) for line in lines
                    ]
                    assert (len(matrix_values) == 3 or len(matrix_values) == 4) and all(
                        len(row) == 4 for row in matrix_values
                    ), "Pose file must contain a 3x4 or 4x4 matrix."

                    if len(matrix_values) == 4:
                        matrix_3x4 = np.array(matrix_values)[:3, :]
                    else:
                        matrix_3x4 = np.array(matrix_values)

                    R_mat = matrix_3x4[:, :3]
                    T_vec = matrix_3x4[:, 3]

                pose[:3, :3] = R_mat
                pose[:3, 3] = T_vec

            poses.append(pose)
        return poses

    def read_intrinsics(self, intri_paths: list, idx: Optional[int] = None) -> list:
        """Read camera intrinsic matrices from file paths.

        Args:
            intri_paths (list): List of intrinsic matrix file paths.

        Returns:
            list: List of intrinsic matrices.
        """
        intrinsics = []
        for path in intri_paths:
            intrinsic = np.eye(3)
            if self.lmdb is not None:
                try:
                    intrinsic = self.lmdb.get(path.encode())
                    intrinsic = np.frombuffer(intrinsic, dtype=np.float64)
                    intrinsic = intrinsic.reshape(3, 3)
                except:
                    with open(path, "r") as f:
                        lines = f.readlines()
                        matrix_values = [
                            list(map(float, line.strip().split())) for line in lines
                        ]
                        intrinsic[0, 0] = matrix_values[0][0]
                        intrinsic[1, 1] = matrix_values[1][1]
                        intrinsic[0, 2] = matrix_values[0][2]
                        intrinsic[1, 2] = matrix_values[1][2]
            else:
                with open(path, "r") as f:
                    lines = f.readlines()
                    matrix_values = [
                        list(map(float, line.strip().split())) for line in lines
                    ]

                intrinsic[0, 0] = matrix_values[0][0]
                intrinsic[1, 1] = matrix_values[1][1]
                intrinsic[0, 2] = matrix_values[0][2]
                intrinsic[1, 2] = matrix_values[1][2]

            intrinsics.append(intrinsic)

        return intrinsics

    def load_data(self):
        """Load images and camera data, assigning them to train/test variables.

        This method should be implemented in derived classes.
        """
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def get_model_path(self, cat: str) -> str:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def process_data(
        self,
        images: list,
        image_paths: list,
        boxes: np.ndarray,
        poses: np.ndarray,
        intrinsics: np.ndarray,
        cat: str,
        query_idx: Optional[int] = None,
        neighbor_idx: Optional[int] = None,
        occlusion_objs: Optional[list] = None,
        cad_models: Optional[str] = None,
    ) -> dict:
        """Process and transform the data into a dictionary suitable for model
        input.

        Args:
            images (list): List of PIL images.
            image_paths (list): List of image file paths.
            boxes (np.ndarray): Array of bounding boxes.
            poses (np.ndarray): Array of pose matrices.
            intrinsics (np.ndarray): Array of intrinsic matrices.
            cat (str): Category name.
            query_idx (Optional[int], optional): Query index. Defaults to None.

        Returns:
            dict: Dictionary containing processed data.
        """
        original_shape = []
        images_processed = []
        crop_parameters = []
        original_intrinsics = copy.deepcopy(intrinsics)  # Non-NDC intrinsics
        image_masks = []
        bbox_3d = []
        bbox_3d_original = []

        new_focal_lengths, new_principal_points = [], []

        # Invert poses for 'co3d' dataset
        if self.dataset == "co3d":
            poses = np.linalg.inv(poses)

        if self.pose_augmentation and self.split == "train":
            # rotate the bbox randomly
            poses = self.augmentor_processor.pose_augmentation_R(poses)

        for idx, img in enumerate(images):
            # save pose for debug
            # np.savetxt(f"pose_{idx}.txt", poses[idx])

            original_shape.append(img.size)
            intri = intrinsics[idx]
            model_path = (
                cad_models[idx] if cad_models is not None else self.get_model_path(cat)
            )

            # Prepare 3D bounding box
            if (
                self.dataset != "co3d" or self.split != "train"
            ) or self.pose_represenation == "bb8":
                bbox3d = prepare_bbox3d(model_path, cat, bbox_3d=self.bbox_3d, split=self.split)
                bbox3d = torch.from_numpy(bbox3d).float()
                bbox_3d.append(bbox3d)
                bbox_3d_original.append(bbox3d)
            else:
                bbox_3d.append(None)
                bbox_3d_original.append(None)

            # apply image rotate augmentation
            if self.split == "train":
                (
                    img,
                    poses[idx],
                    boxes[idx],
                    intri,
                ) = self.augmentor_processor.rotate_image(
                    img, poses[idx], boxes[idx], intri
                )

            mask = make_mask_by_bbox(boxes[idx] if self.use_mask else None, img.size)

            if (
                bbox_3d is not None
                and not self.reconstruction_stage
                and (self.split != "demo" or idx != query_idx)
            ):
                proj_bbox = make_proj_bbox(
                    torch.tensor(poses[idx]).unsqueeze(0),
                    torch.tensor(intri).unsqueeze(0),
                    bbox_3d[0],
                )
                # print(boxes[idx])
                new_box = adjust_bbox_by_proj(proj_bbox[0])
                org_box = copy.deepcopy(
                    boxes[idx]
                )  # smaller box, sometimes from detection or segmentation min max
                boxes[
                    idx
                ] = new_box  # 3d bbox's projection maybe larger than the detection box at most time
                # org_box = None # YCBV debug # Debug
            else:
                org_box = None

            if boxes[idx] is not None:
                # ensure the bbox is valid
                dx, dy = boxes[idx][2] - boxes[idx][0], boxes[idx][3] - boxes[idx][1]
                if dx > img.width * 2 and dy > img.height * 2:
                    boxes[idx] = np.array([0, 0, img.width, img.height])

                # ensure bbox in the image region
                img, padding_info = pad_image_based_on_bbox(img, boxes[idx])

                # ensure have correct intrinsics
                if padding_info:
                    # adjust intrinsic
                    # if we make padding, we need to adjust the intrinsic matrix
                    intri = adjust_camera_intrinsics(intri, padding_info)
                    original_intrinsics[idx] = intri.copy()
                    intrinsics[idx] = intri.copy()
                    boxes[idx] = adjust_bbox_by_proj(
                        make_proj_bbox(
                            torch.tensor(poses[idx]).unsqueeze(0),
                            torch.tensor(intri).unsqueeze(0),
                            bbox_3d[-1],
                        )[0]
                    )

            mask_refine = (
                True
                if (
                    self.mask_augmentation
                    and self.split == "train"
                    and (idx == query_idx or self.ref_aug)
                )
                else False
            )  # only truncate the query image and only for training
            # mask_refine = True # for debug
            occlusion_refine = (
                True
                if (self.split == "train" and (idx == query_idx or self.ref_aug))
                else False
            )  # only truncate the query image and only for training
            if self.bg_images is not None:
                # random select one background image from the list and use pil image
                bg_img = Image.open(
                    self.bg_images[self.split][
                        random.randint(0, len(self.bg_images[self.split]) - 1)
                    ]
                ).convert("RGB")
                # bg_img = self.augmentor_processor.apply_rgb_augmentation(bg_img)
                fg_img = bg_img.copy()
                img = self.augmentor_processor.add_background(img, bg_img)

            if occlusion_objs is not None and occlusion_refine:
                # random select one occlusion object from the list and use pil image
                img = self.augmentor_processor.random_paste_objs(
                    img, occlusion_objs, org_box
                )

            # Pad and resize image
            (
                image_transformed,
                mask_transformed,
                crop_param,
                bbox,
            ) = pad_and_resize_image(
                image=img,
                crop_longest=True,
                img_size=self.image_size,
                mask=mask,
                transform=None,
                bbox_anno=square_bbox(
                    boxes[idx] if self.use_bbox else None
                ),  # ensure the crop is square, so that the resize op has no need to adjust K
                bbox_obj=org_box
                if self.use_bbox and self.mask_bg
                else None,  # no need to make square, used for bg masking
                truncate_augmentation=self.augmentor_processor.random_truncate_image_with_bbox
                if mask_refine
                else None,
                mask_augmentation=self.augmentor_processor.random_mask_image_with_bbox
                if mask_refine
                else None,
                mask_image=fg_img if self.bg_images is not None else None,
            )

            crop_parameters.append(crop_param)
            images_processed.append(image_transformed)
            image_masks.append(mask_transformed)

            # Adjust intrinsic parameters
            fl = torch.tensor([intri[0, 0], intri[1, 1]], dtype=torch.float32)
            pp = intri[:2, 2]
            bbox_xywh = torch.tensor(bbox_xyxy_to_xywh(bbox), dtype=torch.float32)
            fl, pp = convert_pixels_to_ndc(
                fl, pp, torch.tensor(img.size, dtype=torch.float32)
            )

            focal_length_cropped, principal_point_cropped = adjust_camera_to_bbox_crop_(
                fl,
                pp,
                torch.tensor(img.size, dtype=torch.float32),
                bbox_xywh,
            )
            new_focal_length, new_principal_point = adjust_camera_to_image_scale_(
                focal_length_cropped,
                principal_point_cropped,
                torch.tensor(bbox_xywh[2:], dtype=torch.float32),
                torch.tensor(
                    [self.image_size[0], self.image_size[1]], dtype=torch.float32
                ),
            )
            new_focal_lengths.append(new_focal_length)
            new_principal_points.append(new_principal_point)  # NDC intrinsics

        original_shape = torch.tensor(original_shape, dtype=torch.float32)
        images = torch.stack(images_processed)
        crop_parameters = torch.stack(crop_parameters)
        image_masks = torch.stack(image_masks)
        original_intrinsics = torch.tensor(original_intrinsics, dtype=torch.float32)
        bbox_3d = torch.stack(bbox_3d)
        bbox_3d_original = torch.stack(bbox_3d_original)

        original_poses = copy.deepcopy(poses)  # Original poses
        poses = torch.from_numpy(poses).float()
        original_poses = torch.from_numpy(original_poses).float()

        new_focal_lengths = torch.stack(new_focal_lengths)
        new_principal_points = torch.stack(new_principal_points)

        fl = new_focal_lengths.clone()
        pp = new_principal_points.clone()

        intrinsics = torch.from_numpy(intrinsics).float()

        # Update intrinsic matrices
        intrinsics[:, 0, 0] = fl[:, 0].clone()
        intrinsics[:, 1, 1] = fl[:, 1].clone()
        intrinsics[:, :2, 2] = pp.clone()

        non_ndc_intrinsics = intrinsics.clone()

        fl, pp = convert_ndc_to_pixels(
            fl,
            pp,
            torch.tensor([self.image_size[0], self.image_size[1]], dtype=torch.float32),
        )

        non_ndc_intrinsics[:, 0, 0] = fl[:, 0]
        non_ndc_intrinsics[:, 1, 1] = fl[:, 1]
        non_ndc_intrinsics[:, :2, 2] = pp

        batchR = poses[:, :3, :3].clone()
        batchT = poses[:, :3, 3].clone()

        cameras = cameras_from_opencv_projection(
            batchR,
            batchT,
            non_ndc_intrinsics,
            torch.tensor(
                [self.image_size[0], self.image_size[1]], dtype=torch.float32
            ).expand(len(batchR), 2),
        )

        normalized_cameras, _, scale, coordinate_transform = normalize_cameras(
            cameras,
            points=None,
            first_camera=self.coordinate == "first_camera",
            compute_optical=False,  # TODO support optical alignment in the future)
            normalize_trans=False,
            max_norm=self.max_norm,
            query_idx=query_idx,
        )

        if normalized_cameras == -1:
            raise RuntimeError("Error in normalizing cameras: camera scale was 0")
        if isinstance(scale, float):
            scale = torch.tensor([scale, scale, scale], dtype=torch.float32)
            scale = scale.expand_as(normalized_cameras.T)

        elif scale.shape.__len__() == 0:
            # repeat to T shape : B, 3
            scale = scale.expand_as(normalized_cameras.T)

        batchR = normalized_cameras.R
        batchT = normalized_cameras.T

        batchR, batchT, _ = opencv_from_cameras_projection(
            normalized_cameras,
            torch.tensor(
                [self.image_size[0], self.image_size[1]], dtype=torch.float32
            ).expand(len(batchR), 2),
        )

        if self.coordinate == "first_camera":
            coordinate_transform = original_poses[0].clone()
            # rotate R 180 degree around z axis
            rotate_180 = torch.tensor(
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32
            )
            batchR = batchR @ rotate_180
        else:
            coordinate_transform = torch.eye(4)

        poses[:, :3, :3] = batchR
        poses[:, :3, 3] = batchT

        if bbox_3d[0] is not None:
            # Project 3D points to 2D
            bbox_proj_crop = make_proj_bbox(
                poses, non_ndc_intrinsics, bbox_3d
            )  # pixel coordinate
            if self.pose_represenation == "bb8":
                bbox_feat = make_bbox_features(
                    bbox_proj_crop,
                    type=self.bbox_representation,
                    shape=(self.image_size[0], self.image_size[1]),
                )
                # print(f"bbox_feat range: {bbox_feat.min()}, {bbox_feat.max()}")
                # exit()
            else:
                bbox_feat = None
            # # bbox_heatmap = self.make_bbox_features(bbox_proj_crop, type='heatmap', shape=(self.image_size[0], self.image_size[1]))
            # vis = draw_bbox_heatmap(bbox_feat.permute(0, 2, 3, 1), rgb=images.clone())
            # normalized_bbox_feat = bbox_feat.clone() / torch.tensor([self.image_size[0], self.image_size[1]], dtype=torch.float32)
            # log data range
            # print(f"bbox_feat range: {bbox_feat.min()}, {bbox_feat.max()}")
            # print(f"bbox_heatmap range: {bbox_heatmap.min()}, {bbox_heatmap.max()}")
            # print(f"normalized_bbox_feat range: {normalized_bbox_feat.min()}, {normalized_bbox_feat.max()}")
            # exit()

            normalized_bbox_proj_crop = bbox_proj_crop.clone() / torch.tensor(
                [self.image_size[0], self.image_size[1]], dtype=torch.float32
            )
            normalized_bbox_proj_crop = normalized_bbox_proj_crop * 2 - 1
            normalized_bbox_proj_crop = torch.clamp(
                normalized_bbox_proj_crop, min=-5, max=5
            )

        if self.precision in ["16", "16-mixed"]:
            tensor_type = torch.float16
        elif self.precision == "32":
            tensor_type = torch.float32
        elif self.precision == "bf16":
            tensor_type = torch.bfloat16
        else:
            raise ValueError(f"Unsupported precision dtype {self.precision}")

        # poses[query_idx] = torch.eye(4, dtype=tensor_type) # ensure no bug in query pose inference
        batch = {
            "images": array_to_tensor(images, tensor_type),
            "original_images": image_paths,
            "intrinsics": array_to_tensor(intrinsics, tensor_type),  # ndc
            "non_ndc_intrinsics": array_to_tensor(
                non_ndc_intrinsics, tensor_type
            ),  # non-ndc
            "original_intrinsics": array_to_tensor(
                original_intrinsics, tensor_type
            ),  # non-ndc
            "poses": array_to_tensor(poses, tensor_type),  # opencv
            "original_poses": array_to_tensor(
                original_poses, tensor_type
            ),  # opencv. no matter which coordinate system, the original poses are in object coordinate system
            "coordinate_transform": array_to_tensor(
                coordinate_transform, tensor_type
            ),  # coordinate transform matrix)
            "scale": array_to_tensor(scale, tensor_type),
            "crop_parameters": array_to_tensor(crop_parameters, tensor_type),
            "image_masks": array_to_tensor(image_masks, tensor_type),
            "original_shape": array_to_tensor(original_shape, tensor_type),
            "model_path": cad_models
            if cad_models is not None and cad_models[0] is not None
            else "none",
            "query_idx": query_idx if query_idx is not None else "none",
            "neighbor_idx": neighbor_idx if neighbor_idx is not None else "none",
            "dataset": self.dataset,
        }

        if self.split == "test":
            batch["cat"] = cat.split("_")[0]

        if bbox_3d is not None:
            batch["bbox_3d"] = array_to_tensor(bbox_3d, tensor_type)
            batch["bbox_proj_crop"] = array_to_tensor(
                normalized_bbox_proj_crop, tensor_type
            )
            batch["bbox_3d_original"] = array_to_tensor(bbox_3d_original, tensor_type)

            if self.pose_represenation == "bb8":
                batch["bbox_feat"] = array_to_tensor(bbox_feat, tensor_type)

        return batch

    def _get_category(self, idx: int, split: Optional[str] = None) -> Tuple[str, int]:
        """Retrieve the category and index within that category for a given
        dataset split.

        Args:
            idx (int): Index within the split.
            split (Optional[str], optional): Dataset split ('train', 'val', 'test'). Defaults to None.

        Returns:
            Tuple[str, int]: (category, index_within_category)
        """
        if split is None:
            split = self.split
        categories = list(self.cat_len[split].keys())
        cum_counts = np.cumsum([self.cat_len[split][cat] for cat in categories])
        cat_idx = np.searchsorted(cum_counts, idx, side="right")
        category = categories[cat_idx]
        index_within = idx if cat_idx == 0 else idx - cum_counts[cat_idx - 1]
        return category, index_within

    def _get_category_from_split(self, split: str, category: str) -> Tuple[str, int]:
        """Retrieve a random index from a specified split and category.

        Args:
            split (str): Dataset split ('train', 'val', 'test').
            category (str): Category name.

        Returns:
            Tuple[str, int]: (category, random_index_within_category)
        """
        max_idx = self.cat_len[split][category] - 1
        random_idx = np.random.randint(0, max_idx + 1)
        return category, random_idx

    def __getitem__(self, idx_len: Tuple[int, Optional[int]]) -> dict:
        """Retrieve a data sample consisting of a query from the current split
        and references from the training set.

        Args:
            idx_len (Tuple[int, Optional[int]]): Tuple containing index and batch length.

        Returns:
            dict: Processed data containing images, intrinsics, poses, etc.
        """
        if isinstance(idx_len, tuple):
            idx, batch_len = idx_len
        else:
            idx, batch_len = idx_len, None

        # Get category and query index
        if self.reconstruction_stage:
            query_cat, query_q_idx = self._get_category(idx, split="ref")
        else:
            query_cat, query_q_idx = self._get_category(idx, split=self.split)

        # Determine maximum reference index
        max_ref_idx = self.cat_len["ref"][query_cat] - 1

        # Define the length of the reference sequence
        if self.dynamic_length:
            if batch_len is not None:
                length = batch_len
            else:
                raise ValueError("Dynamic length requires batch_len to be specified")
        else:
            assert isinstance(self.length, int), "Length should be an integer"
            length = self.length

        ref_length = length - 1
        if self.reconstruction_stage:
            ref_length = 1

        # Select reference indices
        if not self.reconstruction_stage:
            ref_idxs = select_ref_indices(
                random_stride=self.random_stride,
                fps_sampling=self.fps_sampling,
                uniform_sampling=self.uniform_sampling,
                stride=self.stride,
                dynamic_stride=self.dynamic_stride,
                min_stride=self.min_stride,
                max_stride=self.max_stride,
                ref_length=ref_length,
                max_ref_idx=max_ref_idx,
                images=self.images,
                query_cat=query_cat,
                query_q_idx=query_q_idx,
                dataset=self.dataset,
            )
        else:
            ref_idxs = [query_q_idx]
            ref_idxs = np.array(ref_idxs)

        # Ensure the query index is not in the reference indices (except for Linemod dataset)
        if (
            query_q_idx in ref_idxs
            and self.dataset not in ["linemod", "ycbv", "linemod-o", "onepose"]
            and not self.reconstruction_stage
            and not self.split == "demo"
        ):
            ref_idxs = ref_idxs[ref_idxs != query_q_idx]
            while len(ref_idxs) < ref_length:
                new_idx = np.random.randint(0, max_ref_idx + 1)
                if new_idx not in ref_idxs and new_idx != query_q_idx:
                    ref_idxs = np.append(ref_idxs, new_idx)

        # Shuffle reference indices if in training split
        if self.split == "train":
            np.random.shuffle(ref_idxs)

        query_idx_position = len(ref_idxs)
        # neightbor_id is the index which is the nearest to the query_idx_position in ref_idxs
        neighbor_idx = np.argmin(np.abs(ref_idxs - query_q_idx))

        # Read reference data
        ref_images = self.read_images_pil(
            [self.images["ref"][query_cat][i] for i in ref_idxs]
        )
        ref_boxes = self.read_boxes(
            [self.boxes["ref"][query_cat][i] for i in ref_idxs], "ref"
        )
        ref_poses = self.read_poses(
            [self.poses["ref"][query_cat][i] for i in ref_idxs], ref_idxs
        )
        ref_intrinsics = self.read_intrinsics(
            [self.intrinsics["ref"][query_cat][i] for i in ref_idxs], ref_idxs
        )

        if not self.reconstruction_stage:
            # Read query data
            query_image = self.read_images_pil(
                [self.images[self.split][query_cat][query_q_idx]]
            )
            query_box = self.read_boxes(
                [self.boxes[self.split][query_cat][query_q_idx]], self.split
            )
            query_pose = self.read_poses(
                [self.poses[self.split][query_cat][query_q_idx]], [query_q_idx]
            )
            query_intrinsic = self.read_intrinsics(
                [self.intrinsics[self.split][query_cat][query_q_idx]], [query_q_idx]
            )

            # Combine reference and query data
            images = ref_images + query_image
            boxes = np.array(ref_boxes + query_box)
            poses = np.array(ref_poses + query_pose)
            intrinsics = np.array(ref_intrinsics + query_intrinsic)
            image_paths = [self.images["ref"][query_cat][i] for i in ref_idxs] + [
                self.images[self.split][query_cat][query_q_idx]
            ]
        else:
            images = ref_images
            boxes = np.array(ref_boxes)
            poses = np.array(ref_poses)
            intrinsics = np.array(ref_intrinsics)
            image_paths = [self.images["ref"][query_cat][i] for i in ref_idxs]
        assert (
            len(images) == len(boxes) == len(poses) == len(intrinsics) == length
        ), "Data length mismatch"

        if self.occlusion_objs is not None:
            # random select 1~5 objects to occlude the query image
            indices = np.random.choice(
                len(self.occlusion_objs), size=random.randint(1, 5), replace=False
            )
            occlusion_objs = [self.occlusion_objs[i] for i in indices]
            occlusion_objs = self.read_images_pil(occlusion_objs)
        else:
            occlusion_objs = None

        if (
            query_cat not in self.model_paths
            and not self.reconstruction_stage
            and not self.split == "demo"
        ):
            # not seq specific model path
            ref_cad_models = [self.model_paths["ref"][query_cat][i] for i in ref_idxs]
            query_cad_model = self.model_paths[self.split][query_cat][query_q_idx]
            cad_models = ref_cad_models + [query_cad_model]

        else:
            cad_models = self.model_paths.get(query_cat)
            # expand to a list
            cad_models = [cad_models] * length

        # Process and return the data
        try:
            return self.process_data(
                images=images,
                image_paths=image_paths,
                boxes=boxes,
                poses=poses,
                intrinsics=intrinsics,
                cat=query_cat,
                query_idx=query_idx_position,
                neighbor_idx=neighbor_idx,
                occlusion_objs=occlusion_objs,
                cad_models=cad_models,
            )
        except NotImplementedError as e:
            raise e
        except Exception as e:
            print(f"Error occured while loading data ... \n Error log: {e}")
            print(traceback.format_exc())
            print(
                f"For training/testing stablity, we catch the exception and trying to load data at index:{idx + 1}"
            )
            return self.__getitem__(
                (0 if idx + 1 == self.__len__() else idx + 1, batch_len)
            )

    def set_length(self, length: int):
        """Set the sequence length dynamically.

        Args:
            length (int): New sequence length.
        """
        if self.dynamic_length:
            self.length = length
