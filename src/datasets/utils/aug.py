"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:08:03
Description: Dataset augmentation functions

"""

from PIL import Image, ImageDraw
from .utils_phoaug import *
import numpy as np
import cv2
import random
from typing import List, Tuple, Optional, Union
from .preprocess import adjust_intrinsic_matrix
from pathlib import Path
from itertools import chain
from functools import partial
from loguru import logger
import albumentations as A
import omegaconf


class AugmentationProcessor:
    """AugmentationProcessor class that provides various image augmentation
    techniques."""

    def __init__(self, config):
        self.rgb_augmethods = config["rgb_augmethods"]
        # check if rgb_augmethods is valid or None
        if self.rgb_augmethods is not None:
            for augmethod in self.rgb_augmethods:
                if augmethod not in ["dark", "mobile", "YCBAug", "FDA"]:
                    raise ValueError(f"Unknown augmentation method: {augmethod}")

        self.obj_truncation_ratio = config["obj_truncation_ratio"]
        # ratio should be a float between 0 and 1 or List of floats between 0 and 1 [min, max]
        if isinstance(self.obj_truncation_ratio, float):
            if self.obj_truncation_ratio < 0 or self.obj_truncation_ratio > 1:
                raise ValueError(
                    "obj_truncation_ratio should be a float between 0 and 1."
                )
        elif isinstance(self.obj_truncation_ratio, omegaconf.ListConfig):
            if self.obj_truncation_ratio[0] < 0 or self.obj_truncation_ratio[0] > 1:
                raise ValueError(
                    "obj_truncation_ratio[0] should be a float between 0 and 1."
                )
            if self.obj_truncation_ratio[1] < 0 or self.obj_truncation_ratio[1] > 1:
                raise ValueError(
                    "obj_truncation_ratio[1] should be a float between 0 and 1."
                )

            # convert to normal list
            self.obj_truncation_ratio = list(self.obj_truncation_ratio)

        self.enable_image_rotation = config.get("enable_image_rotation", False)

        self.obj_mask_ratio = config.get("obj_mask_ratio", None)

        if isinstance(self.obj_mask_ratio, float):
            if self.obj_mask_ratio < 0 or self.obj_mask_ratio > 1:
                raise ValueError("obj_mask_ratio should be a float between 0 and 1.")
        elif isinstance(self.obj_mask_ratio, omegaconf.ListConfig):
            if self.obj_mask_ratio[0] < 0 or self.obj_mask_ratio[0] > 1:
                raise ValueError("obj_mask_ratio[0] should be a float between 0 and 1.")
            if self.obj_mask_ratio[1] < 0 or self.obj_mask_ratio[1] > 1:
                raise ValueError("obj_mask_ratio[1] should be a float between 0 and 1.")
            # convert to normal list
            self.obj_mask_ratio = list(self.obj_mask_ratio)

        self.obj_paste_prob = config.get(
            "obj_paste_prob", 0.0
        )  # default is not to paste objects

    def apply_dark_aug(self, image: np.ndarray) -> np.ndarray:
        """Apply dark augmentation by adjusting brightness, contrast, blur,
        motion blur, gamma, and hue/saturation.

        Args:
            image (np.ndarray): Input image in NumPy array format.

        Returns:
            np.ndarray: Augmented image.
        """
        augmentor = A.Compose(
            [
                A.RandomBrightnessContrast(
                    p=0.75, brightness_limit=(-0.6, 0.0), contrast_limit=(-0.5, 0.3)
                ),
                A.Blur(p=0.1, blur_limit=(3, 9)),
                A.MotionBlur(p=0.2, blur_limit=(3, 25)),
                A.RandomGamma(p=0.1, gamma_limit=(15, 65)),
                A.HueSaturationValue(p=0.1, val_shift_limit=(-100, -40)),
            ],
            p=0.75,
        )
        augmented = augmentor(image=image)["image"]
        return augmented

    def apply_mobile_aug(self, image: np.ndarray) -> np.ndarray:
        """Apply mobile-specific augmentations such as motion blur, color
        jitter, random rain, and ISO noise.

        Args:
            image (np.ndarray): Input image in NumPy array format.

        Returns:
            np.ndarray: Augmented image.
        """
        augmentor = A.Compose(
            [
                A.MotionBlur(p=0.25),
                A.ColorJitter(p=0.5),
                A.RandomRain(p=0.1),
                A.ISONoise(p=0.25),
            ],
            p=1.0,
        )
        augmented = augmentor(image=image)["image"]
        return augmented

    def apply_ycb_aug(self, image: np.ndarray) -> np.ndarray:
        """Apply YCB-specific augmentations including ISONoise, GaussNoise, and
        GaussianBlur.

        Args:
            image (np.ndarray): Input image in NumPy array format.

        Returns:
            np.ndarray: Augmented image.
        """
        augmentor = A.Compose(
            [
                A.ISONoise(intensity=(0.4, 0.9), p=0.25),
                A.GaussNoise(var_limit=(100, 300), p=0.7),
                A.GaussianBlur(sigma_limit=10, p=0.7),
            ],
            p=1.0,
        )
        augmented = augmentor(image=image)["image"]
        return augmented

    def apply_stylization(
        self,
        image: np.ndarray,
        ref_root: str = "assets/isrf",
        method: str = "FDA",
        beta_limit: float = 0.05,
        p: float = 0.5,
    ) -> np.ndarray:
        """Apply stylization augmentation using reference images with the FDA
        (Fourier Domain Adaptation) method.

        Args:
            image (np.ndarray): Input image in NumPy array format.
            ref_root (str, optional): Directory containing reference images. Defaults to "assets/isrf".
            method (str, optional): Stylization method to use. Currently only "FDA" is implemented. Defaults to "FDA".
            beta_limit (float, optional): Beta limit for FDA. Defaults to 0.05.
            p (float, optional): Probability of applying FDA. Defaults to 0.5.

        Returns:
            np.ndarray: Augmented image.
        """
        if not hasattr(self, "ref_imgs"):
            self.load_reference_images(ref_root, method)

        stylizer = partial(A.FDA, beta_limit=beta_limit, p=p, read_fn=lambda x: x)
        augmentor = A.Compose([stylizer([random.choice(self.ref_imgs)])])
        augmented = augmentor(image=image)["image"]
        return augmented

    def load_reference_images(self, ref_root: str, method: str):
        """Load reference images for stylization.

        Args:
            ref_root (str): Directory containing reference images.
            method (str): Stylization method.
        """
        logger.info(f"Loading reference images from {ref_root} for method {method}...")
        f_names = list(
            chain(
                *[
                    Path(ref_root).glob(f"**/*.{ext}")
                    for ext in ["png", "jpg", "jpeg", "JPEG"]
                ]
            )
        )
        self.ref_imgs = [
            cv2.cvtColor(cv2.imread(str(fn)), cv2.COLOR_BGR2RGB)
            for fn in f_names
            if cv2.imread(str(fn)) is not None
        ]
        logger.info(f"Loaded {len(self.ref_imgs)} reference images for stylization.")

    def apply_rgb_augmentation(
        self, image: Image.Image, methods: Optional[List[str]] = None, **kwargs
    ) -> Image.Image:
        """Apply selected augmentations to the image.

        Args:
            image (PIL.Image.Image): The image to augment.
            methods (List[str], optional): List of augmentation methods to apply.
                                           Options: "dark", "mobile", "YCBAug", "FDA".
                                           If None, no augmentation is applied.
            **kwargs: Additional keyword arguments for specific augmentation methods.

        Returns:
            PIL.Image.Image: The augmented image.
        """
        if methods is None and self.rgb_augmethods is not None:
            methods = self.rgb_augmethods
        elif methods is None:
            return image

        image_np = np.array(image)

        for method in methods:
            if method == "dark":
                image_np = self.apply_dark_aug(image_np)
            elif method == "mobile":
                image_np = self.apply_mobile_aug(image_np)
            elif method == "YCBAug":
                image_np = self.apply_ycb_aug(image_np)
            elif method == "FDA":
                ref_root = kwargs.get("ref_root", "assets/isrf")
                method_style = kwargs.get("style_method", "FDA")
                beta_limit = kwargs.get("beta_limit", 0.05)
                p = kwargs.get("p", 0.5)
                image_np = self.apply_stylization(
                    image_np,
                    ref_root=ref_root,
                    method=method_style,
                    beta_limit=beta_limit,
                    p=p,
                )
            else:
                raise ValueError(f"Unknown augmentation method: {method}")

        augmented_image = Image.fromarray(image_np)
        return augmented_image

    def random_truncate_image_with_bbox(
        self,
        image: Image.Image,
        bbox: np.ndarray = None,
        mask_ratio: Optional[
            Union[float, Tuple[float, float], None]
        ] = None,  # (0., 0.5)
    ) -> Image.Image:
        """Randomly mask the image with the given bounding box.

        Args:
            image (PIL.Image.Image): Image to be masked
            bbox (np.ndarray): Bounding box coordinates in the format [x_min, y_min, x_max, y_max]
            mask_ratio (float or tuple of floats): Ratio of the masked region to the whole image or bbox region

        Returns:
            PIL.Image.Image: Masked image
        """
        if mask_ratio is None and self.obj_truncation_ratio is not None:
            mask_ratio = self.obj_truncation_ratio
        elif mask_ratio is None:
            # no ratio is given, return the original image
            return image

        width, height = image.size
        if bbox is None:
            x_min, y_min, x_max, y_max = 0, 0, width, height
        else:
            x_min, y_min, x_max, y_max = bbox

        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        # randomly select the side to mask
        mask_side = random.choice(["left", "top", "right", "bottom"])

        if isinstance(mask_ratio, Tuple) or isinstance(mask_ratio, list):
            min_ratio, max_ratio = mask_ratio
            mask_ratio = random.uniform(min_ratio, max_ratio)

        if mask_side == "right":
            x_max = int(x_min + (x_max - x_min) * (1 - mask_ratio))
        elif mask_side == "bottom":
            y_max = int(y_min + (y_max - y_min) * (1 - mask_ratio))
        elif mask_side == "left":
            x_min = int(x_max - (x_max - x_min) * (1 - mask_ratio))
        else:
            y_min = int(y_max - (y_max - y_min) * (1 - mask_ratio))

        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

        # apply the mask to the image
        image = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)

        return image

    def pose_augmentation_R(self, poses: np.ndarray) -> np.ndarray:
        """Apply random rotation on a batch of poses for data augmentation.

        Args:
            poses (np.ndarray): Array of poses with shape (batch_size, 4, 4)

        Returns:
            np.ndarray: Rotated poses.
        """
        # Generate random rotation angles
        rot_x = np.random.uniform(-np.pi, np.pi)
        rot_y = np.random.uniform(-np.pi, np.pi)
        rot_z = np.random.uniform(-np.pi, np.pi)

        # Rotation matrices around each axis
        rot_x_mat = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rot_x), -np.sin(rot_x)],
                [0, np.sin(rot_x), np.cos(rot_x)],
            ]
        )

        rot_y_mat = np.array(
            [
                [np.cos(rot_y), 0, np.sin(rot_y)],
                [0, 1, 0],
                [-np.sin(rot_y), 0, np.cos(rot_y)],
            ]
        )

        rot_z_mat = np.array(
            [
                [np.cos(rot_z), -np.sin(rot_z), 0],
                [np.sin(rot_z), np.cos(rot_z), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix
        rot_mat = rot_x_mat @ rot_y_mat @ rot_z_mat

        poses_rotated = poses.copy()

        for i in range(poses.shape[0]):
            poses_rotated[i, :3, :3] = poses[i, :3, :3] @ rot_mat
            poses_rotated[i, :3, 3] = poses[i, :3, 3]

        return poses_rotated

    def random_crop_and_scale(
        self,
        image: Image.Image,
        K: np.ndarray,
        bbox: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        crop_scale_range: Tuple[float, float] = (0.5, 1.0),
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.33),
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        """Perform random cropping and scaling on an image, adjusting camera
        intrinsics and bounding boxes accordingly.

        Args:
            image (PIL.Image.Image): Input image.
            K (np.ndarray): Original camera intrinsic matrix of shape (3, 3).
            bbox (np.ndarray): Bounding box [x_min, y_min, x_max, y_max].
            scale_range (Tuple[float, float], optional): Range for random scaling factors. Defaults to (0.8, 1.2).
            crop_scale_range (Tuple[float, float], optional): Range for cropping scale relative to the original image. Defaults to (0.5, 1.0).
            aspect_ratio_range (Tuple[float, float], optional): Range for cropping aspect ratio. Defaults to (0.75, 1.33).

        Returns:
            Tuple containing:
                - Transformed image.
                - Adjusted camera intrinsic matrix of shape (3, 3).
                - Adjusted bounding box [x_min, y_min, x_max, y_max].
        """
        img_width, img_height = image.width, image.height
        x_min, y_min, x_max, y_max = bbox

        # Random scaling factor
        scale_factor = random.uniform(*scale_range)

        # Resize image based on scaling factor
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        # Adjust camera intrinsic matrix
        K_scaled = adjust_intrinsic_matrix(
            K=K,
            scale=(scale_factor, scale_factor),
            crop_offset=(0, 0),  # No crop offset initially
        )

        # Scale bounding box
        bbox_scaled = bbox * scale_factor

        # Random cropping parameters
        crop_scale = random.uniform(*crop_scale_range)
        aspect_ratio = random.uniform(*aspect_ratio_range)

        crop_width = int(new_width * crop_scale)
        crop_height = int(crop_width / aspect_ratio)

        # Ensure cropping dimensions do not exceed image size
        if crop_height > new_height:
            crop_height = new_height
            crop_width = int(crop_height * aspect_ratio)

        # Randomly select top-left corner for cropping
        max_left = new_width - crop_width
        max_top = new_height - crop_height

        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0

        # Crop the image
        right = left + crop_width
        bottom = top + crop_height
        image = image.crop((left, top, right, bottom))

        # Adjust camera intrinsic matrix based on cropping
        K_cropped = K_scaled.copy()
        K_cropped[0, 2] -= left
        K_cropped[1, 2] -= top

        # Final intrinsic matrix remains scaled
        K_new = K_cropped

        # Adjust bounding box based on cropping
        bbox_cropped = bbox_scaled - np.array([left, top, left, top])

        # Clamp bounding box to cropping area
        bbox_cropped = np.clip(
            bbox_cropped,
            a_min=0,
            a_max=[crop_width, crop_height, crop_width, crop_height],
        )

        return image, K_new, bbox_cropped

    def add_background(
        self, image: Image.Image, background: Image.Image
    ) -> Image.Image:
        """Add background to the image (only for objaverse / gso datasets which
        have no background).

        Args:
            image (PIL.Image.Image): Image to be added background.
            background (PIL.Image.Image): Background image.

        Returns:
            PIL.Image.Image: Image with background added.
        """
        # Resize the background to match the image size
        background = background.resize((image.width, image.height))
        # Convert images to numpy arrays
        image_np = np.array(image)

        white_threshold = 245

        # Create mask where white pixels are detected
        mask_np = np.all(image_np > white_threshold, axis=-1)
        mask_np = np.uint8(~mask_np * 255)

        mask_foreground = Image.fromarray(mask_np, mode="L")

        # Composite the foreground with the new background
        image = Image.composite(image, background, mask_foreground)

        return image

    def rotate_image(
        self,
        image: Image.Image,
        pose: np.ndarray,
        bbox: Optional[np.ndarray],
        intrinsic_matrix: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Rotate an image by a random angle, adjusting camera pose, intrinsic
        matrix, and bounding boxes accordingly.

        Args:
            image (PIL.Image.Image): Input image.
            pose (np.ndarray): Camera pose of shape (4, 4).
            bbox (np.ndarray): Bounding box [x_min, y_min, x_max, y_max].
            intrinsic_matrix (np.ndarray): Camera intrinsic matrix of shape (3, 3).

        Returns:
            Tuple containing:
                - Rotated image.
                - Adjusted camera pose of shape (4, 4).
                - Adjusted bounding box [x_min, y_min, x_max, y_max].
                - Adjusted camera intrinsic matrix of shape (3, 3).
        """
        if self.enable_image_rotation == False:
            return image, pose, bbox, intrinsic_matrix
        # Step 1: Generate a random rotation angle (e.g., between -45 and 45 degrees)
        angle = np.random.uniform(-45, 45)  # Random rotation angle
        theta = np.radians(angle)  # Convert to radians

        # Step 2: Rotate the image
        w, h = image.size  # Original image dimensions (width, height)
        # Original optical center (c_x, c_y)
        c_x, c_y = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        rotated_image = image.rotate(angle, expand=True, center=(c_x, c_y))
        new_w, new_h = rotated_image.size  # New image dimensions after rotation

        # Step 3: Adjust the pose matrix
        # The pose matrix is a 4x4 matrix: [R | t; 0 0 0 1], where R is a 3x3 rotation matrix
        R_rotate = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )  # Rotation matrix for in-plane rotation

        new_pose = pose.copy()
        # Assuming pose is camera-to-world transformation
        c2w = np.linalg.inv(pose)  # World-to-camera transformation
        # Rotate the world-to-camera transformation
        c2w[:3, :3] = c2w[:3, :3] @ R_rotate
        # Update the camera-to-world transformation
        new_pose = np.linalg.inv(c2w)

        # Step 4: Adjust the intrinsic matrix
        # Calculate the translation due to image rotation with expansion
        translate_x = (new_w - w) / 2
        translate_y = (new_h - h) / 2
        # Update the principal point without rotation
        new_cx = c_x + translate_x
        new_cy = c_y + translate_y
        # Update the intrinsic matrix
        new_intrinsic_matrix = intrinsic_matrix.copy()
        new_intrinsic_matrix[0, 2] = new_cx
        new_intrinsic_matrix[1, 2] = new_cy

        # Step 5: Adjust the bounding box by applying the same rotation
        # bbox format: [x_min, y_min, x_max, y_max]
        # Ensure bbox is in the correct format
        if bbox is None:
            return rotated_image, new_pose, None, new_intrinsic_matrix
        if len(bbox) != 4:
            raise ValueError(
                "Bounding box must be of the format [x_min, y_min, x_max, y_max]."
            )
        x_min, y_min, x_max, y_max = bbox

        # Create a mask image with the same size as the input image
        mask = Image.new("L", image.size, 0)  # Black mask
        draw = ImageDraw.Draw(mask)
        # Draw the bounding box on the mask
        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

        # Rotate the mask
        rotated_mask = mask.rotate(angle, expand=True, center=(c_x, c_y))
        # Get the rotated bounding box
        new_bbox = rotated_mask.getbbox()  # Returns (x_min, y_min, x_max, y_max)

        return rotated_image, new_pose, new_bbox, new_intrinsic_matrix

    def random_paste_objs(
        self, image: Image.Image, objs: list, bbox: Optional[np.ndarray] = None
    ) -> Image.Image:
        """Paste other objects on the image with at least 30% overlap with the
        bounding box.

        Args:
            image (PIL.Image.Image): Image to paste objects on.
            objs (List[PIL.Image.Image]): List of object images to paste.
            bbox (np.ndarray): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].

        Returns:
            PIL.Image.Image: Image with objects pasted.
        """
        if random.random() > self.obj_paste_prob or self.obj_paste_prob is None:
            return image
        # Validate bbox
        image_width, image_height = image.size
        if bbox is None:
            x_min, y_min, x_max, y_max = 0, 0, image_width, image_height
            bbox = [x_min, y_min, x_max, y_max]
        else:
            x_min, y_min, x_max, y_max = bbox
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                "Invalid bbox: x_min must be less than x_max and y_min must be less than y_max"
            )
        if x_min < 0 or y_min < 0 or x_max > image_width or y_max > image_height:
            raise ValueError(
                "Invalid bbox: bbox coordinates must be within image dimensions"
            )

        # Iterate over all objects to paste them on the image
        for obj in objs:
            # Recognize the mask of the object
            if obj.mode == "RGBA":
                # Use alpha channel as mask
                mask_np = np.array(obj)[:, :, 3]
            elif obj.mode == "RGB":
                # Create binary mask based on white threshold
                white_threshold = 245
                mask_np = (
                    np.all(np.array(obj) < white_threshold, axis=-1).astype(np.uint8)
                    * 255
                )

            # Set background to 0 (black) in case the paste will be affected by the background
            obj_np = np.array(obj)
            obj_np[mask_np == 0] = 0

            # Convert back to PIL Image
            obj = Image.fromarray(obj_np)
            obj = obj.crop(
                obj.getbbox()
            )  # Crop to remove unnecessary transparent regions

            # Randomly scale the obj
            scale = random.uniform(0.5, 1.0)
            try:
                obj = obj.resize((int(obj.width * scale), int(obj.height * scale)))
            except Exception as e:
                logger.warning(f"Error resizing object while pasting: {e}")
                continue

            # Randomly rotate the obj
            angle = random.uniform(-45, 45)
            obj = obj.rotate(angle, expand=True)

            # Randomly select region to paste the obj within the image while ensuring 10% overlap with bbox
            paste_bbox = self.generate_paste_bbox_iou(
                image_width, image_height, obj.width, obj.height, bbox
            )

            # Paste the object on the image
            image = self.random_paste_obj_with_bbox(image, obj, paste_bbox)

        return image

    def compute_iou(self, box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1 (list or np.ndarray): The first bounding box in the format [x_min, y_min, x_max, y_max].
            box2 (list or np.ndarray): The second bounding box in the format [x_min, y_min, x_max, y_max].

        Returns:
            float: The IoU between the two bounding boxes.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate the coordinates of the intersection rectangle
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Calculate the width and height of the intersection rectangle
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Calculate the area of both bounding boxes
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate the union area
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0  # Avoid division by zero

        # Calculate the IoU
        iou = inter_area / union_area
        return iou

    def generate_paste_bbox_iou(
        self,
        image_width,
        image_height,
        obj_width,
        obj_height,
        bbox,
        min_iou=0.1,
        max_attempts=100,
    ):
        """Generate a paste bounding box that has an IoU of at least the
        specified minimum with the input bbox.

        Args:
            image_width (int): Width of the image.
            image_height (int): Height of the image.
            obj_width (int): Width of the object to paste.
            obj_height (int): Height of the object to paste.
            bbox (list or np.ndarray): The input bounding box in the format [x_min, y_min, x_max, y_max].
            min_iou (float, optional): The minimum required IoU. Defaults to 0.1.
            max_attempts (int, optional): The maximum number of attempts to find a valid paste bbox. Defaults to 1000.

        Returns:
            list or None: The paste bounding box [x_min, y_min, x_max, y_max] if found; otherwise, None.
        """
        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        required_iou = min_iou

        for attempt in range(max_attempts):
            # Randomly generate the top-left corner coordinates for the paste bbox
            paste_x_min = random.randint(0, max(0, image_width - obj_width))
            paste_y_min = random.randint(0, max(0, image_height - obj_height))
            paste_x_max = paste_x_min + obj_width
            paste_y_max = paste_y_min + obj_height

            paste_bbox = [paste_x_min, paste_y_min, paste_x_max, paste_y_max]

            # Calculate IoU with the input bbox
            iou = self.compute_iou(bbox, paste_bbox)

            if iou >= required_iou:
                return paste_bbox

        # If no valid bbox found after max_attempts, return the patial of bbox
        patial_bbox = [x_min, y_min, x_max, y_max]
        # scale the bbox
        scale = random.uniform(0.5, 1.0)
        patial_bbox[0] = int(patial_bbox[0] + (bbox_width - bbox_width * scale) / 2)
        patial_bbox[1] = int(patial_bbox[1] + (bbox_height - bbox_height * scale) / 2)
        patial_bbox[2] = int(patial_bbox[0] + bbox_width * scale)
        patial_bbox[3] = int(patial_bbox[1] + bbox_height * scale)
        return patial_bbox

    def random_paste_obj_with_bbox(
        self, image: Image.Image, obj_image: Image.Image, bbox: np.ndarray
    ) -> Image.Image:
        """Randomly paste an object image within the bounding box of the input
        image.

        Args:
            image (PIL.Image.Image): Image to paste the object on.
            obj_image (PIL.Image.Image): Object image to paste.
            bbox (np.ndarray): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].

        Returns:
            PIL.Image.Image: Image with object pasted.
        """
        # Extract bbox coordinates
        x_min, y_min, x_max, y_max = bbox

        paste_x = x_min
        paste_y = y_min

        # resize the object image to fit the bounding box
        # obj_image = obj_image.resize((x_max - x_min, y_max - y_min))

        # Check if object image has an alpha channel (RGBA) and use it as a mask
        if obj_image.mode == "RGBA":
            obj_mask = obj_image.split()[-1]  # Alpha channel as mask
        else:
            # If no alpha channel, use a black threshold to generate a binary mask
            obj_np = np.array(obj_image)
            # white_threshold = 245
            # obj_mask = Image.fromarray((np.all(obj_np < white_threshold, axis=-1).astype(np.uint8) * 255))
            black_threshold = 10
            obj_mask = Image.fromarray(
                (np.all(obj_np > black_threshold, axis=-1).astype(np.uint8) * 255)
            )

        # Paste the object onto the image
        image.paste(obj_image, (paste_x, paste_y), obj_mask)

        return image

    def random_mask_image_with_bbox(
        self, image: Image.Image, mask_image: Image.Image, bbox: np.ndarray
    ) -> Image.Image:
        """Randomly mask the image with the given mask image and bounding box.

        Args:
            image (PIL.Image.Image): Image to be masked.
            mask_image (PIL.Image.Image): Mask image.
            bbox (np.ndarray): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].

        Returns:
            PIL.Image.Image: Masked image.
        """
        # Ensure bbox format is correct
        if len(bbox) != 4:
            raise ValueError(
                "Bounding box must be in the format [x_min, y_min, x_max, y_max]."
            )
        if mask_image is None:
            return image
        # Determine masking ratio
        if self.obj_mask_ratio is not None:
            if isinstance(self.obj_mask_ratio, (Tuple, list)):
                min_ratio, max_ratio = self.obj_mask_ratio
                mask_ratio = random.uniform(min_ratio, max_ratio)
            else:
                mask_ratio = self.obj_mask_ratio
        else:
            # No ratio is given, return the original image
            return image

        # Extract bbox coordinates
        x_min, y_min, x_max, y_max = bbox

        # Calculate the mask region dimensions based on mask_ratio
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        mask_width = int(bbox_width * mask_ratio)
        mask_height = int(bbox_height * mask_ratio)

        # Randomly select the top-left corner for the mask region within the bbox
        mask_x_min = random.randint(x_min, x_max - mask_width)
        mask_y_min = random.randint(y_min, y_max - mask_height)
        mask_x_max = mask_x_min + mask_width
        mask_y_max = mask_y_min + mask_height

        # Ensure the mask image has the same size as the original image
        mask_image = mask_image.resize((image.width, image.height))

        # Convert images to NumPy arrays for pixel-wise manipulation
        image_np = np.array(image)
        mask_image_np = np.array(mask_image)

        # Replace the target region in the original image with the corresponding region from the mask image
        image_np[mask_y_min:mask_y_max, mask_x_min:mask_x_max, :] = mask_image_np[
            mask_y_min:mask_y_max, mask_x_min:mask_x_max, :
        ]

        # Convert the modified NumPy array back to a PIL.Image
        masked_image = Image.fromarray(image_np)

        return masked_image
