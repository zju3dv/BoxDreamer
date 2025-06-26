from src.reconstruction.base import BaseReconstructor
import copy
import matplotlib.pyplot as pl
from PIL import Image
from src.reconstruction.base import BaseReconstructor
import numpy as np
import torch
import os
from typing import List, Tuple
import pytorch3d

import os
import torch
import tempfile
from contextlib import nullcontext
import sys

# 3rd party imports
# 3rd dir is under repo root, so we need to add it to sys.path
HERE_PATH = os.path.normpath(os.path.dirname(__file__))
VGGSFM_REPO_PATH = os.path.normpath(os.path.join(HERE_PATH, "../../three"))
VGGSFM_LIB_PATH = os.path.join(VGGSFM_REPO_PATH, "vggsfm")
# check the presence of models directory in repo to be sure its cloned

if os.path.isdir(VGGSFM_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, VGGSFM_LIB_PATH)
else:
    raise ImportError(
        f"mast3r is not initialized, could not find: {VGGSFM_LIB_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )


from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import *
from vggsfm.utils.utils import seed_all_random_engines


class VGGSFMReconstructor(BaseReconstructor):
    def __init__(self, method: str, weight: str, config: dict):
        super().__init__(method)
        self.runner = None
        self.weights = weight
        self.cache_path = config.get("cache_path", "./cache/vggsfm_cache")
        self.config = config

    def _prepare_before_run(self):
        assert self.images is not None, "Image list is not set"

        images = [Image.open(img_path).convert("RGB") for img_path in self.images]
        image_paths = self.images
        masks = (
            [Image.open(mask_path).convert("L") for mask_path in self.masks]
            if self.masks is not None
            else [None] * len(images)
        )
        # seq name is images dir name
        seq_name = os.path.basename(os.path.dirname(self.images[0]))

        return self._prepare_batch(seq_name, images, masks, image_paths)

    def _prepare_batch(
        self,
        sequence_name: str,
        images: list,
        masks: list,
        image_paths: list,
    ) -> dict:
        """Prepare a batch of data for a given sequence.

        This function processes the provided sequence name, metadata, annotations, images, masks, and image paths
        to create a batch of data. It handles the transformation of images and masks, the adjustment of camera parameters,
        and the preparation of ground truth camera data if required.

        Args:
            sequence_name (str): Name of the sequence.
            metadata (list): List of metadata for the sequence.
            annos (list): List of annotations for the sequence.
            images (list): List of images for the sequence.
            masks (list): List of masks for the sequence.
            image_paths (list): List of image paths for the sequence.

        Returns:
            dict: Batch of data containing transformed images, masks, crop parameters, original images, and other relevant information.
        """
        batch = {"seq_name": sequence_name, "frame_num": len(images)}
        crop_parameters, images_transformed, masks_transformed = [], [], []
        original_images = (
            {}
        )  # Dictionary to store original images before any transformations

        for i, image in enumerate(images):
            mask = masks[i]

            if mask is not None:
                mask = np.array(mask) == 0  # Invert the mask
                # to float
                # mask = mask.astype(np.uint8) * 255
                mask = Image.fromarray(mask).convert("L")
                # vggsfm use bg mask, so we need to invert it

            # Store the original image in the dictionary with the basename of the image path as the key
            original_images[os.path.basename(image_paths[i])] = np.array(image)

            # Transform the image and mask, and get crop parameters and bounding box
            (
                image_transformed,
                mask_transformed,
                crop_paras,
                bbox,
            ) = pad_and_resize_image(
                image,
                True,
                1024,
                mask=mask,
                transform=None,
                bbox_anno=None,
            )
            images_transformed.append(image_transformed)
            if mask_transformed is not None:
                # ensure mask in a binary mask
                # mask_transformed = (mask_transformed > 0.1).float()

                # save mask for debug
                mask_for_vis = mask_transformed[0].cpu().numpy() * 255
                mask_for_vis = Image.fromarray(mask_for_vis).convert("L")

                masks_transformed.append(mask_transformed)
            crop_parameters.append(crop_paras)

        images = torch.stack(images_transformed)
        masks = torch.stack(masks_transformed) if masks[0] is not None else None

        batch.update(
            {
                "image": images.clamp(0, 1),
                "crop_params": torch.stack(crop_parameters),
                "scene_dir": os.path.dirname(os.path.dirname(image_paths[0])),
                "masks": masks.clamp(0, 1) if masks is not None else None,
                "original_images": original_images,  # A dict with the image path as the key and the original np image as the value
            }
        )

        return batch

    def run(self):
        self.runner = VGGSfMRunner(self.config)

        data = self._prepare_before_run()
        images = data["image"]
        masks = data["masks"]
        original_images = data["original_images"]
        seq_name = data["seq_name"]
        crop_params = data["crop_params"]
        output_dir = self.cache_path

        # to cuda if available
        images = images.to("cuda")
        if masks is not None:
            masks = masks.to("cuda")
        print(masks.shape)
        print(images.shape)
        predictions = self.runner.run(
            images,
            masks=masks,
            original_images=original_images,
            image_paths=self.images,
            crop_params=crop_params,
            seq_name=seq_name,
            output_dir=output_dir,
        )

        # log predictions keys
        print(f"Predictions keys: {predictions.keys()}")

        extrinsics = predictions["extrinsics_opencv"]
        points3D = predictions["points3D"]

        # log extrinsics and points3D shape
        print(f"Extrinsics shape: {extrinsics.shape}")
        print(f"Points3D shape: {points3D.shape}")

        self.pred_poses = extrinsics
        # invert the poses
        self.pt3d = points3D

        self.pred_intrinsics = predictions["intrinsics_opencv"]
        print(self.pred_intrinsics[0])
        print(predictions["intrinsics_opencv"][0])

        self.draw_3dbbox(self.images)

        # if self._check_gt_poses():
        #     self.align_coordinates()

        return True
