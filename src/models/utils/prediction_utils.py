"""
Author: Yuanhong Yu
Date: 2025-03-17 18:52:04
LastEditTime: 2025-03-17 18:52:15
Description:

"""

import torch
from einops import rearrange
from .box_utils import recover_pose_from_bb8, recover_bb8_corners


def process_prediction(
    pred_poses,
    data,
    query_ret,
    camera_mask,
    pose_representation,
    cameras=None,
    rays=None,
    crop_params=None,
    image_masks=None,
    trans=None,
    K=None,
    bbox_3d=None,
    coordinate=None,
    image_size=None,
    patch_size=None,
    bbox_representation=None,
):
    """Process model predictions to generate final pose estimates.

    Args:
        pred_poses: Current pose predictions
        data, query_ret, camera_mask: Model outputs and mask
        Various configuration parameters and optional inputs

    Returns:
        Updated pose predictions
    """
    if pose_representation == "plucker":
        # Handle plucker representation
        from .rays import Rays
        from .camera_processing import recover_pose_from_rays

        pred_rays = rearrange(data["pred_camera_rays"], "B T C H W -> (B T) C H W")
        camera_rays = Rays.from_spatial(pred_rays)

        # Convert ray representation to pose
        pred_poses = recover_pose_from_rays(
            camera_rays,
            pred_poses,
            cameras,
            crop_params,
            image_masks,
            trans,
            image_size,
            patch_size,
            coordinate,
        )

    elif pose_representation == "bb8":
        # Handle BB8 representation
        query_bbox_copy = query_ret.clone().permute(0, 2, 3, 1)

        # Reshape based on bbox representation
        if bbox_representation == "voting":
            query_bbox_copy = rearrange(
                query_bbox_copy, "B H W (L C) -> B H W L C", L=8, C=2
            )
        elif bbox_representation == "conf_voting":
            query_bbox_copy = rearrange(
                query_bbox_copy, "B H W (L C) -> B H W L C", L=8, C=3
            )

        # Get 3D coordinates and intrinsics
        query_bbox_3d = bbox_3d.clone()[camera_mask]

        # Recover pose using PnP
        query_poses, pred_proj_bbox = recover_pose_from_bb8(
            query_bbox_copy.unsqueeze(1),
            query_bbox_3d.unsqueeze(1),
            K[camera_mask].unsqueeze(1),
            bbox_representation,
        )

        # Update predictions
        pred_poses[camera_mask] = query_poses.clone().squeeze(1).to(pred_poses.dtype)
        data["regression_boxes"] = data["bbox_proj_crop"].clone()
        data["regression_boxes"][camera_mask] = (
            pred_proj_bbox.clone().squeeze(1).to(data["regression_boxes"].dtype)
        )

    elif pose_representation == "vector":
        raise NotImplementedError(
            "Vector representation not implemented for prediction processing"
        )

    # Handle potential numerical issues
    pred_poses = torch.nan_to_num(pred_poses, nan=0.0, posinf=0.0, neginf=0.0)

    return pred_poses


def calculate_bb8_projections(query_ret, data, camera_mask, bbox_representation):
    """Calculate BB8 projections for training.

    Args:
        query_ret: Decoder output
        data: Data dictionary
        camera_mask: Camera mask
        bbox_representation: Type of bounding box representation
    """
    query_bbox_copy = query_ret.clone().permute(0, 2, 3, 1)

    # Reshape based on bbox representation
    if bbox_representation == "voting":
        query_bbox_copy = rearrange(
            query_bbox_copy, "B H W (L C) -> B H W L C", L=8, C=2
        )
    elif bbox_representation == "conf_voting":
        query_bbox_copy = rearrange(
            query_bbox_copy, "B H W (L C) -> B H W L C", L=8, C=3
        )

    # Recover projected bounding box corners
    pred_proj_bbox, _ = recover_bb8_corners(
        query_bbox_copy.unsqueeze(1), bbox_representation
    )

    # Store results in data dictionary
    data["regression_boxes"] = data["bbox_proj_crop"].clone()
    data["regression_boxes"][camera_mask] = (
        pred_proj_bbox.clone().squeeze(1).to(data["regression_boxes"].dtype)
    )
