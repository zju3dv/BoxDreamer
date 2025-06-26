# src/models/modules/data_processing.py

import torch
from einops import rearrange
import numpy as np
from .matching import dino_matching


def filter_by_neighbor_mask(
    data, neighbor_mask, pose_feat, frames, camera_mask, rgb_feature, image_masks
):
    """Filter and reorganize data based on neighbor mask.

    Args:
        data: Input data dictionary
        neighbor_mask: Boolean mask for neighbors to keep
        pose_feat, frames, camera_mask, rgb_feature, image_masks: Input tensors

    Returns:
        Updated data and tensors
    """
    B = frames.shape[0]

    # Extract filtered reference data
    ref_shape = B, frames.shape[1] - 1, *pose_feat.shape[2:]
    reference_pose_feat = (
        pose_feat[~camera_mask]
        .reshape(*ref_shape)[neighbor_mask]
        .reshape(B, -1, *pose_feat.shape[2:])
    )

    ref_frames_shape = B, frames.shape[1] - 1, *frames.shape[2:]
    reference_frames = (
        frames[~camera_mask]
        .reshape(*ref_frames_shape)[neighbor_mask]
        .reshape(B, -1, *frames.shape[2:])
    )

    # Handle RGB features if available
    if rgb_feature is not None:
        ref_rgb_shape = B, frames.shape[1] - 1, *rgb_feature.shape[2:]
        reference_rgb = (
            rgb_feature[~camera_mask]
            .reshape(*ref_rgb_shape)[neighbor_mask]
            .reshape(B, -1, *rgb_feature.shape[2:])
        )
    else:
        reference_rgb = None

    ref_mask_shape = B, frames.shape[1] - 1, *image_masks.shape[2:]
    reference_image_masks = (
        image_masks[~camera_mask]
        .reshape(*ref_mask_shape)[neighbor_mask]
        .reshape(B, -1, *image_masks.shape[2:])
    )

    # Concatenate reference and query data
    query_pose_feat = pose_feat[camera_mask].unsqueeze(1)
    query_frames = frames[camera_mask].unsqueeze(1)
    query_rgb = (
        rgb_feature[camera_mask].unsqueeze(1) if rgb_feature is not None else None
    )
    query_masks = image_masks[camera_mask].unsqueeze(1)

    new_pose_feat = torch.cat([reference_pose_feat, query_pose_feat], dim=1)
    new_frames = torch.cat([reference_frames, query_frames], dim=1)
    new_rgb_feature = (
        torch.cat([reference_rgb, query_rgb], dim=1)
        if rgb_feature is not None
        else None
    )
    new_image_masks = torch.cat([reference_image_masks, query_masks], dim=1)

    # Create new camera mask (marking the last frame as query)
    _, T, _, _, _ = new_frames.shape
    new_camera_mask = torch.zeros(B, T, dtype=torch.bool, device=camera_mask.device)
    new_camera_mask[:, -1] = True

    # Update data dictionary
    update_filtered_data(
        data,
        neighbor_mask,
        new_pose_feat,
        new_frames,
        new_camera_mask,
        T,
        B,
        camera_mask,
    )

    return (
        data,
        new_pose_feat,
        new_frames,
        new_camera_mask,
        new_rgb_feature,
        new_image_masks,
    )


def update_filtered_data(
    data, neighbor_mask, new_pose_feat, new_frames, new_camera_mask, T, B, camera_mask
):
    """Update data dictionary with filtered data."""
    poses = data["poses"].clone()

    # Update core tensors
    data["bbox_feat"] = new_pose_feat.clone()
    data["images"] = new_frames.clone()
    data["query_idx"] = torch.tensor([T - 1], device=poses.device).repeat(B)
    data["camera_mask"] = new_camera_mask.clone()

    # Update poses
    reference_pose = (
        poses[~camera_mask]
        .reshape(B, poses.shape[1] - 1, 4, 4)[neighbor_mask]
        .reshape(B, -1, 4, 4)
    )
    query_pose = poses[camera_mask].reshape(B, 1, 4, 4)
    data["poses"] = torch.cat([reference_pose, query_pose], dim=1)

    # Update other tensor data
    for key, *dims in [
        ("original_poses", 4, 4),
        ("intrinsics", 3, 3),
        ("non_ndc_intrinsics", 3, 3),
        ("original_intrinsics", 3, 3),
        ("scale", 3),
        ("bbox_3d", 8, 3),
        ("bbox_proj_crop", 8, 2),
    ]:
        update_tensor_data(data, B, neighbor_mask, camera_mask, key, *dims)

    # Update original images list
    update_original_images(data, neighbor_mask)


def update_tensor_data(data, B, neighbor_mask, camera_mask, key, *dims):
    """Update tensor data based on neighbor mask."""
    if key not in data:
        return

    ref_tensor = data[key][~camera_mask].reshape(B, -1, *dims)
    query_tensor = data[key][camera_mask].reshape(B, 1, *dims)

    # Filter and concatenate
    filtered_ref = ref_tensor[neighbor_mask].reshape(B, -1, *dims)
    data[key] = torch.cat([filtered_ref, query_tensor], dim=1)


def update_original_images(data, neighbor_mask):
    """Update original images list based on neighbor mask."""
    if "original_images" not in data:
        return

    org_T = len(data["original_images"])
    B = len(data["original_images"][0])
    neighbor_mask_np = neighbor_mask.cpu().numpy()

    # Create new container for images
    N = neighbor_mask.sum(dim=1)[0].item()
    total_frames = N + 1  # add query frame
    new_org_images = [[] for _ in range(total_frames)]

    # Reorganize images
    for b in range(B):
        ref_idx = 0
        for t in range(org_T - 1):
            if neighbor_mask_np[b, t]:
                new_org_images[ref_idx].append(data["original_images"][t][b])
                ref_idx += 1

        # Add query frame
        new_org_images[-1].append(data["original_images"][-1][b])

    data["original_images"] = new_org_images


def process_dense_input(
    data, pose_feat, frames, camera_mask, rgb_feature, image_masks, dense_cfg
):
    """Process dense input data with filtering.

    Args:
        data: Input data dictionary
        pose_feat, frames, camera_mask, rgb_feature, image_masks: Input tensors
        dense_cfg: Dense input configuration

    Returns:
        Processed data and tensors
    """
    # Apply DINO filtering if configured
    if dense_cfg.filter == "dino" and dense_cfg.filter_enable:
        # Reshape reference and query features
        B = frames.shape[0]
        ref_features = rgb_feature[~camera_mask].reshape(
            B, frames.shape[1] - 1, *rgb_feature.shape[2:]
        )
        query_features = rgb_feature[camera_mask]
        ref_images = frames[~camera_mask].reshape(
            B, frames.shape[1] - 1, *frames.shape[2:]
        )
        query_images = frames[camera_mask]

        # Perform feature matching
        neighbor_mask = dino_matching(
            ref_features,
            query_features,
            ref_images,
            query_images,
            topk=dense_cfg.filter_topk,
        )

        # Filter data based on matching results
        return filter_by_neighbor_mask(
            data,
            neighbor_mask,
            pose_feat,
            frames,
            camera_mask,
            rgb_feature,
            image_masks,
        )

    return data, pose_feat, frames, camera_mask, rgb_feature, image_masks


def normalize(x):
    """Normalize input tensor to sum to 1."""
    return x / (x.sum() + 1e-6)
