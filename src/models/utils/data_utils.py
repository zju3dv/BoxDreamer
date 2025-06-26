import torch
from einops import rearrange


def sub_batchify(
    pose_feat, frames, camera_mask, rgb_feature, image_masks, sub_batch_size
):
    """Split reference info into sub-batches for processing.

    Args:
        pose_feat: Pose features tensor
        frames: Image frames tensor
        camera_mask: Camera mask tensor
        rgb_feature: RGB features tensor
        image_masks: Image masks tensor
        sub_batch_size: Size of each sub-batch

    Returns:
        Tuple of tensors with sub-batched data
    """
    # Extract query frames info
    B, T, C, H, W = frames.shape
    query_pose_feat = pose_feat[camera_mask]  # B C H W
    query_frames = frames[camera_mask]  # B C H W
    query_rgb_feature = rgb_feature[camera_mask]  # B L D
    query_image_masks = image_masks[camera_mask]  # B C H W

    # Remove query frames from the input
    pose_feat = pose_feat[~camera_mask].reshape(B, T - 1, *pose_feat.shape[2:])
    frames = frames[~camera_mask].reshape(B, T - 1, *frames.shape[2:])
    rgb_feature = rgb_feature[~camera_mask].reshape(B, T - 1, *rgb_feature.shape[2:])
    image_masks = image_masks[~camera_mask].reshape(B, T - 1, *image_masks.shape[2:])

    # Calculate number of splits
    split_length = (T - 1) // sub_batch_size
    if (T - 1) % sub_batch_size != 0:
        split_length += 1

    # Initialize new tensors for sub-batching
    new_pose_feat = torch.zeros(
        B,
        split_length,
        sub_batch_size + 1,
        *pose_feat.shape[2:],
        device=pose_feat.device
    )
    new_frames = torch.zeros(
        B, split_length, sub_batch_size + 1, *frames.shape[2:], device=frames.device
    )
    new_rgb_feature = torch.zeros(
        B,
        split_length,
        sub_batch_size + 1,
        *rgb_feature.shape[2:],
        device=rgb_feature.device
    )
    new_image_masks = torch.zeros(
        B,
        split_length,
        sub_batch_size + 1,
        *image_masks.shape[2:],
        device=image_masks.device
    )

    # Fill sub-batches with reference and query frames
    for i in range(split_length):
        # Add reference frames (handling potential out-of-bounds)
        slice_end = min((i + 1) * sub_batch_size, T - 1)
        actual_slice_size = slice_end - i * sub_batch_size

        new_pose_feat[:, i, :actual_slice_size] = pose_feat[
            :, i * sub_batch_size : slice_end
        ]
        new_frames[:, i, :actual_slice_size] = frames[:, i * sub_batch_size : slice_end]
        new_rgb_feature[:, i, :actual_slice_size] = rgb_feature[
            :, i * sub_batch_size : slice_end
        ]
        new_image_masks[:, i, :actual_slice_size] = image_masks[
            :, i * sub_batch_size : slice_end
        ]

        # Add query frame at the end of each sub-batch
        new_pose_feat[:, i, sub_batch_size] = query_pose_feat
        new_frames[:, i, sub_batch_size] = query_frames
        new_rgb_feature[:, i, sub_batch_size] = query_rgb_feature
        new_image_masks[:, i, sub_batch_size] = query_image_masks

    # Create camera mask for sub-batches
    new_camera_mask = torch.zeros(
        B, split_length, sub_batch_size + 1, dtype=torch.bool, device=camera_mask.device
    )
    new_camera_mask[:, :, sub_batch_size] = True

    return new_pose_feat, new_frames, new_camera_mask, new_rgb_feature, new_image_masks


def fetch_neighbors_by_pose_similarity(gt_poses, pred_pose, topk=5):
    """Find nearest neighbor frames based on predicted pose.

    Args:
        gt_poses: Ground truth poses
        pred_pose: Predicted pose
        topk: Number of top matches to return

    Returns:
        Indices of top-k nearest neighbors
    """
    B, N = gt_poses.shape[:2]
    gt_poses = gt_poses.clone().reshape(B * N, 4, 4)
    pred_pose = (
        pred_pose.clone().reshape(B, 1, 4, 4).repeat(1, N, 1, 1).reshape(B * N, 4, 4)
    )

    # Calculate rotation and translation differences
    R_gt = gt_poses[:, :3, :3]
    t_gt = gt_poses[:, :3, 3]
    R_pred = pred_pose[:, :3, :3]
    t_pred = pred_pose[:, :3, 3]

    # Compute rotation distance (geodesic distance)
    R_diff = torch.matmul(R_pred, R_gt.transpose(1, 2))
    traces = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
    rot_dist = torch.acos(torch.clamp((traces - 1) / 2, -1, 1))

    # Compute translation distance
    trans_dist = torch.norm(t_pred - t_gt, dim=-1)

    # Combine distances
    total_dist = rot_dist + trans_dist
    total_dist = total_dist.reshape(B, N)

    # Get top-k nearest neighbors
    _, indices = torch.topk(total_dist, k=topk, dim=1, largest=False)

    return indices
