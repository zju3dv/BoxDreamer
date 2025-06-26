import torch
from einops import rearrange
from .data_utils import sub_batchify, fetch_neighbors_by_pose_similarity
from .box_utils import recover_pose_from_dense_bb8
from .data_processing import filter_by_neighbor_mask, normalize


def process_multi_round(
    data,
    pose_feat,
    frames,
    camera_mask,
    rgb_feature,
    image_masks,
    decoder,
    dense_cfg,
    bbox_representation,
):
    """Process data in multiple rounds for dense inputs.

    Args:
        data: Input data dictionary
        pose_feat, frames, camera_mask, rgb_feature, image_masks: Input tensors
        decoder: Model decoder
        dense_cfg: Dense processing configuration
        bbox_representation: Type of bounding box representation

    Returns:
        Processed data or decoder output
    """
    B = frames.shape[0]
    poses = data["poses"].clone()
    K = data["non_ndc_intrinsics"].clone()
    bbox_3d = data["bbox_3d"].clone()

    # Divide data into sub-batches
    (
        new_pose_feat,
        new_frames,
        new_camera_mask,
        new_rgb_feature,
        new_image_masks,
    ) = sub_batchify(
        pose_feat.clone(),
        frames.clone(),
        camera_mask.clone(),
        rgb_feature.clone(),
        image_masks.clone(),
        dense_cfg.sub_batch_size,
    )

    # Process each sub-batch
    if dense_cfg.dense_mem_friendly:
        # Memory-friendly processing (one sub-batch at a time)
        query_rets = []
        for i in range(new_pose_feat.shape[1]):
            query_ret = decoder(
                new_pose_feat[:, i],
                new_frames[:, i],
                new_camera_mask[:, i],
                new_rgb_feature[:, i],
                normalize(new_image_masks[:, i]),
            )
            query_rets.append(query_ret.clone().unsqueeze(1))

        query_rets = torch.cat(query_rets, dim=1)
    else:
        # Process all sub-batches at once
        sub_batch_size = new_pose_feat.shape[1]

        # Flatten the first two dimensions
        flat_pose_feat = new_pose_feat.reshape(
            B * sub_batch_size, *new_pose_feat.shape[2:]
        )
        flat_frames = new_frames.reshape(B * sub_batch_size, *new_frames.shape[2:])
        flat_rgb_feature = new_rgb_feature.reshape(
            B * sub_batch_size, *new_rgb_feature.shape[2:]
        )
        flat_image_masks = new_image_masks.reshape(
            B * sub_batch_size, *new_image_masks.shape[2:]
        )
        flat_camera_mask = new_camera_mask.reshape(
            B * sub_batch_size, *new_camera_mask.shape[2:]
        )

        # Run decoder on flattened data
        query_ret = decoder(
            flat_pose_feat,
            flat_frames,
            flat_camera_mask,
            flat_rgb_feature,
            normalize(flat_image_masks),
        )

        # Reshape back to original dimensions
        query_rets = query_ret.reshape(B, sub_batch_size, *query_ret.shape[1:])

    # Process predictions for pose estimation
    query_ret_for_pose = query_rets.permute(0, 1, 3, 4, 2).unsqueeze(2)
    query_poses, pred_proj_bbox = recover_pose_from_dense_bb8(
        query_ret_for_pose,
        bbox_3d.clone()[camera_mask].unsqueeze(1),
        K[camera_mask].unsqueeze(1),
        bbox_representation,
    )

    # Perform fine-level refinement if enabled
    if dense_cfg.fine_level:
        # Find nearest neighbors by pose similarity
        neighbor_indices = fetch_neighbors_by_pose_similarity(
            poses[~camera_mask].reshape(B, poses.shape[1] - 1, 4, 4),
            query_poses,
            topk=dense_cfg.fine_topk,
        )

        # Create neighbor mask
        neighbor_mask = torch.zeros(
            B, poses.shape[1] - 1, dtype=torch.bool, device=poses.device
        )
        for b in range(B):
            neighbor_mask[b, neighbor_indices[b]] = True

        # Filter data with new neighbors
        (
            data,
            pose_feat,
            frames,
            camera_mask,
            rgb_feature,
            image_masks,
        ) = filter_by_neighbor_mask(
            data,
            neighbor_mask,
            pose_feat,
            frames,
            camera_mask,
            rgb_feature,
            image_masks,
        )

        # Run decoder with refined selection
        return decoder(
            pose_feat, frames, camera_mask, rgb_feature, normalize(image_masks)
        )
    else:
        # Return coarse prediction results
        pred_poses = poses.clone()
        data["pred_bbox"] = pose_feat.clone()
        data["pred_bbox"][camera_mask] = query_rets[:, 0]
        pred_poses[camera_mask] = query_poses.clone().squeeze(1).to(pred_poses.dtype)
        data["regression_boxes"] = data["bbox_proj_crop"].clone()
        data["regression_boxes"][camera_mask] = (
            pred_proj_bbox.clone().squeeze(1).to(data["regression_boxes"].dtype)
        )
        data["pred_poses"] = pred_poses.clone()
        data["pred_intrinsics"] = data["intrinsics"].clone()

        return data
