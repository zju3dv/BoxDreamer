# src/models/modules/camera_processing.py

import torch
from einops import rearrange
from .camera_utils import safe_inverse


def make_camera_rays(pose, K, crop_params, image_size, patch_size, patchify_rays):
    """Create camera rays from pose and intrinsic parameters.

    Args:
        pose: Camera pose matrices [B, T, 4, 4]
        K: Camera intrinsic matrices [B, T, 3, 3]
        crop_params: Crop parameters
        image_size: Image size
        patch_size: Patch size
        patchify_rays: Whether to patchify rays

    Returns:
        rays, cameras, transformation
    """
    from .rays import cameras_to_rays
    from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

    B, T, _, _ = pose.shape

    # Reshape for processing
    pose = rearrange(pose, "B T H W -> (B T) H W")
    K = rearrange(K, "B T H W -> (B T) H W").to(torch.float32)
    crop_params = rearrange(crop_params, "B T L -> (B T) L")

    # Extract camera parameters
    batchR = pose[:, :3, :3].clone().to(torch.float32)
    batchT = pose[:, :3, 3].clone().to(torch.float32)

    # Create PyTorch3D camera objects
    cameras = cameras_from_opencv_projection(
        batchR,
        batchT,
        K,
        torch.tensor([image_size, image_size], dtype=torch.float32).repeat(B * T, 1),
    )

    # Generate rays based on configuration
    if patchify_rays:
        assert (
            image_size % patch_size == 0
        ), "image_size should be divisible by patch_size"
        num_patches = image_size // patch_size
        rays = cameras_to_rays(cameras, None, num_patches, num_patches)
    else:
        rays = cameras_to_rays(cameras, None, image_size, image_size)

    return rays, cameras, None


def recover_pose_from_rays(
    rays, pose, cameras, crop_params, mask, trans, image_size, patch_size, coordinate
):
    """Recover camera pose from ray representations.

    Args:
        rays: Ray representation
        pose: Original pose matrices
        cameras, crop_params, mask, trans: Additional parameters
        image_size, patch_size: Image configuration
        coordinate: Coordinate system

    Returns:
        Recovered poses
    """
    from .rays import rays_to_cameras
    from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection

    # Prepare mask for processing
    mask_flat = rearrange(mask, "B T 1 H W -> (B T) (H W) 1").squeeze(-1)
    crop_params = rearrange(crop_params, "B T L -> (B T) L")
    B, T = pose.shape[:2]

    # Determine patch dimensions
    num_patches = image_size // patch_size if patch_size > 1 else image_size

    # Convert rays back to cameras
    cameras = rays_to_cameras(rays, None, num_patches, num_patches, cameras, None)

    # Extract camera parameters
    batchR, batchT, _ = opencv_from_cameras_projection(
        cameras,
        torch.tensor([image_size, image_size], device=cameras.R.device).repeat(
            B * T, 1
        ),
    )

    # Build recovered poses
    pose = rearrange(pose, "B T H W -> (B T) H W")
    recovered_poses = torch.zeros_like(pose)
    recovered_poses[:, :3, :3] = batchR
    recovered_poses[:, :3, 3] = batchT
    recovered_poses[:, 3, 3] = 1.0

    # Apply transformation if provided
    if trans is not None:
        for i in range(recovered_poses.shape[0]):
            recovered_poses[i] = torch.matmul(trans[i], recovered_poses[i])

    # Reshape back to original dimensions
    recovered_poses = rearrange(recovered_poses, "(B T) H W -> B T H W", B=B, T=T)

    # Transform to first camera coordinate system if needed
    if coordinate == "first_camera":
        for b in range(B):
            recovered_poses[b] = recovered_poses[b] @ safe_inverse(
                recovered_poses[b, 0]
            )

    return recovered_poses


def encode_camera_as_vector(
    pose, K, rotation_type, rotation_length, camera_dim, use_pp
):
    """Encode camera parameters as a vector representation.

    Args:
        pose: Camera pose matrices
        K: Camera intrinsic matrices
        rotation_type, rotation_length, camera_dim, use_pp: Configuration parameters

    Returns:
        Vector representation of camera parameters
    """
    from .pose_utils import make_rotation_representation

    B, T, _, _ = pose.shape
    cameras = torch.zeros((B, T, camera_dim), dtype=pose.dtype, device=pose.device)

    for b in range(B):
        for t in range(T):
            # Extract parameters
            rotation_matrix = pose[b, t, :3, :3].clone()
            translation = pose[b, t, :3, 3].clone()
            r_rep = make_rotation_representation(rotation_matrix, rotation_type)
            focal_length = K[b, t, 0, 0]

            # Fill vector representation
            cameras[b, t, :rotation_length] = torch.tensor(
                r_rep, dtype=pose.dtype, device=pose.device
            )
            cameras[b, t, rotation_length : rotation_length + 3] = translation
            cameras[b, t, rotation_length + 3] = focal_length

            # Add principal point if needed
            if use_pp:
                cameras[b, t, rotation_length + 4 :] = K[b, t, :2, 2]

    return cameras
