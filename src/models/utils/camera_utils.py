"""
Author: Yuanhong Yu
Date: 2025-03-17 16:51:49
LastEditTime: 2025-03-17 16:56:53
Description:

"""
import torch
from einops import rearrange
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils.camera_conversions import (
    cameras_from_opencv_projection,
    opencv_from_cameras_projection,
)


def safe_inverse(a):
    """Safely compute the inverse of a matrix by converting to float32 first.

    Args:
        a: Input tensor to invert

    Returns:
        Inverted tensor with the original dtype
    """
    org_type = a.dtype
    a = a.float()
    a_inv = torch.inverse(a)
    return a_inv.to(org_type)


def adjust_camera_parameters(intrinsics, extrinsics, image_width, image_height):
    """Adjusts camera intrinsic and extrinsic parameters to center the
    principal point.

    Args:
        intrinsics: Torch tensor of shape (B, 3, 3) representing intrinsic matrices
        extrinsics: Torch tensor of shape (B, 4, 4) representing extrinsic matrices
        image_width: Integer representing the width of the image
        image_height: Integer representing the height of the image

    Returns:
        Tuple of (new_intrinsics, new_extrinsics, transformation_matrices)
    """
    B = intrinsics.shape[0]
    new_intrinsics = torch.zeros_like(intrinsics)
    new_extrinsics = torch.zeros_like(extrinsics)
    extrinsics_transformation = []

    # Calculate new principal point (center of the image)
    new_cx, new_cy = image_width / 2, image_height / 2

    for i in range(B):
        K = intrinsics[i]

        # Create the new intrinsic matrix with centered principal point
        K_new = K.clone()
        K_new[0, 2] = new_cx
        K_new[1, 2] = new_cy
        new_intrinsics[i] = K_new

        # Compute the transformation to maintain the projection
        K_new_inv = safe_inverse(K_new)

        # Adjust the extrinsic matrix
        extrinsic = extrinsics[i]
        R_t = extrinsic[:3, :]
        new_R_t = K_new_inv @ K @ R_t

        trans = torch.eye(4, device=extrinsic.device)
        trans[:3, :3] = safe_inverse((K_new_inv @ K))
        extrinsics_transformation.append(trans)

        new_extrinsics[i, :3, :] = new_R_t
        new_extrinsics[i, 3, 3] = 1

    extrinsics_transformation = torch.stack(extrinsics_transformation)

    return new_intrinsics, new_extrinsics, extrinsics_transformation
