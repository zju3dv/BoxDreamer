"""
Author: Yuanhong Yu
Date: 2025-03-17 16:52:22
LastEditTime: 2025-03-17 16:56:44
Description:

"""
import torch
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)


def make_rotation_matrix(r, rotation_type):
    """Convert rotation representation to rotation matrix.

    Args:
        r: Rotation representation tensor
        rotation_type: Type of rotation representation (quat, 6d, euler)

    Returns:
        Rotation matrix
    """
    if rotation_type == "quat":
        return quaternion_to_matrix(r)
    elif rotation_type == "6d":
        return rotation_6d_to_matrix(r)
    elif rotation_type == "euler":
        return euler_angles_to_matrix(r, convention="XYZ")
    else:
        raise NotImplementedError(f"Rotation type {rotation_type} not implemented")


def make_rotation_representation(rotation_matrix, rotation_type):
    """Convert rotation matrix to a specific rotation representation.

    Args:
        rotation_matrix: Rotation matrix tensor
        rotation_type: Type of rotation representation (quat, 6d, euler)

    Returns:
        Rotation representation in the specified format
    """
    if rotation_type == "quat":
        return matrix_to_quaternion(rotation_matrix)
    elif rotation_type == "6d":
        return matrix_to_rotation_6d(rotation_matrix)
    elif rotation_type == "euler":
        return matrix_to_euler_angles(rotation_matrix, convention="XYZ")
    else:
        raise NotImplementedError(f"Rotation type {rotation_type} not implemented")


def normalize(x):
    """Normalize values from [0, 1] to [-1, 1]."""
    return (x * 2) - 1


def unnormalize(x):
    """Unnormalize values from [-1, 1] to [0, 1]."""
    return (x + 1) / 2


def rgb_to_grayscale(images):
    """Converts RGB images to grayscale."""
    if images.shape[1] == 1:
        return images
    elif images.shape[1] == 3:
        weights = torch.tensor(
            [0.2989, 0.5870, 0.1140], device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        grayscale = (images * weights).sum(dim=1, keepdim=True)
        return grayscale
    else:
        raise ValueError(f"Unexpected number of channels: {images.shape[1]}")
