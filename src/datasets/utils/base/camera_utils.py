"""Utility functions for camera operations, including projections and
transformations."""

import torch
import numpy as np
from typing import Tuple, Optional


def reproj_pytorch(
    K: torch.Tensor, pose: torch.Tensor, pts_3d: torch.Tensor
) -> torch.Tensor:
    """Reproject 3D points to 2D points using PyTorch.

    Args:
        K (torch.Tensor): Intrinsic matrix of shape [3, 3] or [3, 4].
        pose (torch.Tensor): Pose matrix of shape [3, 4] or [4, 4].
        pts_3d (torch.Tensor): 3D points of shape [n, 3].

    Returns:
        torch.Tensor: Reprojected 2D points of shape [n, 2].
    """
    assert K.shape in [
        (3, 3),
        (3, 4),
    ], f"K must be of shape [3,3] or [3,4], but got {K.shape}"
    assert pose.shape in [
        (3, 4),
        (4, 4),
    ], f"Pose must be of shape [3,4] or [4,4], but got {pose.shape}"
    assert (
        pts_3d.dim() == 2 and pts_3d.shape[1] == 3
    ), f"pts_3d must be of shape [n, 3], but got {pts_3d.shape}"

    K = K.float()
    pose = pose.float()
    pts_3d = pts_3d.float()

    if K.shape == torch.Size([3, 3]):
        zeros = torch.zeros((3, 1), dtype=K.dtype, device=K.device)
        K_homo = torch.cat([K, zeros], dim=1)  # [3, 4]
    else:
        K_homo = K  # [3, 4]

    if pose.shape == torch.Size([3, 4]):
        last_row = torch.tensor([[0, 0, 0, 1]], dtype=pose.dtype, device=pose.device)
        pose_homo = torch.cat([pose, last_row], dim=0)  # [4, 4]
    else:
        pose_homo = pose  # [4, 4]

    n = pts_3d.shape[0]
    ones = torch.ones((n, 1), dtype=pts_3d.dtype, device=pts_3d.device)
    pts_3d_homo = torch.cat([pts_3d, ones], dim=1).t()  # [4, n]

    reproj_points = K_homo @ pose_homo @ pts_3d_homo  # [3, n]
    z = reproj_points[2:3, :]  # [1, n]
    reproj_points_normalized = reproj_points / z  # [3, n]
    reproj_2d = reproj_points_normalized[:2, :].t()  # [n, 2]

    return reproj_2d  # [n, 2]


def make_proj_bbox(
    pose: torch.Tensor, intrinsic: torch.Tensor, bbox: torch.Tensor
) -> torch.Tensor:
    """Project 3D bounding boxes into 2D image space.

    Args:
        pose (torch.Tensor): Pose matrices of shape [B, 4, 4].
        intrinsic (torch.Tensor): Intrinsic matrices of shape [B, 3, 3].
        bbox (torch.Tensor): 3D bounding box points of shape [8, 3] or [B, 8, 3].

    Returns:
        torch.Tensor: Projected 2D bounding boxes of shape [B, 8, 2].
    """
    L = pose.shape[0]
    bbox_proj = torch.zeros((L, 8, 2), device=pose.device)

    diff_bbox = True if bbox.dim() == 3 else False

    for i in range(L):
        bbox_proj[i] = reproj_pytorch(
            intrinsic[i], pose[i], bbox if not diff_bbox else bbox[i]
        )
    return bbox_proj
