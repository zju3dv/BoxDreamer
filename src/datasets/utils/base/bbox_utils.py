"""Utility functions for bounding box operations."""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
import os
from ..preprocess import generate_cornernet_heatmap


def extract_bboxes(mask_path: str, lmdb=None) -> np.ndarray:
    """Extract bounding boxes from mask images.

    Args:
        mask_path (str): Path to the mask image.
        lmdb: Optional LMDB instance for reading data.

    Returns:
        np.ndarray: Bounding box [x_min, y_min, x_max, y_max].
    """
    if lmdb is not None:
        try:
            mask = lmdb.get(mask_path.encode())
            mask = cv2.imdecode(np.frombuffer(mask, np.uint8), cv2.IMREAD_UNCHANGED)
        except:
            mask = lmdb.get(mask_path.encode())
            mask = cv2.imdecode(np.frombuffer(mask, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask > 0).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(mask)
    return np.array([x, y, x + w, y + h])


def make_mask_by_bbox(
    bbox: Optional[np.array], img_size: Tuple[int, int]
) -> Image.Image:
    """
    Make mask by bounding box
    Args:
        bbox (np.ndarray): Bounding box of shape [x1, y1, x2, y2] or None
        img_size (Tuple[int, int]): Image size (from PIL.Image.size, (width, height))

    Returns:
        Image.Image: Mask image
    """
    if bbox is None:
        return Image.fromarray(
            np.ones((img_size[1], img_size[0]), dtype=np.uint8) * 255
        )
    else:
        mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        # ensure the indices are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask[y1:y2, x1:x2] = 255
        return Image.fromarray(mask)


def adjust_bbox_by_proj(proj_bbox: torch.Tensor) -> np.ndarray:
    """Adjust bounding box coordinates based on the projected bounding box.

    Args:
        proj_bbox (torch.Tensor): Projected bounding box coordinates. 8 points [x, y]. shape [8, 2]

    Returns:
        new_bbox (np.ndarray): 4 points [x1, y1, x2, y2]. shape [4, ]
    """
    x1 = proj_bbox[:, 0].min().item()
    y1 = proj_bbox[:, 1].min().item()
    x2 = proj_bbox[:, 0].max().item()
    y2 = proj_bbox[:, 1].max().item()

    return np.array([x1, y1, x2, y2])


@lru_cache(maxsize=20)
def get_cached_points(model_path: str) -> np.ndarray:
    """Cache and retrieve all 3D points on the model.

    Args:
        model_path (str): Path to the 3D model file.

    Returns:
        np.ndarray: Array of 3D points.
    """
    from src.utils.customize.sample_points_on_cad import get_all_points_on_model

    return get_all_points_on_model(model_path)


def prepare_bbox3d(
    path: Optional[str],
    cat: Optional[str] = None,
    bbox: Optional[np.ndarray] = None,
    intrinsic: Optional[np.ndarray] = None,
    pose: Optional[np.ndarray] = None,
    dataset: str = "",
    split: str = "",
    bbox_3d: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[np.ndarray]:
    """Prepare the 3D bounding box for the given model.

    Args:
        path (Optional[str]): Path to the 3D model file.
        cat (Optional[str]): Category name.
        bbox (Optional[np.ndarray]): 2D bounding box.
        intrinsic (Optional[np.ndarray]): Camera intrinsic matrix.
        pose (Optional[np.ndarray]): Camera pose matrix.
        dataset (str): Dataset name.
        split (str): Dataset split.
        bbox_3d (Optional[Dict]): Dictionary containing 3D bounding box paths.

    Returns:
        Optional[np.ndarray]: 3D bounding box with shape [8, 3] or None.
    """
    from src.lightning.utils.vis.vis_utils import get_3d_bbox_from_pts

    if (
        (
            dataset == "objaverse"
            or path is None
            or path == "none"
            or not os.path.exists(path)
        )
        and split != "demo"
        and bbox_3d is not None
    ):
        # Use provided 3D bounding box
        bbox_3d_path = bbox_3d[split][cat]
        bbox_3d_points = np.loadtxt(bbox_3d_path)
        return bbox_3d_points

    pt3ds = get_cached_points(path)

    if dataset == "co3d" or dataset == "moped":
        centroid = np.mean(pt3ds, axis=0)
        centered_pt3ds = pt3ds - centroid

        # Compute covariance matrix and perform eigen decomposition
        cov_matrix = np.cov(centered_pt3ds, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_indices]

        # Rotate point cloud to align with principal components
        rotated_pt3ds = centered_pt3ds @ eigen_vectors

        # Compute 3D bounding box from rotated point cloud
        bbox_3d_points = get_3d_bbox_from_pts(
            rotated_pt3ds, bbox, intrinsic, pose
        )  # [8, 3]

        # Rotate bounding box back to original coordinate system
        bbox_3d_points = bbox_3d_points @ eigen_vectors.T
        bbox_3d_points += centroid
    else:
        bbox_3d_points = get_3d_bbox_from_pts(pt3ds, bbox, intrinsic, pose)

    return bbox_3d_points


def consist_bbox3d(bbox_3d: torch.Tensor) -> torch.Tensor:
    """Ensure consistent 3D bounding box across batch.

    Args:
        bbox_3d (torch.Tensor): 3D bounding box with shape [B, 8, 3].

    Returns:
        torch.Tensor: Consistent 3D bounding box with shape [B, 8, 3].
    """
    # shape of bbox_3d: B, 8, 3
    # all batch need have the same bbox_3d
    if bbox_3d.ndim != 3 or bbox_3d.shape[1:] != (8, 3):
        raise ValueError(f"Invalid shape of bbox_3d: {bbox_3d.shape}")

    min_coords = torch.min(bbox_3d, dim=1)[0]  # [B, 3]
    max_coords = torch.max(bbox_3d, dim=1)[0]  # [B, 3]

    volumes = torch.prod(max_coords - min_coords, dim=1)  # [B]

    min_volume, min_idx = torch.min(volumes, dim=0)

    selected_min = min_coords[min_idx]  # [3]
    selected_max = max_coords[min_idx]  # [3]

    bits = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ],
        device=bbox_3d.device,
        dtype=torch.float32,
    )  # [8, 3]

    extent = selected_max - selected_min  # [3]

    corners = selected_min.unsqueeze(0) + bits * extent.unsqueeze(0)  # [8, 3]

    bbox_consistent = (
        corners.unsqueeze(0).expand(bbox_3d.shape[0], -1, -1).contiguous()
    )  # [B, 8, 3]

    return bbox_consistent


def make_bbox_features(
    bbox: torch.Tensor,
    type: str = "bbox_map",
    shape: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """Generate bounding box features based on the specified type.

    Args:
        bbox (torch.Tensor): Bounding box coordinates with shape [B, 8, 2] or [B, 9, 2] for center.
        type (str, optional): Type of features to generate. Options: 'heatmap', 'voting'. Defaults to 'heatmap'.
        shape (Tuple[int, int], optional): Shape of the feature map (H, W). Defaults to (64, 64).

    Returns:
        torch.Tensor: Dense features of the bounding box with shape corresponding to the type.
                    - 'voting': [B, 16, H, W]
                    - 'heatmap': [B, 8, H, W]
    """
    B = bbox.shape[0]
    H, W = shape

    if type == "voting":
        # feature shape B,H,W,8,2 (per pixel offset from 8 corners (dx, dy))
        B = bbox.shape[0]
        H, W = shape
        bbox = bbox.view(B, 8, 2)
        # dx = bbox[:, :, 0] - index_x
        # dy = bbox[:, :, 1] - index_y
        bbox_map = torch.zeros((B, H, W, 8, 2), dtype=torch.float32, device=bbox.device)
        index_x = torch.arange(W, dtype=torch.float32, device=bbox.device)
        index_y = torch.arange(H, dtype=torch.float32, device=bbox.device)

        for i in range(8):
            bbox_map[..., i, 0] = bbox[:, i, 0].view(B, 1, 1).expand(
                B, H, W
            ) - index_x.view(1, 1, W).expand(B, H, W)
            bbox_map[..., i, 1] = bbox[:, i, 1].view(B, 1, 1).expand(
                B, H, W
            ) - index_y.view(1, H, 1).expand(B, H, W)

        # normalize the offset
        bbox_map[..., 0] /= W
        bbox_map[..., 1] /= H

        # B, H, W, 8, 2 -> B 16, H, W
        bbox_map = bbox_map.flatten(start_dim=3, end_dim=4).permute(0, 3, 1, 2)

        return bbox_map

    elif type == "heatmap":
        # draw a heat map and the use gaussian kernel to smooth the heat map
        # return shape B, H, W, 1 (heatmap)
        B = bbox.shape[0]
        H, W = shape
        bbox = bbox.view(B, 8, 2)
        bbox_map = torch.zeros((B, H, W, 8), dtype=torch.float32, device=bbox.device)
        index_x = torch.arange(W, dtype=torch.float32, device=bbox.device)
        index_y = torch.arange(H, dtype=torch.float32, device=bbox.device)

        bbox_center = bbox.mean(dim=1)

        for i in range(8):
            dx = bbox[:, i, 0].view(B, 1, 1).expand(B, H, W) - index_x.view(
                1, 1, W
            ).expand(B, H, W)
            dy = bbox[:, i, 1].view(B, 1, 1).expand(B, H, W) - index_y.view(
                1, H, 1
            ).expand(B, H, W)
            # shape : B, H, W
            bbox_map[..., i] = torch.sqrt(dx**2 + dy**2)

            # dis from current point to center
            dis = torch.sqrt(
                (bbox_center[:, 0] - bbox[:, i, 0]) ** 2
                + (bbox_center[:, 1] - bbox[:, i, 1]) ** 2
            )
            scale_factor = (dis / 10) ** 2

            bbox_map[..., i] = torch.exp(
                -bbox_map[..., i] / scale_factor.unsqueeze(-1).unsqueeze(-1)
            )
            # normalize the heatmap
            bbox_map[..., i] = bbox_map[..., i] / bbox_map[..., i].max()
            # to [-1, 1]
            bbox_map[..., i] = bbox_map[..., i] * 2 - 1

        # B, H, W, 1 -> B, 1, H, W
        bbox_map = bbox_map.permute(0, 3, 1, 2)

        return bbox_map

    elif type == "cornernet":
        # Implementation would need to be extracted from original code
        # This is a placeholder for the full implementation
        heatmap = generate_cornernet_heatmap(bbox, H=H, W=W)
        heatmap = heatmap * 2 - 1  # [B, 8, H, W]
        return heatmap

    else:
        raise NotImplementedError(f"Invalid type {type}")
