"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:10:07
Description: dataset preprocess functions

"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as F

from ...utils.camera_transform import *
from .data_io import *
from .data_utils import *


def square_bbox(bbox: np.ndarray, padding: float = 0.1, astype=None) -> np.ndarray:
    """Compute a square bounding box with optional padding.

    Args:
        bbox (np.ndarray): Bounding box in [x_min, y_min, x_max, y_max] format.
        padding (float, optional): Padding factor. Defaults to 0.0.
        astype (type, optional): Data type of the output array. Defaults to None.

    Returns:
        np.ndarray: Square bounding box in [x_min, y_min, x_max, y_max] format.
    """
    if bbox is None:
        return None
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    size = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - size, center[1] - size, center[0] + size, center[1] + size],
        dtype=astype,
    )
    return square_bbox


def adjust_camera_intrinsics(K: np.ndarray, padding: dict) -> np.ndarray:
    K_adjusted = K.copy()
    K_adjusted[0, 2] += padding.get("left", 0)
    K_adjusted[1, 2] += padding.get("top", 0)
    return K_adjusted


def calculate_crop_parameters(
    image: Image.Image, bbox: np.ndarray, crop_dim: int, img_size: int
) -> torch.Tensor:
    """Calculate the parameters needed to crop an image based on a bounding
    box.

    Args:
        image (PIL.Image.Image): The input image.
        bbox (np.ndarray): The bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        crop_dim (int): The dimension to which the image will be cropped.
        img_size (int): The size to which the cropped image will be resized.

    Returns:
        torch.Tensor: A tensor containing the crop parameters, including width, height,
                      crop width, scale, and adjusted bounding box coordinates.
    """
    crop_center = (bbox[:2] + bbox[2:]) / 2
    width, height = image.size
    length = max(width, height)
    scale = length / min(width, height)
    crop_center += (length - np.array([width, height])) / 2

    # Convert to Normalized Device Coordinates (NDC)
    normalized_center = scale - 2 * scale * crop_center / length
    crop_width = 2 * scale * (bbox[2] - bbox[0]) / length
    crop_params = torch.tensor(
        [-normalized_center[0], -normalized_center[1], crop_width, scale]
    )
    return crop_params


def pad_image_based_on_bbox(
    image: Image.Image, bbox: np.ndarray
) -> Tuple[Image.Image, dict]:
    """If the bbox's 4 corners are outside the image, pad the image to include
    the bbox.

    Args:
        image (PIL.Image.Image): Image to be padded
        bbox (np.ndarray): Bounding box coordinates in the format [x_min, y_min, x_max, y_max]

    Returns:
        PIL.Image.Image: Padded image
    """

    width, height = image.size
    x_min, y_min, x_max, y_max = bbox
    padding_info = None

    dx, dy = x_max - x_min, y_max - y_min
    if dx > width and dy > height:
        return image, None

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        # pad the image
        left = max(0, -x_min)
        top = max(0, -y_min)
        right = max(0, x_max - width)
        bottom = max(0, y_max - height)
        padding = (int(left), int(top), int(right), int(bottom))
        image = ImageOps.expand(image, padding, fill=(0, 0, 0))
        # image.save("debug_padding.png")
        # exit()
        padding_info = {"left": left, "top": top, "right": right, "bottom": bottom}

    return image, padding_info


def pad_and_resize_image(
    image: Image.Image,
    crop_longest: bool,
    img_size: int,
    mask: Optional[Image.Image] = None,
    bbox_anno: Optional[np.ndarray] = None,
    bbox_obj: Optional[np.ndarray] = None,
    transform: Optional[transforms.Compose] = None,
    truncate_augmentation: callable = None,
    mask_augmentation: callable = None,
    mask_image: Optional[Image.Image] = None,  # used for mask augmentation
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, np.ndarray]:
    """Pad (through cropping) and resize an image, optionally with a mask.

    Args:
        image (PIL.Image.Image): Image to be processed.
        crop_longest (bool): Flag to indicate if the longest side should be cropped.
        img_size (int): Size to resize the image to.
        mask (Optional[PIL.Image.Image]): Mask to be processed.
        bbox_anno (Optional[np.ndarray]): Bounding box annotations (for crop).
        bbox_obj: object real box
        transform (Optional[transforms.Compose]): Transformations to apply.

    Returns:
        Tuple containing:
            - Transformed image tensor.
            - Transformed mask tensor or None.
            - Crop parameters tensor.
            - Adjusted bounding box coordinates as a NumPy array.
    """
    if transform is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(img_size, antialias=True)]
        )

    w, h = image.width, image.height
    if (crop_longest and bbox_anno is None) or not bbox_anno.any():
        crop_dim = max(h, w)
        top = (h - crop_dim) // 2
        left = (w - crop_dim) // 2
        bbox = np.array([left, top, left + crop_dim, top + crop_dim])
    else:
        bbox = np.array(bbox_anno)
        crop_dim = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if (bbox[2] - bbox[0]) != (bbox[3] - bbox[1]):
            bbox = square_bbox(bbox, padding=0.0, astype=int)

    crop_params = calculate_crop_parameters(image, bbox, crop_dim, img_size)
    if truncate_augmentation is not None:
        image = truncate_augmentation(image, bbox)
    if mask_augmentation is not None:
        image = mask_augmentation(image, mask_image, bbox)
    image = _crop_image(image, bbox, bbox_obj, white_bg=False)
    # image = _crop_image(image, bbox, None, white_bg=False)
    # save for debug
    # name = "debug_image_crop"
    # # maybe have stored some files, if so, add a number to the name
    # i = 0
    # while os.path.exists(f"{name}_{i}.png"):
    #     i += 1
    # image.save(f"{name}_{i}.png")

    image_transformed = transform(image).clamp(0.0, 1.0)
    # tensor_to_ numpy for debug
    # image_transformed_vis = image_transformed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite("debug_image_transformed.png", cv2.cvtColor(image_transformed_vis * 255, cv2.COLOR_RGB2BGR))
    # exit()

    if mask is not None:
        mask = _crop_image(mask, bbox)
        mask_transformed = transform(mask).clamp(0.0, 1.0)
        # use mask on image, set background to 0
        image_transformed = image_transformed * mask_transformed
    else:
        mask_transformed = None

    return image_transformed, mask_transformed, crop_params, bbox


def _crop_image(
    image: Image.Image,
    bbox: np.ndarray,
    bbox_obj: np.ndarray = None,
    white_bg: bool = False,
) -> Image.Image:
    """Crop an image to a bounding box. If the bounding box exceeds image
    dimensions, the image is padded.

    Args:
        image (PIL.Image.Image): Image to be cropped.
        bbox (np.ndarray): Bounding box for the crop [x_min, y_min, x_max, y_max].
        bbox_obj (np.ndarray): Object mask bounding box [x_min, y_min, x_max, y_max],
                               or None if no mask is applied.
        white_bg (bool): Flag to indicate if the background should be white.

    Returns:
        PIL.Image.Image: Cropped image.
    """
    if white_bg:
        # Create a white background if specified
        background = Image.new("RGB", image.size, (255, 255, 255))
    else:
        background = None

    try:
        try:
            if bbox_obj is not None:
                # Apply object mask by creating a mask image
                mask = Image.new(
                    "L", image.size, 0
                )  # "L" mode for grayscale (0: black, 255: white)
                draw = ImageDraw.Draw(mask)

                # Draw the object bounding box on the mask (filled rectangle)
                draw.rectangle(
                    [
                        int(bbox_obj[0]),
                        int(bbox_obj[1]),
                        int(bbox_obj[2]),
                        int(bbox_obj[3]),
                    ],
                    fill=255,
                )

                # Mask the image: keep only the region inside bbox_obj
                image = Image.composite(
                    image, Image.new("RGB", image.size, (0, 0, 0)), mask
                )
        except Exception as e:
            print(f"Mask Error: {e}")
            # import traceback
            # print(traceback.format_exc())
            pass

        # Crop the image to the bbox
        image_crop = F.crop(
            image,
            top=int(bbox[1]),
            left=int(bbox[0]),
            height=int(bbox[3] - bbox[1]),
            width=int(bbox[2] - bbox[0]),
        )
    except Exception as e:
        print(f"Crop Error: {e}")
        return image

    if white_bg and background is not None:
        # Paste the cropped image onto the white background
        background.paste(image_crop, (int(bbox[0]), int(bbox[1])))
        image_crop = background

    return image_crop


def adjust_intrinsic_matrix(
    K: np.ndarray, scale: Tuple[float, float], crop_offset: Tuple[int, int]
) -> np.ndarray:
    """Adjust the camera intrinsic matrix K based on scaling factors and crop
    offsets.

    Args:
        K (np.ndarray): Original camera intrinsic matrix of shape (3, 3).
        scale (Tuple[float, float]): Scaling factors (s_x, s_y).
        crop_offset (Tuple[int, int]): Crop offset from the top-left corner (t_x, t_y).

    Returns:
        np.ndarray: Adjusted camera intrinsic matrix of shape (3, 3).
    """
    s_x, s_y = scale
    t_x, t_y = crop_offset

    K_new = K.copy()
    K_new[0, 0] *= s_x  # f_x
    K_new[1, 1] *= s_y  # f_y
    K_new[0, 2] = K_new[0, 2] * s_x - t_x * s_x
    K_new[1, 2] = K_new[1, 2] * s_y - t_y * s_y

    return K_new


def gaussian2D(shape, sigma=1):
    """Generates a 2D Gaussian kernel.

    Parameters:
        shape (tuple): The shape of the Gaussian kernel (height, width).
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: A 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """Draws a Gaussian peak on the heatmap.

    Parameters:
        heatmap (numpy.ndarray): The heatmap to draw the Gaussian on.
        center (tuple): The (x, y) coordinates of the center.
        radius (int): The radius of the Gaussian kernel.
        k (float): The scaling factor for the Gaussian.

    Returns:
        None: The heatmap is modified in place.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap=0.3):
    """Calculates the Gaussian radius based on the size of the detection and
    minimum overlap.

    Parameters:
        det_size (tuple): The size of the detection (height, width).
        min_overlap (float): The minimum required overlap (IoU).

    Returns:
        float: The calculated Gaussian radius.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def generate_cornernet_heatmap(
    batch_bboxes, H, W, min_overlap=0.3, device="cpu", dtype=torch.float32
):
    """Generates CornerNet-style heatmaps.

    Parameters:
        batch_bboxes (torch.Tensor): Bounding box coordinates, shape [B, 8, 2],
                                     each bbox has 8 corners with (x, y) coordinates.
        H (int): Heatmap height.
        W (int): Heatmap width.
        min_overlap (float): Minimum overlap for Gaussian radius calculation.
        device (str): Device type, e.g., 'cpu' or 'cuda'.
        dtype (torch.dtype): Data type, e.g., torch.float32.

    Returns:
        torch.Tensor: Generated heatmaps, shape [B, 8, H, W].
    """
    B, C, _ = batch_bboxes.shape  # C should be 8, corresponding to 8 corners
    assert C == 8, "Each bounding box should have 8 corners"

    # Initialize heatmap tensor with zeros
    heatmap = torch.zeros(B, C, H, W, dtype=dtype, device=device)

    for b in range(B):
        # Extract bounding box coordinates for the current batch
        bbox = batch_bboxes[b]  # Shape: [8, 2]

        # Compute bounding box size (width and height)
        x_min, y_min = bbox.min(dim=0).values
        x_max, y_max = bbox.max(dim=0).values
        width = (x_max - x_min).item()
        height = (y_max - y_min).item()

        # Calculate Gaussian radius using the original gaussian_radius function
        radius = int(gaussian_radius((height, width), min_overlap))
        if radius <= 0:
            radius = 1  # Minimum radius to ensure Gaussian distribution

        for i in range(C):
            # Extract current corner coordinates and convert to integers
            center_x, center_y = bbox[i].cpu().numpy().astype(int)

            # Ensure coordinates are within heatmap bounds
            if center_x < 0 or center_y < 0 or center_x >= W or center_y >= H:
                continue  # Skip points outside the heatmap

            # Convert the heatmap tensor to numpy array for manipulation
            heatmap_np = heatmap[b, i].cpu().numpy()
            center = (center_x, center_y)

            # Draw Gaussian on the heatmap
            draw_gaussian(heatmap_np, center, radius, k=1)

            # Update the heatmap tensor with the modified numpy array
            heatmap[b, i] = torch.from_numpy(heatmap_np).to(device=device, dtype=dtype)

    # Clamp heatmap values to [0, 1] to maintain valid probability distributions
    heatmap = heatmap.clamp(0, 1)

    return heatmap


def get_gaussian_radius_by_iou(bbox: torch.Tensor, iou: float = 0.3) -> float:
    """Calculate the Gaussian radius based on the IoU of a bounding box.

    Args:
        bbox (torch.Tensor): Bounding box in either [4, 2] format (2D bounding box) or [8, 2] format (3D projection).
                             Each row is a point (x, y).
        iou (float): Desired IoU value.

        2wh-(w-2δ)(h-2δ)/(w-2δ)(h-2δ) >= t

        ((w+h)-sqrt((w+h)^2-4wh(1-t/1+t))) / 4


    Returns:
        float: Gaussian radius for shifting bbox corners such that the IoU of the shifted bbox is at least the desired IoU.
    """
    if bbox.shape not in [(4, 2), (8, 2)]:
        raise ValueError("Bounding box must be in [4, 2] or [8, 2] format.")

    if bbox.shape[0] == 8:
        bbox = _convert_3d_bbox_to_2d(bbox)

    if not (0 < iou < 1):
        raise ValueError("IoU must be a float between 0 and 1 (exclusive).")

    x_min, y_min = bbox.min(dim=0).values
    x_max, y_max = bbox.max(dim=0).values
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box with non-positive width or height.")

    A = width + height
    term_inside_sqrt = (A) ** 2 - 4 * width * height * (1 - iou) / (1 + iou)

    if term_inside_sqrt < 0:
        raise ValueError(
            "No real solution exists for the given IoU and bounding box dimensions."
        )

    sqrt_term = math.sqrt(term_inside_sqrt)
    delta_max = (A - sqrt_term) / 4

    radius = delta_max

    return radius


def _convert_3d_bbox_to_2d(bbox: torch.Tensor) -> torch.Tensor:
    """Convert a 3D bbox (8, 2) to a 2D bbox (4, 2) by finding the min and max
    in x and y directions.

    Args:
        bbox (torch.Tensor): Bounding box in [8, 2] format.

    Returns:
        torch.Tensor: Bounding box in [4, 2] format.
    """
    x_min = bbox[:, 0].min()
    x_max = bbox[:, 0].max()
    y_min = bbox[:, 1].min()
    y_max = bbox[:, 1].max()

    # Construct the 2D bounding box corners
    bbox_2d = torch.tensor(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=bbox.dtype,
        device=bbox.device,
    )

    return bbox_2d
