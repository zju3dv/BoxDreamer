from src.reconstruction.base import BaseReconstructor
from src.reconstruction.dust3r import DUSt3RReconstructor
from src.reconstruction.colmap import COLMAPReconstructor
from src.datasets.custom import CustomDataset
from src.models.BoxDreamerModel import BoxDreamer
from torch.utils.data import DataLoader

import os
import glob
import shutil
import argparse
import yaml
import omegaconf
from tqdm import tqdm
import loguru
import traceback
import numpy as np
from src.lightning.utils.vis.vis_utils import draw_3d_box, reproj
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
from src.demo.utils import *
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import time
import open3d as o3d
import datetime
import json
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from loguru import logger
from pathlib import Path

try:
    import cupy as cp

    CUPY_AVAILABLE = False  # disable gpu for point cloud rendering
    # log cupy is available
    # loguru.logger.info("Cupy is available， use gpu for point cloud rendering")
except Exception:
    CUPY_AVAILABLE = False
    # log cupy is not available
    loguru.logger.info("Cupy is not available， use cpu for point cloud rendering")

try:
    matplotlib.use("TkAgg")
except Exception:
    pass


# ============================================================================
# Logging Configuration
# ============================================================================


def setup_logger():
    """Setup custom logger with clean formatting."""
    loguru.logger.remove()  # Remove default handler

    # Custom format with colors
    log_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )

    # Add handler with custom format
    loguru.logger.add(sys.stderr, format=log_format, level="INFO", colorize=True)


# Initialize logger
setup_logger()


def log_header(text):
    """Print a formatted header."""
    width = 70
    loguru.logger.info("=" * width)
    loguru.logger.info(f"  {text}")
    loguru.logger.info("=" * width)


def log_step(text):
    """Print a step with arrow."""
    loguru.logger.info(f"→ {text}")


def log_success(text):
    """Print success message."""
    loguru.logger.success(f"✓ {text}")


def log_info(text):
    """Print info message."""
    loguru.logger.info(f"  {text}")


def log_warning(text):
    """Print warning message."""
    loguru.logger.warning(f"⚠ {text}")


def log_error(text):
    """Print error message."""
    loguru.logger.error(f"✗ {text}")


# ============================================================================
# Core Functions
# ============================================================================


def get_reconstructor(name: str) -> BaseReconstructor:
    if name == "mast3r":
        raise ValueError("MASt3R is not supported")
    elif name == "vggsfm":
        raise ValueError("VGGSFM is not supported")
    elif name == "dust3r":
        return DUSt3RReconstructor
    elif name == "colmap":
        return COLMAPReconstructor
    else:
        raise ValueError(f"Reconstructor {name} not found")


def run(images, masks, reconstructor: BaseReconstructor):
    reconstructor.set_data(images=images, masks=masks)

    try:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            ret = reconstructor.real_run()

        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()

        if stdout_output.strip():
            logger.debug(f"Reconstructor stdout: {stdout_output}")
        if stderr_output.strip():
            logger.debug(f"Reconstructor stderr: {stderr_output}")

    except Exception as e:
        log_error(f"Reconstruction failed: {str(e)}")
        reconstructor.reset_data()
        return None

    reconstructor.reset_data()
    return ret


def save_original_images(images_dir, output_dir):
    """Save copies of original images before cropping."""
    os.makedirs(output_dir, exist_ok=True)

    images = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(
        os.path.join(images_dir, "*.jpg")
    )

    for img_path in images:
        filename = os.path.basename(img_path).replace("-color", "")
        output_path = os.path.join(output_dir, filename)
        shutil.copy2(img_path, output_path)

    log_info(f"Saved {len(images)} original images")


def get_original_image_size(img_path):
    """Get the original image size."""
    img = Image.open(img_path)
    return img.size


def adjust_intrinsics_for_crop(intrinsics, original_size, crop_size=224):
    """Adjust camera intrinsics from cropped resolution to original resolution."""
    intrinsics_adjusted = intrinsics.copy().astype(np.float32)

    width, height = original_size
    square_size = min(width, height)
    scale_factor = square_size / crop_size

    intrinsics_adjusted[0, 0] *= scale_factor  # fx
    intrinsics_adjusted[1, 1] *= scale_factor  # fy
    intrinsics_adjusted[0, 2] *= scale_factor  # cx
    intrinsics_adjusted[1, 2] *= scale_factor  # cy

    center_x = width // 2
    center_y = height // 2
    left = center_x - square_size // 2
    top = center_y - square_size // 2

    intrinsics_adjusted[0, 2] += left
    intrinsics_adjusted[1, 2] += top

    return intrinsics_adjusted


def interactive_select_reference_images(images_dir, ref_num):
    """Interactive selection of reference images via command line."""
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import random

    images = glob.glob(images_dir + "/*.png")
    images.sort()

    if len(images) < ref_num:
        log_error(f"Insufficient images: {len(images)} available, {ref_num} required")
        return images

    already_sampled = set()
    selected_images = []

    log_step(f"Interactive reference selection ({ref_num} needed)")

    while len(selected_images) < ref_num:
        remaining_count = ref_num - len(selected_images)
        log_info(
            f"Selected: {len(selected_images)}/{ref_num}, need {remaining_count} more"
        )

        candidate_images = [
            img
            for img in images
            if img not in already_sampled and img not in selected_images
        ]

        if not candidate_images:
            log_warning("No more candidates available")
            break

        batch_size = min(5, len(candidate_images))
        batch = random.sample(candidate_images, batch_size)

        cols = min(3, batch_size)
        rows = (batch_size + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle("Candidate Reference Images", fontsize=16)

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for i, img_path in enumerate(batch):
            row, col = i // cols, i % cols
            img = Image.open(img_path)
            axes[row, col].imshow(np.array(img))
            axes[row, col].set_title(f"Image {i+1}\n{os.path.basename(img_path)}")
            axes[row, col].axis("off")

        for i in range(batch_size, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show(block=False)

        print("\nOptions: [1-5] to select | 'a' for all | 'n' to skip | 'q' to quit")
        choice = input("Your choice: ").strip().lower()

        already_sampled.update(batch)
        plt.close(fig)

        if choice == "q":
            break
        elif choice == "a":
            selected_images.extend(batch)
            log_info(f"Added all {batch_size} images")
        elif choice == "n":
            log_info("Skipped batch")
        else:
            try:
                selected_indices = [int(x) - 1 for x in choice.split()]
                valid_indices = [i for i in selected_indices if 0 <= i < batch_size]

                if valid_indices:
                    selected_batch = [batch[i] for i in valid_indices]
                    selected_images.extend(selected_batch)
                    log_info(f"Added {len(selected_batch)} images")
                else:
                    log_warning("Invalid selection")
            except ValueError:
                log_warning("Invalid input")

    if len(selected_images) < ref_num:
        remaining_count = ref_num - len(selected_images)
        remaining_images = [img for img in images if img not in selected_images]
        if remaining_images:
            additional = random.sample(
                remaining_images, min(remaining_count, len(remaining_images))
            )
            selected_images.extend(additional)
            log_info(f"Auto-added {len(additional)} images")

    if len(selected_images) > ref_num:
        selected_images = selected_images[:ref_num]

    log_success(f"Selected {len(selected_images)} reference images")
    display_final_selection(selected_images)

    return selected_images


def display_final_selection(selected_images):
    """Display the final selection of reference images."""
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    cols = min(5, len(selected_images))
    rows = (len(selected_images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle("Final Reference Image Selection", fontsize=16)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, img_path in enumerate(selected_images):
        row, col = i // cols, i % cols
        img = Image.open(img_path)
        axes[row, col].imshow(np.array(img))
        axes[row, col].set_title(f"Ref {i+1}\n{os.path.basename(img_path)}")
        axes[row, col].axis("off")

    for i in range(len(selected_images), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def preprocess_image(
    video_path,
    ref_num=5,
    skip=False,
    query_video=None,
    ref_path=None,
    interactive=False,
    use_grounding_dino=False,
    text_prompt=None,
):
    video_root = os.path.dirname(video_path)

    if ref_path is None:
        ref_path = video_root + "/reference"
        test_path = video_root + "/test"
        os.makedirs(ref_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
    else:
        video_root = os.path.dirname(query_video)
        test_path = video_root + "/test"
        os.makedirs(test_path, exist_ok=True)

    original_images_dir = video_root + "/original_images"
    os.makedirs(original_images_dir, exist_ok=True)

    if skip:
        return ref_path, test_path

    log_step("Preprocessing images")

    if use_grounding_dino:
        from .ov_det import GroundingDinoSegmentationApp as SegmentationApp

        if text_prompt is None:
            raise ValueError("text_prompt required for Grounding DINO")
        log_info(f"Using Grounding DINO with prompt: '{text_prompt}'")
    else:
        from .seg import VideoSegmentationApp as SegmentationApp

        log_info("Using manual segmentation")

    if query_video is not None and ref_path is not None and os.path.exists(ref_path):
        log_info("Processing with separate reference and query videos")

        ref_images = glob.glob(os.path.join(ref_path, "*.png")) + glob.glob(
            os.path.join(ref_path, "*.jpg")
        )
        ref_images.sort()

        if not ref_images:
            raise ValueError(f"No reference images found in {ref_path}")

        log_info(f"Found {len(ref_images)} reference images")

        temp_ref_dir = os.path.join(video_root, "temp_ref")
        os.makedirs(temp_ref_dir, exist_ok=True)

        ref_video_path = os.path.join(temp_ref_dir, "ref_video.mp4")
        create_video_from_images(ref_images, ref_video_path)

        if use_grounding_dino:
            app = SegmentationApp(text_prompt=text_prompt)
        else:
            app = SegmentationApp()
        app.load_video(ref_video_path)
        app.set_box_output_dir(os.path.join(temp_ref_dir, "box"))
        app.set_mask_output_dir(os.path.join(temp_ref_dir, "mask"))
        app.annotate_frame_matplotlib()
        app.process_and_save_video()
        del app

        if use_grounding_dino:
            app = SegmentationApp(text_prompt=text_prompt)
        else:
            app = SegmentationApp()
        app.load_video(query_video)
        app.set_box_output_dir(video_root + "/box")
        app.set_mask_output_dir(video_root + "/mask")
        app.annotate_frame_matplotlib()
        app.process_and_save_video()
        del app

        read_video_to_images(query_video, video_root + "/images")
        save_original_images(video_root + "/images", original_images_dir)

        crop_and_resize_image(video_root + "/images")
        crop_and_resize_image(video_root + "/mask")
        crop_and_resize_image(ref_path)
        crop_and_resize_image(os.path.join(temp_ref_dir, "mask"))

        ref_masks = glob.glob(os.path.join(temp_ref_dir, "mask/*.png"))
        ref_boxes = glob.glob(os.path.join(temp_ref_dir, "box/*.txt"))

        for mask in ref_masks:
            shutil.copy(mask, ref_path)
        for box in ref_boxes:
            shutil.copy(box, ref_path)

        query_images = glob.glob(video_root + "/images/*.png")
        for img in query_images:
            shutil.move(img, test_path)

        query_masks = glob.glob(video_root + "/mask/*.png")
        query_boxes = glob.glob(video_root + "/box/*.txt")

        for mask in query_masks:
            shutil.move(mask, test_path)
        for box in query_boxes:
            shutil.move(box, test_path)

        shutil.rmtree(temp_ref_dir)
        shutil.rmtree(video_root + "/mask", ignore_errors=True)
        shutil.rmtree(video_root + "/images", ignore_errors=True)
        shutil.rmtree(video_root + "/box", ignore_errors=True)

        log_success("Preprocessing complete")
        return ref_path, test_path

    if use_grounding_dino:
        app = SegmentationApp(text_prompt=text_prompt)
    else:
        app = SegmentationApp()
    app.load_video(video_path)
    app.set_box_output_dir(video_root + "/box")
    app.set_mask_output_dir(video_root + "/mask")
    app.annotate_frame_matplotlib()
    app.process_and_save_video()
    del app

    read_video_to_images(video_path, video_root + "/images")
    save_original_images(video_root + "/images", original_images_dir)

    crop_and_resize_image(video_root + "/images")
    crop_and_resize_image(video_root + "/mask")

    images = glob.glob(video_root + "/images/*.png")
    images.sort()

    if interactive:
        ref_images = interactive_select_reference_images(
            images_dir=video_root + "/images", ref_num=ref_num
        )
    else:
        ref_images = auto_select_reference_images(images, ref_num)

    test_images = list(set(images) - set(ref_images))

    for ref_image in ref_images:
        shutil.move(ref_image, ref_path)
    for test_image in test_images:
        shutil.move(test_image, test_path)

    boxes = glob.glob(video_root + "/box/*.txt")
    boxes.sort()
    masks = glob.glob(video_root + "/mask/*.png")
    masks.sort()

    ref_filenames = [
        os.path.basename(img).split(".")[0].split("-")[0] for img in ref_images
    ]

    ref_boxes = [
        box
        for box in boxes
        if os.path.basename(box).split(".")[0].split("-")[0] in ref_filenames
    ]
    test_boxes = list(set(boxes) - set(ref_boxes))

    ref_masks = [
        mask
        for mask in masks
        if os.path.basename(mask).split(".")[0].split("-")[0] in ref_filenames
    ]
    test_masks = list(set(masks) - set(ref_masks))

    for ref_box in ref_boxes:
        shutil.move(ref_box, ref_path)
    for test_box in test_boxes:
        shutil.move(test_box, test_path)
    for ref_mask in ref_masks:
        shutil.move(ref_mask, ref_path)
    for test_mask in test_masks:
        shutil.move(test_mask, test_path)

    shutil.rmtree(video_root + "/mask")
    shutil.rmtree(video_root + "/images")
    shutil.rmtree(video_root + "/box")

    log_success("Preprocessing complete")
    return ref_path, test_path


def auto_select_reference_images(images, ref_num):
    """Automatically select reference images evenly distributed."""
    if len(images) <= ref_num:
        log_warning(f"Using all {len(images)} images (need {ref_num})")
        return images

    indices = np.linspace(0, len(images) - 1, ref_num, dtype=int)
    selected_images = [images[i] for i in indices]

    log_info(
        f"Auto-selected {len(selected_images)} references at indices: {indices.tolist()}"
    )

    return selected_images


def to_gpu(data: dict):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()
    return data


def warp_model(state_dict):
    try:
        state_dict = state_dict["state_dict"]
    except KeyError:
        pass

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("BoxDreamer.", "")] = v
    return new_state_dict


def prepare_reference_images_with_boxes(
    ref_imgs, ref_path, pred_poses, pred_intrinsics, bbox_3d
):
    """Prepare reference images with 3D bounding boxes."""
    ref_images_with_boxes = []

    for i, ref_img_path in enumerate(ref_imgs):
        ref_img = Image.open(ref_img_path).convert("RGB")
        ref_img_np = np.array(ref_img)

        pose = pred_poses[i] if i < len(pred_poses) else pred_poses[0]
        intri = pred_intrinsics[i] if i < len(pred_intrinsics) else pred_intrinsics[0]

        pose = pose.astype(np.float32) if isinstance(pose, np.ndarray) else pose
        intri = intri.astype(np.float32) if isinstance(intri, np.ndarray) else intri

        proj_bbox = reproj(intri, pose, bbox_3d)
        ref_img_with_box = draw_3d_box(ref_img_np, proj_bbox)

        ref_images_with_boxes.append(ref_img_with_box)

    return ref_images_with_boxes


def load_and_render_point_cloud(ply_path, pose, intrinsics, img_shape):
    """Load and render point cloud to 2D image with optimized rendering."""
    if CUPY_AVAILABLE:
        return load_and_render_point_cloud_gpu(ply_path, pose, intrinsics, img_shape)
    else:
        return load_and_render_point_cloud_cpu(ply_path, pose, intrinsics, img_shape)


def load_and_render_point_cloud_cpu(ply_path, pose, intrinsics, img_shape):
    """Load and render point cloud to 2D image with optimized rendering."""
    try:
        pcd = o3d.io.read_point_cloud(ply_path)

        if not pcd.has_points():
            return np.zeros((*img_shape, 3), dtype=np.uint8)

        points = np.asarray(pcd.points).astype(np.float32)
        colors = (
            np.asarray(pcd.colors).astype(np.float32)
            if pcd.has_colors()
            else np.ones_like(points) * 0.5
        )

        pose = (
            pose.astype(np.float32)
            if isinstance(pose, np.ndarray)
            else np.array(pose, dtype=np.float32)
        )
        intrinsics = (
            intrinsics.astype(np.float32)
            if isinstance(intrinsics, np.ndarray)
            else np.array(intrinsics, dtype=np.float32)
        )

        # Transform to camera space
        cam_to_world = pose
        points_homogeneous = np.concatenate(
            [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
        )
        points_cam = (cam_to_world @ points_homogeneous.T).T[:, :3]

        # Project to image
        points_2d = (intrinsics @ points_cam.T).T
        points_2d = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-8)

        # Filter valid points
        valid_depth = points_cam[:, 2] > 0
        valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_shape[1])
        valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_shape[0])
        valid_mask = valid_depth & valid_x & valid_y

        rendered = np.zeros((*img_shape, 3), dtype=np.uint8)

        if np.any(valid_mask):
            valid_points_2d = points_2d[valid_mask].astype(int)
            valid_colors = (colors[valid_mask] * 255).astype(np.uint8)

            for (x, y), color in zip(valid_points_2d, valid_colors):
                cv2.circle(rendered, (x, y), 2, color.tolist(), -1)

        return rendered

    except Exception as e:
        log_error(f"Point cloud rendering failed: {str(e)}")
        return np.zeros((*img_shape, 3), dtype=np.uint8)


def load_and_render_point_cloud_gpu(ply_path, pose, intrinsics, img_shape):
    """Load and render point cloud with GPU acceleration."""
    try:
        pcd = o3d.io.read_point_cloud(ply_path)

        if not pcd.has_points():
            return np.zeros((*img_shape, 3), dtype=np.uint8)

        # Move to GPU
        points = cp.asarray(pcd.points, dtype=cp.float32)
        colors = (
            cp.asarray(pcd.colors, dtype=cp.float32)
            if pcd.has_colors()
            else cp.ones_like(points) * 0.5
        )

        pose = cp.asarray(pose, dtype=cp.float32)
        intrinsics = cp.asarray(intrinsics, dtype=cp.float32)

        # Transform to camera space
        ones = cp.ones((len(points), 1), dtype=cp.float32)
        points_homogeneous = cp.concatenate([points, ones], axis=1)
        points_cam = (pose @ points_homogeneous.T).T[:, :3]

        # Project to image
        points_2d = (intrinsics @ points_cam.T).T
        points_2d = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-8)

        # Filter valid points
        valid_depth = points_cam[:, 2] > 0
        valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_shape[1])
        valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_shape[0])
        valid_mask = valid_depth & valid_x & valid_y

        # Transfer back to CPU for rendering
        valid_points_2d = cp.asnumpy(points_2d[valid_mask]).astype(int)
        valid_colors = cp.asnumpy(colors[valid_mask] * 255).astype(np.uint8)

        rendered = np.zeros((*img_shape, 3), dtype=np.uint8)

        if len(valid_points_2d) > 0:
            y_coords = valid_points_2d[:, 1]
            x_coords = valid_points_2d[:, 0]
            rendered[y_coords, x_coords] = valid_colors

        return rendered

    except Exception as e:
        log_error(f"Point cloud rendering failed: {str(e)}")
        return np.zeros((*img_shape, 3), dtype=np.uint8)


def add_text_label(
    image,
    text,
    position="top",
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    font_scale=0.8,
    thickness=2,
    padding=10,
):
    """Add text label to image with Times New Roman font."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font_size = int(60 * font_scale)
    try:
        font_paths = [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "C:\\Windows\\Fonts\\times.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    h, w = image.shape[:2]

    if position == "top":
        x = (w - text_width) // 2
        y = padding
    elif position == "bottom":
        x = (w - text_width) // 2
        y = h - text_height - padding
    elif position == "top-left":
        x = padding
        y = padding
    elif position == "top-right":
        x = w - text_width - padding
        y = padding
    else:
        x = padding
        y = padding

    bg_color_rgb = tuple(reversed(bg_color))
    text_color_rgb = tuple(reversed(text_color))

    draw.rectangle(
        [
            (max(0, x - padding), max(0, y - padding)),
            (min(w, x + text_width + padding), min(h, y + text_height + padding)),
        ],
        fill=bg_color_rgb,
    )

    draw.text((x, y), text, font=font, fill=text_color_rgb)

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return result


def add_fps_overlay(frame, fps_value):
    """Add FPS overlay to frame."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = 48
    try:
        font_paths = [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "C:\\Windows\\Fonts\\times.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    fps_text = f"FPS: {fps_value:.1f}"

    bbox = draw.textbbox((0, 0), fps_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 10
    x = frame.shape[1] - text_width - padding
    y = padding

    bg_padding = 5
    draw.rectangle(
        [
            (x - bg_padding, y - bg_padding),
            (x + text_width + bg_padding, y + text_height + bg_padding),
        ],
        fill=(0, 0, 0, 153),
    )

    draw.text((x, y), fps_text, font=font, fill=(0, 255, 0, 255))

    pil_img = pil_img.convert("RGBA")
    combined = Image.alpha_composite(pil_img, overlay)

    result = cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)

    return result


def create_academic_layout(
    query_frame, ref_images_with_boxes, point_cloud_render=None, fps_value=None
):
    """Create academic-style video layout."""
    import cv2
    import numpy as np

    query_h, query_w = query_frame.shape[:2]

    query_labeled = add_text_label(
        query_frame,
        "Predicted Object Pose",
        position="top",
        bg_color=(185, 128, 41),
        text_color=(255, 255, 255),
        font_scale=0.9,
        padding=15,
    )

    if point_cloud_render is not None:
        pc_resized = cv2.resize(point_cloud_render, (query_w, query_h))

        pc_labeled = add_text_label(
            pc_resized,
            "Rendered Point Cloud",
            position="top",
            bg_color=(60, 76, 231),
            text_color=(255, 255, 255),
            font_scale=0.9,
            padding=15,
        )

        top_row = np.hstack([query_labeled, pc_labeled])
        top_row_h = query_h
        top_row_w = query_w * 2
    else:
        top_row = query_labeled
        top_row_h = query_h
        top_row_w = query_w

    num_refs = len(ref_images_with_boxes)

    if num_refs == 0:
        combined = top_row
        if fps_value is not None:
            combined = add_fps_overlay(combined, fps_value)
        return combined

    item_width = top_row_w // num_refs
    item_height = item_width

    ref_resized = []
    for i, ref_img in enumerate(ref_images_with_boxes):
        h, w = ref_img.shape[:2]

        scale = min(item_width / w, item_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        ref_resized_temp = cv2.resize(ref_img, (new_w, new_h))

        ref_square = np.zeros((item_height, item_width, 3), dtype=np.uint8)

        y_offset = (item_height - new_h) // 2
        x_offset = (item_width - new_w) // 2
        ref_square[
            y_offset : y_offset + new_h, x_offset : x_offset + new_w
        ] = ref_resized_temp

        ref_labeled = add_text_label(
            ref_square,
            f"Reference {i+1}",
            position="bottom",
            bg_color=(113, 204, 46),
            text_color=(255, 255, 255),
            font_scale=0.8,
            padding=15,
        )
        ref_resized.append(ref_labeled)

    bottom_row = np.hstack(ref_resized)

    if bottom_row.shape[1] != top_row_w:
        bottom_row = cv2.resize(bottom_row, (top_row_w, bottom_row.shape[0]))

    combined = np.vstack([top_row, bottom_row])

    if fps_value is not None:
        combined = add_fps_overlay(combined, fps_value)

    border_color = (200, 200, 200)
    border_thickness = 3
    combined = cv2.copyMakeBorder(
        combined,
        border_thickness,
        border_thickness,
        border_thickness,
        border_thickness,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )

    return combined


def create_video_layout(
    query_frame,
    ref_images_with_boxes,
    layout_type="horizontal",
    point_cloud_render=None,
    fps_value=None,
):
    """Create combined video frame."""
    if layout_type == "academic":
        return create_academic_layout(
            query_frame, ref_images_with_boxes, point_cloud_render, fps_value
        )
    else:
        combined = query_frame

    if fps_value is not None:
        combined = add_fps_overlay(combined, fps_value)

    return combined


def cleanup_intermediate_files(video_root, reference_path, test_path):
    """Clean up intermediate directories."""
    dirs_to_remove = [reference_path, test_path]

    original_images_dir = os.path.join(video_root, "original_images")
    if os.path.exists(original_images_dir):
        dirs_to_remove.append(original_images_dir)

    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                log_warning(f"Cleanup failed for {dir_path}: {str(e)}")


def log_to_rerun(
    rr,
    frame_idx,
    query_frame,
    ref_images,
    point_cloud_render,
    pose,
    intrinsics,
    bbox_3d,
    fps_value,
    ply_path=None,
):
    """Log visualization data to Rerun."""
    rr.set_time_sequence("frame", frame_idx)

    rr.log("query/image", rr.Image(query_frame))

    for i, ref_img in enumerate(ref_images):
        rr.log(f"references/ref_{i}", rr.Image(ref_img))

    if point_cloud_render is not None:
        rr.log("point_cloud/rendered", rr.Image(point_cloud_render))

    if ply_path is not None and os.path.exists(ply_path):
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            if pcd.has_points():
                points = np.asarray(pcd.points).astype(np.float32)
                colors = (
                    np.asarray(pcd.colors).astype(np.float32)
                    if pcd.has_colors()
                    else None
                )

                if colors is not None:
                    colors_uint8 = (colors * 255).astype(np.uint8)
                    rr.log("3d/point_cloud", rr.Points3D(points, colors=colors_uint8))
                else:
                    rr.log("3d/point_cloud", rr.Points3D(points))
        except Exception as e:
            log_warning(f"Rerun point cloud logging failed: {str(e)}")

    if bbox_3d is not None:
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        lines = []
        for edge in edges:
            lines.append([bbox_3d[edge[0]], bbox_3d[edge[1]]])

        rr.log("3d/bbox", rr.LineStrips3D(lines, colors=[255, 0, 0, 255]))

    try:
        rotation_matrix = pose[:3, :3]
        translation = pose[:3, 3]

        rr.log(
            "3d/camera", rr.Transform3D(translation=translation, mat3x3=rotation_matrix)
        )

        if intrinsics is not None:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            height, width = query_frame.shape[:2]

            rr.log(
                "3d/camera/pinhole",
                rr.Pinhole(
                    resolution=[width, height],
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                ),
            )
    except Exception as e:
        log_warning(f"Camera logging failed: {str(e)}")

    try:
        rr.log("stats/fps", rr.TimeSeriesScalar(fps_value))
    except AttributeError:
        try:
            rr.log("stats/fps", rr.Scalar(fps_value))
        except AttributeError:
            rr.log("stats/fps_text", rr.TextLog(f"FPS: {fps_value:.1f}"))


def create_output_directories(
    video_path, output_base=f"{Path(__file__).parent.parent.parent}/cache/BoxDreamer"
):
    """Create timestamped output directory structure."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{timestamp}-{video_name}"
    output_dir = os.path.join(output_base, output_dir_name)

    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "base": output_dir,
        "final_video": os.path.join(output_dir, "final_output.mp4"),
        "final_video_large": os.path.join(output_dir, "final_output_large.mp4"),
        "query_video": os.path.join(output_dir, "query_video.mp4"),
        "query_video_large": os.path.join(output_dir, "query_video_large.mp4"),
        "point_cloud_video": os.path.join(output_dir, "point_cloud_video.mp4"),
        "point_cloud_video_large": os.path.join(
            output_dir, "point_cloud_video_large.mp4"
        ),
        "reference_images": os.path.join(output_dir, "reference_images"),
        "metadata": os.path.join(output_dir, "metadata.json"),
    }

    os.makedirs(paths["reference_images"], exist_ok=True)

    log_info(f"Output: {output_dir}")

    return paths


def save_metadata(paths, args, avg_fps=None, num_frames=None):
    """Save metadata about the processing."""
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_video": args.video,
        "query_video": args.query_video,
        "reference_path": args.ref_path,
        "num_references": args.ref_num,
        "reconstructor": args.reconstructor,
        "layout": args.layout,
        "output_fps": args.fps,
        "show_point_cloud": args.show_point_cloud,
        "use_original_resolution": args.use_original_resolution,
        "num_frames_processed": num_frames,
        "average_inference_fps": avg_fps,
        "checkpoint": args.ckpt,
        "huggingface_model": args.hf,
    }

    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)


def save_reference_images_with_boxes(ref_images_with_boxes, output_dir):
    """Save reference images with bounding boxes."""
    for i, ref_img in enumerate(ref_images_with_boxes):
        output_path = os.path.join(output_dir, f"reference_{i+1:03d}.png")
        ref_img_pil = Image.fromarray(ref_img)
        ref_img_pil.save(output_path)


def create_individual_videos(query_frames, point_cloud_frames, fps, paths):
    """Create individual videos for query and point cloud."""
    if query_frames:
        query_small = [f[0] for f in query_frames]
        if query_small:
            height, width = query_small[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(paths["query_video"], fourcc, fps, (width, height))

            for frame in tqdm(query_small, desc="Writing query video", leave=False):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            video.release()
            log_success(f"Query video: {os.path.basename(paths['query_video'])}")

        query_large = [f[1] for f in query_frames]
        if query_large:
            height, width = query_large[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                paths["query_video_large"], fourcc, fps, (width, height)
            )

            for frame in tqdm(
                query_large, desc="Writing query video (HD)", leave=False
            ):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            video.release()
            log_success(
                f"Query video (HD): {os.path.basename(paths['query_video_large'])}"
            )

    if point_cloud_frames and any(f[0] is not None for f in point_cloud_frames):
        pc_small = [f[0] for f in point_cloud_frames if f[0] is not None]
        if pc_small:
            height, width = pc_small[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                paths["point_cloud_video"], fourcc, fps, (width, height)
            )

            for frame in tqdm(pc_small, desc="Writing point cloud video", leave=False):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            video.release()
            log_success(
                f"Point cloud video: {os.path.basename(paths['point_cloud_video'])}"
            )

        pc_large = [f[1] for f in point_cloud_frames if f[1] is not None]
        if pc_large:
            height, width = pc_large[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                paths["point_cloud_video_large"], fourcc, fps, (width, height)
            )

            for frame in tqdm(
                pc_large, desc="Writing point cloud video (HD)", leave=False
            ):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            video.release()
            log_success(
                f"Point cloud video (HD): {os.path.basename(paths['point_cloud_video_large'])}"
            )


def main():
    parser = argparse.ArgumentParser(description="BoxDreamer Demo")
    parser.add_argument(
        "--use_grounding_dino", action="store_true", help="use Grounding DINO"
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help="text prompt for Grounding DINO"
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.35, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    parser.add_argument(
        "--reconstructor", type=str, default="DUSt3R", help="reconstructor name"
    )
    parser.add_argument("--video", type=str, default="test.mp4", help="video path")
    parser.add_argument(
        "--query_video", type=str, default=None, help="query video path"
    )
    parser.add_argument(
        "--ref_path", type=str, default=None, help="reference image path"
    )
    parser.add_argument(
        "--ref_num", type=int, default=5, help="number of reference images"
    )
    parser.add_argument("--skip", action="store_true", help="skip preprocessing")
    parser.add_argument("--ckpt", type=str, default=None, help="model checkpoint path")
    parser.add_argument(
        "--hf", action="store_true", help="use huggingface", default=True
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default=f"{Path(__file__).parent.parent.parent}/cache/BoxDreamer",
        help="output directory",
    )
    parser.add_argument("--fps", type=int, default=24, help="output video FPS")
    parser.add_argument(
        "--layout",
        type=str,
        default="academic",
        choices=["academic"],
        help="layout type",
    )
    parser.add_argument(
        "--keep_intermediate", action="store_true", help="keep intermediate files"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="interactive reference selection"
    )
    parser.add_argument(
        "--show_point_cloud", action="store_true", help="show point cloud"
    )
    parser.add_argument("--rerun", action="store_true", help="use Rerun visualization")
    parser.add_argument(
        "--use_original_resolution",
        action="store_true",
        help="use original resolution",
        default=True,
    )
    parser.add_argument(
        "--save_individual_videos",
        action="store_true",
        help="save individual videos",
        default=True,
    )
    args = parser.parse_args()

    if args.use_grounding_dino and args.text_prompt is None:
        parser.error("--text_prompt required with --use_grounding_dino")

    log_header("BoxDreamer Demo")

    output_paths = create_output_directories(args.video, args.output_base)

    if args.rerun:
        try:
            import rerun as rr

            rr.init("BoxDreamer", spawn=True)
            log_info("Rerun enabled")
        except ImportError:
            log_error("Rerun not installed: pip install rerun-sdk")
            args.rerun = False

    CONFIG_DIR = Path(__file__).parent / "configs"

    with open(CONFIG_DIR / "data.yaml") as f:
        data_cfgs = omegaconf.OmegaConf.load(f)

    with open(CONFIG_DIR / "reconstructor.yaml") as f:
        recon_cfgs = omegaconf.OmegaConf.load(f)

    with open(CONFIG_DIR / "model.yaml") as f:
        model_cfgs = omegaconf.OmegaConf.load(f)

    data_cfgs.Custom.config.root = os.path.dirname(args.video)

    reference_path, test_path = preprocess_image(
        args.video,
        args.ref_num,
        args.skip,
        args.query_video,
        args.ref_path,
        args.interactive,
        args.use_grounding_dino,
        args.text_prompt,
    )

    if args.query_video is not None and args.ref_path is not None:
        mode = "different-scene"
    else:
        mode = "same-scene"

    ds = CustomDataset(data_cfgs.Custom.config, split="test")

    root = data_cfgs.Custom.config.root
    video_root = os.path.dirname(args.video)
    original_images_dir = os.path.join(video_root, "original_images")

    ref_imgs = (
        glob.glob(os.path.join(reference_path, "*-color.png"))
        + glob.glob(os.path.join(reference_path, "*-color.jpg"))
        + glob.glob(os.path.join(reference_path, "*-color.jpeg"))
    )
    ref_imgs.sort()

    ref_masks = (
        glob.glob(os.path.join(reference_path, "*-mask.png"))
        + glob.glob(os.path.join(reference_path, "*-mask.jpg"))
        + glob.glob(os.path.join(reference_path, "*-mask.jpeg"))
    )
    ref_masks.sort()

    reconstructor_name = args.reconstructor.lower()
    reconstructor = get_reconstructor(reconstructor_name)(
        recon_cfgs[reconstructor_name + "_cfg"]["method"],
        recon_cfgs[reconstructor_name + "_cfg"]["weight"],
        recon_cfgs[reconstructor_name + "_cfg"]["config"],
    )

    log_step("Reconstructing reference images")
    ret = run(ref_imgs, ref_masks, reconstructor)

    if ret is None:
        log_error("Reconstruction failed")
        return

    log_success("Reconstruction complete")

    pred_poses = ret["poses"]
    if isinstance(pred_poses, torch.Tensor):
        pred_poses = pred_poses.detach().cpu().numpy()
    pred_intrinsics = ret["intrinsics"]
    if isinstance(pred_intrinsics, torch.Tensor):
        pred_intrinsics = pred_intrinsics.detach().cpu().numpy()
    model_path = ret["ply_path"]

    pred_poses = pred_poses.astype(np.float32)
    pred_intrinsics = pred_intrinsics.astype(np.float32)

    ds.set_intrinsic(pred_intrinsics[0])
    ds.set_model_path(model_path)
    ds.set_ref_root(reference_path)

    for i in range(len(pred_poses)):
        file_name = ref_imgs[i].split("/")[-1].split(".")[0].split("-")[0]
        with open(os.path.join(reference_path, f"{file_name}-pose.txt"), "w") as f:
            for j in range(4):
                f.write(" ".join([str(x) for x in pred_poses[i][j]]) + "\n")

    for i in range(len(pred_intrinsics)):
        file_name = ref_imgs[i].split("/")[-1].split(".")[0].split("-")[0]
        with open(
            os.path.join(reference_path, f"{file_name}-intrinsics.txt"), "w"
        ) as f:
            for j in range(3):
                f.write(" ".join([str(x) for x in pred_intrinsics[i][j]]) + "\n")

    ds.set_test_root(test_path)

    if mode == "different-scene":
        query_imgs = (
            glob.glob(os.path.join(test_path, "*-color.png"))
            + glob.glob(os.path.join(test_path, "*-color.jpg"))
            + glob.glob(os.path.join(test_path, "*-color.jpeg"))
        )
        query_imgs.sort()
        query_masks = (
            glob.glob(os.path.join(test_path, "*-mask.png"))
            + glob.glob(os.path.join(test_path, "*-mask.jpg"))
            + glob.glob(os.path.join(test_path, "*-mask.jpeg"))
        )
        query_masks.sort()

        ret = run(query_imgs[:1], query_masks[:1], reconstructor)
        if ret is None:
            log_error("Query reconstruction failed")
            return

        pred_intrinsics = ret["intrinsics"]
        if isinstance(pred_intrinsics, torch.Tensor):
            pred_intrinsics = pred_intrinsics.detach().cpu().numpy()
        pred_intrinsics = pred_intrinsics.astype(np.float32)
        ds.set_intrinsic(pred_intrinsics[0])

    ds.load_data()

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    log_step("Initializing BoxDreamer model")
    model = BoxDreamer(model_cfgs).cuda()
    model.eval()

    if args.ckpt is not None:
        log_info(f"Loading checkpoint: {args.ckpt}")
        model.load_state_dict(warp_model(torch.load(args.ckpt)))
        log_success("Checkpoint loaded")
    else:
        assert args.hf, "Provide checkpoint or use --hf"
        if args.hf:
            log_info("Loading from HuggingFace")
            model.load_state_dict(
                warp_model(
                    load_file(
                        hf_hub_download(
                            "yyh929/BoxDreamer", "BoxDreamer-vitb.safetensor"
                        ),
                        device="cuda",
                    )
                )
            )
            log_success("Model loaded from HuggingFace")

    first_data = next(iter(dl))
    bbox_3d = (
        first_data["bbox_3d_original"][0][-1].detach().cpu().numpy().astype(np.float32)
    )

    log_step("Preparing reference images")
    ref_images_with_boxes = prepare_reference_images_with_boxes(
        ref_imgs, reference_path, pred_poses, pred_intrinsics, bbox_3d
    )
    save_reference_images_with_boxes(
        ref_images_with_boxes, output_paths["reference_images"]
    )
    log_success(f"Saved {len(ref_images_with_boxes)} reference images")

    all_frames = []
    all_frames_large = []
    query_frames = []
    point_cloud_frames = []
    frame_times = []

    log_step("Processing frames")

    try:
        for i, data in enumerate(tqdm(dl, desc="Inference")):
            start_time = time.time()

            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                ret = model(to_gpu(data))

            end_time = time.time()
            frame_time = end_time - start_time
            frame_times.append(frame_time)

            recent_times = frame_times[-10:]
            avg_time = np.mean(recent_times)
            current_fps = 1.0 / avg_time if avg_time > 0 else 0.0

            box = ret["regression_boxes"][0][-1].detach().cpu().numpy()
            pose = ret["pred_poses"][0][-1].detach().cpu().numpy().astype(np.float32)
            intri = (
                ret["original_intrinsics"][0][-1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            original_image_path = data["original_images"][-1][0]
            bbox_3d_data = (
                data["bbox_3d_original"][0][-1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            image_filename = os.path.basename(original_image_path).replace("-color", "")
            original_img_path = os.path.join(original_images_dir, image_filename)

            cropped_image = Image.open(original_image_path).convert("RGB")
            cropped_image_np = np.array(cropped_image)

            if args.use_original_resolution:
                original_image = Image.open(original_img_path).convert("RGB")
                original_image_np = np.array(original_image)
                original_size = original_image.size
                intri_large = adjust_intrinsics_for_crop(
                    intri, original_size, crop_size=224
                )
            else:
                original_image_np = cropped_image_np
                intri_large = intri.copy()

            proj_bbox = reproj(intri, pose, bbox_3d_data)
            proj_bbox_large = reproj(intri_large, pose, bbox_3d_data)

            box = ((box + 1) / 2) * 224
            image = data["images"][0][-1].detach().cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            image = np.ascontiguousarray(image)

            query_frame_small = draw_3d_box(image, box)
            query_frame_large = draw_3d_box(
                original_image_np,
                proj_bbox_large,
                linewidth=3 if not args.use_original_resolution else 6,
            )

            query_frames.append((query_frame_small, query_frame_large))

            point_cloud_render_small = None
            point_cloud_render_large = None
            if args.show_point_cloud and model_path is not None:
                point_cloud_render_small = load_and_render_point_cloud(
                    model_path, pose, intri, query_frame_small.shape[:2]
                )
                point_cloud_render_large = load_and_render_point_cloud(
                    model_path, pose, intri_large, query_frame_large.shape[:2]
                )

            point_cloud_frames.append(
                (point_cloud_render_small, point_cloud_render_large)
            )

            if args.rerun:
                log_to_rerun(
                    rr,
                    i,
                    query_frame_large,
                    ref_images_with_boxes,
                    point_cloud_render_large,
                    pose,
                    intri_large,
                    bbox_3d_data,
                    current_fps,
                    model_path,
                )

            combined_frame = create_video_layout(
                query_frame_small,
                ref_images_with_boxes,
                layout_type=args.layout,
                point_cloud_render=point_cloud_render_small,
                fps_value=current_fps,
            )
            combined_frame_large = create_video_layout(
                query_frame_large,
                ref_images_with_boxes,
                layout_type=args.layout,
                point_cloud_render=point_cloud_render_large,
                fps_value=current_fps,
            )

            all_frames.append(combined_frame)
            all_frames_large.append(combined_frame_large)

    except Exception as e:
        log_error(f"Frame processing error: {str(e)}")
        loguru.logger.error(traceback.format_exc())

    avg_fps = None
    if frame_times:
        avg_fps = 1.0 / np.mean(frame_times)
        log_success(f"Average FPS: {avg_fps:.1f}")

    save_metadata(output_paths, args, avg_fps, len(all_frames))

    if args.save_individual_videos:
        log_step("Creating individual videos")
        create_individual_videos(
            query_frames, point_cloud_frames, args.fps, output_paths
        )

    if all_frames:
        log_step("Creating final video")
        height, width, layers = all_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            output_paths["final_video"], fourcc, args.fps, (width, height)
        )

        for frame in tqdm(all_frames, desc="Encoding", leave=False):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

        video.release()
        log_success(f"Final video: {os.path.basename(output_paths['final_video'])}")

    if all_frames_large:
        height, width, layers = all_frames_large[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            output_paths["final_video_large"], fourcc, args.fps, (width, height)
        )

        for frame in tqdm(all_frames_large, desc="Encoding (HD)", leave=False):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

        video.release()
        log_success(
            f"Final video (HD): {os.path.basename(output_paths['final_video_large'])}"
        )

    log_header("Output Summary")
    log_info(f"Directory: {output_paths['base']}")
    log_info(
        f"Videos: {len([k for k in output_paths.keys() if 'video' in k and os.path.exists(output_paths[k])])} files"
    )
    log_info(f"References: {len(ref_images_with_boxes)} images")
    log_info(f"Frames: {len(all_frames)} processed")
    if avg_fps:
        log_info(f"Performance: {avg_fps:.1f} FPS")

    if not args.keep_intermediate:
        log_step("Cleaning up")
        cleanup_intermediate_files(video_root, reference_path, test_path)
        log_success("Cleanup complete")

    log_header("Done!")


if __name__ == "__main__":
    main()
