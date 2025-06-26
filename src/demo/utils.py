import os
import glob
from tqdm import tqdm
from PIL import Image
import cv2
import math
import concurrent.futures
from functools import partial
import loguru


def create_video_from_images(image_paths, output_video_path, fps=1, skip_format=False):
    """Create a video from a list of images.

    Args:
        image_paths: List of paths to images
        output_video_path: Path to save the output video
        fps: Frames per second
    """
    if not image_paths:
        raise ValueError("No images provided")

    img_ext = os.path.splitext(image_paths[0])[1]
    # log img_ext
    loguru.logger.info(f"Image extension: {img_ext}")
    # first, rename all images to 6-digit format xxxxxx-color.png
    if not skip_format:
        for idx, img_path in enumerate(image_paths):
            new_name = f"{idx:06d}-color{img_ext}"
            os.rename(img_path, os.path.join(os.path.dirname(img_path), new_name))

        # reset image paths
        image_paths = glob.glob(
            os.path.join(os.path.dirname(image_paths[0]), "*-color" + img_ext)
        )
        # Sort images by filename
        image_paths.sort()

    # Read the first image to get dimensions
    img = cv2.imread(image_paths[0])
    if img is None:
        raise ValueError(f"Failed to load the first image: {image_paths[0]}")
    h, w, _ = img.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Add each image to video
    failed_images = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            loguru.logger.error(f"Failed to load image: {img_path}")
            continue
        if img.shape[:2] != (h, w):
            loguru.logger.error(
                f"Image {img_path} dimensions {img.shape[:2]} do not match video dimensions ({h}, {w})"
            )
            failed_images.append(img_path)
            continue
        out.write(img)
        loguru.logger.info(f"Frame {idx + 1} written successfully")

    # Release resources
    out.release()
    loguru.logger.info(f"Created video at {output_video_path}")

    # remove failed images
    for img_path in failed_images:
        os.remove(img_path)


def read_video_to_images(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_path, f"{count:06d}-color.png"), frame)
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    return count


def downsample_video(video_path, factor=None, target_fps=None):
    """Downsample the video by reducing its frame rate. The user can specify
    either a downsampling factor or a target frame rate (target_fps). If both
    are provided, target_fps takes precedence.

    Parameters:
    - video_path: str, path to the input video.
    - factor: int, the downsampling factor (e.g., factor=2 means halving the frame rate).
    - target_fps: int or float, the desired frame rate for the output video.

    Returns:
    - written_frames: int, the total number of frames written to the output video.
    """
    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    # Get original video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine the downsampling factor based on target_fps or factor
    if target_fps is not None:
        if target_fps <= 0 or target_fps > original_fps:
            raise ValueError(
                f"Invalid target_fps: {target_fps}. It must be in the range (0, {original_fps}]."
            )
        factor = math.ceil(
            original_fps / target_fps
        )  # Calculate the downsampling factor
        new_fps = original_fps / factor  # Recalculate the actual new FPS
    elif factor is not None:
        if factor <= 0:
            raise ValueError("The downsampling factor must be greater than 0.")
        new_fps = original_fps / factor
        if new_fps <= 0:
            raise ValueError(
                "The downsampling factor is too large, resulting in an invalid frame rate."
            )
    else:
        raise ValueError("Either 'factor' or 'target_fps' must be provided.")

    # Define the codec and create a temporary output video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_output_path = video_path + "_temp.mp4"
    out = cv2.VideoWriter(temp_output_path, fourcc, new_fps, (width, height))

    # Downsample frames
    count = 0
    written_frames = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Write every 'factor'-th frame
            if count % factor == 0:
                out.write(frame)
                written_frames += 1
            count += 1
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Replace the original file with the new downsampled file
    # os.replace(temp_output_path, video_path)

    return temp_output_path


def process_single_image(img_path, size=224):
    """Process a single image: crop and resize."""
    try:
        # Open the image
        img = Image.open(img_path)

        # if image size already matches target size, skip
        if img.size[0] == size and img.size[1] == size:
            return True

        # Convert to RGB if it's not already (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get original width and height
        width, height = img.size

        # Find the center of the image
        center_x = width // 2
        center_y = height // 2

        # Calculate the crop dimensions (the shorter side)
        crop_size = min(width, height)

        # Calculate crop coordinates centered on the image center
        left = center_x - crop_size // 2
        top = center_y - crop_size // 2
        right = left + crop_size
        bottom = top + crop_size

        # Create a new black background image of crop_size (RGB format)
        new_img = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))

        # Calculate paste coordinates
        paste_left = max(0, -left)
        paste_top = max(0, -top)

        # Calculate crop coordinates for original image
        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(width, right)
        crop_bottom = min(height, bottom)

        # Crop the region from original image that falls within its boundaries
        region = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Paste the valid region onto the black background
        new_img.paste(region, (paste_left, paste_top))

        # Resize to target size
        resized_img = new_img.resize((size, size), Image.LANCZOS)

        # Save image (overwrite the original file)
        resized_img.save(img_path)
        return True

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def crop_and_resize_image(root, size=224, max_workers=8):
    """Crop images along the longer side and resize to specified size,
    maintaining the center of the original image.

    Args:
        root: Directory path containing PNG images
        size: Target size for both width and height (default: 224)
        max_workers: Maximum number of worker threads (default: None, which uses CPU count)
    """
    # Get all PNG images
    images = glob.glob(os.path.join(root, "*.png")) + glob.glob(
        os.path.join(root, "*.jpg")
    )
    images.sort()

    print(f"Processing {len(images)} images in {root}...")

    # Create a partial function with the size parameter
    process_func = partial(process_single_image, size=size)

    # Use ThreadPoolExecutor for I/O-bound tasks like image processing
    successful = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get future objects
        future_to_img = {
            executor.submit(process_func, img_path): img_path for img_path in images
        }

        # Use tqdm to show progress as futures complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_img), total=len(images)
        ):
            img_path = future_to_img[future]
            try:
                result = future.result()
                if result:
                    successful += 1
            except Exception as exc:
                print(f"{img_path} generated an exception: {exc}")

    print(f"Completed processing {successful} of {len(images)} images successfully.")
