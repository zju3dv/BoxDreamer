from src.reconstruction.base import BaseReconstructor
from src.reconstruction.dust3r import DUSt3RReconstructor
from src.reconstruction.colmap import COLMAPReconstructor
from src.datasets.custom import CustomDataset
from src.models.BoxDreamerModel import BoxDreamer
from torch.utils.data import DataLoader
from .seg import VideoSegmentationApp
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
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
from src.demo.utils import *


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
    # todo:
    # Colmap, ACE0, MOGE(for monocular setting), Detector-free SFM, Spann3r, Fast3R


def run(images, masks, reconstructor: BaseReconstructor):
    reconstructor.set_data(images=images, masks=masks)

    try:
        ret = reconstructor.real_run()
    except Exception as e:
        loguru.logger.warning(f"Failed to reconstruct reference images")
        print(e)
        loguru.logger.error(traceback.format_exc())
        reconstructor.reset_data()
        return None

    reconstructor.reset_data()
    return ret


def preprocess_image(
    video_path, ref_num=5, skip=False, query_video=None, ref_path=None
):
    # Create output directories
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

    if skip:
        return ref_path, test_path

    # Check if reference images are provided externally
    if query_video is not None and ref_path is not None and os.path.exists(ref_path):
        # CASE: Reference images and query video are provided

        # 1. Create a video from reference images
        ref_images = glob.glob(os.path.join(ref_path, "*.png")) + glob.glob(
            os.path.join(ref_path, "*.jpg")
        )
        ref_images.sort()

        if not ref_images:
            raise ValueError(f"No reference images found in {ref_path}")
        else:
            loguru.logger.info(f"Found {len(ref_images)} reference images")
        # Create temporary directory for reference processing
        temp_ref_dir = os.path.join(video_root, "temp_ref")
        os.makedirs(temp_ref_dir, exist_ok=True)

        # Create video from reference images
        ref_video_path = os.path.join(
            temp_ref_dir, "ref_video.mp4"
        )  # todo: fix last view no seg bug
        create_video_from_images(ref_images, ref_video_path)

        # Process reference video for segmentation
        app = VideoSegmentationApp()
        app.load_video(ref_video_path)
        app.set_box_output_dir(os.path.join(temp_ref_dir, "box"))
        app.set_mask_output_dir(os.path.join(temp_ref_dir, "mask"))
        app.annotate_frame_matplotlib()
        app.process_and_save_video()
        del app

        # Process query video
        app = VideoSegmentationApp()
        app.load_video(query_video)
        app.set_box_output_dir(video_root + "/box")
        app.set_mask_output_dir(video_root + "/mask")
        app.annotate_frame_matplotlib()
        app.process_and_save_video()
        del app

        # Extract images from query video
        read_video_to_images(query_video, video_root + "/images")

        # Crop and resize all images
        crop_and_resize_image(video_root + "/images")
        crop_and_resize_image(video_root + "/mask")
        crop_and_resize_image(ref_path)
        crop_and_resize_image(os.path.join(temp_ref_dir, "mask"))

        # Move reference masks and boxes to reference directory
        ref_masks = glob.glob(os.path.join(temp_ref_dir, "mask/*.png"))
        ref_boxes = glob.glob(os.path.join(temp_ref_dir, "box/*.txt"))

        for mask in ref_masks:
            shutil.copy(mask, ref_path)

        for box in ref_boxes:
            shutil.copy(box, ref_path)

        # Move query images to test directory
        query_images = glob.glob(video_root + "/images/*.png")
        for img in query_images:
            shutil.move(img, test_path)

        # Move query masks and boxes to test directory
        query_masks = glob.glob(video_root + "/mask/*.png")
        query_boxes = glob.glob(video_root + "/box/*.txt")

        for mask in query_masks:
            shutil.move(mask, test_path)

        for box in query_boxes:
            shutil.move(box, test_path)

        # Clean up temporary directories
        shutil.rmtree(temp_ref_dir)
        shutil.rmtree(video_root + "/mask", ignore_errors=True)
        shutil.rmtree(video_root + "/images", ignore_errors=True)
        shutil.rmtree(video_root + "/box", ignore_errors=True)

        loguru.logger.info(f"Preprocess with provided reference and query video done")
        return ref_path, test_path

    # Original flow: segment video and split into reference and test
    app = VideoSegmentationApp()
    app.load_video(video_path)

    app.set_box_output_dir(video_root + "/box")
    app.set_mask_output_dir(video_root + "/mask")
    app.annotate_frame_matplotlib()
    app.process_and_save_video()

    del app
    # Then, split the video into images and save them
    read_video_to_images(video_path, video_root + "/images")

    crop_and_resize_image(video_root + "/images")
    crop_and_resize_image(video_root + "/mask")

    # User interactive selection of reference images
    ref_images = interactive_select_reference_images(video_root + "/images", ref_num)

    # Get all available images
    images = glob.glob(video_root + "/images/*.png")
    images.sort()

    # Test images are all images not selected as reference
    test_images = list(set(images) - set(ref_images))

    # Move reference images to reference directory
    for ref_image in ref_images:
        shutil.move(ref_image, ref_path)

    # Move test images to test directory
    for test_image in test_images:
        shutil.move(test_image, test_path)

    # Process box and mask files
    boxes = glob.glob(video_root + "/box/*.txt")
    boxes.sort()
    masks = glob.glob(video_root + "/mask/*.png")
    masks.sort()

    # Get filenames for matching with boxes and masks
    ref_filenames = [
        os.path.basename(img).split(".")[0].split("-")[0] for img in ref_images
    ]

    # Match boxes and masks with reference images based on filename
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

    # Move box and mask files to appropriate directories
    for ref_box in ref_boxes:
        shutil.move(ref_box, ref_path)
    for test_box in test_boxes:
        shutil.move(test_box, test_path)
    for ref_mask in ref_masks:
        shutil.move(ref_mask, ref_path)
    for test_mask in test_masks:
        shutil.move(test_mask, test_path)

    # Clean up temporary directories
    shutil.rmtree(video_root + "/mask")
    shutil.rmtree(video_root + "/images")
    shutil.rmtree(video_root + "/box")

    loguru.logger.info(f"Preprocess video {video_path} done")
    return ref_path, test_path


def interactive_select_reference_images(images_dir, ref_num):
    """Interactive selection of reference images via command line.

    Args:
        images_dir: Directory containing images
        ref_num: Number of reference images needed

    Returns:
        selected_images: List of selected reference image paths
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import random

    # Get all images
    images = glob.glob(images_dir + "/*.png")
    images.sort()

    if len(images) < ref_num:
        loguru.logger.error(
            f"Not enough images available ({len(images)}) for required reference count ({ref_num})"
        )
        return images

    # Keep track of already shown images to avoid repeating
    already_sampled = set()
    selected_images = []

    while len(selected_images) < ref_num:
        remaining_count = ref_num - len(selected_images)
        loguru.logger.info(
            f"Currently selected {len(selected_images)}/{ref_num} reference images. Need {remaining_count} more."
        )

        # Get candidate images that haven't been evaluated yet
        candidate_images = [
            img
            for img in images
            if img not in already_sampled and img not in selected_images
        ]

        if not candidate_images:
            loguru.logger.warning(
                "No more candidate images available. Using current selection."
            )
            break

        # Randomly select a batch of images to evaluate
        batch_size = min(5, len(candidate_images))
        batch = random.sample(candidate_images, batch_size)

        # Display images in grid layout
        cols = min(3, batch_size)
        rows = (batch_size + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle("Candidate Reference Images", fontsize=16)

        # Ensure axes is always a 2D array
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Display images
        for i, img_path in enumerate(batch):
            row, col = i // cols, i % cols
            img = Image.open(img_path)
            axes[row, col].imshow(np.array(img))
            axes[row, col].set_title(f"Image {i+1}\n{os.path.basename(img_path)}")
            axes[row, col].axis("off")

        # Hide unused subplots
        for i in range(batch_size, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show(block=False)

        # Command line interface for selection
        print("\nSelect images to add to reference set:")
        for i, img_path in enumerate(batch):
            print(f"[{i+1}] {os.path.basename(img_path)}")

        print("\nOptions:")
        print(
            "- Enter image numbers separated by space (e.g., '1 3 5') to select images"
        )
        print("- Enter 'a' to select all images")
        print("- Enter 'n' to skip this batch")
        print("- Enter 'q' to quit with current selection")

        choice = input("\nYour choice: ").strip().lower()

        # Add all evaluated images to already_sampled to avoid showing them again
        already_sampled.update(batch)

        # Close figure
        plt.close(fig)

        if choice == "q":
            loguru.logger.info("Selection completed by user")
            break

        if choice == "a":
            selected_images.extend(batch)
            loguru.logger.info(f"Added all {batch_size} images to reference set")

        elif choice == "n":
            loguru.logger.info("Skipped current batch")

        else:
            try:
                # Parse numbers separated by spaces
                selected_indices = [int(x) - 1 for x in choice.split()]

                # Validate indices
                valid_indices = [i for i in selected_indices if 0 <= i < batch_size]

                if valid_indices:
                    selected_batch = [batch[i] for i in valid_indices]
                    selected_images.extend(selected_batch)
                    loguru.logger.info(
                        f"Added {len(selected_batch)} images to reference set"
                    )
                else:
                    loguru.logger.warning("Invalid selection. Please try again.")
            except ValueError:
                loguru.logger.warning("Invalid input. Please try again.")

    # If we still don't have enough images, fill with random sampling
    if len(selected_images) < ref_num:
        remaining_count = ref_num - len(selected_images)
        loguru.logger.warning(
            f"Still need {remaining_count} more images. Randomly sampling from remaining images."
        )

        remaining_images = [img for img in images if img not in selected_images]
        if remaining_images:
            additional = random.sample(
                remaining_images, min(remaining_count, len(remaining_images))
            )
            selected_images.extend(additional)
            loguru.logger.info(
                f"Added {len(additional)} randomly sampled images to complete the set"
            )

    # If we have more images than needed, trim the list
    if len(selected_images) > ref_num:
        selected_images = selected_images[:ref_num]
        loguru.logger.info(f"Trimmed selection to {ref_num} images")

    loguru.logger.info(
        f"Final selection: {len(selected_images)}/{ref_num} reference images"
    )

    # Show final selection
    display_final_selection(selected_images)

    return selected_images


def display_final_selection(selected_images):
    """Display the final selection of reference images.

    Args:
        selected_images: List of selected reference image paths
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # Display images in grid layout
    cols = min(5, len(selected_images))
    rows = (len(selected_images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle("Final Reference Image Selection", fontsize=16)

    # Ensure axes is always a 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Display images
    for i, img_path in enumerate(selected_images):
        row, col = i // cols, i % cols
        img = Image.open(img_path)
        axes[row, col].imshow(np.array(img))
        axes[row, col].set_title(f"Reference {i+1}\n{os.path.basename(img_path)}")
        axes[row, col].axis("off")

    # Hide unused subplots
    for i in range(len(selected_images), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show(block=True)


def to_gpu(data: dict):
    # for all the tensors in the data, move them to cuda
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()
    return data


def warp_model(state_dict):
    # for all keys, replace "BoxDreamer." with ""
    try:
        state_dict = state_dict["state_dict"]
    except KeyError:
        state_dict = state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("BoxDreamer.", "")] = v
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(description="demo")
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
    parser.add_argument(
        "--skip", action="store_true", help="skip segmentation and image preprocessing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="src/demo/checkpoints/boxdreamer.pth",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="./cache/BoxDreamer/output.mp4",
        help="output video path",
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="frames per second of output video"
    )
    args = parser.parse_args()
    # load yaml config as DictConfig
    with open("src/demo/configs/data.yaml") as f:
        data_cfgs = omegaconf.OmegaConf.load(f)

    # load reconstructor config
    with open("src/demo/configs/reconstructor.yaml") as f:
        recon_cfgs = omegaconf.OmegaConf.load(f)

    # load model config
    with open("src/demo/configs/model.yaml") as f:
        model_cfgs = omegaconf.OmegaConf.load(f)

    data_cfgs.Custom.config.root = os.path.dirname(args.video)

    # preprocess video
    reference_path, test_path = preprocess_image(
        args.video, args.ref_num, args.skip, args.query_video, args.ref_path
    )

    if args.query_video is not None and args.ref_path is not None:
        mode = "different-scene"
    else:
        mode = "same-scene"

    ds = CustomDataset(data_cfgs.Custom.config, split="test")

    root = data_cfgs.Custom.config.root

    # from reference path load all the images
    # support extension: .png, .jpg, .jpeg

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

    reconstructor_name = args.reconstructor
    # to lower case
    reconstructor_name = reconstructor_name.lower()
    reconstructor = get_reconstructor(reconstructor_name)(
        recon_cfgs[reconstructor_name + "_cfg"]["method"],
        recon_cfgs[reconstructor_name + "_cfg"]["weight"],
        recon_cfgs[reconstructor_name + "_cfg"]["config"],
    )

    ret = run(ref_imgs, ref_masks, reconstructor)
    if ret is None:
        loguru.logger.error("Failed to reconstruct reference images")
        return
    else:
        pred_poses = ret["poses"]
        if isinstance(pred_poses, torch.Tensor):
            pred_poses = pred_poses.detach().cpu().numpy()
        pred_intrinsics = ret["intrinsics"]
        if isinstance(pred_intrinsics, torch.Tensor):
            pred_intrinsics = pred_intrinsics.detach().cpu().numpy()
        model_path = ret["ply_path"]

        ds.set_intrinsic(pred_intrinsics[0])
        ds.set_model_path(model_path)
        ds.set_ref_root(reference_path)

        # dump reference poses into txt file
        for i in range(len(pred_poses)):
            file_name = ref_imgs[i].split("/")[-1].split(".")[0].split("-")[0]
            with open(os.path.join(reference_path, f"{file_name}-pose.txt"), "w") as f:
                for j in range(4):
                    f.write(" ".join([str(x) for x in pred_poses[i][j]]) + "\n")

        # dump intrinsics into txt file
        for i in range(len(pred_intrinsics)):
            file_name = ref_imgs[i].split("/")[-1].split(".")[0].split("-")[0]
            with open(
                os.path.join(reference_path, f"{file_name}-intrinsics.txt"), "w"
            ) as f:
                for j in range(3):
                    f.write(" ".join([str(x) for x in pred_intrinsics[i][j]]) + "\n")

        ds.set_test_root(test_path)

        if mode == "different-scene":
            # use first query image and run reconstruction to get intrinsics
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
                loguru.logger.error("Failed to reconstruct query image")
                return
            else:
                pred_intrinsics = ret["intrinsics"]
                if isinstance(pred_intrinsics, torch.Tensor):
                    pred_intrinsics = pred_intrinsics.detach().cpu().numpy()
                ds.set_intrinsic(pred_intrinsics[0])

        ds.load_data()

        # init dataloader
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

        # init BoxDreamer model
        model = BoxDreamer(model_cfgs).cuda()
        model.eval()
        # load checkpoint
        model.load_state_dict(warp_model(torch.load(args.ckpt)))

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_video)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List to store all frames
        all_frames = []
        all_frames_large = []

        # Try to fetch all images with progress bar
        try:
            for i, data in enumerate(tqdm(dl, desc="Processing frames")):
                with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    ret = model(to_gpu(data))
                box = ret["regression_boxes"][0][-1].detach().cpu().numpy()
                pose = ret["pred_poses"][0][-1].detach().cpu().numpy()
                intri = ret["original_intrinsics"][0][-1].detach().cpu().numpy()
                original_image = data["original_images"][-1][0]
                bbox_3d = data["bbox_3d_original"][0][-1].detach().cpu().numpy()
                # load original image
                original_image = Image.open(original_image)
                original_image = original_image.convert("RGB")
                # to numpy
                original_image = np.array(original_image)

                proj_bbox = reproj(intri, pose, bbox_3d)

                # unnorm box
                box = ((box + 1) / 2) * 224
                image = data["images"][0][-1].detach().cpu().numpy().transpose(1, 2, 0)
                image = (image * 255).astype(np.uint8)
                image = np.ascontiguousarray(image)

                fig = draw_3d_box(image, box)
                large_fig = draw_3d_box(original_image, proj_bbox)

                # Store the frame
                all_frames.append(fig)
                all_frames_large.append(large_fig)

                # Optional: show the first frame
                if i == 0:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(fig)
                    plt.title("First frame with 3D bounding box")
                    plt.show()
        except Exception as e:
            loguru.logger.error(f"Error processing frame {i}: {e}")

        # Create video from all frames
        if all_frames:
            loguru.logger.info(
                f"Creating video with {len(all_frames)} frames at {args.fps} FPS"
            )

            # Get frame dimensions
            height, width, layers = all_frames[0].shape

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                args.output_video, fourcc, args.fps, (width, height)
            )

            # Add each frame to video
            for frame in tqdm(all_frames, desc="Writing video"):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            # Release video writer
            video.release()

            loguru.logger.success(f"Video saved to {args.output_video}")

        if all_frames_large:
            loguru.logger.info(
                f"Creating video with {len(all_frames_large)} frames at {args.fps} FPS"
            )

            # Get frame dimensions
            height, width, layers = all_frames_large[0].shape

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(
                args.output_video.replace(".mp4", "_large.mp4"),
                fourcc,
                args.fps,
                (width, height),
            )

            # Add each frame to video
            for frame in tqdm(all_frames_large, desc="Writing video"):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            # Release video writer
            video.release()

            loguru.logger.success(
                f"Video saved to {args.output_video.replace('.mp4', '_large.mp4')}"
            )

        else:
            loguru.logger.error("No frames were processed, cannot create video")


if __name__ == "__main__":
    main()
