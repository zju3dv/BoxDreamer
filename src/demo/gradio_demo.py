ckpt_path = None
use_hf = True
downsample = False
tgt_fps = 1
device = "cuda"


def get_gpu_decorator():
    try:
        import spaces

        if device == "cpu":
            raise ImportError("GPU is not available when device is set to 'cpu'.")

        print("spaces.GPU is available.")

        def gpu_decorator(duration):
            return spaces.GPU(duration=duration)

        return gpu_decorator

    except ImportError:
        print("spaces.GPU is not available. Using a no-op decorator.")

        def gpu_decorator(duration):
            def no_op_decorator(func):
                def wrapper(*args, **kwargs):
                    print(f"Running {func.__name__} without GPU. Duration: {duration}")
                    return func(*args, **kwargs)

                return wrapper

            return no_op_decorator

        return gpu_decorator


GPU = get_gpu_decorator()

import os
import glob
import numpy as np
import cv2
import gradio as gr
from PIL import Image
import tempfile
import shutil
from omegaconf import OmegaConf
import traceback
import loguru
from tqdm import tqdm
import random
import time
import argparse


# Import existing modules

# Create temporary directories for processing
temp_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(temp_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(temp_dir, "mask"), exist_ok=True)
os.makedirs(os.path.join(temp_dir, "box"), exist_ok=True)

# Mode 2 temporary directories
mode2_temp_dir = os.path.join(temp_dir, "mode2")
os.makedirs(os.path.join(mode2_temp_dir, "reference"), exist_ok=True)
os.makedirs(os.path.join(mode2_temp_dir, "test"), exist_ok=True)

# Global state to store UI state
init_state = {
    "first_frame": None,
    "points": [],
    "bbox_start": None,
    "bbox": None,
    "draw_state": "start",
    "mode": "point",
    "video_path": None,
    "segmented_video": None,
    "images_extracted": False,
    "ref_path": None,
    "test_path": None,
    "ref_images": [],
    "app": None,
    "all_frames": [],
    "selected_indices": set(),
    # Mode 2 state
    "ref_images_uploaded": [],
    "test_video": None,
    "mode2_ref_frame": None,
    "mode2_test_frame": None,
    "mode2_ref_points": [],
    "mode2_ref_bbox": None,
    "mode2_test_points": [],
    "mode2_test_bbox": None,
    "mode2_ref_app": None,
    "mode2_test_app": None,
}


def load_video(video_path, state):
    """Load video and display the first frame for annotation."""
    import math
    from src.demo.utils import downsample_video

    if not video_path:
        return None, "🚫 Please upload a video file", state

    if not os.path.exists(video_path):
        return None, "🔍 Cannot find video file", state

    try:
        # Reset state
        # log video path
        loguru.logger.info(f"Video path: {video_path}")
        if downsample:
            video_path = downsample_video(video_path, target_fps=tgt_fps)

        state["app"] = None
        state["images_extracted"] = False
        state["segmented_video"] = None
        state["ref_path"] = None
        state["test_path"] = None
        state["ref_images"] = []
        state["all_frames"] = []
        state["selected_indices"] = set()

        @GPU(duration=15)
        def gpu_op():
            # Initialize VideoSegmentationApp
            from src.demo.seg import VideoSegmentationApp

            # log whether GPU is available
            # loguru.logger.info(f"GPU available: {torch.cuda.is_available()}")

            app = VideoSegmentationApp(device=device)
            app.load_video(video_path)

            return app.frame.copy()

        # app = gpu_op()
        frame = gpu_op()

        # Store app and state
        state["app"] = None
        state["video_path"] = video_path
        state["first_frame"] = frame
        state["points"] = []
        state["bbox"] = None
        state["bbox_start"] = None
        state["mode"] = "point"
        state["draw_state"] = "start"

        return (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            "✅ Video loaded successfully! Add foreground points or bounding box.",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error loading video: {str(e)}", state


def toggle_mode(mode, img, state):
    """Toggle between point mode and bounding box mode."""
    state["mode"] = mode

    if state["first_frame"] is None and img is None:
        return None, "🚫 Please load a video first", state
    elif state["first_frame"] is None:
        state["first_frame"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    frame = state["first_frame"].copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Redraw existing annotations
    for x, y in state["points"]:
        cv2.circle(img, (x, y), 8, (0, 255, 0), -1)

    if state["bbox"] is not None:
        x1, y1, x2, y2 = state["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    message = f"🔄 Switched to {mode} mode"
    if mode == "point":
        message += ". 👆 Click to add foreground points."
    else:
        message += ". 📏 Click twice to define a bounding box (first click for top-left, second for bottom-right)."
        state["draw_state"] = "start"

    return img, message, state


def annotate_image(img, evt: gr.SelectData, state):
    """Add annotations (points or bounding box) to the image."""
    if state["first_frame"] is None:
        return img, "🚫 Please load a video first", state

    frame = img.copy()

    if state["mode"] == "point":
        # Add a point at the clicked position
        x, y = evt.index
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        state["points"].append((x, y))

        return frame, f"✅ Added foreground point at ({x}, {y})", state

    elif state["mode"] == "bbox":
        x, y = evt.index

        if state["draw_state"] == "start":
            # First click (top-left corner)
            state["bbox_start"] = (x, y)
            state["draw_state"] = "end"
            # Draw a point at the first corner
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            return (
                frame,
                f"👆 Selected top-left corner ({x}, {y}). Now click for bottom-right corner.",
                state,
            )

        elif state["draw_state"] == "end":
            # Second click (bottom-right corner)
            x1, y1 = state["bbox_start"]
            x2, y2 = x, y

            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            state["bbox"] = [x1, y1, x2, y2]
            state["draw_state"] = "start"  # Reset for next box

            return frame, f"✅ Added bounding box at ({x1}, {y1}, {x2}, {y2})", state

    return frame, "👆 Click to annotate", state


def reset_annotations(state):
    """Reset all annotations."""
    if state["first_frame"] is None:
        return None, "🚫 Please load a video first", state

    frame = state["first_frame"].copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    state["points"] = []
    state["bbox"] = None
    state["bbox_start"] = None
    state["draw_state"] = "start"

    if state["app"]:
        state["app"].points = []
        state["app"].labels = []
        state["app"].bbox = None

    return img, "🔄 Annotations reset", state


def start_segmentation(state):
    """Start segmentation on the video."""
    if state["video_path"] is None:
        return None, "🚫 Please load a video first", state

    if not state["points"] and state["bbox"] is None:
        return (
            None,
            "⚠️ No annotations provided. Please add points or a bounding box.",
            state,
        )

    try:
        # Set app's points and bounding box
        @GPU(duration=25)
        def gpu_op():
            from src.demo.seg import VideoSegmentationApp

            app = VideoSegmentationApp(device=device)
            app.load_video(state["video_path"])

            if state["points"]:
                app.points = state["points"].copy()
                app.labels = [1] * len(state["points"])  # All foreground

            if state["bbox"] is not None:
                app.bbox = state["bbox"].copy()

            # Set output directories
            box_dir = os.path.join(temp_dir, "box")
            mask_dir = os.path.join(temp_dir, "mask")
            os.makedirs(box_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            app.set_box_output_dir(box_dir)
            app.set_mask_output_dir(mask_dir)

            # Perform segmentation
            app.process_and_save_video()

            # Get segmented video path
            segmented_video = os.path.join(
                app.output_dir,
                "segmented_" + os.path.basename(state["video_path"]),
            )

            # # Free SAM model resources
            # app.release_resources()

            return segmented_video

        state["segmented_video"] = gpu_op()

        return (
            state["segmented_video"],
            f"✨ Video segmentation completed! Output saved.",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error during segmentation: {str(e)}", state


def extract_frames(state):
    """Extract frames from the original video."""

    from src.demo.utils import (
        read_video_to_images,
        crop_and_resize_image,
    )

    if not state["segmented_video"]:
        return "🚫 Please segment the video first", state

    try:
        # Clean previous images
        images_dir = os.path.join(temp_dir, "images")
        shutil.rmtree(images_dir, ignore_errors=True)
        os.makedirs(images_dir, exist_ok=True)

        # Extract frames from video
        frame_count = read_video_to_images(state["video_path"], images_dir)

        # Crop and resize images
        crop_and_resize_image(images_dir)

        # Process masks
        mask_dir = os.path.join(temp_dir, "mask")
        crop_and_resize_image(mask_dir)

        state["images_extracted"] = True

        return (
            f"✅ Successfully extracted {frame_count} frames! Please select reference images next.",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        return f"❌ Error extracting frames: {str(e)}", state


def load_all_frames(ref_num, state):
    """Load all frames for reference image selection."""
    if not state["images_extracted"]:
        return [], "🚫 Please extract video frames first", state

    try:
        # Get all images
        images_dir = os.path.join(temp_dir, "images")
        images = glob.glob(os.path.join(images_dir, "*.png"))
        images.sort()

        if len(images) < ref_num:
            return (
                [],
                f"⚠️ Number of available images ({len(images)}) is less than required reference count ({ref_num})",
                state,
            )

        # Preprocess images for display
        preview_images = []
        for img_path in images:
            img = Image.open(img_path)
            preview_images.append(np.array(img))

        # Save all frames for later use
        state["all_frames"] = images

        return (
            preview_images,
            f"🖼️ Total of {len(images)} frames. Click to select {ref_num} reference images. Currently selected: {len(state['selected_indices'])}.",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        return [], f"❌ Error loading frames: {str(e)}", state


def select_reference_image(evt: gr.SelectData, ref_num, state):
    """Click to select reference images."""
    index = evt.index

    if state["all_frames"] and index < len(state["all_frames"]):
        # Toggle selection state
        if index in state["selected_indices"]:
            state["selected_indices"].remove(index)
            action = "🔄 Deselected"
        else:
            state["selected_indices"].add(index)
            action = "✅ Selected"

        # Prepare preview images
        preview_images = []
        for i, img_path in enumerate(state["all_frames"]):
            img = Image.open(img_path)
            img_array = np.array(img)

            # Add green border for selected images
            if i in state["selected_indices"]:
                border_size = 5
                img_array = cv2.copyMakeBorder(
                    img_array,
                    border_size,
                    border_size,
                    border_size,
                    border_size,
                    cv2.BORDER_CONSTANT,
                    value=[0, 255, 0],  # Green border
                )

            preview_images.append(img_array)

        return (
            preview_images,
            f"{action} image {index}. Currently selected: {len(state['selected_indices'])} reference images, need {ref_num}.",
            state,
        )

    return None, "❌ Invalid selection", state


def reset_selection(state):
    """Reset reference image selection."""
    state["selected_indices"] = set()

    # Prepare preview images
    preview_images = []
    for img_path in state["all_frames"]:
        img = Image.open(img_path)
        preview_images.append(np.array(img))

    return preview_images, "🔄 Reference image selection reset.", state


def confirm_selection(ref_num, state):
    """Confirm user-selected reference images."""
    if not state["all_frames"]:
        return [], "🚫 Please load all frames first", state

    try:
        selected_indices = list(state["selected_indices"])

        if len(selected_indices) < ref_num:
            # If not enough images are selected, randomly choose some to supplement
            unselected = [
                i for i in range(len(state["all_frames"])) if i not in selected_indices
            ]
            additional = random.sample(
                unselected, min(ref_num - len(selected_indices), len(unselected))
            )
            selected_indices.extend(additional)
            state["selected_indices"].update(additional)

        # Get selected image paths
        selected_images = [state["all_frames"][i] for i in selected_indices[:ref_num]]
        state["ref_images"] = selected_images

        # Create reference and test directories
        video_dir = os.path.dirname(state["video_path"])
        ref_path = os.path.join(video_dir, "reference")
        test_path = os.path.join(video_dir, "test")
        os.makedirs(ref_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Clean existing content
        for folder in [ref_path, test_path]:
            for file in glob.glob(os.path.join(folder, "*")):
                os.remove(file)

        # Get all images
        all_images = state["all_frames"]

        # Move selected images to reference directory
        for ref_img in selected_images:
            # Get image name (without path and extension)
            img_name = os.path.basename(ref_img).split(".")[0].split("-")[0]

            # Find corresponding mask
            mask_pattern = os.path.join(temp_dir, "mask", f"{img_name}-mask.png")
            mask_files = glob.glob(mask_pattern)

            # Find corresponding bounding box
            box_pattern = os.path.join(temp_dir, "box", f"{img_name}-box.txt")
            box_files = glob.glob(box_pattern)

            # Copy image to reference directory
            shutil.copy(ref_img, os.path.join(ref_path, f"{img_name}-color.png"))

            # If corresponding mask is found, also copy to reference directory
            for mask_file in mask_files:
                shutil.copy(
                    mask_file, os.path.join(ref_path, os.path.basename(mask_file))
                )

            # If corresponding bounding box is found, also copy to reference directory
            for box_file in box_files:
                shutil.copy(
                    box_file, os.path.join(ref_path, os.path.basename(box_file))
                )

        # Move remaining images to test directory
        test_images = [img for img in all_images if img not in selected_images]
        for test_img in test_images:
            # Get image name (without path and extension)
            img_name = os.path.basename(test_img).split(".")[0].split("-")[0]

            # Find corresponding mask
            mask_pattern = os.path.join(temp_dir, "mask", f"{img_name}-mask.png")
            mask_files = glob.glob(mask_pattern)

            # Find corresponding bounding box
            box_pattern = os.path.join(temp_dir, "box", f"{img_name}-box.txt")
            box_files = glob.glob(box_pattern)

            # Copy image to test directory
            shutil.copy(test_img, os.path.join(test_path, f"{img_name}-color.png"))

            # If corresponding mask is found, also copy to test directory
            for mask_file in mask_files:
                shutil.copy(
                    mask_file, os.path.join(test_path, os.path.basename(mask_file))
                )

            # If corresponding bounding box is found, also copy to test directory
            for box_file in box_files:
                shutil.copy(
                    box_file, os.path.join(test_path, os.path.basename(box_file))
                )

        state["ref_path"] = ref_path
        state["test_path"] = test_path

        # Preview selected reference images
        selected_preview = []
        for img_path in selected_images:
            img = Image.open(img_path)
            selected_preview.append(np.array(img))

        return (
            selected_preview,
            f"✅ Confirmed {len(selected_images)} reference images!",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        return [], f"❌ Error confirming reference images: {str(e)}", state


def start_reconstruction(reconstructor_name, state):
    """Start 3D reconstruction."""
    if not state.get("ref_path") or not state.get("test_path"):
        return None, "🚫 Please complete reference image selection first", state

    reconstructor_name = reconstructor_name.lower()

    try:
        # Load configurations
        with open("src/demo/configs/data.yaml") as f:
            data_cfgs = OmegaConf.load(f)
        with open("src/demo/configs/reconstructor.yaml") as f:
            recon_cfgs = OmegaConf.load(f)

        video_dir = os.path.dirname(state["video_path"])
        data_cfgs.Custom.config.root = video_dir

        # Get reference images and masks
        ref_imgs = glob.glob(
            os.path.join(state["ref_path"], "*-color.png")
        ) + glob.glob(os.path.join(state["ref_path"], "*-color.jpg"))
        ref_imgs.sort()

        ref_masks = glob.glob(
            os.path.join(state["ref_path"], "*-mask.png")
        ) + glob.glob(os.path.join(state["ref_path"], "*-mask.jpg"))
        ref_masks.sort()

        @GPU(duration=30)
        def gpu_op():
            import torch
            from src.demo.demo import (
                get_reconstructor,
                run,
            )

            loguru.logger.info(f"Reconstructor: {reconstructor_name}")
            # Initialize reconstructor
            reconstructor = get_reconstructor(reconstructor_name)(
                recon_cfgs[reconstructor_name + "_cfg"]["method"],
                recon_cfgs[reconstructor_name + "_cfg"]["weight"],
                recon_cfgs[reconstructor_name + "_cfg"]["config"],
            )
            loguru.logger.info(f"Reconstructor: {reconstructor} initialized")

            # Run reconstruction
            ret = run(ref_imgs, ref_masks, reconstructor)
            del reconstructor
            if device == "cuda":
                torch.cuda.empty_cache()
            if ret is None:
                return None, "❌ Failed to reconstruct reference images", state

            # Process results
            pred_poses = ret["poses"]
            if isinstance(pred_poses, torch.Tensor):
                pred_poses = pred_poses.detach().cpu().numpy()
            pred_intrinsics = ret["intrinsics"]
            if isinstance(pred_intrinsics, torch.Tensor):
                pred_intrinsics = pred_intrinsics.detach().cpu().numpy()
            model_path = ret["ply_path"]

            return model_path, pred_poses, pred_intrinsics

        model_path, pred_poses, pred_intrinsics = gpu_op()

        # Ensure model path is absolute and exists
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

        # Verify file exists
        if not os.path.exists(model_path):
            return (
                None,
                f"⚠️ Reconstruction successful but model file not found: {model_path}",
                state,
            )

        # Save state
        state["model_path"] = model_path
        state["poses"] = pred_poses
        state["intrinsics"] = pred_intrinsics

        # Dump reference poses and intrinsics to txt files
        for i in range(len(pred_poses)):
            file_name = ref_imgs[i].split("/")[-1].split(".")[0].split("-")[0]
            with open(
                os.path.join(state["ref_path"], f"{file_name}-pose.txt"), "w"
            ) as f:
                for j in range(4):
                    f.write(" ".join([str(x) for x in pred_poses[i][j]]) + "\n")

            with open(
                os.path.join(state["ref_path"], f"{file_name}-intrinsics.txt"), "w"
            ) as f:
                for j in range(3):
                    f.write(" ".join([str(x) for x in pred_intrinsics[i][j]]) + "\n")

        # Wait a bit to ensure files are written
        time.sleep(1)

        # Convert PLY to OBJ (if needed)
        try:
            import trimesh

            obj_path = model_path.replace(".ply", ".obj")
            mesh = trimesh.load(model_path)
            mesh.export(obj_path)
            model_path = obj_path
        except:
            pass

        return (
            model_path,
            f"🎉 Reconstruction successful! 3D model ready to view.",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error during reconstruction: {str(e)}", state


def run_boxdreamer_prediction(state):
    """Run BoxDreamer for bounding box prediction."""

    if not state.get("model_path"):
        return None, "🚫 Please complete 3D reconstruction first", state

    # Avoid NumPy array boolean value evaluation error
    if not isinstance(state.get("poses"), np.ndarray) or not isinstance(
        state.get("intrinsics"), np.ndarray
    ):
        return (
            None,
            "❌ Invalid pose or intrinsic data, please rerun reconstruction",
            state,
        )

    try:
        # Load configurations
        with open("src/demo/configs/data.yaml") as f:
            data_cfgs = OmegaConf.load(f)
        with open("src/demo/configs/model.yaml") as f:
            model_cfgs = OmegaConf.load(f)

        video_dir = os.path.dirname(state["video_path"])
        data_cfgs.Custom.config.root = video_dir

        # get reference view length( get -color.png files from reference directory)
        ref_imgs = glob.glob(
            os.path.join(state["ref_path"], "*-color.png")
        ) + glob.glob(os.path.join(state["ref_path"], "*-color.jpg"))
        ref_imgs.sort()
        ref_length = len(ref_imgs)

        length = ref_length + 1
        if length < 6:
            data_cfgs.Custom.config.base.length = length

        if length > 6:
            data_cfgs.Custom.config.base.length = length
            model_cfgs.modules.dense_cfg.enable = True

        @GPU(duration=20)
        def gpu_op():
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from src.demo.demo import (
                to_gpu,
                warp_model,
            )
            from src.datasets.custom import CustomDataset
            from src.models.BoxDreamerModel import BoxDreamer
            from torch.utils.data import DataLoader
            from src.lightning.utils.vis.vis_utils import draw_3d_box, reproj

            # Initialize dataset
            ds = CustomDataset(data_cfgs.Custom.config, split="test")
            ds.set_intrinsic(state["intrinsics"][0])
            ds.set_model_path(state["model_path"])
            ds.set_ref_root(state["ref_path"])
            ds.set_test_root(state["test_path"])
            ds.load_data()

            # Initialize data loader
            dl = DataLoader(
                ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
            )

            # Initialize BoxDreamer model
            model = BoxDreamer(model_cfgs).to(torch.float16)
            if device == "cuda":
                model = model.to("cuda")

            if ckpt_path is not None:
                # Load checkpoint - use relative path
                model.load_state_dict(
                    warp_model(torch.load(ckpt_path, map_location="cpu"))
                )
            elif use_hf:
                # Load checkpoint from Hugging Face
                model.load_state_dict(
                    warp_model(
                        load_file(
                            hf_hub_download(
                                "yyh929/BoxDreamer", "BoxDreamer-vitb.safetensor"
                            ),
                            device=device,
                        )
                    )
                )
            else:
                raise ValueError(
                    "Please provide a valid checkpoint path or use Hugging Face"
                )

            model.eval()
            # Process frames
            output_video_path = os.path.join(video_dir, "output.mp4")
            all_frames = []

            try:
                for i, data in enumerate(tqdm(dl, desc="Processing frames")):
                    with torch.no_grad(), torch.amp.autocast(
                        device_type=device, dtype=torch.float16
                    ):
                        if device == "cuda":
                            data = to_gpu(data)
                        ret = model(data)

                    # Get results
                    pose = ret["pred_poses"][0][-1].detach().cpu().numpy()
                    intri = ret["original_intrinsics"][0][-1].detach().cpu().numpy()
                    original_image = data["original_images"][-1][0]
                    bbox_3d = data["bbox_3d_original"][0][-1].detach().cpu().numpy()

                    # Load original image
                    original_image = Image.open(original_image)
                    original_image = original_image.convert("RGB")
                    original_image = np.array(original_image)

                    # Calculate reprojection
                    proj_bbox = reproj(intri, pose, bbox_3d)

                    # Draw 3D bounding box
                    fig = draw_3d_box(original_image, proj_bbox)
                    all_frames.append(fig)

                    # Free memory
                    del data
                    if device == "cuda":
                        torch.cuda.empty_cache()
            except Exception as e:
                loguru.logger.error(f"Error processing frames: {str(e)}")
                traceback.print_exc()
                # return None, f"❌ Error processing frames: {str(e)}"

            # Create video
            if all_frames:
                height, width, layers = all_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if downsample:
                    video = cv2.VideoWriter(
                        output_video_path, fourcc, tgt_fps, (width, height)
                    )
                else:
                    video = cv2.VideoWriter(
                        output_video_path, fourcc, 24, (width, height)
                    )

                for frame in all_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video.write(frame_bgr)

                video.release()

                # Free resources
                # del model
                # if device == "cuda":
                #     torch.cuda.empty_cache()

                return output_video_path

            else:
                return None

        output_video_path = gpu_op()
        if output_video_path:
            return output_video_path, "🎉 Bounding box prediction completed!", state
        else:
            return None, "⚠️ No frames were generated", state

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error during bounding box prediction: {str(e)}", state


# Mode 2 functions - Process uploaded reference images
def process_ref_images(ref_images, state):
    """Process uploaded reference images."""
    if not ref_images:
        return None, "🚫 Please upload reference images", state

    try:
        # Clean previous images
        ref_dir = os.path.join(mode2_temp_dir, "reference")
        for file in glob.glob(os.path.join(ref_dir, "*")):
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)

        # Save uploaded images
        os.makedirs(os.path.join(ref_dir, "images"), exist_ok=True)
        for i, img_path in enumerate(ref_images):
            # Copy image to reference directory
            shutil.copy(img_path, os.path.join(ref_dir, "images", f"{i:06d}-color.png"))

        # reset ref_images
        ref_images = glob.glob(os.path.join(ref_dir, "images", "*.png"))
        ref_images.sort()

        # log image length
        loguru.logger.info(f"Reference image length: {len(ref_images)}")
        # log copied target path
        loguru.logger.info(
            f"Reference image copied to: {os.path.join(ref_dir, 'images')}"
        )

        # Create a combined image from all reference images for annotation
        ref_images_data = []
        for img_path in ref_images:
            img = Image.open(img_path)
            img = img.convert("RGB")
            ref_images_data.append(np.array(img))

        # Use the first image for annotation
        if ref_images_data:
            state["mode2_ref_frame"] = ref_images_data[0].copy()
            state["mode2_ref_points"] = []
            state["mode2_ref_bbox"] = None
            state["ref_images_uploaded"] = ref_images

            return (
                state["mode2_ref_frame"],
                "✅ Reference images uploaded! Please annotate using points or bounding box.",
                state,
            )
        else:
            return None, "❌ Failed to process reference images", state

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error processing reference images: {str(e)}", state


@GPU(duration=60)
def process_test_video(test_video, state):
    """Process uploaded test video."""

    from src.demo.utils import downsample_video

    if not test_video or not os.path.exists(test_video):
        return None, "🚫 Please upload a test video", state

    try:
        if downsample:
            test_video = downsample_video(test_video, target_fps=tgt_fps)
        state["test_video"] = test_video

        def gpu_op():
            from src.demo.seg import VideoSegmentationApp

            app = VideoSegmentationApp(device=device)
            app.load_video(test_video)

            return app.frame.copy()

        #

        # Save first frame for annotation
        state["mode2_test_frame"] = gpu_op()
        state["mode2_test_points"] = []
        state["mode2_test_bbox"] = None
        state["mode2_test_app"] = None

        # Return the first frame for annotation
        return (
            cv2.cvtColor(state["mode2_test_frame"], cv2.COLOR_BGR2RGB),
            f"✅ Query video loaded! Please annotate using points or bounding box.",
            state,
        )

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error processing test video: {str(e)}", state


# Mode 2 annotation functions
def mode2_toggle_ref_mode(mode, state):
    """Toggle between point mode and bounding box mode for reference images."""
    state["mode"] = mode

    if state["mode2_ref_frame"] is None:
        return None, "🚫 Please upload reference images first", state

    frame = state["mode2_ref_frame"].copy()

    # Redraw existing annotations
    for x, y in state["mode2_ref_points"]:
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

    if state["mode2_ref_bbox"] is not None:
        x1, y1, x2, y2 = state["mode2_ref_bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    message = f"🔄 Switched to {mode} mode for reference image"
    if mode == "point":
        message += ". 👆 Click to add foreground points."
    else:
        message += ". 📏 Click twice to define a bounding box (first click for top-left, second for bottom-right)."
        state["draw_state"] = "start"

    return frame, message, state


def mode2_annotate_ref_image(img, evt: gr.SelectData, state):
    """Add annotations to reference image."""
    if state["mode2_ref_frame"] is None:
        return img, "🚫 Please upload reference images first", state

    frame = img.copy()

    if state["mode"] == "point":
        # Add a point at the clicked position
        x, y = evt.index
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        state["mode2_ref_points"].append((x, y))

        return frame, f"✅ Added foreground point at ({x}, {y})", state

    elif state["mode"] == "bbox":
        x, y = evt.index

        if state["draw_state"] == "start":
            # First click (top-left corner)
            state["bbox_start"] = (x, y)
            state["draw_state"] = "end"
            # Draw a point at the first corner
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            return (
                frame,
                f"👆 Selected top-left corner ({x}, {y}). Now click for bottom-right corner.",
                state,
            )

        elif state["draw_state"] == "end":
            # Second click (bottom-right corner)
            x1, y1 = state["bbox_start"]
            x2, y2 = x, y

            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            state["mode2_ref_bbox"] = [x1, y1, x2, y2]
            state["draw_state"] = "start"  # Reset for next box

            return frame, f"✅ Added bounding box at ({x1}, {y1}, {x2}, {y2})", state

    return frame, "👆 Click to annotate reference image", state


def mode2_reset_ref_annotations(state):
    """Reset reference image annotations."""
    if state["mode2_ref_frame"] is None:
        return None, "🚫 Please upload reference images first", state

    frame = state["mode2_ref_frame"].copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    state["mode2_ref_points"] = []
    state["mode2_ref_bbox"] = None
    state["bbox_start"] = None
    state["draw_state"] = "start"

    return img, "🔄 Reference image annotations reset", state


def mode2_toggle_test_mode(mode, state):
    """Toggle between point mode and bounding box mode for test video."""
    state["mode"] = mode

    if state["mode2_test_frame"] is None:
        return None, "🚫 Please upload test video first", state

    frame = state["mode2_test_frame"].copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Redraw existing annotations
    for x, y in state["mode2_test_points"]:
        cv2.circle(img, (x, y), 8, (0, 255, 0), -1)

    if state["mode2_test_bbox"] is not None:
        x1, y1, x2, y2 = state["mode2_test_bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    message = f"🔄 Switched to {mode} mode for test video"
    if mode == "point":
        message += ". 👆 Click to add foreground points."
    else:
        message += ". 📏 Click twice to define a bounding box (first click for top-left, second for bottom-right)."
        state["draw_state"] = "start"

    return img, message, state


def mode2_annotate_test_image(img, evt: gr.SelectData, state):
    """Add annotations to test video frame."""
    if state["mode2_test_frame"] is None:
        return img, "🚫 Please upload test video first", state

    frame = img.copy()

    if state["mode"] == "point":
        # Add a point at the clicked position
        x, y = evt.index
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        state["mode2_test_points"].append((x, y))

        return frame, f"✅ Added foreground point at ({x}, {y})", state

    elif state["mode"] == "bbox":
        x, y = evt.index

        if state["draw_state"] == "start":
            # First click (top-left corner)
            state["bbox_start"] = (x, y)
            state["draw_state"] = "end"
            # Draw a point at the first corner
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            return (
                frame,
                f"👆 Selected top-left corner ({x}, {y}). Now click for bottom-right corner.",
                state,
            )

        elif state["draw_state"] == "end":
            # Second click (bottom-right corner)
            x1, y1 = state["bbox_start"]
            x2, y2 = x, y

            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            state["mode2_test_bbox"] = [x1, y1, x2, y2]
            state["draw_state"] = "start"  # Reset for next box

            return frame, f"✅ Added bounding box at ({x1}, {y1}, {x2}, {y2})", state

    return frame, "👆 Click to annotate test video frame", state


def mode2_reset_test_annotations(state):
    """Reset test video annotations."""
    if state["mode2_test_frame"] is None:
        return None, "🚫 Please upload test video first", state

    frame = state["mode2_test_frame"].copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    state["mode2_test_points"] = []
    state["mode2_test_bbox"] = None
    state["bbox_start"] = None
    state["draw_state"] = "start"

    return img, "🔄 Query video annotations reset", state


def mode2_segment_ref_images(state):
    from src.demo.utils import create_video_from_images

    """Segment reference images using annotations."""
    if state["mode2_ref_frame"] is None or not state["ref_images_uploaded"]:
        return None, "🚫 Please upload and annotate reference images first", state

    if not state["mode2_ref_points"] and state["mode2_ref_bbox"] is None:
        return (
            None,
            "⚠️ No annotations provided for reference images. Please add points or a bounding box.",
            state,
        )

    try:
        # Create reference video from images for segmentation
        ref_dir = os.path.join(mode2_temp_dir, "reference")
        images_dir = os.path.join(ref_dir, "images")
        # log images dir
        loguru.logger.info(f"Images dir: {images_dir}")
        # log images length
        ref_imgs = glob.glob(os.path.join(images_dir, "*-color.png"))
        # sort images
        ref_imgs.sort()
        loguru.logger.info(f"Images length: {len(ref_imgs)}")

        if not ref_imgs:
            return None, "⚠️ No reference images found", state

        # Create reference video
        ref_video_path = os.path.join(ref_dir, "ref_video.mp4")
        create_video_from_images(ref_imgs, ref_video_path, fps=1, skip_format=True)

        @GPU(duration=10)
        def gpu_op():
            from src.demo.seg import VideoSegmentationApp

            app = VideoSegmentationApp()
            app.load_video(ref_video_path)
            app.set_box_output_dir(os.path.join(ref_dir, "box"))
            app.set_mask_output_dir(os.path.join(ref_dir, "mask"))

            # Set annotations
            if state["mode2_ref_points"]:
                app.points = state["mode2_ref_points"].copy()
                app.labels = [1] * len(state["mode2_ref_points"])
            if state["mode2_ref_bbox"] is not None:
                app.bbox = state["mode2_ref_bbox"].copy()

            # Process video
            app.process_and_save_video()

            return app.output_dir

        output_dir = gpu_op()

        # Get segmented video path
        segmented_video = os.path.join(
            output_dir, "segmented_" + os.path.basename(ref_video_path)
        )

        return segmented_video, "✨ Reference images segmentation completed!", state

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error during reference image segmentation: {str(e)}", state


def mode2_segment_test_video(state):
    """Segment test video using annotations."""
    if state["mode2_test_frame"] is None or not state["test_video"]:
        return None, "🚫 Please upload and annotate test video first", state

    if not state["mode2_test_points"] and state["mode2_test_bbox"] is None:
        return (
            None,
            "⚠️ No annotations provided for test video. Please add points or a bounding box.",
            state,
        )

    try:

        @GPU(duration=20)
        def gpu_op():
            from src.demo.seg import VideoSegmentationApp

            app = VideoSegmentationApp()
            app.load_video(state["test_video"])
            app.set_box_output_dir(os.path.join(mode2_temp_dir, "test", "box"))
            app.set_mask_output_dir(os.path.join(mode2_temp_dir, "test", "mask"))

            # Set annotations
            if state["mode2_test_points"]:
                app.points = state["mode2_test_points"].copy()
                app.labels = [1] * len(state["mode2_test_points"])  # All foreground

            if state["mode2_test_bbox"] is not None:
                app.bbox = state["mode2_test_bbox"].copy()

            # Process video
            app.process_and_save_video()

            # Get segmented video path
            segmented_video = os.path.join(
                app.output_dir, "segmented_" + os.path.basename(state["test_video"])
            )

            # Free SAM model resources
            # app.release_resources()
            # app = None

            return segmented_video

        segmented_video = gpu_op()

        return segmented_video, "✨ Query video segmentation completed!", state

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error during test video segmentation: {str(e)}", state


def mode2_process_and_predict(reconstructor_name, state):
    """Process segmented data and run prediction."""

    from src.demo.utils import (
        read_video_to_images,
        crop_and_resize_image,
    )

    if not os.path.exists(
        os.path.join(mode2_temp_dir, "reference", "mask")
    ) or not os.path.exists(os.path.join(mode2_temp_dir, "test", "mask")):
        return (
            None,
            None,
            "🚫 Please segment reference images and test video first",
            state,
        )

    try:
        # Extract frames from test video to test directory
        test_dir = os.path.join(mode2_temp_dir, "test")
        read_video_to_images(state["test_video"], os.path.join(test_dir, "images"))

        # Crop and resize images and masks
        crop_and_resize_image(os.path.join(test_dir, "images"))
        crop_and_resize_image(os.path.join(test_dir, "mask"))

        ref_dir = os.path.join(mode2_temp_dir, "reference")
        crop_and_resize_image(os.path.join(ref_dir, "images"))
        crop_and_resize_image(os.path.join(ref_dir, "mask"))

        ref_path = os.path.join(os.path.dirname(state["test_video"]), "reference")
        os.makedirs(ref_path, exist_ok=True)

        # remove existing files
        for file in glob.glob(os.path.join(ref_path, "*")):
            os.remove(file)

        # Move masks to appropriate directories
        # For reference images
        ref_masks = glob.glob(os.path.join(ref_dir, "mask", "*.png"))
        ref_boxes = glob.glob(os.path.join(ref_dir, "box", "*.txt"))
        ref_masks.sort()
        ref_boxes.sort()

        for img in glob.glob(os.path.join(ref_dir, "images", "*.png")):
            img_name = os.path.basename(img).split(".")[0].split("-")[0]
            shutil.copy(img, os.path.join(ref_path, f"{img_name}-color.png"))

        for mask in ref_masks:
            mask_name = os.path.basename(mask)
            # Adjust to match image name format
            img_name = mask_name.split("-")[0]
            shutil.copy(mask, os.path.join(ref_path, f"{img_name}-mask.png"))

        for box in ref_boxes:
            box_name = os.path.basename(box)
            img_name = box_name.split("-")[0]
            shutil.copy(box, os.path.join(ref_path, f"{img_name}-box.txt"))

        # Set up test directory
        test_path = os.path.join(os.path.dirname(state["test_video"]), "test")
        os.makedirs(test_path, exist_ok=True)

        # remove existing files
        for file in glob.glob(os.path.join(test_path, "*")):
            os.remove(file)

        # Move test images to test directory
        test_images = glob.glob(os.path.join(test_dir, "images", "*.png"))
        test_masks = glob.glob(os.path.join(test_dir, "mask", "*.png"))
        test_boxes = glob.glob(os.path.join(test_dir, "box", "*.txt"))
        test_images.sort()
        test_masks.sort()
        test_boxes.sort()

        for img in test_images:
            img_name = os.path.basename(img).split(".")[0].split("-")[0]
            shutil.copy(img, os.path.join(test_path, f"{img_name}-color.png"))

        for mask in test_masks:
            mask_name = os.path.basename(mask)
            img_name = mask_name.split("-")[0]
            shutil.copy(mask, os.path.join(test_path, f"{img_name}-mask.png"))

        for box in test_boxes:
            box_name = os.path.basename(box)
            img_name = box_name.split("-")[0]
            shutil.copy(box, os.path.join(test_path, f"{img_name}-box.txt"))

        # Load configurations
        with open("src/demo/configs/data.yaml") as f:
            data_cfgs = OmegaConf.load(f)
        with open("src/demo/configs/reconstructor.yaml") as f:
            recon_cfgs = OmegaConf.load(f)
        with open("src/demo/configs/model.yaml") as f:
            model_cfgs = OmegaConf.load(f)

        data_cfgs.Custom.config.root = os.path.dirname(state["test_video"])
        # log data_cfgs.Custom.config.root
        loguru.logger.info("dataloader root: " + data_cfgs.Custom.config.root)

        # Get reference images and masks
        ref_imgs = glob.glob(os.path.join(ref_dir, "images", "*-color.png"))
        ref_masks = glob.glob(os.path.join(ref_dir, "mask", "*-mask.png"))
        # sort images
        ref_imgs.sort()
        ref_masks.sort()

        if not ref_imgs or not ref_masks:
            return None, None, "⚠️ Reference images or masks not found", state

        ref_length = len(ref_imgs)
        length = ref_length + 1
        if length < 6:
            data_cfgs.Custom.config.base.length = length

        if length > 6:
            data_cfgs.Custom.config.base.length = length
            model_cfgs.modules.dense_cfg.enable = True

        # Reconstruction
        reconstructor_name = reconstructor_name.lower()

        @GPU(duration=20)
        def gpu_op():
            import torch
            from src.demo.demo import (
                get_reconstructor,
                run,
            )

            # Initialize reconstructor
            reconstructor = get_reconstructor(reconstructor_name)(
                recon_cfgs[reconstructor_name + "_cfg"]["method"],
                recon_cfgs[reconstructor_name + "_cfg"]["weight"],
                recon_cfgs[reconstructor_name + "_cfg"]["config"],
            )

            # Run reconstruction
            ret = run(ref_imgs, ref_masks, reconstructor)
            # del reconstructor
            # if device == "cuda":
            #     torch.cuda.empty_cache()
            if ret is None:
                return None, None, None

            # Process results
            pred_poses = ret["poses"]
            if isinstance(pred_poses, torch.Tensor):
                pred_poses = pred_poses.detach().cpu().numpy()
            pred_intrinsics = ret["intrinsics"]
            if isinstance(pred_intrinsics, torch.Tensor):
                pred_intrinsics = pred_intrinsics.detach().cpu().numpy()
            model_path = ret["ply_path"]

            return model_path, pred_poses, pred_intrinsics

        result = gpu_op()
        if result is None or len(result) != 3 or any(x is None for x in result):
            return None, "❌ Error: gpu_op did not return expected values", state
        model_path, pred_poses, pred_intrinsics = result

        # Ensure model path is absolute and exists
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

        # Verify file exists
        if not os.path.exists(model_path):
            return (
                None,
                f"⚠️ Reconstruction successful but model file not found: {model_path}",
                state,
            )

        # Dump reference poses and intrinsics to txt files
        for i in range(len(pred_poses)):
            file_name = ref_imgs[i].split("/")[-1].split(".")[0].split("-")[0]
            with open(os.path.join(ref_path, f"{file_name}-pose.txt"), "w") as f:
                for j in range(4):
                    f.write(" ".join([str(x) for x in pred_poses[i][j]]) + "\n")

            with open(os.path.join(ref_path, f"{file_name}-intrinsics.txt"), "w") as f:
                for j in range(3):
                    f.write(" ".join([str(x) for x in pred_intrinsics[i][j]]) + "\n")

        # Get test image intrinsics
        test_imgs = glob.glob(os.path.join(test_path, "*-color.png"))
        test_masks = glob.glob(os.path.join(test_path, "*-mask.png"))

        # sort images
        test_imgs.sort()
        test_masks.sort()

        if not test_imgs or not test_masks:
            return model_path, None, "⚠️ Query images or masks not found", state

        @GPU(duration=15)
        def gpu_op_2():
            import torch
            from src.demo.demo import (
                get_reconstructor,
                run,
            )

            # Initialize reconstructor
            reconstructor = get_reconstructor(reconstructor_name)(
                recon_cfgs[reconstructor_name + "_cfg"]["method"],
                recon_cfgs[reconstructor_name + "_cfg"]["weight"],
                recon_cfgs[reconstructor_name + "_cfg"]["config"],
            )

            # Run reconstruction

            test_ret = run(test_imgs[:1], test_masks[:1], reconstructor)
            del reconstructor
            if test_ret is None:
                return None

            test_intrinsics = test_ret["intrinsics"]
            if isinstance(test_intrinsics, torch.Tensor):
                test_intrinsics = test_intrinsics.detach().cpu().numpy()

            return test_intrinsics

        test_intrinsics = gpu_op_2()

        if test_intrinsics is None:
            return model_path, None, "❌ Failed to reconstruct query images", state

        @GPU(duration=30)
        def gpu_op_3():
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from src.demo.demo import (
                to_gpu,
                warp_model,
            )
            from src.datasets.custom import CustomDataset
            from src.models.BoxDreamerModel import BoxDreamer
            from torch.utils.data import DataLoader
            from src.lightning.utils.vis.vis_utils import draw_3d_box, reproj

            # Set up dataset and run BoxDreamer
            ds = CustomDataset(data_cfgs.Custom.config, split="test")
            ds.set_intrinsic(test_intrinsics[0])
            ds.set_model_path(model_path)
            ds.set_ref_root(ref_path)
            ds.set_test_root(test_path)
            ds.load_data()

            # Initialize data loader
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

            # Initialize BoxDreamer model
            model = BoxDreamer(model_cfgs).to(torch.float16)
            if device == "cuda":
                model = model.to("cuda")

            if ckpt_path is not None:
                # Load checkpoint
                model.load_state_dict(
                    warp_model(torch.load(ckpt_path, map_location="cpu"))
                )
            elif use_hf:
                # Load checkpoint from Hugging Face
                model.load_state_dict(
                    warp_model(
                        load_file(
                            hf_hub_download(
                                "yyh929/BoxDreamer", "BoxDreamer-vitb.safetensor"
                            ),
                            device=device,
                        )
                    )
                )
            else:
                raise ValueError(
                    "Please provide a valid checkpoint path or use Hugging Face"
                )

            model.eval()

            # Process frames
            output_video_path = os.path.join(
                os.path.dirname(state["test_video"]), "output.mp4"
            )
            all_frames = []

            try:
                for i, data in enumerate(tqdm(dl, desc="Processing frames")):
                    with torch.no_grad(), torch.amp.autocast(
                        device_type=device, dtype=torch.float16
                    ):
                        if device == "cuda":
                            data = to_gpu(data)
                        ret = model(data)

                    pose = ret["pred_poses"][0][-1].detach().cpu().numpy()
                    intri = ret["original_intrinsics"][0][-1].detach().cpu().numpy()
                    original_image = data["original_images"][-1][0]
                    bbox_3d = data["bbox_3d_original"][0][-1].detach().cpu().numpy()

                    original_image = Image.open(original_image)
                    original_image = original_image.convert("RGB")
                    original_image = np.array(original_image)

                    proj_bbox = reproj(intri, pose, bbox_3d)

                    fig = draw_3d_box(original_image, proj_bbox)
                    all_frames.append(fig)

                    # # Free memory
                    # del data
                    # if device == "cuda":
                    #     torch.cuda.empty_cache()
            except Exception as e:
                loguru.logger.error(f"Error during processing: {str(e)}")

            # Create video
            if all_frames:
                height, width, layers = all_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if downsample:
                    video = cv2.VideoWriter(
                        output_video_path, fourcc, tgt_fps, (width, height)
                    )
                else:
                    video = cv2.VideoWriter(
                        output_video_path, fourcc, 24, (width, height)
                    )

                for frame in all_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video.write(frame_bgr)

                video.release()

                # Free resources
                # del model
                # if device == "cuda":
                #     torch.cuda.empty_cache()

                # Convert PLY to OBJ (if needed)
                try:
                    import trimesh

                    obj_path = model_path.replace(".ply", ".obj")
                    mesh = trimesh.load(model_path)
                    mesh.export(obj_path)
                    new_model_path = obj_path
                except:
                    pass

                return output_video_path, new_model_path
            else:
                return None, new_model_path

        output_video_path, model_path = gpu_op_3()

        if output_video_path:
            return (
                model_path,
                output_video_path,
                "🎉 Bounding box prediction completed!",
                state,
            )

        else:
            return (
                model_path,
                None,
                "⚠️ No frames were generated, but reference model generated successfully",
                state,
            )

    except Exception as e:
        traceback.print_exc()
        return None, None, f"❌ Error during processing: {str(e)}", state


def main():
    # Define custom CSS for improved appearance
    custom_css = """
    /* Overall container font and background */
    .gradio-container {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background: #f9fafc;
        padding: 20px;
    }

    /* Headers with gradient effect for main title */
    h1 {
        color: #2E4053;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(90deg, #42a5f5, #66bb6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h2 {
        color: #2E4053;
        font-size: 1.8rem;
        border-bottom: 2px solid #42a5f5;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        text-align: center;
    }

    h3 {
        color: #34495e;
        font-size: 1.5rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    h4 {
        color: #42a5f5;
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Panel cards for sections with soft shadows and rounded corners */
    .panel {
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        padding: 15px;
        background-color: #ffffff;
        border: 1px solid #e3e6ea;
    }

    /* Footer design */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e3e6ea;
        color: #95a5a6;
        font-size: 0.9rem;
    }

    /* Tabs container */
    .tabs {
        margin-top: 1rem;
    }

    /* Scrollable gallery container enhancement */
    #scrollable_gallery_container {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e3e6ea;
        border-radius: 8px;
        padding: 10px;
        background-color: #fafafa;
    }

    button.primary {
        background-color: #42a5f5 !important;
        border-color: #1e88e5 !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .small-button {
        min-width: 100px !important;
        font-size: 0.85rem !important;
        padding: 6px 12px !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    button.primary:hover {
        background-color: #1e88e5 !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }

    button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    }

    .gradio-row {
        display: flex;
        align-items: stretch;
    }

    .image-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }

    .status-box {
        border-left: 5px solid #ffa726;
        padding: 12px 20px;
        border-radius: 8px;
        font-size: 0.95rem;
        margin-bottom: 15px;
    }

    /* Animated pulse effect for 3D model display */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .animated-pulse {
        animation: pulse 2s infinite ease-in-out;
    }

    /* Enhanced layout spacing for rows */
    .gradio-row {
        margin-bottom: 20px;
    }

    .author-list {
        font-family: Arial, sans-serif;
        font-size: 14px;
        color: #333;
        text-align: center;
        line-height: 1.6;
        margin-bottom: 10px;
    }

    .author-list a {
        color: #0073e6;
        text-decoration: none;
        font-weight: bold;
        transition: color 0.2s ease-in-out;
    }

    .author-list a:hover {
        color: #005bb5;
        text-decoration: underline;
    }

    .author-list sup {
        font-size: 12px;
        color: #666;
    }

    p {
        font-family: Arial, sans-serif;
        font-size: 13px;
        color: #555;
        text-align: center;
        line-height: 1.6;
        margin: 0;
    }

    /* --- NEW: Custom styles for gr.Examples --- */
    .custom-examples {
        background-color: #e8f0fe; /* Lighter blue background */
        border: 1px dashed #42a5f5; /* Dashed blue border */
        border-radius: 8px;        /* Consistent rounded corners */
        padding: 12px;             /* Internal spacing */
        margin-top: 15px;          /* Space above */
        margin-bottom: 20px;       /* Space below */
    }

    /* Style the label within the custom examples box */
    .custom-examples .label {
        font-weight: bold;
        color: #1a237e; /* Darker blue label color */
        margin-bottom: 10px;
        font-size: 1rem; /* Adjust size if needed */
        display: block; /* Ensure it takes full width */
    }

    /* Style the individual example items (buttons/thumbnails) */
    .custom-examples .examples-items {
        display: flex; /* Or grid */
        flex-wrap: wrap;
        gap: 8px; /* Space between example items */
    }

    /* Style each example item */
    .custom-examples .examples-items > * {
        border: 1px solid #90caf9; /* Light blue border for each item */
        border-radius: 5px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        background-color: #ffffff; /* White background for items */
        padding: 5px; /* Small padding within item if needed */
    }

    /* Hover effect for example items */
    .custom-examples .examples-items > *:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(66, 165, 245, 0.3); /* Blueish shadow */
        border-color: #1e88e5; /* Darker blue border on hover */
        cursor: pointer;
    }

    /* --- NEW: Styles for Accordion workflow --- */
    .workflow-step {
        margin-bottom: 15px;
        border-left: 4px solid #42a5f5;
    }

    /* Make accordion headers more prominent */
    .workflow-step > .gr-accordion-header {
        font-weight: bold;
        font-size: 1.1rem;
        color: #1976d2;
        padding: 12px 15px;
        background-color: #e3f2fd;
        border-radius: 8px 8px 0 0;
        transition: background-color 0.2s ease;
    }

    .workflow-step > .gr-accordion-header:hover {
        background-color: #bbdefb;
    }

    /* Style for help panel */
    .help-panel {
        border-left: 4px solid #ffa726;
        background-color: #fff8e1;
        margin-top: 20px;
    }

    .help-panel > .gr-accordion-header {
        color: #e65100;
        background-color: #fff8e1;
    }

    /* Results tabs styling */
    .results-tabs > .gr-tabs-header {
        background-color: #f1f8e9;
        border-bottom: 2px solid #7cb342;
        padding: 10px 0;
    }

    .results-tabs > .gr-tabs-header > .selected {
        font-weight: bold;
        color: #33691e;
        border-bottom: 3px solid #33691e;
    }
    /* Style for top examples section */
    .top-examples-section {
        margin-bottom: 20px;
        padding: 15px;
        background-color: #e8f5e9;
        border-radius: 12px;
        border: 1px dashed #66bb6a;
    }

    .top-examples-section .custom-examples {
        background-color: transparent;
        border: none;
        padding: 0;
        margin: 0;
        width: 100%;
    }

    .top-examples-section .gr-form {
        border: none;
        background: transparent;
    }

    .top-examples-section .gr-examples-header {
        font-weight: bold;
        color: #2e7d32;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 12px;
    }
    """

    with gr.Blocks(
        title="BoxDreamer Demo",
        theme=gr.themes.Soft(),
        css=custom_css,
        delete_cache=(86400, 86400),
    ) as demo:
        state = gr.State(init_state)

        gr.Markdown(
            "# BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation"
        )

        gr.Markdown(
            """
                        <p>
                            <a href="https://yuanhongyu.xyz/">Yuanhong Yu</a><sup>1,3</sup> &bull;
                            <a href="https://github.com/hxy-123">Xingyi He</a><sup>1,3</sup> &bull;
                            <a href="https://sailor-z.github.io/">Chen Zhao</a><sup>4</sup> &bull;
                            <a href="https://openreview.net/profile?id=~Junhao_Yu3">Junhao Yu</a><sup>5</sup> &bull;
                            <a href="https://yangjiaqihomepage.github.io/">Jiaqi Yang</a><sup>6</sup> &bull;
                            <a href="https://csse.szu.edu.cn/staff/ruizhenhu/">Ruizhen Hu</a><sup>7</sup> &bull;
                            <a href="https://shenyujun.github.io/">Yujun Shen</a><sup>3</sup> &bull;
                            <a href="https://openreview.net/profile?id=~Xing_Zhu2">Xing Zhu</a><sup>3</sup> &bull;
                            <a href="https://www.xzhou.me/">Xiaowei Zhou</a><sup>1</sup> &bull;
                            <a href="https://pengsida.net/">Sida Peng</a><sup>1,2</sup>
                        </p>
                        <p>
                            <sup>1</sup>State Key Lab of CAD & CG, Zhejiang University &nbsp;&nbsp;
                            <sup>2</sup>Xiangjiang Laboratory &nbsp;&nbsp;
                            <sup>3</sup>Ant Group &nbsp;&nbsp;
                            <sup>4</sup>EPFL &nbsp;&nbsp;
                            <sup>5</sup>Chongqing University &nbsp;&nbsp;
                            <sup>6</sup>Northwestern Polytechnical University &nbsp;&nbsp;
                            <sup>7</sup>Shenzhen University &nbsp;&nbsp;
                        </p>

                        <p>
                            <a class="btn btn-light" href="https://arxiv.org/abs/2504.07955" role="button" target="_blank">
                                Paper
                            </a>
                            &nbsp;&nbsp;
                            <a class="btn btn-light" href="https://zju3dv.github.io/boxdreamer/" role="button" target="_blank">
                                Project Page
                            </a>
                            &nbsp;&nbsp;
                            <a class="btn btn-light" id="code" href="https://github.com/zju3dv/BoxDreamer" role="button" target="_blank">
                                Code
                            </a>
                        </p>
            """,
        )

        gr.Markdown(
            """
                This interactive demo allows you to estimate 6DoF object poses with in-the-wild images.
            """
        )

        with gr.Tab("🎬 Mode 1: Single Video Processing"):
            with gr.Row():
                # Left panel - Main workflow
                with gr.Column(scale=1, min_width=300):
                    # Status information at the top
                    status = gr.Textbox(
                        label="Status Information",
                        interactive=False,
                        elem_classes=["status-box"],
                    )

                    # Step-by-step accordion interface
                    with gr.Accordion(
                        "📤 STEP 1: Upload & Annotate Video",
                        open=True,
                        elem_classes=["workflow-step", "panel"],
                    ):
                        video_input = gr.Video(
                            label="Upload Video", elem_classes=["image-container"]
                        )

                        load_btn = gr.Button(
                            "🎬 Load Video", variant="primary", elem_id="load-btn"
                        )

                        base_dir = os.path.abspath("src/demo/examples")
                        examples = [
                            [os.path.join(base_dir, "mode1", f)]
                            for f in os.listdir(os.path.join(base_dir, "mode1"))
                            if f.endswith(".mp4")
                        ]

                        with gr.Column(elem_classes=["custom-examples"]):
                            gr.Examples(
                                examples=examples,
                                inputs=[video_input],
                                label="Try an Example Video",
                            )

                        gr.Markdown("##### 🎨 Annotation Tools")
                        with gr.Row(equal_height=True):
                            point_btn = gr.Button(
                                "👆 Point Mode", elem_classes=["small-button"]
                            )
                            bbox_btn = gr.Button(
                                "📏 Bounding Box Mode", elem_classes=["small-button"]
                            )
                            reset_btn = gr.Button(
                                "🔄 Reset", elem_classes=["small-button"]
                            )
                        segment_btn = gr.Button(
                            "✨ Start Segmentation (SAM2 Tiny)", variant="primary"
                        )

                    with gr.Accordion(
                        "🖼️ STEP 2: Extract & Select Frames",
                        open=False,
                        elem_classes=["workflow-step", "panel"],
                    ):
                        extract_btn = gr.Button("📸 Extract Frames", variant="primary")
                        ref_num = gr.Slider(
                            minimum=1,
                            maximum=15,
                            value=5,
                            step=1,
                            label="Number of Reference Images",
                        )
                        load_frames_btn = gr.Button(
                            "📂 Load All Frames", variant="primary"
                        )
                        with gr.Row():
                            reset_select_btn = gr.Button(
                                "🔄 Reset Selection", elem_classes=["small-button"]
                            )
                            confirm_select_btn = gr.Button(
                                "✅ Confirm Selection", variant="primary"
                            )

                    with gr.Accordion(
                        "🔮 STEP 3: 3D Reconstruction & Prediction",
                        open=False,
                        elem_classes=["workflow-step", "panel"],
                    ):
                        reconstructor_choice = gr.Dropdown(
                            choices=["DUSt3R"], value="DUSt3R", label="Reconstructor"
                        )
                        recon_btn = gr.Button(
                            "🏗️ Start Reconstruction", variant="primary"
                        )
                        predict_btn = gr.Button(
                            "🎯 Predict Bounding Boxes", variant="primary"
                        )

                    # Quick help box
                    with gr.Accordion(
                        "⚠️ Help & Tips",
                        open=False,
                        elem_classes=["help-panel", "panel"],
                    ):
                        gr.Markdown(
                            """
                            - **Point Mode**: Click to add foreground points
                            - **Bounding Box Mode**: First click sets top-left, second click sets bottom-right
                            - Only one object is supported for annotation and prediction
                            - Camera intrinsics are estimated from reference images using DUSt3R
                        """
                        )

                # Right panel - Visual workspace
                with gr.Column(scale=2):
                    # Interactive workspace with tabs
                    with gr.Tabs() as interactive_tabs:
                        with gr.TabItem("Annotation Canvas", id="annotation_tab"):
                            image_canvas = gr.Image(
                                label="👆 Click on image to annotate",
                                interactive=True,
                                elem_classes=["image-container"],
                                height="300px",
                            )
                        with gr.TabItem(
                            "Reference Image Selection", id="reference_tab"
                        ):
                            with gr.Column(elem_id="scrollable_gallery_container"):
                                reference_gallery = gr.Gallery(
                                    label="👆 Click images to select/deselect (green border = selected)",
                                    columns=4,
                                    rows=None,
                                    height="300px",
                                    object_fit="contain",
                                    show_label=True,
                                    elem_id="ref_gallery",
                                )
                            ref_selection_status = gr.Textbox(
                                label="Selection Status",
                                interactive=False,
                                elem_classes=["status-box"],
                            )

                    # Selected references display
                    with gr.Row():
                        selected_gallery = gr.Gallery(
                            label="✅ Selected Reference Images",
                            columns=5,
                            rows=None,
                            object_fit="contain",
                            show_label=True,
                            height="200px",
                        )

                    # Results display area
                    with gr.Tabs(elem_classes=["results-tabs"]):
                        with gr.TabItem("Segmentation"):
                            segmented_video_output = gr.Video(
                                label="🎬 Segmented Video",
                                elem_classes=["image-container"],
                                height="250px",
                            )
                        with gr.TabItem("3D Model"):
                            model_3d_view = gr.Model3D(
                                label="🏗️ Reconstructed 3D Model",
                                clear_color=[0.0, 0.0, 0.0, 0.0],
                                elem_classes=["image-container"],
                                height="250px",
                            )
                        with gr.TabItem("Object Pose"):
                            video_result = gr.Video(
                                label="🎯 Pose Prediction",
                                elem_classes=["animated-pulse"],
                                height="250px",
                                interactive=False,
                            )

            load_btn.click(
                load_video,
                inputs=[video_input, state],
                outputs=[image_canvas, status, state],
            )
            point_btn.click(
                lambda img, state: toggle_mode("point", img, state),
                inputs=[image_canvas, state],
                outputs=[image_canvas, status, state],
            )
            bbox_btn.click(
                lambda img, state: toggle_mode("bbox", img, state),
                inputs=[image_canvas, state],
                outputs=[image_canvas, status, state],
            )
            reset_btn.click(
                reset_annotations, inputs=[state], outputs=[image_canvas, status, state]
            )
            image_canvas.select(
                annotate_image,
                inputs=[image_canvas, state],
                outputs=[image_canvas, status, state],
            )
            segment_btn.click(
                start_segmentation,
                inputs=[state],
                outputs=[segmented_video_output, status, state],
            )
            extract_btn.click(extract_frames, inputs=[state], outputs=[status, state])
            load_frames_btn.click(
                load_all_frames,
                inputs=[ref_num, state],
                outputs=[reference_gallery, ref_selection_status, state],
            )
            reset_select_btn.click(
                reset_selection,
                inputs=[state],
                outputs=[reference_gallery, ref_selection_status, state],
            )
            reference_gallery.select(
                select_reference_image,
                inputs=[ref_num, state],
                outputs=[reference_gallery, ref_selection_status, state],
            )
            confirm_select_btn.click(
                confirm_selection,
                inputs=[ref_num, state],
                outputs=[selected_gallery, status, state],
            )
            recon_btn.click(
                start_reconstruction,
                inputs=[reconstructor_choice, state],
                outputs=[model_3d_view, status, state],
            )
            predict_btn.click(
                run_boxdreamer_prediction,
                inputs=[state],
                outputs=[video_result, status, state],
            )

        with gr.Tab("🔄 Mode 2: Reference Images + Query Video"):
            # Status information below examples
            status2 = gr.Textbox(
                label="Status Information",
                interactive=False,
                elem_classes=["status-box"],
            )

            with gr.Row():
                # Left panel - Main workflow
                with gr.Column(scale=1, min_width=300):
                    # Step-by-step accordion interface
                    with gr.Accordion(
                        "📤 STEP 1: Reference Images",
                        open=True,
                        elem_classes=["workflow-step", "panel"],
                    ):
                        ref_images = gr.File(
                            label="Upload Reference Images",
                            file_types=["image"],
                            file_count="multiple",
                        )
                        upload_ref_btn = gr.Button(
                            "📸 Process Reference Images", variant="primary"
                        )

                        gr.Markdown("##### 🎨 Reference Annotation Tools")
                        with gr.Row(equal_height=True):
                            ref_point_btn = gr.Button(
                                "👆 Point Mode", elem_classes=["small-button"]
                            )
                            ref_bbox_btn = gr.Button(
                                "📏 Bounding Box Mode", elem_classes=["small-button"]
                            )
                            ref_reset_btn = gr.Button(
                                "🔄 Reset", elem_classes=["small-button"]
                            )
                        ref_segment_btn = gr.Button(
                            "✨ Segment Reference Images", variant="primary"
                        )

                    with gr.Accordion(
                        "📹 STEP 2: Query Video",
                        open=False,
                        elem_classes=["workflow-step", "panel"],
                    ):
                        test_video = gr.Video(
                            label="Upload Query Video", elem_classes=["image-container"]
                        )
                        upload_test_btn = gr.Button(
                            "🎬 Process Query Video", variant="primary"
                        )

                        gr.Markdown("##### 🎨 Query Annotation Tools")
                        with gr.Row(equal_height=True):
                            test_point_btn = gr.Button(
                                "👆 Point Mode", elem_classes=["small-button"]
                            )
                            test_bbox_btn = gr.Button(
                                "📏 Bounding Box Mode", elem_classes=["small-button"]
                            )
                            test_reset_btn = gr.Button(
                                "🔄 Reset", elem_classes=["small-button"]
                            )
                        test_segment_btn = gr.Button(
                            "✨ Segment Query Video (SAM2 Tiny)", variant="primary"
                        )

                    base_dir = os.path.abspath("src/demo/examples")
                    examples = [
                        [
                            [
                                os.path.join(base_dir, "mode2/set1/ref/1.png"),
                                os.path.join(base_dir, "mode2/set1/ref/2.png"),
                                os.path.join(base_dir, "mode2/set1/ref/3.png"),
                                os.path.join(base_dir, "mode2/set1/ref/4.png"),
                                os.path.join(base_dir, "mode2/set1/ref/5.png"),
                            ],
                            os.path.join(base_dir, "mode1/mode1-1.mp4"),
                        ],
                    ]
                    with gr.Column(elem_classes=["custom-examples"]):
                        gr.Examples(
                            examples=examples,
                            inputs=[ref_images, test_video],
                            label="Try an Example (Reference Images + Query Video)",
                        )

                    with gr.Accordion(
                        "🔮 STEP 3: Process and Predict",
                        open=False,
                        elem_classes=["workflow-step", "panel"],
                    ):
                        reconstructor_choice2 = gr.Dropdown(
                            choices=["DUSt3R"], value="DUSt3R", label="Reconstructor"
                        )
                        process_btn2 = gr.Button(
                            "🚀 Process and Predict", variant="primary"
                        )

                    # Quick help box
                    with gr.Accordion(
                        "⚠️ Help & Tips",
                        open=False,
                        elem_classes=["help-panel", "panel"],
                    ):
                        gr.Markdown(
                            """
                            - **Only one object is supported** for annotation and object pose prediction
                            - Camera intrinsics are assumed different between reference views and query video
                            - Intrinsics for query video are estimated from its first frame using DUSt3R
                            - Use the same annotation approach (points or bbox) for both reference and query
                        """
                        )

                # Right panel - Visual workspace
                with gr.Column(scale=2):
                    # Interactive workspace with tabs
                    with gr.Tabs() as interactive_tabs2:
                        with gr.TabItem(
                            "Reference Image Annotation", id="ref_annotation_tab"
                        ):
                            ref_canvas = gr.Image(
                                label="👆 Annotate Reference Image",
                                interactive=True,
                                elem_classes=["image-container"],
                                height="300px",
                            )
                        with gr.TabItem(
                            "Query Video Annotation", id="test_annotation_tab"
                        ):
                            test_canvas = gr.Image(
                                label="👆 Annotate Query Video",
                                interactive=True,
                                elem_classes=["image-container"],
                                height="300px",
                            )

                    # Results display area with tabs
                    with gr.Tabs(elem_classes=["results-tabs"]):
                        with gr.TabItem("Reference Segmentation"):
                            ref_segmented_video = gr.Video(
                                label="Segmented Reference Images",
                                elem_classes=["image-container"],
                                height="250px",
                            )
                        with gr.TabItem("Query Segmentation"):
                            test_segmented_video = gr.Video(
                                label="Segmented Query Video",
                                elem_classes=["image-container"],
                                height="250px",
                            )
                        with gr.TabItem("3D Model"):
                            model_3d_view2 = gr.Model3D(
                                label="🏗️ Reconstructed 3D Model",
                                clear_color=[0.0, 0.0, 0.0, 0.0],
                                elem_classes=["image-container"],
                                height="250px",
                            )
                        with gr.TabItem("Object Pose"):
                            video_result2 = gr.Video(
                                label="🎯 Pose Prediction",
                                elem_classes=["animated-pulse"],
                                height="250px",
                                interactive=False,
                            )

            upload_ref_btn.click(
                process_ref_images,
                inputs=[ref_images, state],
                outputs=[ref_canvas, status2, state],
            )
            ref_point_btn.click(
                lambda state: mode2_toggle_ref_mode("point", state),
                inputs=[state],
                outputs=[ref_canvas, status2, state],
            )
            ref_bbox_btn.click(
                lambda state: mode2_toggle_ref_mode("bbox", state),
                inputs=[state],
                outputs=[ref_canvas, status2, state],
            )
            ref_reset_btn.click(
                mode2_reset_ref_annotations,
                inputs=[state],
                outputs=[ref_canvas, status2, state],
            )
            ref_canvas.select(
                mode2_annotate_ref_image,
                inputs=[ref_canvas, state],
                outputs=[ref_canvas, status2, state],
            )
            ref_segment_btn.click(
                mode2_segment_ref_images,
                inputs=[state],
                outputs=[ref_segmented_video, status2, state],
            )
            upload_test_btn.click(
                process_test_video,
                inputs=[test_video, state],
                outputs=[test_canvas, status2, state],
            )
            test_point_btn.click(
                lambda state: mode2_toggle_test_mode("point", state),
                inputs=[state],
                outputs=[test_canvas, status2, state],
            )
            test_bbox_btn.click(
                lambda state: mode2_toggle_test_mode("bbox", state),
                inputs=[state],
                outputs=[test_canvas, status2, state],
            )
            test_reset_btn.click(
                mode2_reset_test_annotations,
                inputs=[state],
                outputs=[test_canvas, status2, state],
            )
            test_canvas.select(
                mode2_annotate_test_image,
                inputs=[test_canvas, state],
                outputs=[test_canvas, status2, state],
            )
            test_segment_btn.click(
                mode2_segment_test_video,
                inputs=[state],
                outputs=[test_segmented_video, status2, state],
            )
            process_btn2.click(
                mode2_process_and_predict,
                inputs=[reconstructor_choice2, state],
                outputs=[model_3d_view2, video_result2, status2, state],
            )

        gr.Markdown(
            """
                <div class="footer" align="center">
                    <p>📦✨ BoxDreamer Demo - Estimating 6DoF object pose in the wild!</p>
                    <p>© 2025 - Built with Gradio | Powered by <a href="https://github.com/zju3dv" target="_blank">ZJU3DV</a></p>
                </div>
            """,
            elem_classes=["footer"],
        )

    try:
        demo.launch(share=True)
    finally:
        # Clean up temporary directory on program exit
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary directory cleaned up")


if __name__ == "__main__":
    # set ckpt path
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--hf", action="store_true", default=False)
    parser.add_argument("--downsample", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    args = parser.parse_args()

    if args.ckpt is not None:
        if os.path.exists(args.ckpt):
            ckpt_path = args.ckpt
        else:
            raise FileNotFoundError(f"Checkpoint path {args.ckpt} not found")
    elif args.hf is not None:
        use_hf = True
    else:
        raise ValueError("Please provide a checkpoint path")

    if args.downsample:
        downsample = True

    if args.cpu:
        device = "cpu"

    main()
