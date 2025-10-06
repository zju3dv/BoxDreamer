# use grounding-dino to detect objects

import os
import torch
import cv2
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
import tempfile
from tqdm import tqdm
import shutil
from PIL import Image
import supervision as sv
from typing import List, Optional, Union
import sys
from loguru import logger
from pathlib import Path


# ============================================================================
# Logging Configuration
# ============================================================================


def setup_logger():
    """Setup custom logger with clean formatting."""
    logger.remove()  # Remove default handler

    # Custom format with colors
    log_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )

    # Add handler with custom format
    logger.add(sys.stderr, format=log_format, level="INFO", colorize=True)


# Initialize logger
setup_logger()


def log_header(text):
    """Print a formatted header."""
    width = 70
    logger.info("=" * width)
    logger.info(f"  {text}")
    logger.info("=" * width)


def log_step(text):
    """Print a step with arrow."""
    logger.info(f"→ {text}")


def log_success(text):
    """Print success message."""
    logger.success(f"✓ {text}")


def log_info(text):
    """Print info message."""
    logger.info(f"  {text}")


def log_warning(text):
    """Print warning message."""
    logger.warning(f"⚠ {text}")


def log_error(text):
    """Print error message."""
    logger.error(f"✗ {text}")


# ============================================================================
# Grounding DINO Segmentation App
# ============================================================================


class GroundingDinoSegmentationApp:
    """
    Grounding DINO + SAM2 based video segmentation app.
    Compatible interface with VideoSegmentationApp but uses text prompts instead of manual annotation.
    """

    def __init__(
        self,
        sam_model_name="facebook/sam2-hiera-tiny",
        grounding_model_name="IDEA-Research/grounding-dino-tiny",
        text_prompt=None,
        device="cuda",
    ):
        self.device = device

        log_step("Initializing models")

        # Initialize SAM2
        log_info(f"Loading SAM2: {sam_model_name}")
        self.sam_predictor = SAM2VideoPredictor.from_pretrained(
            sam_model_name, device=device
        )
        log_success("SAM2 loaded")

        # Initialize Grounding DINO
        self._init_grounding_dino(grounding_model_name)

        # State variables (compatible with VideoSegmentationApp)
        self.state = None
        self.video_path = None
        self.current_frame_idx = 0
        self.frame_width = 0
        self.frame_height = 0
        self.output_dir = f"{Path(__file__).parent.parent.parent}/cache/segs"
        self.mask_output_dir = None
        self.box_output_dir = None
        self.bboxes = {}

        # Grounding DINO specific
        self.text_prompt = text_prompt
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        self.detected_boxes = []
        self.frame = None

        os.makedirs(self.output_dir, exist_ok=True)

    def _init_grounding_dino(self, model_name):
        """Initialize Grounding DINO model."""
        try:
            from groundingdino.util.inference import load_model, load_image, predict

            log_info(f"Loading Grounding DINO: {model_name}")

            # Download and load model
            if "tiny" in model_name.lower():
                config_path = (
                    Path(__file__).parent.parent.parent
                    / "three/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
                )
                checkpoint_path = (
                    Path(__file__).parent.parent.parent
                    / "weights/groundingdino_swint_ogc.pth"
                )
            else:
                config_path = (
                    Path(__file__).parent.parent.parent
                    / "three/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py"
                )
                checkpoint_path = (
                    Path(__file__).parent.parent.parent
                    / "weights/groundingdino_swinb_cogcoor.pth"
                )

            # Try to load from local path first, fallback to huggingface
            try:
                self.grounding_model = load_model(
                    config_path, checkpoint_path, device=self.device
                )
                self.use_transformers = False
                log_success("Grounding DINO loaded (official)")
            except Exception as e:
                log_warning(f"Official Grounding DINO failed: {str(e)}")
                log_info("Falling back to Transformers implementation")

                from transformers import (
                    AutoProcessor,
                    AutoModelForZeroShotObjectDetection,
                )

                self.grounding_processor = AutoProcessor.from_pretrained(model_name)
                self.grounding_model = (
                    AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(
                        self.device
                    )
                )
                self.use_transformers = True
                log_success("Grounding DINO loaded (Transformers)")

        except ImportError as e:
            log_error("Grounding DINO not installed")
            raise ImportError(
                "Grounding DINO not installed. Please install with:\n"
                "  pip install groundingdino-py\n"
                "or\n"
                "  pip install transformers"
            )

    def to_device(self, device):
        """Move models to specified device."""
        log_step(f"Moving models to {device}")
        self.device = device
        self.sam_predictor.to(device)
        if hasattr(self, "grounding_model"):
            self.grounding_model.to(device)
        log_success(f"Models moved to {device}")

    def set_mask_output_dir(self, mask_output_dir):
        """Set mask output directory."""
        self.mask_output_dir = mask_output_dir
        log_info(f"Mask output: {mask_output_dir}")

    def set_box_output_dir(self, box_output_dir):
        """Set bounding box output directory."""
        self.box_output_dir = box_output_dir
        log_info(f"Box output: {box_output_dir}")

    def load_video(self, video_path):
        """Load video for processing."""
        log_step("Loading video")

        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        log_info(f"Resolution: {self.frame_width}x{self.frame_height}")
        log_info(f"Frames: {frame_count} ({fps:.1f} FPS)")

        cap.release()

        # Initialize SAM2 state
        log_info("Initializing SAM2 state")
        self.state = self.sam_predictor.init_state(video_path, True, True)

        # Get first frame for detection
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        cap.release()

        if not ret:
            log_error("Could not read first frame")
            raise ValueError("Could not read the first frame")

        self.current_frame_idx = 0
        log_success("Video loaded")

    def set_text_prompt(
        self,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        Set text prompt for Grounding DINO detection.

        Args:
            text_prompt: Text description of object to detect (e.g., "a cat", "person . car", "red ball")
            box_threshold: Confidence threshold for box detection
            text_threshold: Confidence threshold for text matching
        """
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        log_info(f"Prompt: '{text_prompt}'")
        log_info(f"Thresholds: box={box_threshold:.2f}, text={text_threshold:.2f}")

    def detect_objects_transformers(self, image):
        """Detect objects using Transformers-based Grounding DINO."""
        from PIL import Image as PILImage

        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image

        # Prepare inputs
        inputs = self.grounding_processor(
            images=image_pil, text=self.text_prompt, return_tensors="pt"
        ).to(self.device)

        # Run detection
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # Post-process
        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        # Convert to xyxy format
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        return boxes, scores, labels

    def detect_objects_official(self, image):
        """Detect objects using official Grounding DINO."""
        from groundingdino.util.inference import predict
        from groundingdino.util import box_ops

        # Save image temporarily
        temp_image_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(temp_image_path, image)

        # Load image
        from groundingdino.util.inference import load_image

        image_source, image_transformed = load_image(temp_image_path)

        # Predict
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=image_transformed,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        # Convert boxes from cxcywh (normalized) to xyxy (absolute)
        h, w = image.shape[:2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h])
        boxes_xyxy = boxes_xyxy.cpu().numpy()

        # Clean up
        os.unlink(temp_image_path)

        return boxes_xyxy, logits.cpu().numpy(), phrases

    def detect_objects(self, image):
        """Detect objects in the image using Grounding DINO."""
        if self.text_prompt is None:
            log_error("Text prompt not set")
            raise ValueError("Text prompt not set. Call set_text_prompt() first.")

        if self.use_transformers:
            return self.detect_objects_transformers(image)
        else:
            return self.detect_objects_official(image)

    def annotate_frame_matplotlib(self, auto_detect=True):
        """
        Annotate frame using Grounding DINO detection.
        Compatible with VideoSegmentationApp interface.

        Args:
            auto_detect: If True, automatically detect and show results.
                        If False, just store the frame for later processing.
        """
        if self.text_prompt is None:
            log_error("Text prompt required for detection")
            raise ValueError(
                "Text prompt not set. Call set_text_prompt() before annotation.\n"
                "Example: app.set_text_prompt('a cat . a dog')"
            )

        if auto_detect:
            log_step("Running object detection")

            # Run detection
            boxes, scores, labels = self.detect_objects(self.frame)
            self.detected_boxes = boxes

            if len(boxes) == 0:
                log_warning("No objects detected")
            else:
                log_success(f"Detected {len(boxes)} objects")
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    log_info(f"  [{i+1}] {label}: {score:.3f}")

            # Visualize detection results
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

            # Draw boxes
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 5,
                    f"{label}: {score:.2f}",
                    color="red",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

            plt.title(f"Detection Results - Prompt: '{self.text_prompt}'")
            plt.axis("off")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        else:
            log_info("Frame loaded (detection pending)")

    def extract_bbox_from_mask(self, mask):
        """Extract bounding box from binary mask."""
        y_indices, x_indices = np.where(mask > 0)

        if len(y_indices) == 0 or len(x_indices) == 0:
            return None

        x0 = int(np.min(x_indices))
        y0 = int(np.min(y_indices))
        x1 = int(np.max(x_indices))
        y1 = int(np.max(y_indices))

        return [x0, y0, x1, y1]

    def process_without_gui(
        self, text_prompt=None, box_threshold=0.35, text_threshold=0.25
    ):
        """
        Process video without GUI (compatible with VideoSegmentationApp interface).

        Args:
            text_prompt: Text description of object to detect
            box_threshold: Confidence threshold for box detection
            text_threshold: Confidence threshold for text matching
        """
        if text_prompt is not None:
            self.set_text_prompt(text_prompt, box_threshold, text_threshold)

        self.process_and_save_video()

        video_name = os.path.basename(self.video_path)
        return os.path.join(self.output_dir, "segmented_" + video_name)

    def release_resources(self):
        """Release GPU resources."""
        log_step("Releasing resources")

        if hasattr(self, "sam_predictor"):
            del self.sam_predictor
        if hasattr(self, "grounding_model"):
            del self.grounding_model
        self.state = None

        if self.device == "cuda":
            torch.cuda.empty_cache()

        log_success("Resources released")

    @classmethod
    def initialize_once(
        cls,
        sam_model_name="facebook/sam2-hiera-tiny",
        grounding_model_name="IDEA-Research/grounding-dino-tiny",
        device="cuda",
    ):
        """Initialize once and reuse (compatible with VideoSegmentationApp)."""
        if not hasattr(cls, "_instance"):
            cls._instance = cls(sam_model_name, grounding_model_name, device)
        return cls._instance

    def process_and_save_video(self):
        """Process video and save segmentation results."""
        if self.text_prompt is None:
            log_error("Text prompt not set")
            raise ValueError("Text prompt not set. Call set_text_prompt() first.")

        if self.frame_width <= 0 or self.frame_height <= 0:
            log_error(f"Invalid dimensions: {self.frame_width}x{self.frame_height}")
            return

        # Detect objects in first frame if not already done
        if len(self.detected_boxes) == 0:
            log_step("Detecting objects in first frame")
            boxes, scores, labels = self.detect_objects(self.frame)
            self.detected_boxes = boxes

            if len(boxes) == 0:
                log_warning("No objects detected")
                log_info("Try adjusting thresholds or text prompt")
                return

            log_success(f"Detected {len(boxes)} objects")
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                log_info(f"  [{i+1}] {label}: {score:.3f}")

        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        os.makedirs(self.mask_output_dir, exist_ok=True)
        os.makedirs(self.box_output_dir, exist_ok=True)

        # Prepare for video output
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_filename = temp_file.name
        temp_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            temp_filename, fourcc, fps, (self.frame_width, self.frame_height)
        )

        log_step("Processing video with SAM2")
        self.bboxes = {}

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            # Add detected boxes as prompts to SAM2
            first_box = self.detected_boxes[0]
            bbox_tensor = torch.tensor([first_box], dtype=torch.float32)

            log_info(
                f"Using box: [{first_box[0]:.0f}, {first_box[1]:.0f}, {first_box[2]:.0f}, {first_box[3]:.0f}]"
            )

            frame_idx, object_ids, masks = self.sam_predictor.add_new_points_or_box(
                self.state, 0, obj_id=0, box=bbox_tensor
            )

            log_success(f"Initial segmentation (frame {frame_idx})")

            # Create progress bar
            pbar = tqdm(total=frame_count, desc="Segmenting", unit="frame")

            # Propagate segmentation through video
            processed_frames = 0
            for frame_idx, object_ids, masks in self.sam_predictor.propagate_in_video(
                self.state
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    log_warning(f"Could not read frame {frame_idx}")
                    break

                combined_mask = np.zeros(
                    (self.frame_height, self.frame_width), dtype=np.uint8
                )

                # Apply mask overlay
                for i, obj_id in enumerate(object_ids):
                    mask = masks[i].cpu().numpy()

                    if mask.size == 0:
                        continue

                    if len(mask.shape) == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)

                    binary_mask = (mask > 0.0).astype(np.uint8)

                    if (
                        binary_mask.shape[0] != self.frame_height
                        or binary_mask.shape[1] != self.frame_width
                    ):
                        binary_mask = cv2.resize(
                            binary_mask,
                            (self.frame_width, self.frame_height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    combined_mask = np.logical_or(combined_mask, binary_mask).astype(
                        np.uint8
                    )

                    # Create overlay
                    color_mask = np.zeros_like(frame)
                    color_mask[:, :, 1] = binary_mask * 255  # Green channel

                    alpha = 0.5
                    frame = cv2.addWeighted(frame, 1, color_mask, alpha, 0)

                # Save mask and bbox
                if np.any(combined_mask):
                    mask_filename = f"{frame_idx:06d}-mask.png"
                    mask_path = os.path.join(self.mask_output_dir, mask_filename)
                    cv2.imwrite(mask_path, combined_mask * 255)

                    bbox = self.extract_bbox_from_mask(combined_mask)
                    if bbox is not None:
                        self.bboxes[frame_idx] = bbox
                        bbox_filename = f"{frame_idx:06d}-box.txt"
                        bbox_path = os.path.join(self.box_output_dir, bbox_filename)
                        with open(bbox_path, "w") as f:
                            f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

                out.write(frame)
                pbar.update(frame_idx - processed_frames)
                processed_frames = frame_idx

            pbar.close()

        cap.release()
        out.release()

        log_success(f"Processed {processed_frames} frames")

        # Save output video
        output_path = os.path.join(
            self.output_dir, "segmented_" + os.path.basename(self.video_path)
        )
        try:
            shutil.copy2(temp_filename, output_path)
            os.unlink(temp_filename)
            log_success(f"Video saved: {output_path}")
        except Exception as e:
            log_error(f"Failed to save output: {str(e)}")
            log_info(f"Temporary file: {temp_filename}")


def main():
    """Example usage of GroundingDinoSegmentationApp."""
    log_header("Grounding DINO Segmentation Demo")

    app = GroundingDinoSegmentationApp()

    # Get video path and text prompt from user
    video_path = input("\nEnter video path: ")
    text_prompt = input("Enter text prompt (e.g., 'a cat', 'person . car'): ")

    # Load video and run application
    try:
        app.load_video(video_path)
        app.set_box_output_dir(
            f"{Path(__file__).parent.parent.parent}/cache/segs/boxes"
        )
        app.set_mask_output_dir(
            f"{Path(__file__).parent.parent.parent}/cache/segs/masks"
        )
        app.set_text_prompt(text_prompt)
        app.annotate_frame_matplotlib()
        app.process_and_save_video()

        log_header("Processing Complete!")

    except Exception as e:
        log_error(f"Processing failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
