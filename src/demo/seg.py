import os
import torch
import cv2
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
import tempfile
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

try:
    matplotlib.use("TkAgg")
except Exception as e:
    print(f"Error setting matplotlib backend: {e}")
    pass
import shutil


class VideoSegmentationApp:
    def __init__(self, model_name="facebook/sam2-hiera-tiny", device="cuda"):
        self.device = device
        self.predictor = SAM2VideoPredictor.from_pretrained(model_name, device=device)
        self.points = []
        self.labels = []
        self.bbox = None
        self.frame = None
        self.state = None
        self.video_path = None
        self.current_frame_idx = 0
        self.frame_width = 0
        self.frame_height = 0
        self.output_dir = "./cache/segs"
        self.mask_output_dir = None
        self.box_output_dir = None
        self.fig = None
        self.ax = None
        self.rs = None
        self.mode = "point"  # 'point' or 'bbox'
        self.bboxes = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def to_device(self, device):
        self.device = device
        self.predictor.to(device)

    def set_mask_output_dir(self, mask_output_dir):
        self.mask_output_dir = mask_output_dir
        print(f"Mask output directory: {self.mask_output_dir}")

    def set_box_output_dir(self, box_output_dir):
        self.box_output_dir = box_output_dir
        print(f"Box output directory: {self.box_output_dir}")

    def load_video(self, video_path):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video dimensions: {self.frame_width}x{self.frame_height}")
        # load video length
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video length: {frame_count} frames")
        cap.release()

        # with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
        self.state = self.predictor.init_state(video_path, True, True)

        # Get first frame for annotation
        cap = cv2.VideoCapture(video_path)
        ret, self.frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Could not read the first frame")

        self.current_frame_idx = 0

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if self.mode == "point":
            x, y = int(event.xdata), int(event.ydata)
            if event.button == 1:  # Left click - foreground
                self.points.append((x, y))
                self.labels.append(1)
                self.ax.plot(x, y, "go", markersize=8)  # Green circle for foreground
                plt.draw()
                print(f"Added foreground point at ({x}, {y})")

    def line_select_callback(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Make sure x1 < x2 and y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Draw rectangle on plot
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="r", linewidth=2
        )
        self.ax.add_patch(rect)
        plt.draw()

        self.bbox = [x1, y1, x2, y2]
        print(f"Set bounding box at ({x1}, {y1}, {x2}, {y2})")

    def toggle_selector(self, event):
        if event.key == "b":
            if self.mode == "point":
                self.mode = "bbox"
                if self.rs is None:
                    self.rs = RectangleSelector(self.ax, self.line_select_callback)
                self.rs.set_active(True)
                print("Switched to bounding box mode. Click and drag to draw a box.")
            else:
                self.mode = "point"
                if self.rs is not None:
                    self.rs.set_active(False)
                print("Switched to point mode. Click to add a foreground point.")
        elif event.key == "r":
            self.points = []
            self.labels = []
            self.bbox = None
            self.ax.clear()
            self.ax.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            plt.draw()
            print("Reset all annotations.")
        elif event.key == " ":  # Space to confirm
            plt.close()

    def annotate_frame_matplotlib(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("key_press_event", self.toggle_selector)

        instruction = "Click to add a foreground point.\n"
        instruction += "Press 'b' to toggle bounding box mode.\n"
        instruction += "Press 'r' to reset.\n"
        instruction += "Press SPACE to start segmentation."

        plt.title(instruction)
        plt.tight_layout()
        plt.show()

    def extract_bbox_from_mask(self, mask):
        y_indices, x_indices = np.where(mask > 0)

        if len(y_indices) == 0 or len(x_indices) == 0:
            return None

        x0 = int(np.min(x_indices))
        y0 = int(np.min(y_indices))
        x1 = int(np.max(x_indices))
        y1 = int(np.max(y_indices))

        return [x0, y0, x1, y1]

    def process_without_gui(self, points=None, bbox=None):
        if points is not None:
            self.points = points
            self.labels = [1] * len(points)

        if bbox is not None:
            self.bbox = bbox

        self.process_and_save_video()

        video_name = os.path.basename(self.video_path)
        return os.path.join(self.output_dir, "segmented_" + video_name)

    def release_resources(self):
        if hasattr(self, "predictor"):
            del self.predictor
            self.state = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @classmethod
    def initialize_once(cls, model_name="facebook/sam2-hiera-tiny", device="cuda"):
        if not hasattr(cls, "_instance"):
            cls._instance = cls(model_name, device)
        return cls._instance

    def process_and_save_video(self):
        if not self.points and self.bbox is None:
            print("No annotations provided. Please select a point or a bounding box.")
            return

        if self.frame_width <= 0 or self.frame_height <= 0:
            print(f"Invalid frame dimensions: {self.frame_width}x{self.frame_height}")
            return

        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        os.makedirs(self.mask_output_dir, exist_ok=True)
        os.makedirs(self.box_output_dir, exist_ok=True)

        print(f"Mask output directory: {self.mask_output_dir}")
        print(f"Box output directory: {self.box_output_dir}")

        # Convert points and labels to tensors
        if self.points:
            points_normalized = [(x, y) for x, y in self.points]
            points_tensor = torch.tensor(points_normalized)
            labels_tensor = torch.tensor(self.labels)
            print(f"Using {len(self.points)} points as prompts")

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

        print("Processing video...")
        self.bboxes = {}

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            # Add prompts
            if self.bbox is not None:
                bbox_tensor = torch.tensor([self.bbox])
                print(f"Using bounding box as prompt: {self.bbox}")
                frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                    self.state, 0, obj_id=0, box=bbox_tensor
                )
            else:
                frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                    self.state, 0, obj_id=0, points=points_tensor, labels=labels_tensor
                )

            print(f"Initial segmentation completed for frame {frame_idx}")
            print(f"Object IDs: {object_ids}")
            if len(masks) > 0:
                print(f"First mask shape: {masks[0].shape}, dtype: {masks[0].dtype}")

            # Create a progress bar
            pbar = tqdm(total=frame_count)

            # Propagate the prompts
            processed_frames = 0
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(
                self.state
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    print(f"Could not read frame {frame_idx}")
                    break

                combined_mask = np.zeros(
                    (self.frame_height, self.frame_width), dtype=np.uint8
                )

                # Apply mask overlay
                for i, obj_id in enumerate(object_ids):
                    # Get mask and ensure it has the right shape
                    mask = masks[i].cpu().numpy()

                    # Check if mask is not empty and has proper shape
                    if mask.size == 0:
                        print(f"Empty mask for object {obj_id} on frame {frame_idx}")
                        continue

                    # Handle different mask dimensions
                    if len(mask.shape) == 3 and mask.shape[0] == 1:
                        # If mask is 3D with first dimension of 1, squeeze it
                        mask = mask.squeeze(0)

                    # Create a binary mask (1 where mask > 0.0, 0 elsewhere)
                    binary_mask = (mask > 0.0).astype(np.uint8)

                    # Now check dimensions after processing
                    if (
                        binary_mask.shape[0] != self.frame_height
                        or binary_mask.shape[1] != self.frame_width
                    ):
                        try:
                            # Ensure target dimensions are not empty
                            if self.frame_width > 0 and self.frame_height > 0:
                                binary_mask = cv2.resize(
                                    binary_mask,
                                    (self.frame_width, self.frame_height),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            else:
                                print(
                                    f"Invalid target dimensions: {self.frame_width}x{self.frame_height}"
                                )
                                continue
                        except cv2.error as e:
                            print(f"Error resizing mask: {e}")
                            print(
                                f"Mask shape: {binary_mask.shape}, Target size: {self.frame_width}x{self.frame_height}"
                            )
                            continue

                    combined_mask = np.logical_or(combined_mask, binary_mask).astype(
                        np.uint8
                    )

                    # Create a 3-channel mask for overlay
                    color_mask = np.zeros_like(frame)
                    # Set the color (green) where the binary mask is 1
                    color_mask[:, :, 1] = binary_mask * 255  # Green channel

                    # Blend original frame with color mask
                    alpha = 0.5
                    frame = cv2.addWeighted(frame, 1, color_mask, alpha, 0)

                if np.any(combined_mask):
                    mask_filename = f"{frame_idx:06d}-mask.png"
                    mask_path = os.path.join(self.mask_output_dir, mask_filename)
                    cv2.imwrite(mask_path, combined_mask * 255)

                    bbox = self.extract_bbox_from_mask(combined_mask)
                    bbox_filename = f"{frame_idx:06d}-box.txt"
                    bbox_path = os.path.join(self.box_output_dir, bbox_filename)
                    if bbox is not None:
                        self.bboxes[frame_idx] = bbox
                        with open(bbox_path, "w") as f:
                            # write into txt file
                            f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

                out.write(frame)
                pbar.update(frame_idx - processed_frames)
                processed_frames = frame_idx

            pbar.close()

        cap.release()
        out.release()

        output_path = os.path.join(
            self.output_dir, "segmented_" + os.path.basename(self.video_path)
        )
        try:
            shutil.copy2(temp_filename, output_path)
            os.unlink(temp_filename)
            print(f"Segmentation completed! Output saved to: {output_path}")
        except Exception as e:
            print(f"Error copying output file: {e}")
            print(f"Temporary file is still available at: {temp_filename}")

        # Show a preview of the first frame with mask
        preview_frame = None
        cap = cv2.VideoCapture(output_path)
        ret, preview_frame = cap.read()
        cap.release()

        if ret:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB))
            plt.title("First frame of the segmented video")
            plt.axis("off")
            plt.show()


def main():
    app = VideoSegmentationApp()

    # Get video path from user
    video_path = input("Enter the path to your video file: ")

    # Load video and run application
    try:
        app.load_video(video_path)
        app.set_box_output_dir("./cache/segs/boxes")
        app.set_mask_output_dir("./cache/segs/masks")
        app.annotate_frame_matplotlib()
        app.process_and_save_video()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
