"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:15:32
Description: Preprocess OnePose Videos for Full RGB images

"""
import cv2
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def find_video_files(path, extensions=(".m4v",)):
    video_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(extensions):
                video_files.append(os.path.join(root, file))
    return video_files


def process_video(video):
    try:
        father = os.path.dirname(video)
        # check if the video is already processed
        if os.path.exists(os.path.join(father, "color_full")) and os.path.exists(
            os.path.join(father, "intrinsics_full")
        ):
            # check if the number of frames in the video and the number of frames in the color_full directory match
            color_full_dir = os.path.join(father, "color_full")
            num_frames = len(os.listdir(color_full_dir))
            color_small_dir = os.path.join(father, "color")
            num_frames_small = len(os.listdir(color_small_dir))
            intrinsics_full_dir = os.path.join(father, "intrinsics_full")
            if num_frames == num_frames_small and num_frames == len(
                os.listdir(intrinsics_full_dir)
            ):
                return (video, "Already processed")
        # Read all frames from the video
        frames = read_video(video)
        num_frames = len(frames)

        # Read existing color frames count to assert
        color_dir = os.path.join(father, "color")
        if not os.path.exists(color_dir):
            raise FileNotFoundError(f"Color directory not found: {color_dir}")

        color_small_num = os.listdir(color_dir)
        if len(color_small_num) != num_frames:
            raise ValueError(
                f"Frame count mismatch in {father}: color/{len(color_small_num)} vs video/{num_frames}"
            )

        # Prepare directories for full RGB frames and intrinsics
        color_full_dir = os.path.join(father, "color_full")
        intrinsics_full_dir = os.path.join(father, "intrinsics_full")
        os.makedirs(color_full_dir, exist_ok=True)
        os.makedirs(intrinsics_full_dir, exist_ok=True)

        # Save all frames as PNG images
        for i, frame in enumerate(frames):
            frame_path = os.path.join(color_full_dir, f"{i}.png")
            cv2.imwrite(frame_path, frame)

        # Load intrinsics and save a copy for each frame
        intrinsics_path = os.path.join(father, "intrinsics.txt")
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")

        # intrinsics like:
        """
            fx: 1606.313951302609
            fy: 1606.313951302609
            cx: 956.4425804809623
            cy: 715.9485629659316
        """

        # need to be saved in each frame's intrinsics file as a 3x3 matrix

        with open(intrinsics_path, "r") as f:
            intrinsics = f.readlines()
            fx = float(intrinsics[0].split(": ")[1])
            fy = float(intrinsics[1].split(": ")[1])
            cx = float(intrinsics[2].split(": ")[1])
            cy = float(intrinsics[3].split(": ")[1])

        for i in range(num_frames):
            intrinsics_frame_path = os.path.join(intrinsics_full_dir, f"{i}.txt")
            with open(intrinsics_frame_path, "w") as f:
                f.write(f"{fx} 0 {cx}\n0 {fy} {cy}\n0 0 1")

        return (video, "Success")

    except Exception as e:
        return (video, f"Failed: {str(e)}")


def dump_full_rgb_parallel(video_files, max_workers=8):
    # if max_workers is None:
    #     max_workers = multiprocessing.cpu_count()

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video processing tasks
        future_to_video = {
            executor.submit(process_video, video): video for video in video_files
        }

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(future_to_video), total=len(video_files)):
            video, status = future.result()
            results.append((video, status))

    # Optionally, handle results (e.g., log failures)
    failures = [v for v, s in results if s != "Success" and s != "Already processed"]
    if failures:
        print(f"Processed {len(video_files)} videos with {len(failures)} failures.")
        for video in failures:
            print(f"Failed to process: {video}")
    else:
        print("All videos processed successfully.")


def main():
    roots = [
        "data/onepose/val_data",
        "data/onepose/train_data",
        "data/onepose/test_data",
    ]
    for root in roots:
        if not os.path.exists(root):
            raise FileNotFoundError(f"Directory not found: {root}")
        else:
            print(f"Processing videos in {root}...")
        all_videos = find_video_files(root)
        print(f"Found {len(all_videos)} video(s) to process.")
        dump_full_rgb_parallel(all_videos, max_workers=4)


if __name__ == "__main__":
    main()
