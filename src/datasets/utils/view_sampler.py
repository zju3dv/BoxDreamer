"""
Author: Yuanhong Yu
Date: 2025-03-16 15:56:56
LastEditTime: 2025-03-17 15:10:50
Description: Sample reference views from the dataset

"""


import numpy as np
import argparse
import os
import shutil
from typing import List
from tqdm import tqdm
from src.utils.customize.sample_points_on_cad import get_all_points_on_model
from PIL import Image, ImageDraw
import concurrent.futures
from functools import partial
import cv2
import tempfile
import subprocess


def uniform_sample(num_views: int, total_views: int) -> np.ndarray:
    return np.linspace(0, total_views - 1, num_views, dtype=int)


def _fps_sample(
    poses: List[np.ndarray], query_idx: int, fps_num: int = 20, object_center=None
) -> List[int]:
    """Apply FPS (Farthest Point Sampling) to a list of poses.

    Args:
        poses: List of poses as 4x4 transformation matrices (numpy arrays).
        query_idx: Index of the initial query pose.
        fps_num: Number of poses to sample.

    Returns:
        List of indices of the FPS sampled poses.
    """
    num_poses = len(poses)

    if not (0 <= query_idx < num_poses or query_idx == -1):
        raise ValueError(
            f"query_idx {query_idx} is out of bounds for poses of length {num_poses}."
        )

    fps_num = min(fps_num, num_poses - 1)
    if fps_num <= 0:
        return []

    if object_center is None:
        translations = np.array(
            [pose[:3, 3] for pose in poses]
        )  # Shape: (num_poses, 3)
    else:
        # translate the poses to the object center
        translations = np.array(
            [pose[:3, 3] - object_center for pose in poses]
        )  # Shape: (num_poses, 3)
    if query_idx == -1:
        # that means the center point is reference of fps sample
        # so the first point farthest from the original point is the reference point

        # get the farthest point from the center
        cameras_center = np.mean(translations, axis=0)
        cur_point = cameras_center
        fps_indices = []
    else:
        fps_indices = [query_idx]
        cur_point = translations[fps_indices[0]]

    distance = np.full(num_poses, 1e8)

    for _ in range(fps_num):
        cur_distance = np.linalg.norm(cur_point[None, :] - translations, 2, 1)
        distance = np.min(np.stack([cur_distance, distance], 1), 1)
        cur_index = np.argmax(distance)
        cur_point = translations[cur_index]
        fps_indices.append(cur_index)

    if query_idx != -1:
        sampled_indices = fps_indices[1:]
    else:
        sampled_indices = fps_indices
    return [int(idx) for idx in sampled_indices]


def pose_inverse(pose):
    R = pose[:3, :3].T
    t = -R @ pose[:3, 3:]
    return np.concatenate(
        [np.concatenate([R, t], axis=1), np.array([[0, 0, 0, 1]])], axis=0
    )


def read_pose_file(path: str) -> np.ndarray:
    # read the pose file
    # return the poses as numpy array (4*4)
    pose = np.loadtxt(path)
    # to fp64 high precision
    pose = pose.astype(np.float64)
    # 3*3 -> 4*4
    h_pose = np.eye(4)

    h_pose[:3, :3] = pose[:3, :3]
    h_pose[:3, 3] = pose[:3, 3]

    # inverse the pose
    h_pose = pose_inverse(h_pose)

    return h_pose


def fps_sample(
    num_views: int, ref_idx: int, pose_files: List[str], cad_model=None
) -> np.ndarray:
    # read the pose files
    poses = [read_pose_file(pose_file) for pose_file in pose_files]

    if cad_model is not None:
        pts = get_all_points_on_model(cad_model)  # Nx3
        center = np.mean(pts, axis=0)  # ,3
    else:
        center = None

    # get the fps indices
    fps_indices = _fps_sample(poses, ref_idx, num_views, center)

    return fps_indices


def sample_linemod_reference_views(
    method: str = "uniform", num_views: int = 5, root: str = "data/lm"
) -> None:
    """Sample reference views for the linemod dataset :param method: the method
    to sample the reference views :param num_views: the number of reference
    views :param root: the root path of the dataset."""
    if method not in ["uniform", "fps"]:
        raise ValueError("The method is not supported yet")

    # The intrinsic parameter of linemod dataset
    K = np.array(
        [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    training_data_dir = os.path.join(root, "real_train")
    tgt_sample_dir = os.path.join(root, f"real_train_{method}_{num_views}")
    # log target directory
    print(f"Target directory: {tgt_sample_dir}")
    models_root = os.path.join(root, "models")

    objs = os.listdir(training_data_dir)

    # check whether the target directory exists, if exists, remove it
    if os.path.exists(tgt_sample_dir):
        shutil.rmtree(tgt_sample_dir)

    os.makedirs(tgt_sample_dir)

    for obj in objs:
        # make tgt directory
        print(f"Object: {obj}")
        os.makedirs(os.path.join(tgt_sample_dir, obj))
        model_path = os.path.join(models_root, obj, f"{obj}.ply")

        # list all files end with .png to get the file prefix index
        files = os.listdir(os.path.join(training_data_dir, obj))
        files = [f for f in files if f.endswith("-color.png")]
        file_prefix = [f.split(".")[0].split("-")[0] for f in files]

        # sort the file prefix
        file_prefix = sorted(file_prefix)

        # get the step size
        if method == "uniform":
            reference_idx = uniform_sample(num_views, len(file_prefix))
        elif method == "fps":
            ref_idx = -1
            pose_files = [
                os.path.join(training_data_dir, obj, f"{file_prefix[i]}-pose.txt")
                for i in range(len(file_prefix))
            ]
            reference_idx = fps_sample(num_views, ref_idx, pose_files, None)
        else:
            raise ValueError("The method is not supported yet")

        assert len(reference_idx) == num_views

        # sample the reference views
        for i in reference_idx:
            # -box.txt, -color.png, -depth.png, -pose.txt -coor.pkl -intrinsic.txt -label.png
            suffixes = [
                "-box.txt",
                "-color.png",
                "-depth.png",
                "-pose.txt",
                "-coor.pkl",
                "-intrisic.txt",
                "-label.png",
            ]
            print(f"Reference view: {file_prefix[i]}")
            for suffix in suffixes:
                src_file = os.path.join(training_data_dir, obj, file_prefix[i] + suffix)
                tgt_file = os.path.join(tgt_sample_dir, obj, file_prefix[i] + suffix)
                if suffix == "-intrisic.txt":
                    with open(tgt_file, "w") as f:
                        for row in K:
                            # Format each element in scientific notation with 18 decimal places
                            formatted_row = " ".join(f"{x:.18e}" for x in row)
                            f.write(formatted_row + "\n")
                else:
                    shutil.copyfile(src_file, tgt_file)

    print("Sample reference views done")


def sample_onepose_reference_views(
    method: str = "uniform",
    num_views: int = 5,
    root: str = "data/onepose",
    sub_dir: str = "test_data",
) -> None:
    """Sample reference views for the linemod dataset :param method: the method
    to sample the reference views :param num_views: the number of reference
    views :param root: the root path of the dataset."""

    if method not in ["uniform", "fps"]:
        raise ValueError("The method is not supported yet")

    sample_src_dir = os.path.join(root, sub_dir)
    sample_tgt_dir = os.path.join(root, f"{sub_dir}_{method}_{num_views}")

    # check whether the target directory exists, if exists, remove it
    if os.path.exists(sample_tgt_dir):
        shutil.rmtree(sample_tgt_dir)

    os.makedirs(sample_tgt_dir)

    objs = os.listdir(sample_src_dir)

    for obj in tqdm(objs, desc="Sampling reference views", total=len(objs)):
        os.makedirs(os.path.join(sample_tgt_dir, obj))
        seqs = os.listdir(os.path.join(sample_src_dir, obj))
        for seq in seqs:
            if seq == "box3d_corners.txt":
                # copy and continue
                shutil.copyfile(
                    os.path.join(sample_src_dir, obj, seq),
                    os.path.join(sample_tgt_dir, obj, seq),
                )
                continue
            if seq == ".DS_Store":
                continue
            try:
                _, seq_id = seq.split("-")
                if seq_id != "1":
                    # directly make a soft link
                    # use absolute path
                    src = os.path.join(sample_src_dir, obj, seq)
                    abs_src = os.path.abspath(src)
                    os.symlink(
                        abs_src,
                        os.path.join(sample_tgt_dir, obj, seq),
                        target_is_directory=True,
                    )
                    continue
            except:
                pass
            os.makedirs(os.path.join(sample_tgt_dir, obj, seq))
            data_dirs = ["color", "intrin_ba", "poses_ba", "reproj_box"]
            files = os.listdir(os.path.join(sample_src_dir, obj, seq, data_dirs[0]))
            file_prefix = [f.split(".")[0] for f in files]

            # exclude the filenames that is not number
            file_prefix = [f for f in file_prefix if f.isdigit()]

            # file name is str number
            file_prefix = [int(f) for f in file_prefix]
            file_prefix = sorted(file_prefix)
            # back to str
            file_prefix = [str(f) for f in file_prefix]

            if method == "uniform":
                reference_idx = uniform_sample(num_views, len(file_prefix))
            elif method == "fps":
                # add path prefix
                pose_files = [
                    os.path.join(
                        sample_src_dir, obj, seq, data_dirs[2], f"{file_prefix[i]}.txt"
                    )
                    for i in range(len(file_prefix))
                ]
                reference_idx = fps_sample(num_views, -1, pose_files)
            else:
                raise ValueError("The method is not supported yet")

            suffixes = [".png", ".txt", ".txt", ".txt"]
            for i in reference_idx:
                for data_dir, suffix in zip(data_dirs, suffixes):
                    file = f"{file_prefix[i]}{suffix}"
                    # make data_dir directory
                    os.makedirs(
                        os.path.join(sample_tgt_dir, obj, seq, data_dir), exist_ok=True
                    )

                    src_file = os.path.join(sample_src_dir, obj, seq, data_dir, file)
                    tgt_file = os.path.join(sample_tgt_dir, obj, seq, data_dir, file)
                    shutil.copyfile(src_file, tgt_file)

    print("Sample reference views done")


def read_rotation(pose_file):
    """Reads a pose file and returns the rotation matrix.

    Assumes the pose file contains a 4x4 transformation matrix.
    """
    try:
        with open(pose_file, "r") as f:
            lines = f.readlines()
            pose = np.array([list(map(float, line.strip().split())) for line in lines])
        return pose[:3, :3]  # Return rotation matrix
    except Exception as e:
        print(f"Error reading pose file {pose_file}: {e}")
        return None


def compute_average_angular_distance(pose_files):
    """Computes the average pairwise angular distance between all poses in a
    sequence."""
    rotations = []
    for pf in pose_files:
        R = read_rotation(pf)
        if R is not None:
            rotations.append(R)

    num_rotations = len(rotations)
    if num_rotations < 2:
        # Not enough rotations to compute pairwise distances
        return 0

    angles = []
    for i in range(num_rotations):
        for j in range(i + 1, num_rotations):
            R1 = rotations[i]
            R2 = rotations[j]
            R = np.dot(R1.T, R2)
            trace = np.trace(R)
            # Clamp the trace to avoid numerical issues with arccos
            trace = np.clip(trace, -1.0, 3.0)
            angle = np.arccos((trace - 1) / 2)  # Angle in radians
            angles.append(angle)

    if not angles:
        return 0
    return np.mean(angles)


def process_sequence(seq, obj_dir):
    """Worker function to process a single sequence. Computes the average
    angular distance for the sequence.

    :param seq: Sequence name
    :param obj_dir: Directory path of the object
    :return: Tuple of (sequence name, average angular distance)
    """
    seq_path = os.path.join(obj_dir, seq)
    pose_files = [
        os.path.join(seq_path, f'{f.split("-")[0]}-pose.txt')
        for f in os.listdir(seq_path)
        if f.endswith("-color.png")
    ]
    if not pose_files:
        return (seq, 0)
    avg_angle = compute_average_angular_distance(pose_files)
    return (seq, avg_angle)


def select_best_sequence(obj_dir, max_workers=None):
    """Selects the sequence with the highest average angular distance (best
    coverage) using concurrent processing.

    :param obj_dir: Directory path of the object
    :param max_workers: Maximum number of worker processes (optional)
    :return: Tuple of (best sequence name, sorted list of sequences with
        scores)
    """
    seqs = os.listdir(obj_dir)
    best_seq = None
    best_score = -1
    seq_score = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Prepare the partial function with obj_dir argument
        worker = partial(process_sequence, obj_dir=obj_dir)
        # Submit all sequences to the executor
        future_to_seq = {executor.submit(worker, seq): seq for seq in seqs}

        for future in tqdm(
            concurrent.futures.as_completed(future_to_seq),
            desc="Selecting best sequence",
            total=len(future_to_seq),
        ):
            seq = future_to_seq[future]
            try:
                seq_name, avg_angle = future.result()
                seq_score[seq_name] = avg_angle
                if avg_angle > best_score:
                    best_score = avg_angle
                    best_seq = seq_name
            except Exception as exc:
                print(f"Sequence {seq} generated an exception: {exc}")

    # Sort the sequences by their scores in descending order
    sorted_seq_list = sorted(seq_score.items(), key=lambda x: x[1], reverse=True)

    return best_seq, sorted_seq_list


def sample_ycbv_reference_views(
    method: str = "uniform",
    num_views: int = 5,
    root: str = "data/ycbv",
    sub_dir: str = "ref_views_16_processed",
) -> None:
    """Sample reference views for the YCB-Video (YCBV) dataset by selecting the
    sequence with the most comprehensive pose coverage for each object.

    :param method: the method to sample the reference views ("uniform"
        or "fps")
    :param num_views: the number of reference views to sample
    :param root: the root path of the dataset
    :param sub_dir: the subdirectory containing the reference views
    """

    if method not in ["uniform", "fps"]:
        raise ValueError("The method is not supported yet")

    sample_src_dir = os.path.join(root, sub_dir)
    sample_tgt_dir = os.path.join(root, f"{sub_dir}_{method}_{num_views}")
    pre_selected = {
        "024_bowl": "0007",
        "025_mug": "0070",
        "006_mustard_bottle": "0008",
        "002_master_chef_can": "0014",
        "011_banana": "0010",
        "003_cracker_box": "0007",
        "052_extra_large_clamp": "0003",
        "051_large_clamp": "0010",
        "035_power_drill": "0010",
        "036_wood_block": "0081",
        "019_pitcher_base": "0041",  # need improvement
        "040_large_marker": "0089",
        "009_gelatin_box": "0000",
        "061_foam_brick": "0081",  # fix, origin pre select 0071 but hard for reconstruction
        "010_potted_meat_can": "0014",
        "008_pudding_box": "0076",
        "037_scissors": "0016",
        "007_tuna_fish_can": "0039",  # 0072 is another good one
        "021_bleach_cleanser": "0006",
        "005_tomato_soup_can": "0003",
        "004_sugar_box": "0074",
    }

    good_objs = [
        "024_bowl",
        "025_mug",
        "006_mustard_bottle",
        "002_master_chef_can",
        "011_banana",
        "003_cracker_box",
        "052_extra_large_clamp",
        "051_large_clamp",
        "035_power_drill",
        "036_wood_block",
        "019_pitcher_base",
        "040_large_marker",
        "009_gelatin_box",
        "010_potted_meat_can",
        "008_pudding_box",
        "021_bleach_cleanser",
        "005_tomato_soup_can",
        "004_sugar_box",
    ]
    # most overlapping sequence
    """
        024_bowl: 0007
        025_mug: 0007
        006_mustard_bottle: 0008
        002_master_chef_can: 0091
        011_banana: 0010
        003_cracker_box: 0007
        052_extra_large_clamp: 0003
        051_large_clamp: 0010
        035_power_drill: 0010
        036_wood_block: 0090
        019_pitcher_base: 0009
        040_large_marker: 0010
        009_gelatin_box: 0003
        061_foam_brick: 0081
        010_potted_meat_can: 0008
        008_pudding_box: 0070
        037_scissors: 0010
        007_tuna_fish_can: 0008
        021_bleach_cleanser: 0008
        005_tomato_soup_can: 0008
        004_sugar_box: 0089
    """

    mode = "most-overlapping"  # Options: "interact", "first", "most-overlapping" or "pre-selected"

    # Check whether the target directory exists; if it exists, remove it
    if os.path.exists(sample_tgt_dir):
        shutil.rmtree(sample_tgt_dir)

    os.makedirs(sample_tgt_dir)

    objs = os.listdir(sample_src_dir)

    obj_seq_mapping = {}

    for obj in tqdm(
        objs, desc="Sampling reference views for each object", total=len(objs)
    ):
        obj_src_dir = os.path.join(sample_src_dir, obj)

        if mode != "first" and mode != "pre-selected":
            best_seq, ranked_list = select_best_sequence(obj_src_dir)

            if best_seq is None:
                print(f"No valid sequences found for object '{obj}'. Skipping.")
                continue
        else:
            best_seq = sorted(os.listdir(obj_src_dir))[0]

        if mode == "interact":
            selection = False
            for seq, score in ranked_list:
                if obj in good_objs:
                    selection = True
                    break
                print(f"Sequence: {seq}, Score: {score}")

                best_seq_path = os.path.join(obj_src_dir, seq)
                img_files = sorted(
                    [f for f in os.listdir(best_seq_path) if f.endswith("-color.png")]
                )

                if not img_files:
                    print(f"No color images found in sequence '{seq}'. Skipping.")
                    continue

                processed_frames = []
                print(f"Processing sequence '{seq}' with {len(img_files)} images...")

                for img_file in tqdm(
                    img_files, desc=f"Processing images for sequence '{seq}'"
                ):
                    img_path = os.path.join(best_seq_path, img_file)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        draw = ImageDraw.Draw(img)

                        # Read the bounding box file
                        bbox_file = img_file.replace("-color.png", "-box.txt")
                        bbox_path = os.path.join(best_seq_path, bbox_file)
                        if not os.path.exists(bbox_path):
                            print(
                                f"Bounding box file '{bbox_file}' not found. Skipping bbox drawing for this image."
                            )
                            # Convert PIL Image to NumPy array
                            cv_img = np.array(img)[:, :, ::-1]  # RGB to BGR
                            processed_frames.append(cv_img)
                            continue

                        with open(bbox_path, "r") as f:
                            line = f.readline().strip()
                            bbox = list(map(float, line.split()))
                            if len(bbox) != 4:
                                print(
                                    f"Invalid bounding box format in '{bbox_file}'. Expected 4 values, got {len(bbox)}."
                                )
                                cv_img = np.array(img)[:, :, ::-1]  # RGB to BGR
                                processed_frames.append(cv_img)
                                continue

                        # Draw the bounding box
                        draw.rectangle(bbox, outline="red", width=2)

                        # Convert PIL Image to NumPy array
                        cv_img = np.array(img)[:, :, ::-1]  # RGB to BGR
                        processed_frames.append(cv_img)
                    except Exception as e:
                        print(f"Error processing image '{img_file}': {e}")
                        continue

                if not processed_frames:
                    print(f"No frames processed for sequence '{seq}'. Skipping.")
                    continue

                # Create a temporary video file
                with tempfile.NamedTemporaryFile(
                    suffix=".avi", delete=False
                ) as temp_video:
                    temp_video_path = temp_video.name

                # Define video parameters
                height, width, layers = processed_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec
                fps = 24  # Frames per second

                video_writer = cv2.VideoWriter(
                    temp_video_path, fourcc, fps, (width, height)
                )

                print(f"Creating video at '{temp_video_path}'...")
                for frame in processed_frames:
                    video_writer.write(frame)
                video_writer.release()

                print(
                    f"Video for sequence '{seq}' created at '{temp_video_path}'. Opening video with default player..."
                )

                # Open video with default video player
                if os.name == "nt":  # Windows
                    os.startfile(temp_video_path)
                elif os.name == "posix":
                    subprocess.run(["xdg-open", temp_video_path])
                else:
                    print("Automatic video playback not supported on this OS.")

                # Ask user whether to select this sequence
                user_input = (
                    input(
                        f"Select sequence '{seq}' as the best for object '{obj}'? (y/n): "
                    )
                    .strip()
                    .lower()
                )
                if user_input == "y":
                    best_seq = seq
                    selection = True
                    # delete the temporary video file
                    os.remove(temp_video_path)
                    break  # Exit the sequence loop

                # delete the temporary video file
                os.remove(temp_video_path)

            if not selection:
                # Ask user whether to use the best sequence from the ranked list
                user_input = (
                    input(
                        f"Use the best sequence '{best_seq}' for object '{obj}'? (y/n): "
                    )
                    .strip()
                    .lower()
                )
                if user_input == "n":
                    print(f"Skipping object '{obj}'.")
                    continue
        elif mode == "most-overlapping":
            pass  # use the best sequence from the ranked list
        elif mode == "pre-selected":
            best_seq = pre_selected[obj]

        obj_seq_mapping[obj] = best_seq

        tgt_obj_dir = os.path.join(sample_tgt_dir, obj)
        os.makedirs(tgt_obj_dir, exist_ok=True)

        best_seq_path = os.path.join(obj_src_dir, best_seq)
        tgt_seq_dir = os.path.join(tgt_obj_dir, best_seq)
        os.makedirs(tgt_seq_dir, exist_ok=True)

        # List all files ending with -color.png to get the file prefix index
        files = sorted(
            [f for f in os.listdir(best_seq_path) if f.endswith("-color.png")]
        )
        file_prefix = [f.split(".")[0].split("-")[0] for f in files]

        # Sort the file prefix numerically
        try:
            file_prefix = sorted(file_prefix, key=lambda x: int(x))
        except ValueError:
            # If not purely numeric, sort lexicographically
            file_prefix = sorted(file_prefix)

        total_views = len(file_prefix)
        if total_views == 0:
            print(
                f"No color images found in sequence '{best_seq}' for object '{obj}'. Skipping."
            )
            continue

        # Get the reference indices based on the sampling method
        if method == "uniform":
            reference_idx = uniform_sample(num_views, total_views)
        elif method == "fps":
            ref_idx = -1  # Modify as needed based on FPS implementation
            pose_files = [
                os.path.join(best_seq_path, f"{file_prefix[i]}-pose.txt")
                for i in range(total_views)
            ]
            reference_idx = fps_sample(num_views, ref_idx, pose_files, None)
        else:
            raise ValueError("The method is not supported yet")

        if len(reference_idx) != num_views:
            print(
                f"Number of sampled indices ({len(reference_idx)}) does not match num_views ({num_views}) for object '{obj}', sequence '{best_seq}'."
            )
            continue

        # Sample the reference views
        for i in reference_idx:
            if i < 0 or i >= total_views:
                print(
                    f"Invalid index {i} for object '{obj}', sequence '{best_seq}'. Skipping this index."
                )
                continue
            # Define the suffixes to copy
            suffixes = ["-box.txt", "-color.png", "-pose.txt", "-intrinsics.txt"]
            print(f"Reference view: {file_prefix[i]}")
            for suffix in suffixes:
                src_file = os.path.join(best_seq_path, f"{file_prefix[i]}{suffix}")
                tgt_file = os.path.join(tgt_seq_dir, f"{file_prefix[i]}{suffix}")
                if os.path.exists(src_file):
                    shutil.copyfile(src_file, tgt_file)
                else:
                    print(f"Source file '{src_file}' does not exist. Skipping.")

        # Only sample the best overlapping sequence for each object

    print("Sample reference views done.")
    # Log YCBV object and sequence mapping
    print("Object and sequence mapping:")
    for obj, seq in obj_seq_mapping.items():
        print(f"{obj}: {seq}")


def main(dataset, method, num_views, root, sub_dir=None):
    if dataset == "linemod":
        sample_linemod_reference_views(method, num_views, root)
    elif dataset == "onepose":
        sample_onepose_reference_views(method, num_views, root, sub_dir)
    elif dataset == "ycbv":
        sample_ycbv_reference_views(method, num_views, root, sub_dir)
    else:
        raise ValueError("The dataset is not supported yet")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Sample reference views")

    # Define command line arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="linemod",
        choices=["linemod", "onepose", "ycbv"],
        help="Dataset name (default: linemod)",
    )
    parser.add_argument(
        "--method", type=str, default="fps", help="Sampling method (default: fps)"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=32,
        help="Number of views to sample (default: 32)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/lm",
        help="Root directory of dataset (default: data/lm)",
    )
    parser.add_argument(
        "--sub_dir",
        type=str,
        default=None,
        help="Sub-directory for specific dataset (for onepose/onepose-lowtexture: test_data, for ycbv: train)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set default sub_dir based on dataset type if not provided
    if args.sub_dir is None:
        if args.dataset in ["onepose", "onepose-lowtexture"]:
            args.sub_dir = "test_data"
        elif args.dataset == "ycbv":
            args.sub_dir = "train"
        # linemod doesn't need sub_dir, so leave as None

    # Call main function with parsed arguments
    main(args.dataset, args.method, args.num_views, args.root, args.sub_dir)
