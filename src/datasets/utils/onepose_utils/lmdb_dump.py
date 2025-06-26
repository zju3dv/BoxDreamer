"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:17:18
Description: Create lmdb dataset for onepose dataset, which may speed up the data loading process.

"""
import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List


def get_sequences(root: str) -> set[str]:
    sequences = set()
    # traverse the root directory until we find the color directory
    for root, _, files in os.walk(root):
        if "color" in root:
            # add father directory to the list
            sequences.add(os.path.dirname(root))

    return sequences


def sort_by_filename_idx(files: List[str]) -> List[str]:
    # sort the files by the filename index
    return sorted(files, key=lambda x: int(x.split(".")[0]))


def accumulate_files(seq: str):
    # color directory: color
    # pose directory: poses_ba 4*4 png
    # intrinsics directory: intrin_ba 3*3 txt
    # box directory: reproj_box 8*2 txt

    # get the color images
    color_dir = os.path.join(seq, "color")
    color_files = sort_by_filename_idx(os.listdir(color_dir))
    color_files = [os.path.join(color_dir, file) for file in color_files]

    # get the poses
    pose_dir = os.path.join(seq, "poses_ba")
    pose_files = sort_by_filename_idx(os.listdir(pose_dir))
    pose_files = [os.path.join(pose_dir, file) for file in pose_files]

    # get the intrinsics
    intrin_dir = os.path.join(seq, "intrin_ba")
    intrin_files = sort_by_filename_idx(os.listdir(intrin_dir))
    intrin_files = [os.path.join(intrin_dir, file) for file in intrin_files]

    # get the boxes
    box_dir = os.path.join(seq, "reproj_box")
    box_files = sort_by_filename_idx(os.listdir(box_dir))
    box_files = [os.path.join(box_dir, file) for file in box_files]

    # # read the data
    # intrinsics = [np.loadtxt(file) for file in intrin_files]
    # poses = [np.loadtxt(file) for file in pose_files]
    # boxes = [np.loadtxt(file) for file in box_files]

    return color_files, pose_files, intrin_files, box_files


def dump_all_data_into_lmdb(
    path: str,
    color_files: List[str],
    pose_files: List[str],
    intrin_files: List[str],
    box_files: List[str],
):
    env = lmdb.open(f"{path}/data.lmdb", map_size=1099511627776)
    txn = env.begin(write=True)

    for color_file, pose_file, intrin_file, box_file in tqdm(
        zip(color_files, pose_files, intrin_files, box_files),
        total=len(color_files),
        desc="Dumping data to lmdb",
    ):
        # use path as key

        color = cv2.imread(color_file)
        pose = np.loadtxt(pose_file)
        intrin = np.loadtxt(intrin_file)
        box = np.loadtxt(box_file)

        # use absolute path as key
        color_path = Path(color_file).absolute()
        pose_path = Path(pose_file).absolute()
        intrin_path = Path(intrin_file).absolute()
        box_path = Path(box_file).absolute()

        _, encoded = cv2.imencode(".png", color)

        # dump 4 kinds of data separately
        txn.put(str(color_path).encode(), encoded.tobytes())
        txn.put(str(pose_path).encode(), pose.tobytes())
        txn.put(str(intrin_path).encode(), intrin.tobytes())
        txn.put(str(box_path).encode(), box.tobytes())

        # commit per 500 data for efficiency
        if int(color_file.split("/")[-1].split(".")[0]) % 500 == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()


def debug_read_lmdb(path: str):
    env = lmdb.open(path, map_size=1099511627776)
    txn = env.begin()
    cursor = txn.cursor()

    for key, value in cursor:
        print(key)
        print(np.frombuffer(value, dtype=np.uint8).shape)
        print()

    env.close()


if __name__ == "__main__":
    root = "data/onepose/train_data"
    sequences = get_sequences(root)
    color_files, pose_files, intrin_files, box_files = [], [], [], []
    with tqdm(total=len(sequences)) as pbar:
        for seq in sequences:
            (
                seq_color_files,
                seq_pose_files,
                seq_intrin_files,
                seq_box_files,
            ) = accumulate_files(seq)
            color_files.extend(seq_color_files)
            pose_files.extend(seq_pose_files)
            intrin_files.extend(seq_intrin_files)
            box_files.extend(seq_box_files)
            pbar.update(1)

    dump_all_data_into_lmdb(root, color_files, pose_files, intrin_files, box_files)
