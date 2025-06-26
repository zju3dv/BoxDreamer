"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:12:09
Description: YCBV dataset parser

"""
import os
import shutil
import json
import numpy as np
import tqdm as tqdm
from pytorch3d.transforms import quaternion_to_matrix
from pathlib import Path
import torch


def process_models(path: str):
    pass


def get_meta_data(path: str):
    classes_file = os.path.join(path, "classes.txt")
    train_ids_file = os.path.join(path, "train.txt")
    test_ids_file = os.path.join(path, "val.txt")
    val_ids_file = os.path.join(path, "trainval.txt")

    # load train ids
    with open(train_ids_file, "r") as f:
        train_ids = f.readlines()
    train_ids = [t[:-1] for t in train_ids]
    # load test ids
    with open(test_ids_file, "r") as f:
        test_ids = f.readlines()
    test_ids = [t[:-1] for t in test_ids]
    # load val ids
    with open(val_ids_file, "r") as f:
        val_ids = f.readlines()
        # eliminate '\n' character
    val_ids = [v[:-1] for v in val_ids]

    # dataset id format is seq_id/frame_id like 0011/000001

    # load classes
    with open(classes_file, "r") as f:
        classes = f.readlines()
        # exclude new line character
        classes = [c[:-1] for c in classes]

    # classes are in the format of 0xx_class_name which including 21 ycbv classes

    return train_ids, test_ids, val_ids, classes


def process_dataset(root: str, split: str, ids: list, classes: list):
    working_dir = os.path.join(root, split)
    org_data_dir = os.path.join(
        root, "YCB_Video_Dataset/data"
    )  # data stored like seq_id/xxxxxx-color.png etc.
    seqs = os.listdir(org_data_dir)
    seqs.sort()

    pose_dir = os.path.join(
        root, "YCB_Video_Dataset/poses"
    )  # pose stored class_id/class_id.txt

    camera_cmu = json.load(
        open(os.path.join(root, "YCB_Video_Dataset/cameras/asus-cmu.json"))
    )
    camera_uw = json.load(
        open(os.path.join(root, "YCB_Video_Dataset/cameras/asus-uw.json"))
    )

    camera_cmu = camera_cmu["rig"]["camera"][0]["camera_model"][
        "params"
    ]  # fu, fv, u0, v0, k1, k2, k3
    camera_uw = camera_uw["rig"]["camera"][0]["camera_model"][
        "params"
    ]  # fu, fv, u0, v0, k1, k2, k3
    # The camera parameters used to capture the videos. asus-uw.json for video 0000 ~ 0059, asus-cmu.json for video 0060 ~ 0091.
    # we need dump data in the format of split/class_id/seq_id/xxxxxx-color.png etc.
    # for storage consideration, we only make soft links to the original data
    obj_rgb_mappping = {}
    obj_pose_mapping = {}
    obj_box_mapping = {}
    obj_intrinsics_mapping = {}

    obj_poses_data = {}

    for class_id in classes:
        pose_path = os.path.join(pose_dir, class_id + ".txt")
        # load pose data (per pose per line)
        obj_pose_data = open(
            pose_path, "r"
        ).readlines()  # 7 dimension data (quaternion + translation)
        obj_poses_data[class_id] = {}
        pose_cursor = 0
        obj_poses_list = []

        for line in obj_pose_data:
            pose = line.split("\n")[0].split(" ")
            obj_poses_list.append(pose)

        for seq in seqs:
            box_files = [
                f
                for f in os.listdir(os.path.join(org_data_dir, seq))
                if f.endswith("-box.txt")
            ]
            box_files.sort()

            for box_file in box_files:
                box_data = open(
                    os.path.join(org_data_dir, seq, box_file), "r"
                ).readlines()
                frame_prefix = box_file.split("-")[0]
                index = seq + "/" + frame_prefix
                for line in box_data:
                    obj_id = line.split(" ")[0]
                    if obj_id == class_id:
                        obj_poses_data[class_id][index] = obj_poses_list[pose_cursor]
                        pose_cursor += 1

    for seq in tqdm.tqdm(seqs, desc="Processing dataset", total=len(seqs)):
        org_seq_dir = os.path.join(org_data_dir, seq)
        seq_intrinsics = camera_uw if int(seq) < 60 else camera_cmu
        matrix_intrinsics = np.array(
            [
                [seq_intrinsics[0], 0, seq_intrinsics[2]],
                [0, seq_intrinsics[1], seq_intrinsics[3]],
                [0, 0, 1],
            ]
        )
        # get all files end with -box.txt
        box_files = [f for f in os.listdir(org_seq_dir) if f.endswith("-box.txt")]
        # sort the files
        box_files.sort()

        for idx, box_file in enumerate(box_files):
            frame_prefix = box_file.split("-")[0]
            index = seq + "/" + frame_prefix
            if index not in ids:
                continue
            # load the box data
            box_data = open(os.path.join(org_seq_dir, box_file), "r").readlines()
            # per instance per line
            for line in box_data:
                # get the instance id
                obj_id = line.split(" ")[0]
                # get the box data
                box = line.split("\n")[0].split(" ")[1:]
                # store the box data
                if obj_id not in obj_box_mapping:
                    obj_box_mapping[obj_id] = {}
                    obj_intrinsics_mapping[obj_id] = {}
                    obj_pose_mapping[obj_id] = {}
                    obj_rgb_mappping[obj_id] = {}

                if seq not in obj_box_mapping[obj_id]:
                    obj_box_mapping[obj_id][seq] = []
                    obj_intrinsics_mapping[obj_id][seq] = []
                    obj_pose_mapping[obj_id][seq] = []
                    obj_rgb_mappping[obj_id][seq] = []

                # obj_box_mapping[obj_id].append(box)
                # obj_intrinsics_mapping[obj_id].append(matrix_intrinsics)
                # obj_rgb_mappping[obj_id].append(Path(os.path.join(org_seq_dir, box_file.replace('-box.txt', '-color.png'))).absolute())
                # obj_pose_mapping[obj_id].append(obj_poses_data[obj_id][index])

                obj_box_mapping[obj_id][seq].append(box)
                obj_intrinsics_mapping[obj_id][seq].append(matrix_intrinsics)
                obj_rgb_mappping[obj_id][seq].append(
                    Path(
                        os.path.join(
                            org_seq_dir, box_file.replace("-box.txt", "-color.png")
                        )
                    ).absolute()
                )
                obj_pose_mapping[obj_id][seq].append(obj_poses_data[obj_id][index])

    for obj_id in tqdm.tqdm(
        obj_rgb_mappping.keys(),
        desc="Creating soft links",
        total=len(obj_rgb_mappping.keys()),
    ):
        # for rgb file we need to create soft links
        # for others, make a txt file to store the data, file format is like: xxxxxx-color.png xxxxxx-box.txt xxxxxx-pose.txt xxxxxx-intrinsics.txt
        obj_dir = os.path.join(working_dir, obj_id)
        os.makedirs(obj_dir, exist_ok=True)

        for seq in obj_rgb_mappping[obj_id]:
            obj_seq_dir = os.path.join(obj_dir, seq)
            os.makedirs(obj_seq_dir, exist_ok=True)

            for idx, rgb_file in enumerate(obj_rgb_mappping[obj_id][seq]):
                # check if the soft link exists, if exists, delete it
                if os.path.lexists(
                    os.path.join(obj_seq_dir, str(idx).zfill(6) + "-color.png")
                ):
                    print(
                        f"Removing existing soft link {os.path.join(obj_seq_dir, str(idx).zfill(6) + '-color.png')}"
                    )
                    os.remove(
                        os.path.join(obj_seq_dir, str(idx).zfill(6) + "-color.png")
                    )
                os.symlink(
                    rgb_file,
                    os.path.join(obj_seq_dir, str(idx).zfill(6) + "-color.png"),
                )
                with open(
                    os.path.join(obj_seq_dir, str(idx).zfill(6) + "-box.txt"), "w"
                ) as f:
                    f.write(" ".join(obj_box_mapping[obj_id][seq][idx]))
                with open(
                    os.path.join(obj_seq_dir, str(idx).zfill(6) + "-pose.txt"), "w"
                ) as f:
                    # convert pose data to 4*4 matrix
                    pose = obj_pose_mapping[obj_id][seq][idx]
                    # convert to float
                    pose = [float(x) for x in pose]
                    quat = pose[:4]
                    t = pose[4:]
                    R = quaternion_to_matrix(
                        torch.from_numpy(np.array(quat).reshape(1, 4))
                    ).reshape(3, 3)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t
                    # 4 data per line
                    for i in range(4):
                        f.write(" ".join([str(x) for x in T[i]]) + "\n")

                with open(
                    os.path.join(obj_seq_dir, str(idx).zfill(6) + "-intrinsics.txt"),
                    "w",
                ) as f:
                    intri = obj_intrinsics_mapping[obj_id][seq][idx]  # numpy array
                    for i in range(3):
                        f.write(" ".join([str(x) for x in intri[i]]) + "\n")


def main(root: str):
    meta_path = os.path.join(root, "YCB_Video_Dataset/image_sets")
    train_ids, test_ids, val_ids, classes = get_meta_data(meta_path)

    # create directories
    # if directory exists, remove it
    if os.path.exists(os.path.join(root, "train")):
        shutil.rmtree(os.path.join(root, "train"))
    if os.path.exists(os.path.join(root, "test")):
        shutil.rmtree(os.path.join(root, "test"))
    if os.path.exists(os.path.join(root, "val")):
        shutil.rmtree(os.path.join(root, "val"))
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "test"), exist_ok=True)
    os.makedirs(os.path.join(root, "val"), exist_ok=True)

    # process train
    process_dataset(root, "train", train_ids, classes)
    process_dataset(root, "val", val_ids, classes)
    process_dataset(root, "test", test_ids, classes)


if __name__ == "__main__":
    root = "data/ycbv"
    main(root)
