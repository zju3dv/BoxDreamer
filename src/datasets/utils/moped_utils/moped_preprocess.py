"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:21:49
Description: MOPED dataset parser

"""
import os
import pytorch3d
import cv2
import shutil
import json

import sys
import numpy as np
import tqdm as tqdm
from pytorch3d.transforms import quaternion_to_matrix
from pathlib import Path
import torch

import json
from pathlib import Path

import numpy as np


def inverse_transform(trans):
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


def handle_seq(seq_dir: str, src_seq_root: str, tgt_seq_root: str, obj: str):
    for seq in seq_dir:
        # check whether the seq is valid
        if not os.path.isdir(os.path.join(src_seq_root, seq)):
            continue

        else:
            # make seq dir under obj dir
            os.makedirs(os.path.join(tgt_seq_root, obj, seq))

            # make color dir and pose dir and intrinsics dir and mask dir
            os.makedirs(os.path.join(tgt_seq_root, obj, seq, "pose"))
            os.makedirs(os.path.join(tgt_seq_root, obj, seq, "intrinsics"))

            # # mask and color could be cp directly
            # if exists, delete it
            # color and mask is dir and has multiple files
            if os.path.exists(os.path.join(tgt_seq_root, obj, seq, "mask")):
                shutil.rmtree(os.path.join(tgt_seq_root, obj, seq, "mask"))
            if os.path.exists(os.path.join(tgt_seq_root, obj, seq, "color")):
                shutil.rmtree(os.path.join(tgt_seq_root, obj, seq, "color"))

            # if mask-plane in mask dir, use mask-plane as mask dir
            mask_dir = "mask-plane"
            if not os.path.exists(os.path.join(src_seq_root, seq, mask_dir)):
                mask_dir = "mask"

            shutil.copytree(
                os.path.join(src_seq_root, seq, mask_dir),
                os.path.join(tgt_seq_root, obj, seq, "mask"),
            )
            shutil.copytree(
                os.path.join(src_seq_root, seq, "color"),
                os.path.join(tgt_seq_root, obj, seq, "color"),
            )

            # check the tgt color and mask dir and selete all file which is not end with .jpg formated like .xxxxx.jpg

            for file in os.listdir(os.path.join(tgt_seq_root, obj, seq, "color")):
                # if file have two ., it is not a jpg file
                if file.count(".") != 1:
                    os.remove(os.path.join(tgt_seq_root, obj, seq, "color", file))

            for file in os.listdir(os.path.join(tgt_seq_root, obj, seq, "mask")):
                if file.count(".") != 1:
                    os.remove(os.path.join(tgt_seq_root, obj, seq, "mask", file))

            # handle pose and intrinsics
            pose_file = os.path.join(src_seq_root, seq, "scene/trajectory.log")
            intri_file = os.path.join(src_seq_root, seq, "intrinsics.json")

            pose_align_file = os.path.join(
                src_seq_root, seq, "scene/refined_registration_optimized.json"
            )

            with open(pose_align_file, "r") as f:
                base_trans = json.load(f)
                try:
                    base_trans = np.array(base_trans["edges"][0]["transformation"])
                    base_trans = base_trans.reshape(4, 4).transpose()
                except:
                    base_trans = np.eye(4)

            # load intrinsics
            with open(intri_file, "r") as f:
                intri = json.load(f)
                intri = np.array(intri["intrinsic_matrix"])
                # reshape to 3*3
                intri = intri.reshape(3, 3).transpose()

            # pose file like:
            """
            0 0 1
            1.00000000 -0.00000000 0.00000000 0.00000000
            0.00000000 1.00000000 0.00000000 0.00000000
            0.00000000 0.00000000 1.00000000 0.00000000
            0.00000000 0.00000000 0.00000000 1.00000000
            1 1 2
            0.99999610 0.00266277 0.00084324 -0.00154903
            -0.00266354 0.99999603 0.00091640 0.00086867
            -0.00084079 -0.00091864 0.99999922 0.00124078
            0.00000000 0.00000000 0.00000000 1.00000000
            """
            # per 5 lines is a pose

            with open(pose_file, "r") as f:
                lines = f.readlines()
                for i in range(0, len(lines), 5):
                    pose = np.array(
                        [
                            list(map(float, line.split()))
                            for line in lines[i + 1 : i + 5]
                        ]
                    )

                    pose = inverse_transform(pose)

                    # pose file name like: 000000.txt which has 6 digits
                    pose_file_name = str(i // 5).zfill(6) + ".txt"

                    with open(
                        os.path.join(tgt_seq_root, obj, seq, "pose", pose_file_name),
                        "w",
                    ) as f:
                        for j in range(4):
                            f.write(" ".join([str(x) for x in pose[j]]) + "\n")

                    intri_file_name = str(i // 5).zfill(6) + ".txt"
                    with open(
                        os.path.join(
                            tgt_seq_root, obj, seq, "intrinsics", intri_file_name
                        ),
                        "w",
                    ) as f:
                        for k in range(3):
                            f.write(" ".join([str(x) for x in intri[k]]) + "\n")


def main(root: str, tgt: str):
    objs = os.listdir(root)
    # if tgt exists, remove it
    if os.path.exists(tgt):
        shutil.rmtree(tgt)
    os.makedirs(tgt, exist_ok=True)

    # make models directory
    models_dir = os.path.join(tgt, "models")
    os.makedirs(models_dir, exist_ok=True)

    # traverse each object and cp pointcloud.ply to models directory
    for obj in objs:
        if obj == "models" or obj == "reference" or obj == "test":
            continue
        obj_dir = os.path.join(root, obj, "reference")
        if not os.path.isdir(obj_dir):
            continue
        for seq in os.listdir(obj_dir):
            if not os.path.isdir(os.path.join(obj_dir, seq)):
                continue
            ply_file = os.path.join(obj_dir, seq, "scene/integrated_cropped.ply")
            if os.path.exists(ply_file):
                shutil.copy(
                    ply_file, os.path.join(models_dir, f"reference-{obj}-{seq}.ply")
                )

        obj_dir = os.path.join(root, obj, "evaluation")
        if not os.path.isdir(obj_dir):
            continue
        for seq in os.listdir(obj_dir):
            if not os.path.isdir(os.path.join(obj_dir, seq)):
                continue
            ply_file = os.path.join(obj_dir, seq, "scene/integrated_cropped.ply")
            if os.path.exists(ply_file):
                shutil.copy(ply_file, os.path.join(models_dir, f"test-{obj}-{seq}.ply"))

    # make reference and test directories

    # if already exists, remove it
    if os.path.exists(os.path.join(tgt, "reference")):
        shutil.rmtree(os.path.join(tgt, "reference"))
    if os.path.exists(os.path.join(tgt, "test")):
        shutil.rmtree(os.path.join(tgt, "test"))

    ref_dir = os.path.join(tgt, "reference")
    test_dir = os.path.join(tgt, "test")

    # handle reference and test split

    for obj in objs:
        obj_dir = os.path.join(root, obj)

        if not os.path.isdir(obj_dir):
            continue

        reference_dir = os.path.join(obj_dir, "reference")
        evaluation_dir = os.path.join(obj_dir, "evaluation")

        # make obj dir under reference and test
        os.makedirs(os.path.join(ref_dir, obj), exist_ok=True)
        os.makedirs(os.path.join(test_dir, obj), exist_ok=True)

        # handle reference first
        ref_seqs = os.listdir(reference_dir)
        handle_seq(ref_seqs, reference_dir, ref_dir, obj)

        # handle test
        test_seqs = os.listdir(evaluation_dir)
        handle_seq(test_seqs, evaluation_dir, test_dir, obj)


if __name__ == "__main__":
    main("data/moped", "data/moped_preprocessed")
