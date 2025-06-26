"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:22:30
Description: Preprocess linemod-occlusion dataset

"""

import os
import numpy as np
import shutil
import json
import tqdm

# we need xxxxxx-pose.txt, xxxxxx-rgb.png, xxxxxx-box.txt
root = "data/lmo"  # linemod-occlusion dataset root
ob_id_to_names = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "water_pour",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
# bop19_meta = os.path.join(root, 'lmo/train_pbr/test_targets_bop19.json')

scene_meta = "data/lmo/test/000002/scene_gt_info.json"
scene_gt = "data/lmo/test/000002/scene_gt.json"
scene_meta = json.load(open(scene_meta))
scene_gt = json.load(open(scene_gt))

dump_dir = "data/lmo/test-preprocessed"
if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)
else:
    shutil.rmtree(dump_dir)
    os.makedirs(dump_dir)

for scene_id in tqdm.tqdm(
    scene_meta.keys(),
    total=len(scene_meta),
    desc="Processing linemod-occlusion dataset",
):
    # each scene has multiple objects
    for idx, obj in enumerate(scene_gt[scene_id]):
        obj_id = obj["obj_id"]
        obj_name = ob_id_to_names[obj_id]
        # if obj_name not exists in dump_dir, create it
        obj_dir = os.path.join(dump_dir, obj_name)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)

        rgb_path = os.path.join(root, "test/000002/rgb", f"{int(scene_id):06d}.png")
        tgt_rgb_path = os.path.join(obj_dir, f"{int(scene_id):06d}-color.png")

        camera_R_m2c = np.array(obj["cam_R_m2c"]).reshape(3, 3)
        camera_t_m2c = np.array(obj["cam_t_m2c"]).reshape(3, 1)

        # mm -> m
        camera_t_m2c /= 1000

        bbox = scene_meta[scene_id][idx]["bbox_obj"]

        pose_tgt_path = os.path.join(obj_dir, f"{int(scene_id):06d}-pose.txt")
        with open(pose_tgt_path, "w") as f:
            f.write(
                f"{camera_R_m2c[0, 0]} {camera_R_m2c[0, 1]} {camera_R_m2c[0, 2]} {camera_t_m2c[0, 0]}\n"
            )
            f.write(
                f"{camera_R_m2c[1, 0]} {camera_R_m2c[1, 1]} {camera_R_m2c[1, 2]} {camera_t_m2c[1, 0]}\n"
            )
            f.write(
                f"{camera_R_m2c[2, 0]} {camera_R_m2c[2, 1]} {camera_R_m2c[2, 2]} {camera_t_m2c[2, 0]}\n"
            )

        bbox_tgt_path = os.path.join(obj_dir, f"{int(scene_id):06d}-box.txt")
        with open(bbox_tgt_path, "w") as f:
            f.write(f"{bbox[0]}\n{bbox[1]}\n{bbox[2]}\n{bbox[3]}\n")

        # cp rgb to dump_dir

        os.system(f"cp {rgb_path} {tgt_rgb_path}")
