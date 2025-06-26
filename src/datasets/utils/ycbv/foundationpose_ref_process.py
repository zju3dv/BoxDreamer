"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:11:21
Description: foudationpose reference database parser

"""
import os
import glob
import shutil
from PIL import Image
import numpy as np
import open3d as o3d

root = "data/ycbv"

ref_path = os.path.join(root, "ref_views_4")

tgt_path = os.path.join(root, "ref_views_4_processed")

tgt_model_path = os.path.join(root, "models_ref4")

ycbv_id_obj_map = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}

# make tgt dir
if os.path.exists(tgt_path):
    shutil.rmtree(tgt_path)

os.makedirs(tgt_path, exist_ok=True)

for obj_id in os.listdir(ref_path):
    id = int(obj_id.split("_")[1])
    obj_name = ycbv_id_obj_map[id]

    # make obj dir
    obj_dir = os.path.join(tgt_path, obj_name)
    os.makedirs(obj_dir, exist_ok=True)

    # make a fake seq dir
    seq_dir = os.path.join(obj_dir, "0001")
    os.makedirs(seq_dir, exist_ok=True)

    # we need cp rgb. intrinsics, pose, box to the seq dir

    src_pose_dir = os.path.join(ref_path, obj_id, "cam_in_ob")
    # sort the files
    src_pose_files = sorted(glob.glob(os.path.join(src_pose_dir, "*.txt")))
    src_color_dir = os.path.join(ref_path, obj_id, "rgb")
    src_mask_dir = os.path.join(ref_path, obj_id, "mask")

    intrinsics_file = os.path.join(ref_path, obj_id, "K.txt")

    for idx, file in enumerate(src_pose_files):
        # shutil.copyfile(file, os.path.join(seq_dir, str(idx).zfill(6) + '-pose.txt'))
        # load pose (4*4) and write the inverse to file
        pose = np.loadtxt(file)
        # reshape to 4*4
        pose = pose.reshape(4, 4)
        # get the inverse
        pose_inv = np.linalg.inv(pose)
        # write to file
        with open(os.path.join(seq_dir, str(idx).zfill(6) + "-pose.txt"), "w") as f:
            for row in range(4):
                for col in range(4):
                    f.write(f"{pose_inv[row, col]} ")
                f.write("\n")

        rgb_file = file.replace("cam_in_ob", "rgb").replace("txt", "png")
        shutil.copyfile(
            rgb_file, os.path.join(seq_dir, str(idx).zfill(6) + "-color.png")
        )
        mask_file = file.replace("cam_in_ob", "mask").replace("txt", "png")

        # read mask and make bbox (x0, y0, x1, y1)
        mask = Image.open(mask_file)
        mask = mask.convert("L")
        bbox = mask.getbbox()
        # to numpy
        bbox = np.array(bbox)
        # write to file
        with open(os.path.join(seq_dir, str(idx).zfill(6) + "-box.txt"), "w") as f:
            f.write(" ".join([str(x) for x in bbox]))

        # copy intrinsics
        shutil.copyfile(
            intrinsics_file,
            os.path.join(seq_dir, str(idx).zfill(6) + "-intrinsics.txt"),
        )

    # make model dir
    model_dir = os.path.join(tgt_model_path, obj_name)
    os.makedirs(model_dir, exist_ok=True)

    src_model_path = os.path.join(ref_path, obj_id, "model/model.obj")
    xyz_model_path = os.path.join(model_dir, f"points.xyz")

    # obj to pcd and dump to xyz format file
    mesh = o3d.io.read_triangle_mesh(src_model_path)
    pcd = mesh.sample_points_poisson_disk(5000)
    o3d.io.write_point_cloud(xyz_model_path, pcd)

    print(f"Processed {obj_name}")

print("All done")
