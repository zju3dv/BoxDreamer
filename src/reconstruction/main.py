from src.reconstruction.base import BaseReconstructor

# from src.reconstruction.mast3r import MASt3RReconstructor
from src.reconstruction.dust3r import DUSt3RReconstructor
from src.reconstruction.colmap import COLMAPReconstructor

import hydra
from omegaconf import DictConfig
import os
import glob
import sys
import shutil
from pytorch3d.io import save_ply
import argparse
import yaml
from src.reconstruction.dataset import get_dl
import omegaconf
import tqdm
import copy
import loguru
import traceback
import open3d as o3d
import numpy as np
import trimesh

ycbv_id_model = {
    "002": "002_master_chef_can",
    "003": "003_cracker_box",
    "004": "004_sugar_box",
    "005": "005_tomato_soup_can",
    "006": "006_mustard_bottle",
    "007": "007_tuna_fish_can",
    "008": "008_pudding_box",
    "009": "009_gelatin_box",
    "010": "010_potted_meat_can",
    "011": "011_banana",
    "019": "019_pitcher_base",
    "021": "021_bleach_cleanser",
    "024": "024_bowl",
    "025": "025_mug",
    "035": "035_power_drill",
    "036": "036_wood_block",
    "037": "037_scissors",
    "040": "040_large_marker",
    "051": "051_large_clamp",
    "052": "052_extra_large_clamp",
    "061": "061_foam_brick",
}
ycbv_model_root = "data/ycbv"
onepose_model_root = "data/onepose"
linemod_model_root = "data/lm"


def get_reconstructor(name: str) -> BaseReconstructor:
    if name == "mast3r":
        raise ValueError("MASt3R is not supported")
    elif name == "vggsfm":
        raise ValueError("VGGSFM is not supported")
        return VGGSFMReconstructor
    elif name == "dust3r":
        return DUSt3RReconstructor
    elif name == "colmap":
        return COLMAPReconstructor
    else:
        raise ValueError(f"Reconstructor {name} not found")
    # todo:
    # Colmap, ACE0, MOGE(for monocular setting), Detector-free SFM, Spann3r, Fast3R


def run(obj_data, reconstructor: BaseReconstructor):
    obj = obj_data[0]["cat"][0]
    reconstructor.set_processed_data(obj_data)

    try:
        path = reconstructor.run()
    except Exception as e:
        loguru.logger.warning(f"Failed to reconstruct {obj}")
        print(e)
        loguru.logger.error(traceback.format_exc())
        reconstructor.reset_data()
        return "none"

    reconstructor.reset_data()
    return path


def convert_ply_to_xyz(input_ply_file, output_xyz_file):
    mesh = trimesh.load(input_ply_file, file_type="ply")
    if mesh.is_empty:
        print(f"Empty mesh {input_ply_file}")
        return

    vertices = np.array(mesh.vertices)

    np.savetxt(output_xyz_file, vertices, fmt="%.6f", delimiter=" ")


def to_same_format(org_file, tgt_format):
    # this method is used to convert the pts file to tgt_format
    # tgt_format: ply, obj, glb, xyz
    # org_file: mostly is ply
    if tgt_format == "ply":
        return org_file  # no need to convert
    elif tgt_format == "xyz":
        # convert to xyz
        convert_ply_to_xyz(org_file, f"{org_file.split('.')[0]}.xyz")
        return f"{org_file.split('.')[0]}.xyz"
    else:
        raise ValueError(f"Format {tgt_format} not supported")


if __name__ == "__main__":
    # get dataset name from command line

    parser = argparse.ArgumentParser(description="dataset.py")
    parser.add_argument("--dataset", type=str, default="LINEMOD", help="dataset name")
    parser.add_argument(
        "--reconstructor", type=str, default="DUSt3R", help="reconstructor name"
    )
    parser.add_argument(
        "--ref_suffix",
        type=str,
        default="null",
        help="reference databse suffix (default without suffix, use full database)",
    )

    args = parser.parse_args()

    reconstructor_name = args.reconstructor
    # to lower case
    reconstructor_name = reconstructor_name.lower()

    # load yaml config as DictConfig
    with open("src/reconstruction/configs/data.yaml") as f:
        cfgs = omegaconf.OmegaConf.load(f)

    # load reconstructor config
    with open("src/reconstruction/configs/reconstructor.yaml") as f:
        recon_cfgs = omegaconf.OmegaConf.load(f)

    if args.ref_suffix != "null":
        # update the reference database
        cfgs[args.dataset]["config"]["reference_suffix"] = args.ref_suffix
    dl = get_dl(cfgs, args.dataset)

    # log data length
    print(len(dl))

    # tracverse the dataset and calculate each objs files
    # for each obj, call the reconstructor
    objs = {}
    sum_imgs = 0
    last_obj = None
    # init reconstructor
    reconstructor = get_reconstructor(reconstructor_name)(
        recon_cfgs[reconstructor_name + "_cfg"]["method"],
        recon_cfgs[reconstructor_name + "_cfg"]["weight"],
        recon_cfgs[reconstructor_name + "_cfg"]["config"],
    )
    obj_data = []
    debug_mode = False
    for data in dl:
        sum_imgs += 1
        if data["cat"][0] not in objs:
            objs[data["cat"][0]] = 1
            if last_obj is None:
                last_obj = data["cat"][0]
            else:
                # means first obj is done, because no shuffle applied
                loguru.logger.info(
                    f"Reconstructing {last_obj} with {objs[last_obj]} images"
                )
                # call the reconstructor
                if debug_mode:
                    # ask user whether to continue
                    print(
                        f"Continue with {last_obj} with {objs[last_obj]} images? (y/n)"
                    )
                    user_input = input()
                    if user_input.lower() == "y":
                        run(obj_data, reconstructor)
                    else:
                        pass
                else:
                    data_0 = copy.deepcopy(obj_data[0])
                    # log key
                    if data_0["dataset"][0] == "ycbv":
                        loguru.logger.info(
                            f"Reconstructing {ycbv_id_model[data_0['cat'][0]]} with {objs[data_0['cat'][0]]} images"
                        )
                        dump_path = os.path.join(
                            ycbv_model_root,
                            f"models_{reconstructor_name}_{objs[data_0['cat'][0]]}",
                            f"{ycbv_id_model[data_0['cat'][0]]}",
                        )
                        os.makedirs(dump_path, exist_ok=True)
                    elif data_0["dataset"][0] == "onepose":
                        loguru.logger.info(
                            f"Reconstructing {data_0['cat'][0]} with {objs[data_0['cat'][0]]} images"
                        )
                        dump_path = os.path.join(
                            onepose_model_root,
                            f"models_{reconstructor_name}_{objs[data_0['cat'][0]]}",
                            f"{data_0['cat'][0]}",
                        )
                        os.makedirs(dump_path, exist_ok=True)
                    elif data_0["dataset"][0] == "linemod":
                        loguru.logger.info(
                            f"Reconstructing {data_0['cat'][0]} with {objs[data_0['cat'][0]]} images"
                        )
                        dump_path = os.path.join(
                            linemod_model_root,
                            f"models_{reconstructor_name}_{objs[data_0['cat'][0]]}",
                            f"{data_0['cat'][0]}",
                        )
                        os.makedirs(dump_path, exist_ok=True)
                    else:
                        raise ValueError("Not supported dataset")
                    obj_path = run(obj_data, reconstructor)
                    if obj_path != "none":
                        # check if the file exists
                        model_path = data_0["model_path"]
                        if model_path != "none" and model_path[0][0] != "none":
                            model_name = model_path[0][0].split("/")[-1]
                        else:
                            model_name = "model.ply"

                        model_prefix = model_name.split(".")[0]
                        # rename the obj file
                        os.rename(
                            obj_path,
                            obj_path.replace(
                                obj_path.split("/")[-1], f"{model_prefix}.ply"
                            ),
                        )
                        obj_path = obj_path.replace(
                            obj_path.split("/")[-1], f"{model_prefix}.ply"
                        )
                        obj_path = to_same_format(obj_path, model_name.split(".")[-1])

                        obj_file_name = os.path.basename(obj_path)
                        if os.path.exists(dump_path + f"/{obj_file_name}"):
                            loguru.logger.warning(
                                f"File {obj_file_name} exists, removing"
                            )
                            os.remove(dump_path + f"/{obj_file_name}")

                        shutil.move(obj_path, dump_path)

                obj_data = []

                last_obj = data["cat"][0]  # set to current obj
        else:
            objs[data["cat"][0]] += 1

        obj_data.append(copy.deepcopy(data))

        # handle the last obj
        # check have next data or not
        if sum_imgs == len(dl):
            loguru.logger.info(
                f"Reconstructing {last_obj} with {objs[last_obj]} images"
            )
            data_0 = copy.deepcopy(obj_data[0])
            if data_0["dataset"][0] == "ycbv":
                loguru.logger.info(
                    f"Reconstructing {ycbv_id_model[data_0['cat'][0]]} with {objs[data_0['cat'][0]]} images"
                )
                dump_path = os.path.join(
                    ycbv_model_root,
                    f"models_{reconstructor_name}_{objs[data_0['cat'][0]]}",
                    f"{ycbv_id_model[data_0['cat'][0]]}",
                )
                os.makedirs(dump_path, exist_ok=True)
            elif data_0["dataset"][0] == "onepose":
                loguru.logger.info(
                    f"Reconstructing {data_0['cat'][0]} with {objs[data_0['cat'][0]]} images"
                )
                dump_path = os.path.join(
                    onepose_model_root,
                    f"models_{reconstructor_name}_{objs[data_0['cat'][0]]}",
                    f"{data_0['cat'][0]}",
                )
                os.makedirs(dump_path, exist_ok=True)
            elif data_0["dataset"][0] == "linemod":
                loguru.logger.info(
                    f"Reconstructing {data_0['cat'][0]} with {objs[data_0['cat'][0]]} images"
                )
                dump_path = os.path.join(
                    linemod_model_root,
                    f"models_{reconstructor_name}_{objs[data_0['cat'][0]]}",
                    f"{data_0['cat'][0]}",
                )
                os.makedirs(dump_path, exist_ok=True)
            else:
                raise ValueError("Not supported dataset")
            obj_path = run(obj_data, reconstructor)
            if obj_path != "none":
                # check if the file exists
                model_path = data_0["model_path"]

                if model_path != "none":
                    model_name = model_path[0][0].split("/")[-1]
                else:
                    model_name = "model.ply"

                model_prefix = model_name.split(".")[0]
                # rename the obj file
                os.rename(
                    obj_path,
                    obj_path.replace(obj_path.split("/")[-1], f"{model_prefix}.ply"),
                )
                obj_path = obj_path.replace(
                    obj_path.split("/")[-1], f"{model_prefix}.ply"
                )
                obj_path = to_same_format(obj_path, model_name.split(".")[-1])

                obj_file_name = os.path.basename(obj_path)
                if os.path.exists(dump_path + f"/{obj_file_name}"):
                    loguru.logger.warning(f"File {obj_file_name} exists, removing")
                    os.remove(dump_path + f"/{obj_file_name}")

                shutil.move(obj_path, dump_path)
