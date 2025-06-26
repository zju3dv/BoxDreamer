"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:19:38
Description:

"""
import hydra.core
import hydra.core.hydra_config
from objaverese_downloader import ObjaverseHandler
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import itertools


@hydra.main(config_path="../../configs/preprocess", config_name="objaverse_download")
def main(config: DictConfig):
    # convert to dict
    config = OmegaConf.to_container(config, resolve=True)
    handler = ObjaverseHandler(config)

    local_objs = handler.init_local_objs()  # local objs is a dict

    # handler.get_obj_alignment_annotations()
    # handler.get_lvis_annotations()
    # handler.get_annotations_metadata()
    sampled_objs = dict(itertools.islice(local_objs.items(), 0, 20))
    # list = handler.download()
    handler.set_render_objs(sampled_objs)
    handler.render_parallel()


# only for debug
def npy2glb(path):
    npy = np.load(path)

    print(npy.shape)

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)

    # dump to ply
    o3d.io.write_point_cloud(
        "/home/yuyuanhong/data/objaverse/rendered/sketchfab/5dd474a6-3cd8-536c-9fe5-3ae700789c03/point_cloud/000.ply",
        pcd,
    )


def exr2png(path):
    pass


if __name__ == "__main__":
    main()
    # npy2glb("/home/yuyuanhong/data/objaverse/rendered/sketchfab/5dd474a6-3cd8-536c-9fe5-3ae700789c03/point_cloud/pt3d.npy")
