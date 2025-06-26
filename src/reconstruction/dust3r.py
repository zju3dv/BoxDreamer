"""
Author: Yuanhong Yu
Date: 2025-03-13 20:52:54
LastEditTime: 2025-03-17 15:40:26
Description:

"""
from src.reconstruction.base import BaseReconstructor
import numpy as np
import torch
import os

import os
import torch
import sys
import time

# 3rd party imports
# 3rd dir is under repo root, so we need to add it to sys.path
HERE_PATH = os.path.normpath(os.path.dirname(__file__))

DUSt3R_REPO_PATH = os.path.normpath(os.path.join(HERE_PATH, "../../three"))
DUSt3R_LIB_PATH = os.path.join(DUSt3R_REPO_PATH, "dust3r")
# check the presence of models directory in repo to be sure its cloned
if os.path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, DUSt3R_LIB_PATH)
else:
    raise ImportError(
        f"dust3r is not initialized, could not find: {DUSt3R_LIB_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )


from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

import copy
import matplotlib.pyplot as pl
from PIL import Image
import trimesh


class DUSt3RReconstructor(BaseReconstructor):
    def __init__(self, methods="DUSt3R", weights=None, config=None):
        super().__init__(methods)
        assert weights is not None, "Please specify the weights for the model"
        self.weights = weights
        self.cache_path = config.get("cache_path", "./cache/dust3r_cache")
        self.cache_root = self.cache_path

    def _square_bbox(
        self, bbox: np.ndarray, padding: float = 0.1, astype=None
    ) -> np.ndarray:
        """Compute a square bounding box with optional padding.

        Args:
            bbox (np.ndarray): Bounding box in [x_min, y_min, x_max, y_max] format.
            padding (float, optional): Padding factor. Defaults to 0.0.
            astype (type, optional): Data type of the output array. Defaults to None.

        Returns:
            np.ndarray: Square bounding box in [x_min, y_min, x_max, y_max] format.
        """
        if bbox is None:
            return None
        if astype is None:
            astype = type(bbox[0])
        bbox = np.array(bbox)
        center = (bbox[:2] + bbox[2:]) / 2
        extents = (bbox[2:] - bbox[:2]) / 2
        size = max(extents) * (1 + padding)
        square_bbox = np.array(
            [center[0] - size, center[1] - size, center[0] + size, center[1] + size],
            dtype=astype,
        )
        return square_bbox

    def _hash_str(self, s):
        import hashlib

        return hashlib.md5(s.encode()).hexdigest()

    def _preprare_before_run(self):
        # load images and use mask or bbox to mask or crop the images
        # and dump the images to the cache folder

        # reset the cache folder based on the current timestamp(with hash) to avoid conflict
        self.cache_path = os.path.join(
            self.cache_root, self._hash_str(str(time.time()))
        )
        # create cache folder first
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        assert self.images is not None, "Please set the data first"
        if isinstance(self.images, torch.Tensor or np.ndarray):
            # this means the data has been loaded from the dataloader
            # directly dump the images to the cache folder
            paths = []
            # self.images shape is B, C, H, W
            bs = self.images.shape[0]
            for i in range(bs):
                img = self.images[i]
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
                # to uint8
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                # create cache folder first
                if not os.path.exists(self.cache_path):
                    os.makedirs(self.cache_path)
                img_path = os.path.join(self.cache_path, f"image_{i}.png")
                img.save(img_path)
                paths.append(img_path)

            return paths, None

        else:
            # if self.masks is None:
            # only load the images
            # directly dump the images to the cache folder
            paths = []
            for img_path in self.images:
                img = Image.open(img_path)
                # create cache folder first
                if not os.path.exists(self.cache_path):
                    os.makedirs(self.cache_path)
                img.save(os.path.join(self.cache_path, os.path.basename(img_path)))
                paths.append(os.path.join(self.cache_path, os.path.basename(img_path)))
            if self.masks is None:
                return paths, None
            else:
                mask_paths = []
                for mask_path in self.masks:
                    print(mask_path)
                    img = Image.open(mask_path)
                    # create cache folder first
                    if not os.path.exists(self.cache_path):
                        os.makedirs(self.cache_path)
                    img.save(os.path.join(self.cache_path, os.path.basename(mask_path)))
                    mask_paths.append(
                        os.path.join(self.cache_path, os.path.basename(mask_path))
                    )

                return paths, mask_paths
            # else:
            #     for img_path, mask_path in zip(self.images, self.masks):
            #         print(img_path, mask_path)
            #         img = Image.open(img_path)

            #         # if mask ends with .png or .jpg, then it is a mask image
            #         # or if it is a bbox end with .txt (x0, y0, x1, y1)

            #         if mask_path.endswith(".png") or mask_path.endswith(".jpg"):
            #             mask = Image.open(mask_path)
            #             # log mask data range
            #             mask = mask.convert("L")
            #             # apply mask on the image
            #             img = Image.composite(
            #                 img, Image.new("RGB", img.size, (0, 0, 0)), mask
            #             )

            #             # convert to RGBA
            #             img = img.convert("RGBA")

            #             # make bbox from mask
            #             bbox = np.array(mask.getbbox())
            #             bbox = self._square_bbox(bbox, padding=0.1)
            #             # crop the image based on bbox
            #             img = img.crop(bbox)

            #         elif mask_path.endswith(".txt"):
            #             bbox = np.loadtxt(mask_path)
            #             # bbox = [int(x) for x in bbox]
            #             try:
            #                 bbox = self._square_bbox(bbox, padding=0.1, astype=int)
            #                 img = img.crop(bbox)
            #             except:
            #                 # bbox if x0, y0, w, h format
            #                 new_bbox = [
            #                     bbox[0],
            #                     bbox[1],
            #                     bbox[0] + bbox[2],
            #                     bbox[1] + bbox[3],
            #                 ]
            #                 new_bbox = self._square_bbox(
            #                     new_bbox, padding=0.1, astype=int
            #                 )
            #                 img = img.crop(new_bbox)
            #         else:
            #             raise ValueError("Invalid mask or bbox file")

            #         # create cache folder first
            #         if not os.path.exists(self.cache_path):
            #             os.makedirs(self.cache_path)

            #         img.save(os.path.join(self.cache_path, os.path.basename(img_path)))

            #     # returen new image paths
            #     return [
            #         os.path.join(self.cache_path, os.path.basename(img_path))
            #         for img_path in self.images
            #     ]

    def run(self):
        assert self.images is not None, "Please set the data first"

        # run the model
        model = AsymmetricCroCo3DStereo.from_pretrained(self.weights).to(self.device)

        img_paths, _ = self._preprare_before_run()

        imgs = load_images(img_paths, size=224)
        if len(imgs) == 0:
            raise ValueError("No images found in the folder")
        elif len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]["idx"] = 1
            filelist = [img_paths[0], img_paths[0] + "_2"]
        else:
            filelist = img_paths

        pairs = make_pairs(
            imgs, scene_graph="complete", prefilter=None, symmetrize=True
        )

        output = inference(pairs, model, device=self.device, batch_size=1)

        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PointCloudOptimizer,
            optimize_pp=True,
        )
        if self._check_gt_poses():
            cam_2_world = torch.tensor(self.gt_poses, device=self.device)
            # inverse
            cam_2_world = torch.inverse(cam_2_world)
            scene.preset_pose(cam_2_world)
        if self.intinsics is not None:
            print(self.intinsics.shape)
            scene.preset_principal_point(self.intinsics[:, :2, 2].clone())
            scene.preset_focal(self.intinsics[:, 0, 0].clone())

        loss = scene.compute_global_alignment(
            init="known_poses", niter=300, schedule="cosine", lr=0.01
        )

        # get glb file
        glb_file = os.path.join(self.cache_path, "scene.glb")

        # scene.show()

        # get pts3d
        pt3d = scene.get_pts3d()
        mask = scene.get_masks()

        # pt3d's structure is a list of H, W, 3
        # convert to N, 3, where N = H * W * len(pt3d)
        all_pts3d = []
        color = []
        for pt, m, col in zip(pt3d, mask, imgs):
            # all_pts3d.append(pt.reshape(-1, 3).cpu().numpy())
            # apply mask
            all_pts3d.append(pt[m].reshape(-1, 3).detach().cpu().numpy())
            # log col type
            if self.device == "cuda":
                img = col["img"][0].cuda()
            else:
                img = col["img"][0]
            img = (img.permute(1, 2, 0) + 1) / 2

            color.append(
                (img[m].reshape(-1, 3) * 255).detach().cpu().numpy().astype(np.uint8)
            )

        # to N, 3 numpy array
        pt3d = np.concatenate(all_pts3d, axis=0)
        # dump pts3d to cache folder as ply file
        ply_file = os.path.join(self.cache_path, "scene.ply")
        # use trimesh to save the ply file
        mesh = trimesh.PointCloud(pt3d)
        # add color
        mesh.colors = np.concatenate(color, axis=0)

        mesh.export(ply_file)

        new_path = self._self_pruning(ply_file)

        # vis the 3d points
        # self.visualize_ply(ply_file)
        return new_path

    def real_run(self):
        assert self.images is not None, "Please set the data first"

        # run the model
        model = AsymmetricCroCo3DStereo.from_pretrained(self.weights).to(self.device)

        img_paths, mask_paths = self._preprare_before_run()

        imgs = load_images(img_paths, size=224)
        if mask_paths is not None:
            masks = load_images(mask_paths, size=224)
            self.masks = masks
        if len(imgs) == 0:
            raise ValueError("No images found in the folder")
        elif len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]["idx"] = 1
            filelist = [img_paths[0], img_paths[0] + "_2"]
        else:
            filelist = img_paths

        pairs = make_pairs(
            imgs, scene_graph="complete", prefilter=None, symmetrize=True
        )

        output = inference(pairs, model, device=self.device)
        if self.images.__len__() == 1:
            # monoocular mode
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.PairViewer,
            )
        else:
            scene = global_aligner(
                output,
                device=self.device,
                mode=GlobalAlignerMode.PointCloudOptimizer,
                optimize_pp=False,
            )

        if self.intinsics is not None:
            print(self.intinsics.shape)
            scene.preset_principal_point(self.intinsics[:, :2, 2].clone())
            scene.preset_focal(self.intinsics[:, 0, 0].clone())

        loss = scene.compute_global_alignment(
            init="mst", niter=300, schedule="cosine", lr=0.01
        )

        # get glb file
        glb_file = os.path.join(self.cache_path, "scene.glb")

        # scene.show()

        # get pts3d
        pt3d = scene.get_pts3d()
        mask = scene.get_masks()
        pred_poses = scene.get_im_poses()
        focal = scene.get_focals()
        pp = scene.get_principal_points()

        pred_intrinsics = torch.zeros((focal.shape[0], 3, 3), device=self.device)
        pred_intrinsics[:, 0, 0] = focal.squeeze()
        pred_intrinsics[:, 1, 1] = focal.squeeze()
        pred_intrinsics[:, 2, 2] = 1
        pred_intrinsics[:, :2, 2] = pp

        self.pred_poses = torch.inverse(pred_poses)
        self.pred_intrinsics = pred_intrinsics

        # pt3d's structure is a list of H, W, 3
        # convert to N, 3, where N = H * W * len(pt3d)
        all_pts3d = []
        color = []
        idx = 0
        for pt, m, col in zip(pt3d, mask, imgs):
            # all_pts3d.append(pt.reshape(-1, 3).cpu().numpy())
            # apply mask
            if self.masks is not None:
                seg = self.masks[idx]["img"][0].cpu().numpy()
                seg = np.transpose(seg, (1, 2, 0))
                # RGB to Gray
                seg = np.mean(seg, axis=2)
                seg = seg > 0.5
                m = m & torch.tensor(seg, device=self.device)

            all_pts3d.append(pt[m].reshape(-1, 3).detach().cpu().numpy())
            # log col type
            if self.device == "cuda":
                img = col["img"][0].cuda()
            else:
                img = col["img"][0]

            img = (img.permute(1, 2, 0) + 1) / 2

            color.append(
                (img[m].reshape(-1, 3) * 255).detach().cpu().numpy().astype(np.uint8)
            )
            if self.images.__len__() == 1:
                break
            idx += 1

        # to N, 3 numpy array
        pt3d = np.concatenate(all_pts3d, axis=0)
        self.pt3d = torch.tensor(pt3d, device=self.device)
        # dump pts3d to cache folder as ply file
        ply_file = os.path.join(self.cache_path, "scene.ply")
        # use trimesh to save the ply file
        mesh = trimesh.PointCloud(pt3d)
        # add color
        mesh.colors = np.concatenate(color, axis=0)

        mesh.export(ply_file)

        if self.masks is None:
            new_path = self._self_pruning(ply_file, True)
        else:
            new_path = ply_file
            new_path = self._to_object_coordinate(new_path)
        # vis the 3d points
        # self.visualize_ply(new_path)

        # exit()
        # return new_path

        self.draw_3dbbox(self.images)

        ret = {
            "poses": self.pred_poses,
            "intrinsics": self.pred_intrinsics,
            "ply_path": new_path,
        }

        return ret

    def visualize_ply(self, ply_path):
        """Visualize the PLY point cloud using trimesh.

        Args:
            ply_path (str): Path to the PLY file.
        """
        try:
            point_cloud = trimesh.load(ply_path)
            if isinstance(point_cloud, trimesh.Scene):
                # If loaded as a Scene, merge all geometries
                point_cloud = trimesh.util.concatenate(point_cloud.dump())
            point_cloud.show()
        except Exception as e:
            print(f"Failed to visualize PLY file: {e}")
