from src.reconstruction.base import BaseReconstructor
import numpy as np
import torch
import os
from typing import List, Tuple
import pytorch3d

import os
import torch
import tempfile
from contextlib import nullcontext
import sys

# 3rd party imports
# 3rd dir is under repo root, so we need to add it to sys.path
HERE_PATH = os.path.normpath(os.path.dirname(__file__))
MAST3R_REPO_PATH = os.path.normpath(os.path.join(HERE_PATH, '../../three'))
MAST3R_LIB_PATH = os.path.join(MAST3R_REPO_PATH, 'mast3r')
# check the presence of models directory in repo to be sure its cloned

if os.path.isdir(MAST3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, MAST3R_LIB_PATH)
else:
    raise ImportError(f"mast3r is not initialized, could not find: {MAST3R_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")

from mast3r.model import AsymmetricMASt3R
from mast3r.demo import get_3D_model_from_scene, SparseGAState
from mast3r.utils.misc import hash_md5
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

DUSt3R_REPO_PATH = os.path.normpath(os.path.join(HERE_PATH, '../../three'))
DUSt3R_LIB_PATH = os.path.join(DUSt3R_REPO_PATH, 'mast3r/dust3r')
# check the presence of models directory in repo to be sure its cloned
if os.path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, DUSt3R_LIB_PATH)
else:
    raise ImportError(f"dust3r is not initialized, could not find: {DUSt3R_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")


from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes

import copy
import matplotlib.pyplot as pl
from PIL import Image
class MASt3RReconstructor(BaseReconstructor):
    def __init__(self, methods='MASt3R', weights=None, config=None):
        super().__init__(methods)
        assert weights is not None, "Please specify the weights for the model"
        self.weights = weights
        
        self.cache_path = config.get('cache_path', './cache/mast3r_cache')
        self.device = config.get('device', 'cuda')
        self.lr1 = config.get('lr1', 0.07)
        self.lr2 = config.get('lr2', 0.014)
        self.niter1 = config.get('niter1', 500)
        self.niter2 = config.get('niter2', 200)
        self.optim_level = config.get('optim_level', 'refine+depth')
        self.shared_intrinsics = config.get('shared_intrinsics', True)
        self.matching_conf_thr = config.get('matching_conf_thr', 5.0)
        
        self.scenegraph_type = config.get('scenegraph_type', 'swin')
        self.winsize = config.get('winsize', 1)
        self.win_cyclic = config.get('win_cyclic', False)
    
    
    def _square_bbox(self, bbox: np.ndarray, padding: float = 0.1, astype=None) -> np.ndarray:
        """
        Compute a square bounding box with optional padding.

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
        
    def _preprare_before_run(self):
        # load images and use mask or bbox to mask or crop the images
        # and dump the images to the cache folder
        assert self.images is not None, "Please set the data first"
        
        for img_path, mask_path in zip(self.images, self.masks):
            print(img_path, mask_path)
            img = Image.open(img_path)
            
            # if mask ends with .png or .jpg, then it is a mask image
            # or if it is a bbox end with .txt (x0, y0, x1, y1)
            
            if mask_path.endswith('.png') or mask_path.endswith('.jpg'):
                mask = Image.open(mask_path)
                # log mask data range
                mask = mask.convert('L')
                # apply mask on the image
                img = Image.composite(img, Image.new('RGB', img.size, (0, 0, 0)), mask)
                
                # convert to RGBA 
                img = img.convert('RGBA')
                
                # make bbox from mask
                bbox = np.array(mask.getbbox())
                bbox = self._square_bbox(bbox, padding=0.1)
                # crop the image based on bbox
                img = img.crop(bbox)
                
            elif mask_path.endswith('.txt'):
                bbox = np.loadtxt(mask_path)
                # bbox = [int(x) for x in bbox]
                try:
                    bbox = self._square_bbox(bbox, padding=0.1, astype=int)
                    img = img.crop(bbox)
                except:
                    # bbox if x0, y0, w, h format
                    new_bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    new_bbox = self._square_bbox(new_bbox, padding=0.1, astype=int)
                    img = img.crop(new_bbox)
            else:
                raise ValueError("Invalid mask or bbox file")
            
            # create cache folder first
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            
            img.save(os.path.join(self.cache_path, os.path.basename(img_path)))
        
        # returen new image paths
        return [os.path.join(self.cache_path, os.path.basename(img_path)) for img_path in self.images]

    def run(self):
        assert self.images is not None, "Please set the data first"

        # run the model
        model = AsymmetricMASt3R.from_pretrained(self.weights).to('cuda')
        self.chkpt_tag = hash_md5(self.weights)
        
        img_paths = self._preprare_before_run()

        imgs = load_images(img_paths, size=224)
        if len(imgs) == 0:
            raise ValueError("No images found in the folder")
        elif len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
            filelist = [img_paths[0], img_paths[0] + '_2']
        else:
            filelist = img_paths
        
        scene_graph_params = [self.scenegraph_type]
        if self.scenegraph_type in ["swin", "logwin"]:
            scene_graph_params.append(str(self.winsize))
        elif self.scenegraph_type == "oneref":
            scene_graph_params.append(str(0))
        if self.scenegraph_type in ["swin", "logwin"] and not self.win_cyclic:
            scene_graph_params.append('noncyclic')
        
        scene_graph = '-'.join(scene_graph_params)
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        
        other_cfgs = {'matching_conf_thr': self.matching_conf_thr}
        
        scene = sparse_global_alignment(filelist, pairs, self.cache_path,
                                    model, lr1=self.lr1, niter1=self.niter1, lr2=self.lr2, niter2=self.niter2, device=self.device,
                                    opt_depth='depth' in self.optim_level, shared_intrinsics=self.shared_intrinsics, **other_cfgs)
        scene_state = SparseGAState(scene, False, self.cache_path, self.cache_path + '/scene.glb')
        glb_file = get_3D_model_from_scene(False, scene_state=scene_state)
        print("GLB file saved at:", glb_file)
        
        self.pred_poses = scene.get_im_poses()
        self.pt3d = scene.get_sparse_pts3d()[0]
        # inverse the poses
        for i in range(len(self.pred_poses)):
            self.pred_poses[i] = torch.inverse(self.pred_poses[i])
        

        focal = scene.get_focals() # B
        principal = scene.get_principal_points() # Bx2
        
        # make intrinsic matrix
        intrinsics = torch.zeros(len(focal), 3, 3)
        intrinsics[:, 0, 0] = focal
        intrinsics[:, 1, 1] = focal
        intrinsics[:, 0, 2] = principal[:, 0]
        intrinsics[:, 1, 2] = principal[:, 1]
        intrinsics[:, 2, 2] = 1.0
        
        
        self.pred_intrinsics = intrinsics
        
        self.draw_3dbbox(img_paths)
        
        if self._check_gt_poses():
            self.align_coordinates()
            
        return True