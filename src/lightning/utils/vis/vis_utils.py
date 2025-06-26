import os
import math
import gc
import cv2
import matplotlib.figure
import torch
import imageio
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
from PIL import Image

from wis3d import Wis3D as Vis3D
from src.utils.log import INFO
from .plot_utils import *
from .mesh_utils import *
import natsort
from pytorch3d.renderer import (
    PerspectiveCameras,
    RayBundle)
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection, opencv_from_cameras_projection
from pytorch3d.vis.plotly_vis import plot_scene
from src.models.utils.rays import *
import plotly
import wandb
import open3d as o3d
cmap = plt.cm.get_cmap('jet')
class VisUtils:
    """
    Visualization Utilities Class.

    This class provides methods to handle and visualize various types of data,
    including bounding boxes, RGB images, camera rays, depth maps, reprojections,
    3D scenes, and video generation.
    """

    def __init__(self, config) -> None:
        """
        Initialize the VisUtils instance.

        Args:
            config: Configuration object containing visualization settings.
        """
        self.config = config
        self.val_results = {}
        self.data = []
        self.metrics = []
        self.dataloader = set()

        INFO(f"VisUtils initialized with vis types: {self.config.vis_types}")

    def reset(self):
        """
        Reset the visualization data and results.
        """
        self.val_results = {}
        self.data.clear()
        self.metrics.clear()
        self.dataloader.clear()
        gc.collect()

    def add_data(self, batch, dataloader_id: int = 0):
        """
        Add a batch of data for visualization.

        Args:
            batch: The data batch to add.
            dataloader_id (int): Identifier for the dataloader source.
        """
        if dataloader_id not in self.dataloader:
            # First time adding data from this dataloader
            self.dataloader.add(dataloader_id)
            self.data.append(batch)
            return
        elif len(self.data) >= self.config.save_n_batches:
            return
        else:
            self.data.append(batch)
            
    def add_metrics(self, metrics, dataloader_id: int = 0):
        '''
        Add metrics for visualization.
        Args:
            metrics(dict): The metrics to add.
            dataloader_id (int): Identifier for the dataloader source.
        '''
        
        if len(self.metrics) >= self.config.save_n_batches and len(self.metrics) == len(self.data):
            return
        else:
            self.metrics.append(metrics)

    def add_val_result(self, key: str, value):
        """
        Add a validation result to the results dictionary.

        Args:
            key (str): The key under which to store the result.
            value: The value to store, can be various types (e.g., images, scenes).
        """
        # Convert matplotlib figures to numpy arrays
        if isinstance(value, plt.Figure):
            value.canvas.draw()
            width, height = value.get_size_inches() * value.get_dpi()
            value = np.frombuffer(value.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # Convert trimesh.Scene to a single mesh
        if isinstance(value, trimesh.Scene):
            value = value.dump(concatenate=True)

        # Append or create a new entry in val_results
        if key in self.val_results:
            if isinstance(value, list):
                self.val_results[key].extend(value)
            else:
                self.val_results[key].append(value)
        else:
            self.val_results[key] = value if isinstance(value, list) else [value]

    def visualize_results(self):
        """
        Visualize all collected results using multiple processors.
        """
        for bid, batch in enumerate(self.data):
            batch_size = batch['images'].shape[0]

            for b in range(batch_size):
                query_idx = batch['query_idx'][b]

                # Visualize bounding boxes
                if 'bbox' in self.config.vis_types:
                    self._visualize_bbox(batch, b, query_idx)

                # Visualize RGB reconstructions
                if "rgb_reconstruction" in self.config.vis_types:
                    self._visualize_rgb_reconstruction(batch, b, query_idx)

                # Visualize camera rays
                if 'ray' in self.config.vis_types:
                    self._visualize_camera_rays(batch, b, query_idx)

            # Visualize depth, reprojection, scene, and video if specified
            if 'depth' in self.config.vis_types:
                self._visualize_depth(batch)

            if 'reprojection' in self.config.vis_types:
                reproj_fig = draw_reprojection_pair(batch, visual_color_type='conf')
                self.add_val_result('reprojection', reproj_fig)

            if 'scene' in self.config.vis_types:
                scenes = self.create_mesh(batch)
                for idx, scene in enumerate(scenes):
                    dump_path = Path(f"scene_{bid}_{idx}.glb")
                    scene.export(dump_path)

            if 'video' in self.config.vis_types:
                if self.data.__len__() == self.metrics.__len__():
                    process_video_frames(batch, self.metrics[bid], bid)
                else:
                    process_video_frames(batch, None, bid)
                    
            if "pt3d_scene" in self.config.vis_types:
                self._visualize_pt3d_scene(batch)
                
            if "bbox_feat" in self.config.vis_types:
                self._visualize_bbox_feat(batch)

        INFO("Visualizing results done!")
        
    def _visualize_bbox_feat(self, batch):
        """
        Visualize bounding box features.

        Args:
            batch: The current data batch.
        """
        B, T, C, H, W = batch["pred_bbox"].shape
        if C == 16:
            bbox_feat = batch["pred_bbox"].clone().permute(0, 1, 3, 4, 2).reshape(B, T, H, W, 8, 2)
            gt_bbox_feat = batch["bbox_feat"].clone().permute(0, 1, 3, 4, 2).reshape(B, T, H, W, 8, 2)
        elif C == 24:
            bbox_feat = batch["pred_bbox"].clone().permute(0, 1, 3, 4, 2).reshape(B, T, H, W, 8, 3)
            gt_bbox_feat = batch["bbox_feat"].clone().permute(0, 1, 3, 4, 2).reshape(B, T, H, W, 8, 3)
        elif C == 8:
            bbox_feat = batch["pred_bbox"].clone().permute(0, 1, 3, 4, 2) # B T H W 8
            gt_bbox_feat = batch["bbox_feat"].clone().permute(0, 1, 3, 4, 2) # B T H W 8
            

        query_idx = batch['query_idx'].clone()
        query_feat = bbox_feat[np.arange(B), query_idx]
        gt_query_feat = gt_bbox_feat[np.arange(B), query_idx]
        images = batch['images'].clone()
        for b in range(B):
            batch_query_feat = query_feat[b].unsqueeze(0)
            batch_query_rgb = images[b][query_idx[b]].unsqueeze(0)
            batch_gt_query_feat = gt_query_feat[b].unsqueeze(0)
            if C == 16:
                vis = draw_bbox_voting_as_heatmap(batch_query_feat.clone(), batch_query_rgb.clone())
                self.add_val_result(f"bbox_voting_heatmap", vis)
                vis = draw_bbox_voting_as_heatmap(batch_gt_query_feat.clone(), batch_query_rgb.clone())
                self.add_val_result(f"bbox_voting_heatmap_gt", vis)
            elif C == 8:
                vis = draw_bbox_heatmap(batch_query_feat.clone(), batch_query_rgb.clone()) 
                self.add_val_result(f"bbox_heatmap", vis)
                vis = draw_bbox_heatmap(batch_gt_query_feat.clone(), batch_query_rgb.clone())
                self.add_val_result(f"bbox_heatmap_gt", vis)
            elif C == 24:
                raise NotImplementedError("Not implemented yet")
        
        return     
        
    def _visualize_pt3d_scene(self, batch):
        """
        Visualize 3D scene using PyTorch3D.

        Args:
            batch: The current data batch.
            """
        # get in/extrinsics -> make perspective cameras -> (optional) make ray bundle -> plot
        B,T,C,H,W = batch['images'].shape
        figs = []
        # per batch per scene
        for b in range(B):
            query_idx = batch['query_idx'][b]
            gt_poses = batch['original_poses'][b].clone()
            pred_poses = batch['pred_poses'][b].clone()
            intrinsics = batch['non_ndc_intrinsics'][b].clone()
            scale = batch['scale'][b].clone()
            # recover pred_poses scale
            pred_poses[:, :3, 3] *= scale
            coordinate_transform = batch['coordinate_transform'][b]
            pred_poses = pred_poses @ coordinate_transform
            # make cameras(from opencv to pt3d)
            gt_R = gt_poses[:, :3, :3]
            gt_T = gt_poses[:, :3, 3]
            pred_R = pred_poses[:, :3, :3]
            pred_T = pred_poses[:, :3, 3]

            
            gt_cameras = cameras_from_opencv_projection(
                gt_R, gt_T, intrinsics, torch.tensor([H, W]).expand(T, 2)
            )
            pred_cameras = cameras_from_opencv_projection(
                pred_R, pred_T, intrinsics, torch.tensor([H, W]).expand(T, 2)
                )
            gt_rays = cameras_to_rays(gt_cameras[-1], None, num_patches_x=16, num_patches_y=16)
            pred_rays = cameras_to_rays(pred_cameras[-1], None, num_patches_x=16, num_patches_y=16)
            gt_rays_bundle = gt_rays.to_ray_bundle()
            pred_rays_bundle = pred_rays.to_ray_bundle()

            fig = self._create_plotly_cameras_visualization(gt_cameras, pred_cameras, T, gt_rays=gt_rays_bundle, pred_rays=pred_rays_bundle)
            # convert to matplotlib for vis
            fig.show()
            fig = wandb.Plotly(fig)
            self.add_val_result(f"pt3d_scene_{b}", fig)
            
        
    def _create_plotly_cameras_visualization(self, cameras_gt, cameras_pred, num, gt_rays=None, pred_rays=None)->plotly.graph_objs.Figure:
        num_frames = cameras_gt.R.shape[0]
        name = f"Vis {num - 1} Ref & Query Cameras"
        camera_scale = 0.05

        # Cameras_pred is already a 2D list of unbatched cameras
        # But cameras_gt is a 1D list of batched cameras
        scenes = {f"Vis {num - 1} Ref & Query Cameras": {}}
        for i in range(num_frames - 1):
            scenes[name][f"Ref Camera {i}"] = PerspectiveCameras(
                R=cameras_gt.R[i].unsqueeze(0), T=cameras_gt.T[i].unsqueeze(0)
            )
        scenes[name][f"Query Camera(pred)"] = PerspectiveCameras(
            R=cameras_pred.R[-1].unsqueeze(0), T=cameras_pred.T[-1].unsqueeze(0)
        )
        scenes[name][f"Query Camera(GT)"] = PerspectiveCameras(
            R=cameras_gt.R[-1].unsqueeze(0), T=cameras_gt.T[-1].unsqueeze(0)
            )
        
        if gt_rays is not None and pred_rays is not None:
            scenes[name][f"Query Ray Bundle(GT)"] = gt_rays
            scenes[name][f"Query Ray Bundle(pred)"] = pred_rays

        fig = plot_scene(
            scenes,
            camera_scale=camera_scale,
        )
        fig.update_scenes(aspectmode="data")
        fig.update_layout(height=800, width=800)

        for i in range(num_frames - 1):
            fig.data[i].line.color = matplotlib.colors.to_hex(cmap(0.0))
            fig.data[i].line.width = 4

        fig.data[-1].line.color = matplotlib.colors.to_hex(cmap(1.0))
        fig.data[-1].line.width = 4
        fig.data[-2].line.dash = "dash"
        fig.data[-2].line.color = matplotlib.colors.to_hex(cmap(1.0))
        fig.data[-2].line.width = 4    


        return fig
    
    def _perturb_pose(self, pose: np.ndarray, t_xyz: tuple, r_xyz: tuple):
        '''
            pose: [4, 4] numpy array
            t_xyz: [3] # means translation in xyz, unit in meters
            r_xyz: [3] # means rotation in xyz, unit in degrees
        '''
        
        # translation
        pose[:3, 3] += np.array(t_xyz)
        # rotation
        r_xyz = np.radians(r_xyz)
        R = pose[:3, :3]
        R_x = np.array([[1, 0, 0], [0, np.cos(r_xyz[0]), -np.sin(r_xyz[0])], [0, np.sin(r_xyz[0]), np.cos(r_xyz[0])]])
        R_y = np.array([[np.cos(r_xyz[1]), 0, np.sin(r_xyz[1])], [0, 1, 0], [-np.sin(r_xyz[1]), 0, np.cos(r_xyz[1])]])
        R_z = np.array([[np.cos(r_xyz[2]), -np.sin(r_xyz[2]), 0], [np.sin(r_xyz[2]), np.cos(r_xyz[2]), 0], [0, 0, 1]])
        R = R @ R_x @ R_y @ R_z
        
        pose[:3, :3] = R
        return pose
        
    
        

    def _visualize_bbox(self, batch, b, query_idx):
        """
        Visualize bounding boxes for a single instance in the batch.

        Args:
            batch: The current data batch.
            b (int): Index of the item in the batch.
            query_idx (int): Query index for the current item.
        """
        if 'bbox_figs' not in batch:
            batch['bbox_figs'] = []
        if 'bbox_3d_original' not in batch:
            bbox_3d = self._load_and_compute_bbox(batch, b)
        else:
            bbox_3d = batch['bbox_3d_original'][b][query_idx]

        pred_pose = batch['pred_poses'][b][query_idx].clone().numpy()
        coordinate_transform = batch['coordinate_transform'][b].clone().numpy()
        scale = batch['scale'][b][query_idx].numpy()
        pred_pose[:3, 3] *= scale
        pred_pose = pred_pose @ coordinate_transform
        
        proj_bbox = reproj(batch['original_intrinsics'][b][query_idx].numpy(), pred_pose, bbox_3d)
        proj_bbox_gt = reproj(batch['original_intrinsics'][b][query_idx].numpy(),
                              batch['original_poses'][b][query_idx].numpy(), bbox_3d)

        original_image_path = batch['original_images'][query_idx][b]
        original_image = np.array(Image.open(original_image_path).convert('RGB'))
        
        # proj_bbox = reproj(batch['non_ndc_intrinsics'][b][query_idx].numpy(), pred_pose, bbox_3d)
        # proj_bbox_gt = reproj(batch['non_ndc_intrinsics'][b][query_idx].numpy(),
        #                       batch['original_poses'][b][query_idx].numpy(), bbox_3d)
        
        # image = batch['images'][b][query_idx].clone().permute(1, 2, 0).numpy()
        # image = (image * 255).astype(np.uint8)
        # image = np.array(Image.fromarray(image).convert('RGB'))

        bbox_fig = draw_3d_box(original_image, proj_bbox_gt)
        try:
            bbox_fig = draw_3d_box(bbox_fig, proj_bbox, color='b')
        except:
            pass
        # try:
        #     bbox_fig = draw_coordinate_on_image(bbox_fig, pred_pose, batch['original_intrinsics'][b][query_idx].numpy(), (255, 0, 0))
        #     bbox_fig = draw_coordinate_on_image(bbox_fig, batch['original_poses'][b][query_idx].numpy(), batch['original_intrinsics'][b][query_idx].numpy(), (0, 255, 0))
        # except:
        #     pass
        
        batch['bbox_figs'].append(bbox_fig)
        self.add_val_result('bbox', bbox_fig)

        if "regression_boxes" in batch:
            if 'pred_bbox_vis' not in batch:
                batch['pred_bbox_vis'] = []
            self._visualize_regression_boxes(batch, b, query_idx)

    def _visualize_regression_boxes(self, batch, b, q_idx):
        """
        Visualize regression bounding boxes for all images in the batch.

        Args:
            batch: The current data batch.
            b (int): Index of the item in the batch.
        """
        # Initialize a list to store visualizations for all images
        pred_bbox_vis = []
        # Iterate over all images in the batch for the given index `b`
        for query_idx in range(len(batch['images'][b])):
            # Clone and preprocess the current image
            croped_images = batch['images'][b][query_idx].clone().permute(1, 2, 0).numpy()
            bbox_proj_crop = batch['bbox_proj_crop'][b][query_idx].clone().numpy()
            regressed_bbox = batch['regression_boxes'][b][query_idx].clone().numpy()
            bbox_scale = croped_images.shape[:2]
            croped_images = (croped_images * 255).astype(np.uint8)
            # Convert to continuous space
            croped_images = np.ascontiguousarray(croped_images)

            # Scale bbox_proj_crop and regressed_bbox to image dimensions
            bbox_proj_crop = ((bbox_proj_crop + 1) / 2) * bbox_scale
            regressed_bbox = ((regressed_bbox + 1) / 2) * bbox_scale

            # Draw the projected and regressed bounding boxes
            croped_bbox_fig = draw_3d_box(croped_images, bbox_proj_crop)
            croped_bbox_fig = draw_3d_box(croped_bbox_fig, regressed_bbox, color='b')

            # Store the visualization
            pred_bbox_vis.append(croped_bbox_fig)

        # Add the visualizations to the validation results
        self.add_val_result('croped_bbox', pred_bbox_vis[q_idx])
        batch['pred_bbox_vis'].append(pred_bbox_vis)

    def _load_and_compute_bbox(self, batch, b):
        """
        Load a 3D model and compute its bounding box.

        Args:
            batch: The current data batch.
            b (int): Index of the item in the batch.

        Returns:
            np.ndarray: 3D bounding box corners.
        """
        model_path = batch['model_path'][b]
        assert os.path.exists(model_path), f"Model path {model_path} does not exist!"

        mesh = trimesh.load(model_path)
        pt3ds = np.array(mesh.vertices)

        if batch["dataset"][0] == "co3d" or batch["dataset"][0] == "moped":
            centroid = np.mean(pt3ds, axis=0)
            centered_pt3ds = pt3ds - centroid
            cov_matrix = np.cov(centered_pt3ds, rowvar=False)
            eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
            sorted_indices = np.argsort(eigen_values)[::-1]
            eigen_vectors = eigen_vectors[:, sorted_indices]
            rotation_matrix = eigen_vectors
            rotated_pt3ds = centered_pt3ds @ rotation_matrix
            bbox_3d = get_3d_bbox_from_pts(rotated_pt3ds)
            bbox_3d = bbox_3d @ rotation_matrix.T
            bbox_3d += centroid
        else:
            bbox_3d = get_3d_bbox_from_pts(pt3ds)

        return bbox_3d

    def _visualize_rgb_reconstruction(self, batch, b, query_idx):
        """
        Visualize RGB reconstructions for a single instance.

        Args:
            batch: The current data batch.
            b (int): Index of the item in the batch.
            query_idx (int): Query index for the current item.
        """
        gt_image = batch['images'][b][query_idx].numpy().transpose(1, 2, 0)
        pred_image = batch['pred_images'][b][query_idx].numpy().transpose(1, 2, 0)

        self.add_val_result('rgb_gt', gt_image)
        self.add_val_result('rgb_pred', pred_image)

    def _visualize_camera_rays(self, batch, b, query_idx):
        """
        Visualize camera rays for a single instance.

        Args:
            batch: The current data batch.
            b (int): Index of the item in the batch.
            query_idx (int): Query index for the current item.
        """
        gt_ray = batch['camera_rays'][b][query_idx].permute(1, 2, 0)
        pred_ray = batch['pred_camera_rays'][b][query_idx].permute(1, 2, 0)

        self.add_val_result('ray_gt', visualize_direction_as_color(gt_ray))
        self.add_val_result('ray_pred', visualize_direction_as_color(pred_ray))

    def _visualize_depth(self, batch):
        """
        Visualize depth maps and confidence maps.

        Args:
            batch: The current data batch.
        """
        depth_gt, _ = visualize_results(batch, is_gt=True)
        depth_pred, conf_pred = visualize_results(batch)

        self.add_val_result('depth_gt', depth_gt)
        self.add_val_result('depth', depth_pred)
        self.add_val_result('conf', conf_pred)

    def create_mesh(self, batch) -> List[trimesh.Scene]:
        """
        Create 3D meshes from the batch data.

        Args:
            batch: The current data batch.

        Returns:
            List[trimesh.Scene]: A list of 3D scenes.
        """
        B, N = batch['images'].shape[0], batch['images'].shape[1]
        pose = batch['pred_poses'].clone().numpy()        # [B, N, 4, 4] object to camera
        coordinate_transform = batch['coordinate_transform'].clone().numpy()  # [B, 4, 4]
        scales = batch['scale'].numpy()           # [B, N]
        scaled_poses = pose.copy()
        scaled_poses[:, :, :3, 3] *= scales
        scaled_poses = scaled_poses @ coordinate_transform[:, None]
        gt_pose = batch['original_poses'].numpy() # [B, N, 4, 4]
        intrinsics = batch['original_intrinsics'].numpy().astype(np.float32) # [B, N, 3, 3]
        dataset = batch['dataset'][0]
        scenes = []

        for b in range(B):
            scene = trimesh.Scene()
            model_path = batch['model_path'][b]
            assert os.path.exists(model_path), f"Model path {model_path} does not exist!"
            mesh = trimesh.load(model_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            pt3ds = np.array(mesh.vertices)
            if dataset == "objaverse":
                # normalize the point cloud
                min_coords = pt3ds.min(axis=0)
                max_coords = pt3ds.max(axis=0)
                offset = (min_coords + max_coords) / 2
                scale = np.linalg.norm(max_coords - min_coords) / 2
                pt3ds = (pt3ds - offset) / scale
                # make new mesh
                pt3ds[:, 1:] = -pt3ds[:, 1:]  # Adjust axes if necessary
                mesh.vertices = pt3ds
                scene.add_geometry(mesh)
            else:
                scene.add_geometry(mesh)
            
            if dataset == "co3d":
                pt3ds[:, :2] = -pt3ds[:, :2]  # Adjust axes if necessary
            # log point cloud range
            # INFO(f"Point cloud range: {pt3ds.min(axis=0)} - {pt3ds.max(axis=0)}")
            # # log pose translation range
            # INFO(f"Pose translation range: {gt_pose[b, :, :3, 3].min(axis=0)} - {gt_pose[b, :, :3, 3].max(axis=0)}")
            

            begin_color = np.array([255, 0, 0, 255])
            end_color = np.array([0, 0, 255, 255])
            step = (end_color - begin_color) / N

            for i in range(N):
                scene_scale = np.linalg.norm(pt3ds.max(axis=0) - pt3ds.min(axis=0))

                # Integrate predicted cameras
                if i != batch['query_idx'][b]:
                    color = begin_color + i * step - np.array([0, 0, 0, 40])
                else:
                    color = np.array([0, 255, 0, 255])  # Green for query camera

                integrate_camera_into_scene(
                    scene,
                    transform=np.linalg.inv(scaled_poses[b, i]),
                    face_colors=color,
                    scene_scale=scene_scale
                )

                # Integrate ground truth cameras
                gt_color = begin_color + i * step
                integrate_camera_into_scene(
                    scene,
                    transform=np.linalg.inv(gt_pose[b, i]),
                    face_colors=gt_color,
                    scene_scale=scene_scale
                )

            aligned_scene = apply_scene_alignment(scene, gt_pose[b])
            scenes.append(aligned_scene)

        return scenes

    def get_results(self):
        """
        Retrieve the collected validation results.

        Returns:
            dict: Dictionary containing all validation results.
        """
        return self.val_results


def visualize_direction_as_color(rays: torch.Tensor) -> np.ndarray:
    """
    Map direction vectors to RGB colors for visualization.

    Args:
        rays (torch.Tensor): Tensor of shape (H, W, 3) representing direction vectors.

    Returns:
        np.ndarray: RGB image representing the directions.
    """
    directions = rays[..., :3]

    # Normalize directions to [0, 1]
    directions_min = directions.min(dim=0, keepdim=True)[0]
    directions_max = directions.max(dim=0, keepdim=True)[0]
    directions_norm = (directions - directions_min) / (directions_max - directions_min + 1e-8)

    directions_norm = torch.clamp(directions_norm, 0, 1).numpy()
    img_rgb = directions_norm.reshape(rays.shape[0], rays.shape[1], 3)

    return img_rgb


def visualize_moment_as_color(rays: torch.Tensor) -> np.ndarray:
    """
    Map moment vectors to RGB colors for visualization.

    Args:
        rays (torch.Tensor): Tensor of shape (H, W, 3) representing moment vectors.

    Returns:
        np.ndarray: RGB image representing the moments.
    """
    directions = rays[..., 3:]

    # Normalize directions to [0, 1]
    directions_min = directions.min(dim=0, keepdim=True)[0]
    directions_max = directions.max(dim=0, keepdim=True)[0]
    directions_norm = (directions - directions_min) / (directions_max - directions_min + 1e-8)

    directions_norm = torch.clamp(directions_norm, 0, 1).numpy()
    img_rgb = directions_norm.reshape(rays.shape[0], rays.shape[1], 3)

    return img_rgb


def create_ref_matrix(ref_images: np.ndarray, matrix_size: int, single_img_shape: tuple) -> np.ndarray:
    """
    Arrange reference images into a grid matrix.

    Args:
        ref_images (np.ndarray): Array of reference images with shape [N, H, W, C].
        matrix_size (int): Number of images per row/column in the grid.
        single_img_shape (tuple): Shape of a single image (H, W, C).

    Returns:
        np.ndarray: Concatenated reference image matrix.
    """
    H, W, C = single_img_shape
    grid_height = matrix_size * H
    grid_width = matrix_size * W
    ref_matrix = np.zeros((grid_height, grid_width, C), dtype=ref_images.dtype)

    for idx, ref in enumerate(ref_images):
        row = idx // matrix_size
        col = idx % matrix_size
        y_start = row * H
        y_end = y_start + H
        x_start = col * W
        x_end = x_start + W
        ref_matrix[y_start:y_end, x_start:x_end, :] = ref

    return ref_matrix


def add_text_to_image(image: np.ndarray, text: str, position: tuple, font=cv2.FONT_HERSHEY_TRIPLEX, 
                     font_scale: float = 1.0, color: tuple = (255, 255, 255), thickness: int = 2) -> np.ndarray:
    """
    Add text to an image at a specified position.

    Args:
        image (np.ndarray): The image to draw text on.
        text (str): The text string to add.
        position (tuple): Bottom-left corner of the text string in the image.
        font: Font type.
        font_scale (float): Font scale factor.
        color (tuple): Text color in BGR.
        thickness (int): Thickness of the text strokes.

    Returns:
        np.ndarray: Image with text added.
    """
    return cv2.putText(image.copy(), text, position, font, font_scale, color, thickness, cv2.LINE_AA)


def add_label(image: np.ndarray, text: str, height: int = 50, font_scale: float = 0.7, 
             color: tuple = (255, 255, 255), thickness: int = 2) -> np.ndarray:
    """
    Add a text label above the given image.

    Args:
        image (np.ndarray): Image to label.
        text (str): Text for the label.
        height (int): Reserved height for the label.
        font_scale (float): Scale of the font.
        color (tuple): Text color in BGR.
        thickness (int): Thickness of the text.

    Returns:
        np.ndarray: Image with the label added above.
    """
    label_image = np.ones((height, image.shape[1], 3), dtype=image.dtype) * 0  # Black background
    label_with_text = add_text_to_image(
        image=label_image,
        text=text,
        position=(10, int(height * 0.7)),
        font_scale=font_scale,
        color=color,
        thickness=thickness
    )
    label_with_text = label_with_text
    return np.concatenate([label_with_text, image], axis=0)


def arrange_images_grid(images: np.ndarray, grid_size: int, image_shape: tuple) -> np.ndarray:
    """
    Arrange images into a grid and resize them.

    Args:
        images (np.ndarray): Array of images to arrange, shape [N, H, W, C].
        grid_size (int): Size of the grid (number of rows and columns).
        image_shape (tuple): Desired shape (H, W, C) for each image.

    Returns:
        np.ndarray: Grid image.
    """
    ref_matrix = create_ref_matrix(images, grid_size, image_shape)
    target_width = image_shape[1] * 2  # Example resizing logic
    resized_matrix = cv2.resize(ref_matrix, (target_width, target_width))
    return resized_matrix


def visualize_and_label_rays(gt_rays: torch.Tensor, pred_rays: torch.Tensor, single_img_shape: tuple) -> np.ndarray:
    """
    Visualize ground truth and predicted camera rays with labels.

    Args:
        gt_rays (torch.Tensor): Ground truth camera rays.
        pred_rays (torch.Tensor): Predicted camera rays.

    Returns:
        np.ndarray: Concatenated ray visualizations with labels.
    """
    gt_ray_d = visualize_direction_as_color(gt_rays)
    gt_ray_m = visualize_moment_as_color(gt_rays)
    pred_ray_d = visualize_direction_as_color(pred_rays)
    pred_ray_m = visualize_moment_as_color(pred_rays)
    
    # make all images have same shape
    gt_ray_d = cv2.resize(gt_ray_d, (single_img_shape[1], single_img_shape[0]))
    gt_ray_m = cv2.resize(gt_ray_m, (single_img_shape[1], single_img_shape[0]))
    pred_ray_d = cv2.resize(pred_ray_d, (single_img_shape[1], single_img_shape[0]))
    pred_ray_m = cv2.resize(pred_ray_m, (single_img_shape[1], single_img_shape[0]))

    ray_gt = np.concatenate([gt_ray_d, gt_ray_m], axis=1)
    ray_pred = np.concatenate([pred_ray_d, pred_ray_m], axis=1)

    label_height = 50
    ray_gt_with_label = add_label(ray_gt, "GT Camera Rays", height=label_height)
    ray_pred_with_label = add_label(ray_pred, "Pred Camera Rays", height=label_height)

    return np.concatenate([ray_gt_with_label, ray_pred_with_label], axis=0)


def process_video_frames(batch: dict, metrics: dict = None, bid: int = 0):
    """
    Process the batch to create video frames and save as a video file.

    Args:
        batch (dict): Batch containing necessary tensors.
    """
    # Extract and clone necessary tensors
    gt_images = batch['images'].clone()
    query_idx = batch['query_idx'].clone()
    if "pred_images" in batch:
        pred_images = batch['pred_images'].clone()
    else:
        pred_images = None
    if "pred_camera_rays" in batch:
        pred_camera_rays = batch['pred_camera_rays'].clone()
        gt_camera_rays = batch['camera_rays'].clone()
    else:
        pred_camera_rays = None
        gt_camera_rays = None
    
    
    
    bbox_figs = batch.get('bbox_figs', None)
    pred_bbox_fig = batch.get('pred_bbox_vis', None)
    if pred_bbox_fig is not None:
        pred_bbox_fig_np = np.stack(pred_bbox_fig, axis=0)
    else:
        pred_bbox_fig_np = None
    B, T, C, H, W = gt_images.shape

    # Move to CPU and convert to NumPy arrays
    gt_images = gt_images.cpu().numpy()
    if pred_images is not None:
        pred_images = pred_images.cpu().numpy()
    query_idx = query_idx.cpu().numpy()
    if pred_camera_rays is not None:
        pred_camera_rays = pred_camera_rays.cpu()
        gt_camera_rays = gt_camera_rays.cpu()

    # Select the queried frames
    indices = np.arange(B)
    gt_selected = gt_images[indices, query_idx].copy()
    if pred_images is not None:
        pred_selected = pred_images[indices, query_idx]
    else:
        pred_selected = None
    if pred_camera_rays is not None:
        gt_camera_rays = gt_camera_rays[indices, query_idx]
        pred_camera_rays = pred_camera_rays[indices, query_idx]

    # Transpose to [B, H, W, C] for visualization
    gt_selected = np.transpose(gt_selected, (0, 2, 3, 1))
    if pred_images is not None:
        pred_selected = np.transpose(pred_selected, (0, 2, 3, 1))
    
    if pred_camera_rays is not None:
        gt_camera_rays = gt_camera_rays.permute(0, 2, 3, 1)
        pred_camera_rays = pred_camera_rays.permute(0, 2, 3, 1)

    if bbox_figs is not None:
        # ensure bbox_figs is a list of numpy arrays having same shape
        for i in range(len(bbox_figs)):
            bbox_figs[i] = cv2.resize(bbox_figs[i], (H, W))
        
        gt_selected = np.stack(bbox_figs, axis=0)
        gt_selected = gt_selected.astype(np.float32)
    
    if pred_bbox_fig_np is not None:
        # replace gt image with pred_bbox_fig_np
        pred_bbox_fig_np = pred_bbox_fig_np.astype(np.float32)
        # reshape from [B, T, H, W, C] to [B, T, C, H, W]
        pred_bbox_fig_np = np.transpose(pred_bbox_fig_np, (0, 1, 4, 2, 3))
        gt_images = pred_bbox_fig_np.copy()
        pred_selected = pred_bbox_fig_np[indices, query_idx].copy()
        pred_selected = np.transpose(pred_selected, (0, 2, 3, 1))
        
    video_frames = []
    
    # get metrics(if have)
    if metrics is not None:
        batch_metrics = []
        for i in range(B):
            metrics_dict = {}
            for key in metrics.keys():
                
                if i >= len(metrics[key]):
                    raise ValueError(f"Batch size {B} does not match metrics size {len(metrics[key])}")
                
                metrics_dict[key] = metrics[key][i]
            batch_metrics.append(metrics_dict)
    else:
        batch_metrics = [None] * B
        

    for i in range(B):
        metrics_i = batch_metrics[i]
            
        # Process reference images
        ref_images = np.transpose(np.delete(gt_images[i].copy(), query_idx[i], axis=0), (0, 2, 3, 1))
        ref_num = ref_images.shape[0]
        # if ref num > 15, only keep first 15 frames
        if ref_num>15:
            ref_num = 15
            ref_images = ref_images[:ref_num]
            
        grid_size = math.ceil(math.sqrt(ref_num))
        ref_matrix = arrange_images_grid(ref_images, grid_size, ref_images.shape[1:])

        # Resize reference matrix
        target_width = gt_selected.shape[2] * 2
        ref_matrix_resized = cv2.resize(ref_matrix, (target_width, target_width))

        # Add label to reference matrix
        ref_matrix_with_label = add_label(ref_matrix_resized, "Reference Images")
        
        # if "gt_neighbor_idx" in batch.keys():
        #     gt_neighbor_idx = batch["gt_neighbor_idx"][i]
        #     pred_neighbor_idx = batch["pred_neighbor_idx"][i]
        #     gt_neighbor_ref = gt_images[i][gt_neighbor_idx] # C, H, W
        #     pred_neighbor_ref = gt_images[i][pred_neighbor_idx] # C, H, W
        #     gt_neighbor_ref = np.transpose(gt_neighbor_ref, (1, 2, 0))
        #     pred_neighbor_ref = np.transpose(pred_neighbor_ref, (1, 2, 0))
        #     gt_neighbor_ref_with_label = add_label(gt_neighbor_ref, "GT Neighbor")
        #     pred_neighbor_ref_with_label = add_label(pred_neighbor_ref, "Pred Neighbor")

        #     neighbor_row = np.concatenate([gt_neighbor_ref_with_label, pred_neighbor_ref_with_label], axis=1)
        #     ref_matrix_with_label = np.concatenate([ref_matrix_with_label, neighbor_row], axis=0)

        # Prepare ground truth and predicted images with labels
        # if metrics_i is not None and 'psnr' not in metrics_i:
        #     # set get view black
        #     gt_with_label = add_label(gt_selected[i], "Pred Pose")
        #     if pred_selected is not None:
        #         pred_selected[i] = np.zeros_like(pred_selected[i])
        #         pred_with_label = add_label(pred_selected[i], "w/o nvs")
        #     else:
        #         pred_with_label = add_label(np.zeros_like(gt_selected[i]), "w/o nvs")
        # else:
        gt_with_label = add_label(gt_selected[i], "Pred Pose")
        pred_with_label = add_label(pred_selected[i], "Pred bbox")

        # Concatenate ground truth and predicted images side by side
        concatenated_images = np.concatenate([gt_with_label, pred_with_label], axis=1)
        
        if metrics_i is not None:
            # Add metrics to the image
            for key, value in metrics_i.items():
                # show only 3 float point
                if isinstance(value, torch.Tensor):
                    value = value.item()
                elif isinstance(value, np.float32):
                    value = value.item()
                value = round(value, 3)
                concatenated_images = add_label(concatenated_images, f"{key}: {value}")
        

        # Combine reference images and concatenated images
        combined_image = np.concatenate([ref_matrix_with_label, concatenated_images], axis=0)

        # Visualize and label camera rays
        # make shure gt_camera_rays and pred_camera_rays have correct shape(single image shape)
        if pred_camera_rays is not None:
            ray_concatenated = visualize_and_label_rays(gt_camera_rays[i], pred_camera_rays[i], single_img_shape=(H, W))
            # Concatenate all parts vertically
            final_frame = np.concatenate([combined_image, ray_concatenated], axis=0)
        else:
            final_frame = combined_image

        # Scale to [0, 255] and convert to uint8
        final_frame = (final_frame * 255).astype(np.uint8) if final_frame.max() <= 1.0 else final_frame.astype(np.uint8)

        video_frames.append(final_frame)

    # Define the output video path
    dataset = batch.get('dataset', 0)
    if dataset != 0:
        dataset = dataset[0]
    video_path = Path(f"video_{dataset}.mp4")
    height, width, channels = video_frames[0].shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 8  # Adjust frames per second as needed
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for frame in video_frames:
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    # use ffmepg to convert video to h264 format
    new_video_path = Path(f"video_{dataset}_{bid}_h264.mp4")
    os.system(f"ffmpeg -i {video_path} -vcodec libx264 {new_video_path}")
    # Remove the original video and rename the h264 video
    os.remove(video_path)
    logger.info(f"Video saved to {new_video_path}")


def visualize_results(data: dict, is_gt: bool = False) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Visualize depth and confidence maps as images.

    Args:
        data (dict): The data dictionary containing depth and confidence information.
        is_gt (bool): Flag indicating whether to visualize ground truth data.

    Returns:
        Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
            - List of depth images.
            - List of confidence images or None.
    """
    pts3d = data['pts3d_gt'] if is_gt else data['pts3d']  # [n, H, W, C]
    conf = data.get('conf', None)

    pts3d = pts3d.numpy()

    if conf is not None:
        conf = conf.numpy()

    # Convert 3D points to depth map (z-axis)
    depth_map = pts3d[..., 2]

    # Visualize depth map as grayscale images
    depth_images = []
    valid_mask = data['valid_mask'].numpy()  # [n, H, W]
    for i in range(depth_map.shape[0]):
        fig, ax = plt.subplots()
        # Normalize the depth map to [0, 1]
        depth_normalized = (depth_map[i] - depth_map[i].min()) / (depth_map[i].max() - depth_map[i].min())
        depth_normalized *= valid_mask[i]
        ax.imshow(depth_normalized, cmap='gray')
        plt.axis('off')  # Turn off axis
        fig.canvas.draw()

        # Convert plot to numpy array
        width, height = fig.get_size_inches() * fig.get_dpi()
        depth_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        depth_images.append(depth_img)
        plt.close(fig)

    # Visualize confidence map as heatmap images if available
    conf_images = None
    if conf is not None:
        conf_images = []
        for i in range(conf.shape[0]):
            fig, ax = plt.subplots()
            # Normalize the confidence map to [0, 1]
            conf_normalized = (conf[i] - conf[i].min()) / (conf[i].max() - conf[i].min())
            conf_normalized *= valid_mask[i]

            ax.imshow(conf_normalized, cmap='viridis')
            plt.axis('off')  # Turn off axis
            fig.canvas.draw()

            # Convert plot to numpy array
            width, height = fig.get_size_inches() * fig.get_dpi()
            conf_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            conf_images.append(conf_img)
            plt.close(fig)

    return depth_images, conf_images


def visualize_depth_to_3d(depth: np.ndarray, mapping: torch.Tensor, path: Optional[str] = None):
    """
    Visualize a depth map and its corresponding 3D point cloud.

    Args:
        depth (np.ndarray): Depth map.
        mapping (torch.Tensor): Mapping tensor to convert depth to 3D points.
        path (Optional[str]): Path to save the visualization image. If None, displays the plot.
    """
    if isinstance(mapping, torch.Tensor):
        mapping = mapping.numpy()

    height, width = depth.shape
    step = 10  # Sampling step

    valid_mask = np.linalg.norm(mapping, axis=2) > 0.1
    sampled_mask = valid_mask[::step, ::step]
    sampled_depth = depth[::step, ::step]
    sampled_mapping = mapping[::step, ::step]

    y, x = np.where(sampled_mask)
    sampled_points = sampled_mapping[sampled_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot depth image with sampled points
    ax1.imshow(depth, cmap='gray')
    ax1.scatter(x * step, y * step, color='red', s=20)
    ax1.set_title('Subsampled Depth Image with Valid Points')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')

    # Plot 3D point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                color='red', s=50, label='Sampled Valid Points')
    ax2.set_title('3D Point Cloud with Sampled Valid Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


def vis_mapping(depth: np.ndarray, mapping: torch.Tensor, path: Optional[str] = None):
    """
    Visualize depth mapping and corresponding 3D point cloud.

    Args:
        depth (np.ndarray): Depth map.
        mapping (torch.Tensor): Mapping tensor to convert depth to 3D points.
        path (Optional[str]): Path to save the visualization image.
    """
    if isinstance(mapping, torch.Tensor):
        mapping = mapping.numpy()
    if isinstance(depth, torch.Tensor):
        depth = depth.numpy()

    valid_mask = np.any(mapping != 0, axis=2)
    x, y = np.meshgrid(np.arange(mapping.shape[1]), np.arange(mapping.shape[0]))
    x = x[valid_mask]
    y = y[valid_mask]
    points = mapping[valid_mask]

    fig = plt.figure(figsize=(12, 6))

    # Plot 2D depth image
    ax1 = fig.add_subplot(121)
    ax1.imshow(depth, cmap='gray')
    ax1.set_title('2D Depth Image')
    ax1.axis('off')

    # Plot 3D point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis')
    ax2.set_title('3D Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    if path:
        plt.savefig(path)
    else:
        plt.show()


def get_3d_bbox_from_pts(pts: np.ndarray, bbox: Optional[np.ndarray] = None, intrinsic: Optional[np.ndarray] = None, pose: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute a 3D bounding box from 3D points.

    Args:
        pts (np.ndarray): Array of points with shape [n, 3].
        bbox (Optional[np.ndarray]): 2D detection bounding box with format [x0, y0, x1, y1].

    Returns:
        np.ndarray: Array of bounding box corners with shape [8, 3].
    """
    if bbox is not None:
        # proj all 3d pts to 2d
        if intrinsic is None or pose is None:
            raise ValueError("Intrinsic matrix and pose matrix must be provided.")
        pts_2d = reproj(intrinsic, pose, pts)
        
        # eliminate pts outside bbox (x0, y0, x1, y1)
        x0, y0, x1, y1 = bbox
        mask = (pts_2d[:, 0] >= x0) & (pts_2d[:, 0] <= x1) & (pts_2d[:, 1] >= y0) & (pts_2d[:, 1] <= y1)
        sub_pts = pts[mask]
    else:
        sub_pts = pts
        
    # if sub_pts.shape[0] < 10:
    #     return get_3d_bbox_from_pts(pts)
    # else:
    #     return get_3d_bbox_from_pts(sub_pts)
        
    
    max_point = np.max(pts, axis=0)
    min_point = np.min(pts, axis=0)

    corners = np.array([
        [min_point[0], min_point[1], min_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [min_point[0], max_point[1], max_point[2]],
        [max_point[0], max_point[1], max_point[2]],
        [max_point[0], min_point[1], max_point[2]]
    ])
    # return get_3d_bbox_from_pts_pruning(pts)
    
    return corners

def get_3d_bbox_from_pts_pruning(pts: np.ndarray) -> np.ndarray:
    """
    Compute a 3D bounding box from 3D points with noise filtering.

    Args:
        pts (np.ndarray): Array of points with shape [n, 3].

    Returns:
        np.ndarray: Array of bounding box corners with shape [8, 3].
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
    filtered_pcd = pcd.select_by_index(ind)
    filtered_pts = np.asarray(filtered_pcd.points)
    
    if filtered_pts.shape[0] < 10:
        filtered_pts = pts
    
    max_point = np.max(filtered_pts, axis=0)
    min_point = np.min(filtered_pts, axis=0)
    
    corners = np.array([
        [min_point[0], min_point[1], min_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [min_point[0], max_point[1], max_point[2]],
        [max_point[0], max_point[1], max_point[2]],
        [max_point[0], min_point[1], max_point[2]]
    ])

    return corners

def reproj(K: np.ndarray, pose: np.ndarray, pts_3d: np.ndarray) -> np.ndarray:
    """
    Reproject 3D points to 2D image plane.

    Args:
        K (np.ndarray): Intrinsic matrix [3, 3] or [3, 4].
        pose (np.ndarray): Pose matrix [3, 4] or [4, 4].
        pts_3d (np.ndarray): 3D points with shape [n, 3].

    Returns:
        np.ndarray: 2D reprojected points with shape [n, 2].
    """
    assert K.shape in [(3, 3), (3, 4)], "Intrinsic matrix K must be shape [3,3] or [3,4]."
    assert pose.shape in [(3, 4), (4, 4)], "Pose matrix must be shape [3,4] or [4,4]."

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1).T  # [4, n]
    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points /= reproj_points[2, :]  # Normalize by z

    return reproj_points[:2, :].T  # [n, 2]


def draw_coordinate_on_image(
    image: np.ndarray,
    pose: np.ndarray,
    intri: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255)  # Default color: Red in BGR
) -> np.ndarray:
    """
    Draws the camera's world coordinate system origin and its three axes on the image using a single color.

    Parameters:
    - image (np.ndarray): The original image in BGR format.
    - pose (np.ndarray): Camera extrinsic parameters, a 4x4 transformation matrix [R|t; 0 0 0 1] or a 3x4 matrix [R|t].
    - intri (np.ndarray): Camera intrinsic matrix, a 3x3 matrix.
    - color (Tuple[int, int, int], optional): A BGR color tuple for all axes and labels.
        Example: (255, 0, 0) for blue.
        Default: (0, 0, 255) which is red in BGR.

    Returns:
    - np.ndarray: The image with the coordinate axes drawn.
    """
    # Create a copy of the original image to draw on
    img = image.copy()

    # Define the length of the coordinate axes (in world units, e.g., meters)
    axis_length = 0.05  # Adjust as needed

    # Define the 3D points for the origin and the end points of the X, Y, Z axes
    axes_3D = np.float32([
        [0, 0, 0],                         # Origin
        [axis_length, 0, 0],               # X-axis
        [0, axis_length, 0],               # Y-axis
        [0, 0, axis_length]                # Z-axis
    ]).reshape(-1, 3)

    # Separate rotation matrix and translation vector from the pose
    if pose.shape == (4, 4):
        R = pose[:3, :3]
        t = pose[:3, 3].reshape(3, 1)
    elif pose.shape == (3, 4):
        R = pose[:, :3]
        t = pose[:, 3].reshape(3, 1)
    else:
        raise ValueError("Pose must be a 4x4 or 3x4 transformation matrix.")

    # Transform the 3D points from world coordinates to camera coordinates: Xc = R * Xw + t
    axes_cam = R @ axes_3D.T + t  # Shape: 3 x N

    # Project the 3D points onto the 2D image plane using the intrinsic matrix
    axes_proj = intri @ axes_cam  # Shape: 3 x N

    # Convert from homogeneous coordinates to 2D
    axes_proj /= axes_proj[2, :]

    # Transpose and extract the first two coordinates for image points
    axes_2D = axes_proj[:2, :].T  # Shape: N x 2

    # Convert points to tuples of two Python integers
    # Using map to ensure each coordinate is a Python int
    origin = tuple(map(int, axes_2D[0]))
    x_axis = tuple(map(int, axes_2D[1]))
    y_axis = tuple(map(int, axes_2D[2]))
    z_axis = tuple(map(int, axes_2D[3]))

    # Define line thickness and font settings
    line_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2

    # Optional: Validate if points are within image boundaries
    height, width = img.shape[:2]
    for point in [origin, x_axis, y_axis, z_axis]:
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            # print(f"Warning: Point {point} is outside the image boundaries.")
            # maybe 3 lines are out of image, clip them to image boundary
            point = (max(0, min(point[0], width - 1)), max(0, min(point[1], height - 1)))

    # Draw the X-axis
    cv2.line(img, origin, x_axis, color, line_thickness, cv2.LINE_AA)
    cv2.putText(img, 'X', x_axis, font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Draw the Y-axis
    cv2.line(img, origin, y_axis, color, line_thickness, cv2.LINE_AA)
    cv2.putText(img, 'Y', y_axis, font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Draw the Z-axis
    cv2.line(img, origin, z_axis, color, line_thickness, cv2.LINE_AA)
    cv2.putText(img, 'Z', z_axis, font, font_scale, color, font_thickness, cv2.LINE_AA)

    return img

def draw_3d_box(image: np.ndarray, corners_2d: np.ndarray, linewidth: int = 3, color: str = "g") -> np.ndarray:
    """
    Draw a 3D bounding box on an image.

    Args:
        image (np.ndarray): Image to draw on.
        corners_2d (np.ndarray): 2D coordinates of the bounding box corners [8, 2].
        linewidth (int): Line thickness.
        color (str): Color of the box edges ('g', 'r', 'b').

    Returns:
        np.ndarray: Image with the bounding box drawn.
    """
    lines = np.array([
        [0, 1], [1, 5], [5, 4], [4, 0],
        [3, 2], [2, 6], [6, 7], [7, 3],
        [0, 3], [1, 2], [5, 6], [4, 7]
    ])

    colors = {"g": (0, 255, 0), "r": (255, 0, 0), "b": (0, 0, 255)}
    color_rgb = colors.get(color, (42, 97, 247))  # Default color if not specified

    # Normalize image if necessary
    if image.max() <= 1 and image.min() >= 0:
        image = (image * 255).astype(np.uint8)
    elif image.max() <= 1 and image.min() >= -1:
        image = ((image + 1) / 2 * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    for line in lines:
        pt1, pt2 = corners_2d[line[0]].astype(int), corners_2d[line[1]].astype(int)
        image = cv2.line(image, tuple(pt1), tuple(pt2), color_rgb, linewidth)

    return image


def draw_2d_box(image: np.ndarray, corners_2d: np.ndarray, linewidth: int = 3) -> None:
    """
    Draw a 2D bounding box on an image.

    Args:
        image (np.ndarray): Image to draw on.
        corners_2d (np.ndarray): 2D coordinates [x_left, y_top, x_right, y_bottom].
        linewidth (int): Line thickness.
    """
    x1, y1, x2, y2 = corners_2d.astype(int)
    box_pts = [
        [(x1, y1), (x1, y2)],
        [(x1, y2), (x2, y2)],
        [(x2, y2), (x2, y1)],
        [(x2, y1), (x1, y1)],
    ]

    for pt1, pt2 in box_pts:
        cv2.line(image, pt1, pt2, (0, 0, 255), linewidth)


def add_pointcloud_to_vis3d(pointcloud_pth: str, dump_dir: str, save_name: str):
    """
    Add a point cloud to Vision3D for visualization.

    Args:
        pointcloud_pth (str): Path to the point cloud file.
        dump_dir (str): Directory to dump the visualization files.
        save_name (str): Name for the saved visualization.
    """
    vis3d = Vis3D(dump_dir, save_name)
    vis3d.add_point_cloud(pointcloud_pth, name="filtered_pointcloud")


def save_demo_image(pose_pred: np.ndarray, K: np.ndarray, image_path: str, box3d: Optional[str] = None, 
                   draw_box: bool = True, save_path: Optional[str] = None) -> np.ndarray:
    """
    Project a 3D bounding box using the predicted pose and visualize it on the image.

    Args:
        pose_pred (np.ndarray): Predicted pose matrix.
        K (np.ndarray): Intrinsic camera matrix.
        image_path (str): Path to the image file.
        box3d (Optional[str or np.ndarray]): Path to the box3d file or the box3d array.
        draw_box (bool): Flag to draw the bounding box.
        save_path (Optional[str]): Path to save the resulting image.

    Returns:
        np.ndarray: Image with the projected bounding box.
    """
    if isinstance(box3d, str):
        box3d = np.loadtxt(box3d)

    image_full = cv2.imread(image_path)

    if draw_box and box3d is not None:
        reproj_box_2d = reproj(K, pose_pred, box3d)
        image_full = draw_3d_box(image_full, reproj_box_2d, color='b', linewidth=10)

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(save_path, image_full)

    return image_full


def save_demo_image_with_multibox(pose_preds: List[np.ndarray], colors: List[str], K: np.ndarray, 
                                  image_path: str, box3d: Optional[str] = None, draw_box: bool = True, 
                                  save_path: Optional[str] = None) -> np.ndarray:
    """
    Project multiple 3D bounding boxes using predicted poses and visualize them on the image.

    Args:
        pose_preds (List[np.ndarray]): List of predicted pose matrices.
        colors (List[str]): List of colors for each bounding box.
        K (np.ndarray): Intrinsic camera matrix.
        image_path (str): Path to the image file.
        box3d (Optional[str or np.ndarray]): Path to the box3d file or the box3d array.
        draw_box (bool): Flag to draw the bounding boxes.
        save_path (Optional[str]): Path to save the resulting image.

    Returns:
        np.ndarray: Image with the projected bounding boxes.
    """
    if isinstance(box3d, str):
        box3d = np.loadtxt(box3d)

    image_full = cv2.imread(image_path)
    for pose_pred, color in zip(pose_preds, colors):
        if draw_box and box3d is not None:
            reproj_box_2d = reproj(K, pose_pred, box3d)
            image_full = draw_3d_box(image_full, reproj_box_2d, color=color, linewidth=10)

    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(save_path, image_full)

    return image_full


def make_video(image_dir: str, output_video_path: str):
    """
    Generate a video from a directory of images.

    Args:
        image_dir (str): Directory containing image files.
        output_video_path (str): Path to save the output video.
    """
    images = natsort.natsorted(os.listdir(image_dir))
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(output_video_path).exists():
        Path(output_video_path).unlink()

    first_image = cv2.imread(os.path.join(image_dir, images[0]))
    H, W, C = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24  # Frames per second
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    for image_name in images:
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            video_writer.write(image)
        else:
            logger.warning(f"Unable to read image: {image_path}")

    video_writer.release()
    logger.info(f"Demo video saved to: {output_video_path}")
    

def draw_bbox_vector_map(bbox_map: torch.Tensor, rgb: torch.Tensor = None) -> torch.Tensor:
    '''
    Draw bbox vector map
    
    Args:
        bbox_map (torch.Tensor): Bounding box coordinates of shape [B, H, W, 8, 2]
        shape (Tuple[int, int]): Shape of the output image
        rgb (torch.Tensor): RGB values for the bbox
        
    Returns:
        numpy.ndarray: Image of the bbox vector map
    '''
    # draw strategy: random select some points in the range of H, W, then draw the bbox vector
    B, H, W, _, _ = bbox_map.shape
    sample_points = torch.randint(0, H*W, (B, 1), device=bbox_map.device)
    vis_map = bbox_map.clone()
    vis_map = vis_map.view(B, -1, 8, 2)
    # recover the dx, dy to the original scale
    vis_map[:, :, :, 0] = vis_map[:, :, :, 0] * W
    vis_map[:, :, :, 1] = vis_map[:, :, :, 1] * H
    
    # each sample have 8 proposals, which is the dx, dy from the 8 corners
    # draw the arrow from the sample point to the proposal point
    # convert tensor to numpy for better visualization
    vis_map = vis_map.cpu().numpy()
    if rgb is None:
        # draw arrow on the black background
        rgb = np.zeros((B, H, W, 3), dtype=np.uint8)
    else:
        rgb = rgb.cpu().numpy()
        # reshape to B, H, W, 3
        rgb = rgb.transpose(0, 2, 3, 1) * 255
        # convert to contiguous
        rgb = np.ascontiguousarray(rgb)
    
    # pink line
    line_color_pink = (255, 0, 255)
        
    for i in range(B):
        for j in range(1):
            idx = sample_points[i, j]
            x, y = idx % W, idx // W
            for k in range(8):
                dx, dy = vis_map[i, idx, k]
                x1, y1 = x, y
                # convert to int 
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x + dx), int(y + dy)
                rgb[i] = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
                cv2.arrowedLine(rgb[i], (x1, y1), (x2, y2), line_color_pink, 2)
                # draw start point
                cv2.circle(rgb[i], (x1, y1), 2, (0, 255, 0), 3)
                
    # # save for debug
    # for i in range(B):
    #     cv2.imwrite(f'bbox_vector_{i}.png', rgb[i])
                
    return rgb

def draw_bbox_voting_as_heatmap(bbox_map: torch.Tensor, rgb: torch.Tensor = None) -> torch.Tensor:
    '''
    Draw bbox voting heatmap
    
    Args:
        bbox_map (torch.Tensor): Bounding box heatmap of shape [B, H, W, 8, 2]
        rgb (torch.Tensor): RGB values for the bbox
        
    Returns:
        numpy.ndarray: Image of the bbox heatmap
    
    '''
    # perpixel voting 8 corners' of bbox, counting votes for each pixel and thne convert to heatmap
    
    B, H, W, _, _ = bbox_map.shape
    copy_map = bbox_map.clone()
    # recover the dx, dy to the original scale
    copy_map = copy_map.view(B, -1, 8, 2)
    copy_map[:, :, :, 0] = copy_map[:, :, :, 0] * W
    copy_map[:, :, :, 1] = copy_map[:, :, :, 1] * H
    copy_map = copy_map.cpu().numpy()
    
    if rgb is None:
        # draw heatmap on the black background
        rgb = np.zeros((B, H, W, 3), dtype=np.uint8)
    else:
        rgb = rgb.cpu().numpy()
        # reshape to B, H, W, 3
        rgb = rgb.transpose(0, 2, 3, 1) * 255
        # convert to contiguous
        rgb = np.ascontiguousarray(rgb).astype(np.uint8)
        
    # perpixel voting
    for i in range(B):
        heatmap = np.zeros((H, W))
        for j in range(H*W):
            x, y = j % W, j // W
            for k in range(8):
                dx, dy = copy_map[i, j, k]
                x1, y1 = int(x + dx), int(y + dy)
                if x1 >= 0 and x1 < W and y1 >= 0 and y1 < H:
                    heatmap[y1, x1] += 1
        # log heatmap range 
        # print(f'heatmap range: {heatmap.min()}, {heatmap.max()}')
        # normalize and smooth the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        heatmap = heatmap / heatmap.max()
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        rgb[i] = cv2.addWeighted(rgb[i], 0.5, heatmap, 0.5, 0)
        
    # return visualization
    return rgb
    
    
    
def draw_bbox_heatmap(bbox_map: torch.Tensor, rgb: torch.Tensor = None, 
                     cmap: str = "Spectral") -> torch.Tensor:
    '''
    Draw bbox heatmap using Matplotlib's colormap visualization method
    
    Args:
        bbox_map (torch.Tensor): Bounding box heatmap of shape [B, H, W, 8]
        rgb (torch.Tensor): RGB values for the background image (optional)
        cmap (str): Matplotlib colormap name to use (e.g., "Spectral", "plasma", "viridis")
        
    Returns:
        numpy.ndarray: Image of the bbox heatmap
    '''
    B, H, W, C = bbox_map.shape
    
    # Convert from -1, 1 to 0, 1
    bbox_map = (bbox_map + 1) / 2
    
    # Create background if none provided
    if rgb is None:
        # Use black background
        rgb = np.zeros((B, H, W, 3), dtype=np.uint8)
    else:
        rgb = rgb.cpu().numpy().astype(np.float32)
        # Reshape to B, H, W, 3
        rgb = rgb.transpose(0, 2, 3, 1) * 255
        # Convert to contiguous array
        rgb = np.ascontiguousarray(rgb).astype(np.uint8)
    
    # Create output image
    result = np.copy(rgb)
    
    # Get the matplotlib colormap
    cm = matplotlib.colormaps[cmap]
    
    # Process each image in the batch
    for i in range(B):
        # Create a combined heatmap with maximum values
        combined_heatmap = np.zeros((H, W), dtype=np.float32)
        
        # Process each channel
        for c in range(C):
            # Get current channel
            channel_data = bbox_map[i, :, :, c].cpu().numpy()
            
            # Take maximum value across channels
            combined_heatmap = np.maximum(combined_heatmap, channel_data)
        
        # Normalize heatmap to 0-1 range using our original method
        if np.max(combined_heatmap) > 0:
            combined_heatmap = combined_heatmap / np.max(combined_heatmap)
        
        # Apply colormap using matplotlib's method
        colored_heatmap = cm(combined_heatmap, bytes=False)[:, :, 0:3]  # value from 0 to 1
        colored_heatmap = (colored_heatmap * 255.0).astype(np.uint8)
        
        # Simply overlay the heatmap on the background with 50-50 blend
        result[i] = cv2.addWeighted(rgb[i], 0.55, colored_heatmap, 0.45, 0)
        
        # Save for debug
        if not os.path.exists(f'bbox_heatmap_{i}.png'):
            cv2.imwrite(f'bbox_heatmap_{i}.png', cv2.cvtColor(result[i], cv2.COLOR_RGB2BGR))
    
    return result
    
def draw_bbox_map_rgb(bbox_map: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    '''
    Draw bbox map with RGB values (dx,dy,conf), conf default to 1
    # draw 8 corners' bbox map separately, then combine them to a single image
    
    Args:
        bbox_map (torch.Tensor): Bounding box coordinates of shape [B, H, W, 8, 2]
        shape (Tuple[int, int]): Shape of the output image
        
    Returns:
        torch.Tensor: Image of the bbox map
        B, 8, H, W, 3
    
    '''
    
    B, H, W, _, _ = bbox_map.shape
    device = bbox_map.device
    bbox_map = bbox_map.view(B, -1, 8, 2)
    
    # convert to numpy for better visualization
    bbox_map = bbox_map.cpu().numpy()
    rgb = np.zeros((B, 8, H, W, 3), dtype=np.uint8)
    
    # draw strategy: draw each corner's bbox map separately, then combine them to a single image
    for i in range(B):
        for j in range(8):
            ret = bbox_map[i, :, j] # (H*W, 2)
            dx, dy = ret[:, 0], ret[:, 1]
            rgb[i, j, :, :, 0] = (dx * 255).astype(np.uint8).reshape(H, W)
            rgb[i, j, :, :, 1] = (dy * 255).astype(np.uint8).reshape(H, W)
            rgb[i, j, :, :, 2] = 255 # confidence channel, set to 1
            
    # # save for debug
    # for i in range(B):
    #     for j in range(8):
    #         cv2.imwrite(f'bbox_map_{i}_{j}.png', rgb[i, j])
            
    return torch.from_numpy(rgb).to(device)