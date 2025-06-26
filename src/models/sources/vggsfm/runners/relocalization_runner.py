# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import copy
import torch
import datetime

import numpy as np
from visdom import Visdom
from torch.cuda.amp import autocast
from hydra.utils import instantiate
from lightglue import SuperPoint, SIFT, ALIKED

from collections import defaultdict
from vggsfm.utils.visualizer import Visualizer
from vggsfm.two_view_geo.estimate_preliminary import (
    estimate_preliminary_cameras,
)

from vggsfm.utils.utils import (
    write_array,
    generate_rank_by_midpoint,
    generate_rank_by_dino,
    generate_rank_by_interval,
    calculate_index_mappings,
    extract_dense_depth_maps,
    align_dense_depth_maps,
    switch_tensor_order,
    average_camera_prediction,
    create_video_with_reprojections,
    save_video_with_reprojections,
)
from .runner import (
    VGGSfMRunner,
    move_to_device,
    add_batch_dimension,
    predict_tracks,
    get_query_points,
)

from loguru import logger
from vggsfm.utils.align import align_camera_extrinsics, apply_transformation
from vggsfm.utils.metric import rotation_angle, translation_angle, translation_meters, add_metric, projection_2d_error

# Optional imports
try:
    import poselib
    from vggsfm.two_view_geo.estimate_preliminary import (
        estimate_preliminary_cameras_poselib,
    )

    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")

try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.vis.plotly_vis import plot_scene
    from pytorch3d.renderer.cameras import (
        PerspectiveCameras as PerspectiveCamerasVisual,
    )
except Exception as e:
    print("PyTorch3d is not available. Please disable visdom.") 
    #raise e

class RelocalizationRunner(VGGSfMRunner):
    def __init__(self, cfg):
        super().__init__(cfg)

        # assert (
        #     self.cfg.shared_camera == True
        # ), "Currently only shared camera is supported for video runner"
        logger.info("RelocalizationRunner initialized")


    def run(
        self,
        images,
        masks=None,
        original_images=None,
        image_paths=None,
        crop_params=None,
        query_frame_num=None,
        seq_name=None,
        output_dir=None,
        relocalization_method="align_gt",
        gt_poses=None,
        gt_depths=None,
        query_poses=None,
        model_path=None,
    ):
        supported_methods = ["align_gt"]
        
        self.relocalization_method = relocalization_method
        self.gt_poses = gt_poses
        self.gt_depths = gt_depths
        self.query_poses = query_poses
        self.model_path = model_path
        
        if self.relocalization_method not in supported_methods:
            raise ValueError(
                f"Unsupported relocalization method: {self.relocalization_method}"
            )        
        # demo code:
        
        if output_dir is None:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M")
            output_dir = f"{seq_name}_{timestamp}"

        with torch.no_grad():
            images = move_to_device(images, self.device)
            masks = move_to_device(masks, self.device)
            crop_params = move_to_device(crop_params, self.device)
            self.gt_poses = move_to_device(self.gt_poses, self.device)
            self.gt_depths = move_to_device(self.gt_depths, self.device)
            self.query_poses = move_to_device(self.query_poses, self.device)
            

            # Add batch dimension if necessary
            if len(images.shape) == 4:
                images = add_batch_dimension(images)
                masks = add_batch_dimension(masks)
                crop_params = add_batch_dimension(crop_params)

            if query_frame_num is None:
                query_frame_num = self.cfg.query_frame_num

            # Perform sparse reconstruction
            predictions = self.sparse_reconstruct(
                images,
                masks=masks,
                crop_params=crop_params,
                image_paths=image_paths,
                query_frame_num=query_frame_num,
                seq_name=seq_name,
                output_dir=output_dir,
            )

            # Save the sparse reconstruction results
            if self.cfg.save_to_disk:
                self.save_sparse_reconstruction(
                    predictions, seq_name, output_dir
                )

            # Extract sparse depth and point information if needed for further processing
            if self.cfg.dense_depth or self.cfg.make_reproj_video:
                predictions = (
                    self.extract_sparse_depth_and_point_from_reconstruction(
                        predictions
                    )
                )

            # Perform dense reconstruction if enabled
            if self.cfg.dense_depth:
                predictions = self.dense_reconstruct(
                    predictions, image_paths, original_images
                )

                # Save the dense depth maps
                if self.cfg.save_to_disk:
                    self.save_dense_depth_maps(
                        predictions["depth_dict"], output_dir
                    )
            
            if self.relocalization_method == "align_gt":
                predictions = self.align_gt(predictions, output_dir)

            # Create reprojection video if enabled
            if self.cfg.make_reproj_video:
                max_hw = crop_params[0, :, :2].max(dim=0)[0].long()
                video_size = (max_hw[0].item(), max_hw[1].item())
                img_with_circles_list = self.make_reprojection_video(
                    predictions, video_size, image_paths, original_images
                )
                predictions["reproj_video"] = img_with_circles_list
                if self.cfg.save_to_disk:
                    self.save_reprojection_video(
                        img_with_circles_list, video_size, output_dir
                    )

            # Visualize the 3D reconstruction if enabled
            if self.cfg.viz_visualize:
                self.visualize_3D_in_visdom(predictions, seq_name, output_dir)

            if self.cfg.gr_visualize:
                self.visualize_3D_in_gradio(predictions, seq_name, output_dir)

            return predictions
        
    def align_gt(self, predictions, output_dir=None):
        """
        Align the predicted camera poses and depths with the ground truth.
        """
        # Align the camera poses
        
        if self.cfg.filter_invalid_frame:
            valid_frame_mask = predictions["valid_frame_mask"]
            self.gt_poses = self.gt_poses[valid_frame_mask]
        
        predictions["aligned_camera_poses"] = self.gt_poses
        
        if self.gt_poses is not None:
            # last image as query
            # select frame besides the last one
            if predictions["extrinsics_opencv"].shape[0] == self.gt_poses.shape[0]:
                gt_pose = self.gt_poses[:-1]
            else:
                gt_pose = self.gt_poses
            
            align_t_R, align_t_T, align_t_s = align_camera_extrinsics(
                predictions["extrinsics_opencv"][:-1],
                gt_pose,
                estimate_scale=True,
            )
            # log scale
            logger.debug(f"scale: {align_t_s}")
            
            # only calulate the last frame error
            predictions["extrinsics_opencv"] = apply_transformation(
                predictions["extrinsics_opencv"],
                align_t_R,
                align_t_T,
                align_t_s,
            )
            if predictions["extrinsics_opencv"].shape[0] == self.gt_poses.shape[0]:
                err_R = rotation_angle(
                    self.gt_poses[-1][:3, :3].unsqueeze(0),
                    predictions["extrinsics_opencv"][-1][:3, :3].unsqueeze(0),
                )
                err_t_degree = translation_angle(
                    self.gt_poses[-1][:3, 3].unsqueeze(0),
                    predictions["extrinsics_opencv"][-1][:3, 3].unsqueeze(0),
                )
                
                err_t = translation_meters(
                    self.gt_poses[-1][:3, 3].unsqueeze(0),
                    predictions["extrinsics_opencv"][-1][:3, 3].unsqueeze(0),
                    input_unit="m",
                )
                
                logger.info(f"query frame rotation error: {err_R}")
                logger.info(f"query frame translation error: {err_t}")
                
                if self.model_path is not None:
                    # calucate the add0.1d and proj2d error
                    ADD = add_metric(
                        self.model_path,
                        predictions["extrinsics_opencv"][-1].unsqueeze(0).cpu().numpy(),
                        self.gt_poses[-1].unsqueeze(0).cpu().numpy())
                    
                    
                    proj2d = projection_2d_error(
                        self.model_path,
                        predictions["extrinsics_opencv"][-1].unsqueeze(0).cpu().numpy(),
                        self.gt_poses[-1].unsqueeze(0).cpu().numpy())
                else:
                    ADD = None
                    proj2d = None
                    
                        
                
                # save error to output_dir if provided
                if output_dir is not None:
                    # save metrcis as json
                    metrics = {
                        "rotation_error": err_R.item(),
                        "translation_error_degree": err_t_degree.item(),
                        "translation_error_meter": err_t.item(),
                        "add": ADD,
                        "proj2d": proj2d,
                    }
                    import json
                    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                        json.dump(metrics, f)
            
            predictions["points3D"] = self.transform_points(
                predictions["points3D"],
                align_t_R,
                align_t_T,
                align_t_s,
            )
            
            predictions['metrics'] = metrics
            
            
        else:
            logger.warning("No ground truth camera poses provided")
            
        
        return predictions
    
    
    def transform_points(self, points, R, T, s):
        """
        Transform the points using the given rotation, translation, and scale.
        """
        points = points * s
        points = torch.matmul(points, R) + T

        return points.squeeze(0)
    
    
    def visualize_3D_in_visdom(
        self, predictions, seq_name=None, output_dir=None
    ):
        """
        This function takes the predictions from the reconstruction process and visualizes
        the 3D point cloud and camera positions in Visdom. It handles both sparse and dense
        reconstructions if available. Requires a running Visdom server and PyTorch3D library.

        Args:
            predictions (dict): Reconstruction results including 3D points and camera parameters.
            seq_name (str, optional): Sequence name for visualization.
            output_dir (str, optional): Directory for saving output files.
        """

        if "points3D_rgb" in predictions:
            pcl = Pointclouds(
                points=predictions["points3D"][None],
                features=predictions["points3D_rgb"][None],
            )
        else:
            pcl = Pointclouds(points=predictions["points3D"][None])

        extrinsics_opencv = predictions["extrinsics_opencv"]

        # From OpenCV/COLMAP to PyTorch3D
        rot_PT3D = extrinsics_opencv[:, :3, :3].clone().permute(0, 2, 1)
        trans_PT3D = extrinsics_opencv[:, :3, 3].clone()
        trans_PT3D[:, :2] *= -1
        rot_PT3D[:, :, :2] *= -1
        visual_cameras = PerspectiveCamerasVisual(
            R=rot_PT3D, T=trans_PT3D, device=trans_PT3D.device
        )

        visual_dict = {"scenes": {"points": pcl, "cameras": visual_cameras}}

        unproj_dense_points3D = predictions["unproj_dense_points3D"]
        if unproj_dense_points3D is not None:
            unprojected_rgb_points_list = []
            for unproj_img_name in sorted(unproj_dense_points3D.keys()):
                unprojected_rgb_points = torch.from_numpy(
                    unproj_dense_points3D[unproj_img_name]
                )
                unprojected_rgb_points_list.append(unprojected_rgb_points)

                # Separate 3D point locations and RGB colors
                point_locations = unprojected_rgb_points[0]  # 3D point location
                rgb_colors = unprojected_rgb_points[1]  # RGB color

                # Create a mask for points within the specified range
                valid_mask = point_locations.abs().max(-1)[0] <= 512

                # Create a Pointclouds object with valid points and their RGB colors
                point_cloud = Pointclouds(
                    points=point_locations[valid_mask][None],
                    features=rgb_colors[valid_mask][None],
                )

                # Add the point cloud to the visual dictionary
                visual_dict["scenes"][f"unproj_{unproj_img_name}"] = point_cloud

        fig = plot_scene(visual_dict, camera_scale=0.05)

        env_name = f"demo_visual_{seq_name}"
        print(f"Visualizing the scene by visdom at env: {env_name}")

        self.viz.plotlyplot(fig, env=env_name, win="3D")
