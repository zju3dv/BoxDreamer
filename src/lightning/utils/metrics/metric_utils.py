import copy
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from time import time

import numpy as np
import torch
from scipy import spatial

from src.utils.customize.sample_points_on_cad import get_all_points_on_model

from src.utils.log import INFO, ERROR

# import torchmetrics
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio


class Metrics:
    """A class to aggregate and compute various metrics for pose estimation.

    Attributes:
        metrics_config (dict): Configuration for selecting which metrics to calculate.
        metrics_result (dict): Stores the computed metrics.
        data (dict): Holds the current batch of data being processed.
        dataloader_id (int): Identifier for the current dataloader.
        dataloader_id_set (set): Set of all processed dataloader IDs.
    """

    def __init__(self, metrics_config=None):
        """Initializes the Metrics handler with the given configuration.

        Args:
            metrics_config (dict): Configuration dict specifying which metrics to compute.
        """
        assert metrics_config is not None, "Metrics config is None!"
        self.metrics_config = metrics_config

        for metric in self.metrics_config.metrics_list:
            INFO(f"Metrics: {metric} will be calculated!")

        self.metrics_result = {}
        self.data = None
        self.dataloader_id = 0
        self.dataloader_id_set = set()

        if "image" in self.metrics_config.metrics_list:
            # self.fid = FrechetInceptionDistance()
            # self.kid = KernelInceptionDistance()
            self.psnr = PeakSignalNoiseRatio()

        INFO("Metrics handler is initialized!")

    def reset(self):
        """Resets the metrics results and cleans up memory."""
        self.metrics_result = {}
        self.data = None
        self.dataloader_id_set = set()
        gc.collect()

    def reset_config(self, metrics_config):
        """Updates the metrics configuration.

        Args:
            metrics_config (dict): New configuration dict.
        """
        self.metrics_config = metrics_config

    def set_data(self, data):
        """Sets the current batch data using a deep copy to avoid unintended
        modifications.

        Args:
            data (dict): Batch data.
        """
        self.data = copy.deepcopy(data)

    def get_metrics(self):
        """Retrieves the computed metrics.

        Returns:
            dict: The metrics results.
        """
        return self.metrics_result

    def set_metrics(self, metrics):
        """Sets the metrics results.

        Args:
            metrics (dict): Metrics to set.
        """
        self.metrics_result = metrics

    def compute_metrics(self, data, dataloader_id=0):
        """Computes the specified metrics for the given data.

        Args:
            data (dict): Batch data containing predictions and ground truths.
            dataloader_id (int): Identifier for the dataloader.
        """
        batch_results = {}

        self.dataloader_id = dataloader_id
        self.dataloader_id_set.add(dataloader_id)

        if "pose_error" not in self.metrics_config.metrics_list:
            ERROR("No pose error in the metrics list, Metrics handler will not work!")
            self.reset()
            return

        R_err, t_err, inplane_R_err = self.compute_query_pose_errors_mp(data)
        batch_results["R_err"] = R_err
        batch_results["t_err"] = t_err
        batch_results["inplane_R_err"] = inplane_R_err
        if "proj2d" in self.metrics_config.metrics_list:
            self.projection_2d_error_mp(copy.deepcopy(data))
        if "ADD" in self.metrics_config.metrics_list:
            self.add_metric_mp(copy.deepcopy(data))
        if "image" in self.metrics_config.metrics_list:
            image_metrics = self.rgb_metrics(copy.deepcopy(data), dataloader_id)

            for key in image_metrics:
                batch_results[key] = image_metrics[key]

        return batch_results

    def rgb_metrics(self, data, dataloader_id=0):
        # calculate psnr
        result_key = f"psnr_{dataloader_id}"
        if "cat" in data:
            for cat in data["cat"]:
                self.metrics_result.setdefault(result_key, {}).setdefault(cat, [])
            self.metrics_result[result_key].setdefault("all", [])
        else:
            self.metrics_result.setdefault(result_key, [])

        bs = len(data["pred_images"])
        ret = {}
        ret["psnr"] = []
        for i in range(bs):
            query_idx = data["query_idx"][i].numpy()
            pred_img = data["pred_images"][i]
            gt_img = data["images"][i]

            psnr = self.psnr(
                pred_img[query_idx].unsqueeze(0), gt_img[query_idx].unsqueeze(0)
            )

            if "cat" in data:
                self.metrics_result[result_key][data["cat"][i]].append(psnr)
                self.metrics_result[result_key]["all"].append(psnr)
            else:
                self.metrics_result[result_key].append(psnr)

            ret["psnr"].append(psnr)

        return ret

    def query_pose_error(self, pose_pred, pose_gt):
        """Calculates the angular and translation errors between predicted and
        ground truth poses.

        Args:
            pose_pred (np.ndarray): Predicted pose matrix (3x4 or 4x4).
            pose_gt (np.ndarray): Ground truth pose matrix (3x4 or 4x4).

        Returns:
            tuple: (angular_distance in degrees, translation_error in specified units)
        """
        # Ensure poses are 3x4
        pose_pred = pose_pred[:3] if pose_pred.shape[0] == 4 else pose_pred
        pose_gt = pose_gt[:3] if pose_gt.shape[0] == 4 else pose_gt

        # Calculate translation error based on scaling
        t_scale = self.metrics_config.t_scale
        translation_error = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
        if t_scale == "m":
            translation_error *= 100  # Convert meters to centimeters
        elif t_scale == "mm":
            translation_error /= 10  # Convert millimeters to centimeters

        # Calculate rotational difference using rotation matrices
        rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
        trace = np.clip(np.trace(rotation_diff), -1.0, 3.0)
        angular_distance = np.rad2deg(
            np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        )

        # Handle numerical issues
        angular_distance = (
            0.0
            if np.isnan(angular_distance) or np.isinf(angular_distance)
            else angular_distance
        )
        translation_error = (
            0.0
            if np.isnan(translation_error) or np.isinf(translation_error)
            else translation_error
        )

        # calculate in-plane rotation error
        in_plane_rotation_error = np.rad2deg(
            np.arctan2(rotation_diff[1, 0], rotation_diff[0, 0])
        )
        in_plane_rotation_error = np.abs(in_plane_rotation_error)

        return angular_distance, translation_error, in_plane_rotation_error

    @lru_cache(maxsize=None)
    def get_cached_points(self, model_path: str):
        """Caches and retrieves all 3D points on the model.

        Args:
            model_path (str): Path to the 3D model file.

        Returns:
            np.ndarray: Array of 3D points.
        """
        return get_all_points_on_model(model_path)

    def project_optimized(self, xyz: np.ndarray, K: np.ndarray, RT: np.ndarray):
        """Projects 3D points to 2D using the camera intrinsics and extrinsics.

        Args:
            xyz (np.ndarray): 3D points (N x 3).
            K (np.ndarray): Camera intrinsic matrix (3 x 3).
            RT (np.ndarray): Camera extrinsic matrix (3 x 4).

        Returns:
            np.ndarray: 2D projected points (N x 2).
        """
        rotated = xyz @ RT[:, :3].T
        translated = rotated + RT[:, 3]
        projected = translated @ K.T
        xy = projected[:, :2] / projected[:, 2, np.newaxis]
        return xy

    def project(self, xyz: np.ndarray, K: np.ndarray, RT: np.ndarray) -> np.ndarray:
        """Projects 3D points to 2D using the camera projection matrix.

        Args:
            xyz (np.ndarray): 3D points (N x 3).
            K (np.ndarray): Camera intrinsic matrix (3 x 3).
            RT (np.ndarray): Camera extrinsic matrix (3 x 4).

        Returns:
            np.ndarray: 2D projected points (N x 2).
        """
        projected = self.project_optimized(xyz, K, RT)
        return projected

    def process_single_bs_2d(
        self, bs: int, data: dict, metrics_result: dict, dataloader_id: int
    ) -> None:
        """Processes a single batch sample for 2D projection error.

        Args:
            bs (int): Batch index.
            data (dict): Batch data.
            metrics_result (dict): Dictionary to store metrics.
            dataloader_id (int): Identifier for the dataloader.
        """
        query_idx = data["query_idx"][bs].numpy()
        model_path = data["model_path"][query_idx][bs]
        assert os.path.exists(model_path), f"Model path does not exist: {model_path}"
        suffix = model_path.split("/")[-3]
        model_path = model_path.replace(suffix, "models")
        model_3D_pts = self.get_cached_points(model_path)
        pose_gt = data["original_poses"][bs][query_idx].numpy()
        pose_pred = data["pred_poses"][bs][query_idx].numpy()
        coordinates_transform = data["coordinate_transform"][bs].numpy()
        scale = data["scale"][bs][query_idx].numpy()
        K = data["original_intrinsics"][bs][query_idx].numpy()
        cat = data.get("cat", None)
        if cat is not None:
            cat = data["cat"][bs]

        # Apply scaling to translation components
        pose_pred[:3, 3] *= scale
        pose_pred = pose_pred @ coordinates_transform

        if pose_pred.shape[0] == 4:
            pose_pred = pose_pred[:3]
        if pose_gt.shape[0] == 4:
            pose_gt = pose_gt[:3]

        # Project points using predicted and ground truth poses
        model_2d_pred = self.project(model_3D_pts, K, pose_pred)
        model_2d_gt = self.project(model_3D_pts, K, pose_gt)

        # Compute mean Euclidean distance between projections
        proj_diff = model_2d_pred - model_2d_gt
        proj_mean_diff = np.mean(np.linalg.norm(proj_diff, axis=1))

        # Store the metric
        result_key = f"proj2D_metric_{dataloader_id}"
        if cat is not None:
            metrics_result.setdefault(result_key, {}).setdefault(cat, []).append(
                proj_mean_diff
            )
            metrics_result[result_key].setdefault("all", []).append(proj_mean_diff)
        else:
            metrics_result.setdefault(result_key, []).append(proj_mean_diff)

    def projection_2d_error_mp(self, data):
        """Computes 2D projection errors in parallel.

        Args:
            data (dict): Batch data.
        """
        batch_size = len(data["pred_poses"])
        with ThreadPoolExecutor(
            max_workers=min(batch_size, os.cpu_count())
        ) as executor:
            futures = [
                executor.submit(
                    self.process_single_bs_2d,
                    bs,
                    data,
                    self.metrics_result,
                    self.dataloader_id,
                )
                for bs in range(batch_size)
            ]
            for future in futures:
                future.result()  # Ensure all tasks are completed

    def process_single_bs_add(
        self, bs: int, data: dict, metrics_result: dict, dataloader_id: int
    ):
        """Processes a single batch sample for ADD metric.

        Args:
            bs (int): Batch index.
            data (dict): Batch data.
            metrics_result (dict): Dictionary to store metrics.
            dataloader_id (int): Identifier for the dataloader.
        """
        percentage = 0.1
        syn = False  # Set to True if using synthetic data
        model_unit = self.metrics_config.t_scale

        query_idx = data["query_idx"][bs].numpy()
        model_path = data["model_path"][query_idx][bs]
        # model path format :
        # xxx/xxx/models_suffix/obj_id/obj_id.ply
        # replace models_suffix with models to get gt cad model path
        suffix = model_path.split("/")[-3]
        model_path = model_path.replace(suffix, "models")

        assert os.path.exists(model_path), f"Model path does not exist: {model_path}"
        model_3D_pts = self.get_cached_points(model_path)
        pose_gt = data["original_poses"][bs][query_idx].numpy()
        pose_pred = data["pred_poses"][bs][query_idx].numpy()
        coordinates_transform = data["coordinate_transform"][bs].numpy()
        scale = data["scale"][bs][query_idx].numpy()
        K = data["original_intrinsics"][bs][query_idx].numpy()
        cat = data.get("cat", None)
        if cat is not None:
            cat = data["cat"][bs]

        # Apply scaling to translation components
        pose_pred[:3, 3] *= scale
        pose_pred = pose_pred @ coordinates_transform

        if pose_pred.shape[0] == 4:
            pose_pred = pose_pred[:3]
        if pose_gt.shape[0] == 4:
            pose_gt = pose_gt[:3]

        # Compute model points in world coordinates
        model_pred = (model_3D_pts @ pose_pred[:, :3].T) + pose_pred[:, 3]
        model_gt = (model_3D_pts @ pose_gt[:, :3].T) + pose_gt[:, 3]

        # Use KDTree for nearest neighbor distance if synthetic
        kd_tree = spatial.cKDTree(model_pred)
        adds_mean_dist, _ = kd_tree.query(model_gt, k=1)
        adds_mean_dist = np.mean(adds_mean_dist)

        # Compute mean Euclidean distance
        add_mean_dist = np.mean(np.linalg.norm(model_pred - model_gt, axis=-1))

        # Compute diameter threshold
        diameter = np.linalg.norm(
            np.max(model_3D_pts, axis=0) - np.min(model_3D_pts, axis=0)
        )
        diameter_thres = diameter * percentage

        # absolute_thres = 0.1 if model_unit == 'm' else 10.0  # 10 cm or 0.1 m

        # Determine if the mean distance is below the threshold # ADD-0.1d
        add_score = 1.0 if add_mean_dist < diameter_thres else 0.0
        adds_score = 1.0 if adds_mean_dist < diameter_thres else 0.0

        # Store the metric
        result_key = f"ADD_0.1d_{dataloader_id}"
        adds_result_key = f"ADDs_0.1d_{dataloader_id}"
        result_raw_key = f"ADD_raw_{dataloader_id}"
        adds_raw_key = f"ADDs_raw_{dataloader_id}"
        if cat is not None:
            metrics_result.setdefault(result_key, {}).setdefault(cat, []).append(
                add_score
            )
            metrics_result[result_key].setdefault("all", []).append(add_score)
            metrics_result.setdefault(result_raw_key, {}).setdefault(cat, []).append(
                add_mean_dist
            )
            metrics_result[result_raw_key].setdefault("all", []).append(add_mean_dist)
            metrics_result.setdefault(adds_result_key, {}).setdefault(cat, []).append(
                adds_score
            )
            metrics_result[adds_result_key].setdefault("all", []).append(adds_score)
            metrics_result.setdefault(adds_raw_key, {}).setdefault(cat, []).append(
                adds_mean_dist
            )
            metrics_result[adds_raw_key].setdefault("all", []).append(adds_mean_dist)
        else:
            metrics_result.setdefault(result_key, []).append(add_score)
            metrics_result.setdefault(result_raw_key, []).append(add_mean_dist)
            metrics_result.setdefault(adds_result_key, []).append(adds_score)
            metrics_result.setdefault(adds_raw_key, []).append(adds_mean_dist)

    def add_metric_mp(self, data):
        """Computes ADD metrics in parallel.

        Args:
            data (dict): Batch data.
        """
        batch_size = len(data["pred_poses"])
        with ThreadPoolExecutor(
            max_workers=min(batch_size, os.cpu_count())
        ) as executor:
            futures = [
                executor.submit(
                    self.process_single_bs_add,
                    bs,
                    data,
                    self.metrics_result,
                    self.dataloader_id,
                )
                for bs in range(batch_size)
            ]
            for future in futures:
                future.result()  # Ensure all tasks are completed

    @torch.no_grad()
    def process_single_batch_regression(self, pose_pred, pose_gt):
        """Processes a single batch sample for pose regression errors.

        Args:
            pose_pred (np.ndarray): Predicted pose matrix.
            pose_gt (np.ndarray): Ground truth pose matrix.

        Returns:
            tuple: (rotational error, translational error)
        """
        return self.query_pose_error(pose_pred, pose_gt)

    @torch.no_grad()
    def compute_query_pose_errors_mp(self, data):
        """Computes pose errors in parallel for the entire batch.

        Args:
            data (dict): Batch data.
        """
        query_idx = data["query_idx"].numpy()
        batch_indices = torch.arange(query_idx.shape[0])
        pose_gt = data["original_poses"][batch_indices, query_idx].numpy()
        pose_pred = data["pred_poses"][batch_indices, query_idx].clone().numpy()
        original_paths = np.array(data["original_images"])[query_idx, batch_indices]
        scales = data["scale"][batch_indices, query_idx].numpy()
        coordinate_transform = data["coordinate_transform"][
            batch_indices
        ].numpy()  # (4, 4)
        cat = data.get("cat", None)

        # Apply scaling to translation components
        pose_pred[:, :3, 3] *= scales
        # apply coordinate transform
        pose_pred = pose_pred @ coordinate_transform

        if cat is not None:
            result_R_key = f"R_errs_{self.dataloader_id}"
            result_t_key = f"t_errs_{self.dataloader_id}"
            result_inplane_R_key = f"inplane_R_errs_{self.dataloader_id}"
            result_pose_key = f"pred_poses_{self.dataloader_id}"
            original_paths_key = f"original_paths_{self.dataloader_id}"
            self.metrics_result.setdefault(result_R_key, {}).setdefault("all", [])
            self.metrics_result.setdefault(result_t_key, {}).setdefault("all", [])
            self.metrics_result.setdefault(result_inplane_R_key, {}).setdefault(
                "all", []
            )
            self.metrics_result.setdefault(result_pose_key, {}).setdefault("all", [])
            self.metrics_result.setdefault(original_paths_key, {}).setdefault("all", [])
            categories = data["cat"]
        else:
            result_R_key = f"R_errs_{self.dataloader_id}"
            result_t_key = f"t_errs_{self.dataloader_id}"
            result_inplane_R_key = f"inplane_R_errs_{self.dataloader_id}"
            result_pose_key = f"pred_poses_{self.dataloader_id}"
            original_paths_key = f"original_paths_{self.dataloader_id}"
            self.metrics_result.setdefault(result_R_key, [])
            self.metrics_result.setdefault(result_t_key, [])
            self.metrics_result.setdefault(result_inplane_R_key, [])
            self.metrics_result.setdefault(result_pose_key, [])
            self.metrics_result.setdefault(original_paths_key, [])
        results = []
        with ThreadPoolExecutor(
            max_workers=min(len(pose_pred), os.cpu_count())
        ) as executor:
            futures = [
                executor.submit(
                    self.process_single_batch_regression, pose_pred[bs], pose_gt[bs]
                )
                for bs in range(len(pose_pred))
            ]
            for future in futures:
                results.append(future.result())

        R_errs, t_errs, inplane_R_errs = zip(*results)

        if cat is not None:
            for bs, category in enumerate(cat):
                self.metrics_result[result_R_key].setdefault(category, []).append(
                    R_errs[bs]
                )
                self.metrics_result[result_t_key].setdefault(category, []).append(
                    t_errs[bs]
                )
                self.metrics_result[result_inplane_R_key].setdefault(
                    category, []
                ).append(inplane_R_errs[bs])
                self.metrics_result[result_pose_key].setdefault(category, []).append(
                    pose_pred[bs]
                )
                self.metrics_result[original_paths_key].setdefault(category, []).append(
                    original_paths[bs]
                )
                self.metrics_result[result_R_key]["all"].append(R_errs[bs])
                self.metrics_result[result_t_key]["all"].append(t_errs[bs])
                self.metrics_result[result_inplane_R_key]["all"].append(
                    inplane_R_errs[bs]
                )
        else:
            self.metrics_result[result_R_key].extend(R_errs)
            self.metrics_result[result_t_key].extend(t_errs)
            self.metrics_result[result_inplane_R_key].extend(inplane_R_errs)
            self.metrics_result[result_pose_key].extend(pose_pred)
            self.metrics_result[original_paths_key].extend(original_paths)

        return R_errs, t_errs, inplane_R_errs

    def aggregate_metrics(self):
        """Aggregates all computed metrics into a final summary.

        Returns:
            dict: Aggregated metrics for the entire dataset.
        """
        agg_metric = {}
        for dataloader_id in self.dataloader_id_set:
            prefix = f"_{dataloader_id}"
            R_key = f"R_errs{prefix}"
            t_key = f"t_errs{prefix}"
            inplane_R_key = f"inplane_R_errs{prefix}"
            ADD_key = f"ADD_0.1d_{dataloader_id}"
            ADDs_key = f"ADDs_0.1d_{dataloader_id}"
            ADD_raw_key = f"ADD_raw_{dataloader_id}"
            ADDs_raw_key = f"ADDs_raw_{dataloader_id}"
            proj2D_key = f"proj2D_metric_{dataloader_id}"
            psnr_key = f"psnr_{dataloader_id}"
            eval_size_key = f"eval size_{dataloader_id}"
            pred_pose_key = f"pred_poses_{dataloader_id}"
            original_paths_key = f"original_paths_{dataloader_id}"

            if isinstance(self.metrics_result[R_key], dict):
                path_pose_dict = {}
                path_error_dict = {}
                for cat, R_errs in self.metrics_result[R_key].items():
                    t_errs = np.array(self.metrics_result[t_key][cat])
                    R_errs = np.array(R_errs)
                    inplane_R_errs = np.array(self.metrics_result[inplane_R_key][cat])
                    eval_length = len(R_errs)
                    unit = "cm" if self.metrics_config.t_scale else "degree"

                    for threshold in self.metrics_config.pose_error.pose_thresholds:
                        condition = (
                            (R_errs < threshold) & (t_errs < threshold)
                        ).astype(np.float32)
                        metric_name = (
                            f"{threshold}{unit}@{threshold}degree_{dataloader_id}"
                        )
                        agg_metric.setdefault(metric_name, {})[cat] = np.mean(condition)

                    if ADD_key in self.metrics_result:
                        ADD_metric = np.array(self.metrics_result[ADD_key][cat])
                        agg_metric.setdefault(f"ADD-0.1d {dataloader_id}", {})[
                            cat
                        ] = np.mean(ADD_metric)

                        ADDs_metric = np.array(self.metrics_result[ADDs_key][cat])
                        agg_metric.setdefault(f"ADDs-0.1d {dataloader_id}", {})[
                            cat
                        ] = np.mean(ADDs_metric)

                        ADD_raw = np.array(self.metrics_result[ADD_raw_key][cat])
                        add_auc_10cm = auc_add(ADD_raw)
                        agg_metric.setdefault(f"ADD-AUC(10cm) {dataloader_id}", {})[
                            cat
                        ] = add_auc_10cm

                        # calculate ADD auc
                        add_auc = compute_auc_sklearn(ADD_raw)
                        agg_metric.setdefault(f"ADD-AUC {dataloader_id}", {})[
                            cat
                        ] = add_auc

                        adds_raw = np.array(self.metrics_result[ADDs_raw_key][cat])
                        adds_auc_10cm = auc_add(adds_raw)
                        agg_metric.setdefault(f"ADDs-AUC(10cm) {dataloader_id}", {})[
                            cat
                        ] = adds_auc_10cm

                        # calculate ADDs auc
                        adds_auc = compute_auc_sklearn(adds_raw)
                        agg_metric.setdefault(f"ADDs-AUC {dataloader_id}", {})[
                            cat
                        ] = adds_auc

                    if proj2D_key in self.metrics_result:
                        proj2D_metric = np.array(self.metrics_result[proj2D_key][cat])
                        condition_2d = (
                            proj2D_metric < self.metrics_config.proj2d.proj2d_thres
                        ).astype(np.float32)
                        agg_metric.setdefault(f"proj2D@5px {dataloader_id}", {})[
                            cat
                        ] = np.mean(condition_2d)

                        # calculate proj2d auc
                        proj2d_auc = auc_proj2d(proj2D_metric)
                        agg_metric.setdefault(f"proj2D-AUC(40px) {dataloader_id}", {})[
                            cat
                        ] = proj2d_auc

                    if psnr_key in self.metrics_result:
                        psnr_metric = np.array(self.metrics_result[psnr_key][cat])
                        agg_metric.setdefault(f"psnr_{dataloader_id}", {})[
                            cat
                        ] = np.mean(psnr_metric)

                    agg_metric.setdefault(eval_size_key, {})[cat] = eval_length

                    # Average errors
                    agg_metric.setdefault(f"avg_err_R_{dataloader_id}", {})[
                        cat
                    ] = np.mean(R_errs)
                    agg_metric.setdefault(f"avg_err_t_{dataloader_id}", {})[
                        cat
                    ] = np.mean(t_errs)
                    agg_metric.setdefault(f"avg_err_inplane_R_{dataloader_id}", {})[
                        cat
                    ] = np.mean(inplane_R_errs)

                    pred_poses = self.metrics_result[pred_pose_key][cat]
                    original_paths = self.metrics_result[original_paths_key][cat]

                    # assign path pose dict
                    path_pose_dict[cat] = {}
                    path_error_dict[cat] = {}
                    for i, path in enumerate(original_paths):
                        path_pose_dict[cat][path.item()] = pred_poses[i]
                        path_error_dict[cat][path.item()] = R_errs[i]

                # save path pose dict
                np.save(f"path_pose_dict_{dataloader_id}.npy", path_pose_dict)
                np.save(f"path_error_dict_{dataloader_id}.npy", path_error_dict)

            else:
                R_errs = np.array(self.metrics_result[R_key])
                t_errs = np.array(self.metrics_result[t_key])
                inplane_R_errs = np.array(self.metrics_result[inplane_R_key])
                eval_length = len(R_errs)
                unit = "cm" if self.metrics_config.t_scale else "degree"

                for threshold in self.metrics_config.pose_error.pose_thresholds:
                    condition = ((R_errs < threshold) & (t_errs < threshold)).astype(
                        np.float32
                    )
                    metric_name = f"{threshold}{unit}@{threshold}degree_{dataloader_id}"
                    agg_metric[metric_name] = np.mean(condition)

                if ADD_key in self.metrics_result:
                    ADD_metric = np.array(self.metrics_result[ADD_key])
                    agg_metric[f"ADD metric_{dataloader_id}"] = np.mean(ADD_metric)

                if proj2D_key in self.metrics_result:
                    proj2D_metric = np.array(self.metrics_result[proj2D_key])
                    condition_2d = (
                        proj2D_metric < self.metrics_config.proj2d.proj2d_thres
                    ).astype(np.float32)
                    agg_metric[f"proj2D metric_{dataloader_id}"] = np.mean(condition_2d)

                if psnr_key in self.metrics_result:
                    psnr_metric = np.array(self.metrics_result[psnr_key])
                    agg_metric[f"psnr_{dataloader_id}"] = np.mean(psnr_metric)

                agg_metric[eval_size_key] = eval_length

                # Average errors
                agg_metric[f"avg_err_R_{dataloader_id}"] = np.mean(R_errs)
                agg_metric[f"avg_err_t_{dataloader_id}"] = np.mean(t_errs)
                agg_metric[f"avg_err_inplane_R_{dataloader_id}"] = np.mean(
                    inplane_R_errs
                )

        return agg_metric


def convert_pose2T(pose):
    """Converts a pose consisting of rotation and translation into a 4x4
    transformation matrix.

    Args:
        pose (tuple): (R, t) where R is 3x3 rotation matrix and t is 3-element translation vector.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    R, t = pose
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def angle_error_vec(v1, v2):
    """Computes the angular error between two vectors in degrees.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.

    Returns:
        float: Angular error in degrees.
    """
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / norm_product, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    """Computes the angular error between two rotation matrices in degrees.

    Args:
        R1 (np.ndarray): First rotation matrix.
        R2 (np.ndarray): Second rotation matrix.

    Returns:
        float: Angular error in degrees.
    """
    cos_angle = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical issues
    return np.rad2deg(np.abs(np.arccos(cos_angle)))


import sklearn


def auc_add(metrics):
    thresholds = np.linspace(0.0, 0.10, 1000)
    x_range = thresholds.max() - thresholds.min()
    results = metrics
    accuracies = [(results <= t).sum() / len(results) for t in thresholds]
    auc = sklearn.metrics.auc(thresholds, accuracies) / x_range
    return auc


def auc_proj2d(metrics):
    thresholds = np.linspace(0, 40.0, 1000)
    x_range = thresholds.max() - thresholds.min()
    results = metrics
    accuracies = [(results <= t).sum() / len(results) for t in thresholds]
    auc = sklearn.metrics.auc(thresholds, accuracies) / x_range
    return auc


def compute_auc_sklearn(errs, max_val=0.1, step=0.001):
    from sklearn import metrics

    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val + step, step)
    Y = np.ones(len(X))
    for i, x in enumerate(X):
        y = (errs <= x).sum() / len(errs)
        Y[i] = y
        if y >= 1:
            break
    auc = metrics.auc(X, Y) / (max_val * 1)
    return auc
