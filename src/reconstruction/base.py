# base class for large number of (in the future) reconstruction methods

import numpy as np
import torch
import os
import cv2
from src.lightning.utils.vis.vis_utils import reproj, draw_3d_box, get_3d_bbox_from_pts
from src.utils.customize.sample_points_on_cad import get_all_points_on_model
import loguru
import trimesh


class BaseReconstructor:
    def __init__(self, methods=None):
        self.methods = methods
        assert self.methods is not None, "Please specify the reconstruction methods"

        self.images = None
        self.masks = None
        self.intinsics = None
        self.gt_poses = None  # use this to align the predicted poses to the ground truth poses coordinates
        self.pred_poses = None
        self.aligned_poses = None
        self.pt3d = None
        self.gt_pt3d = None
        self.pred_intrinsics = None
        self.model_path = None
        self.device = "cuda"

    def set_device(self, device):
        self.device = device

    def run(self):
        raise NotImplementedError("Please implement the run method")

    def real_run(self):
        raise NotImplementedError("Please implement the real_run method")

    def set_data(self, images, masks=None, gt_poses=None, intinsics=None, model=None):
        self.images = images
        self.masks = masks
        self.gt_poses = gt_poses
        self.intinsics = intinsics
        self.model_path = model

        if self.intinsics is not None:
            # load intrinsics from txt file
            intrinsics = []
            for path in self.intinsics:
                if not os.path.exists(path):
                    intrinsics.append(None)
                    continue
                with open(path, "r") as f:
                    lines = f.readlines()
                    intrinsics.append(
                        np.array(
                            [list(map(float, line.strip().split())) for line in lines]
                        )
                    )

            # to tensor
            self.intinsics = torch.tensor(intrinsics, dtype=torch.float32)

    def set_processed_data(self, data_list):
        # input is a list of batch input which preprocessed by the dataloader
        # all data keys:
        # images, non_ndc_intrinsics, poses, model_path, cat (name of obj)
        self._reset_data_to_list()
        obj = data_list[0]["cat"][0]
        for batch in data_list:
            self.images.append(batch["images"])
            self.intinsics.append(batch["non_ndc_intrinsics"])
            self.gt_poses.append(batch["poses"])
            self.model_path = batch["model_path"]  # same obj, same model path

        self.images = torch.cat(self.images, dim=0).squeeze(1)
        self.intinsics = torch.cat(self.intinsics, dim=0).squeeze(1)
        self.gt_poses = torch.cat(self.gt_poses, dim=0).squeeze(1)
        loguru.logger.info(f"Set processed {obj}'s data with {len(self.images)} images")
        # log shape
        loguru.logger.info(f"Images shape: {self.images.shape}")
        return True

    def _reset_data_to_list(self):
        self.images = []
        self.masks = []
        self.intinsics = []
        self.gt_poses = []
        self.pred_poses = []
        self.aligned_poses = []
        self.pt3d = []
        self.gt_pt3d = []
        self.pred_intrinsics = []

    def reset_data(self):
        self.images = None
        self.masks = None
        self.intinsics = None
        self.gt_poses = None
        self.pred_poses = None
        self.aligned_poses = None
        self.pt3d = None
        self.gt_pt3d = None
        self.pred_intrinsics = None

    def _align_camera_extrinsics(
        self,
        cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
        cameras_tgt: torch.Tensor,  # Bx3x4 tensor representing [R | t]
        estimate_scale: bool = True,
        eps: float = 1e-9,
    ):
        """Align the source camera extrinsics to the target camera extrinsics.
        NOTE Assume OPENCV convention.

        Args:
            cameras_src (torch.Tensor): Bx3x4 tensor representing [R | t] for source cameras.
            cameras_tgt (torch.Tensor): Bx3x4 tensor representing [R | t] for target cameras.
            estimate_scale (bool, optional): Whether to estimate the scale factor. Default is True.
            eps (float, optional): Small value to avoid division by zero. Default is 1e-9.

        Returns:
            align_t_R (torch.Tensor): 1x3x3 rotation matrix for alignment.
            align_t_T (torch.Tensor): 1x3 translation vector for alignment.
            align_t_s (float): Scaling factor for alignment.
        """
        cameras_src = cameras_src.to(torch.float64)
        cameras_tgt = cameras_tgt.to(torch.float64)

        R_src = cameras_src[:, :, :3]  # Extracting the rotation matrices from [R | t]
        R_tgt = cameras_tgt[:, :, :3]  # Extracting the rotation matrices from [R | t]

        RRcov = torch.bmm(R_tgt.transpose(2, 1), R_src).mean(0)
        U, _, V = torch.svd(RRcov)
        align_t_R = V @ U.t()

        T_src = cameras_src[:, :, 3]  # Extracting the translation vectors from [R | t]
        T_tgt = cameras_tgt[:, :, 3]  # Extracting the translation vectors from [R | t]

        A = torch.bmm(T_src[:, None], R_src)[:, 0]
        B = torch.bmm(T_tgt[:, None], R_src)[:, 0]

        Amu = A.mean(0, keepdim=True)
        Bmu = B.mean(0, keepdim=True)

        if estimate_scale and A.shape[0] > 1:
            # get the scaling component by matching covariances
            # of centered A and centered B
            Ac = A - Amu
            Bc = B - Bmu
            align_t_s = (Ac * Bc).mean() / (Ac**2).mean().clamp(eps)
        else:
            # set the scale to identity
            align_t_s = 1.0

        # get the translation as the difference between the means of A and B
        align_t_T = Bmu - align_t_s * Amu

        align_t_R = align_t_R[None]
        return align_t_R, align_t_T, align_t_s

    def _apply_transformation(
        self,
        cameras_src: torch.Tensor,  # Bx3x4 tensor representing [R | t]
        align_t_R: torch.Tensor,  # 1x3x3 rotation matrix
        align_t_T: torch.Tensor,  # 1x3 translation vector
        align_t_s: float,  # Scaling factor
        return_extri: bool = True,
    ) -> torch.Tensor:
        """Align and transform the source cameras using the provided rotation,
        translation, and scaling factors. NOTE Assume OPENCV convention.

        Args:
            cameras_src (torch.Tensor): Bx3x4 tensor representing [R | t] for source cameras.
            align_t_R (torch.Tensor): 1x3x3 rotation matrix for alignment.
            align_t_T (torch.Tensor): 1x3 translation vector for alignment.
            align_t_s (float): Scaling factor for alignment.

        Returns:
            aligned_R (torch.Tensor): Bx3x3 tensor representing the aligned rotation matrices.
            aligned_T (torch.Tensor): Bx3 tensor representing the aligned translation vectors.
        """

        R_src = cameras_src[:, :, :3]
        T_src = cameras_src[:, :, 3]

        aligned_R = torch.bmm(R_src, align_t_R.expand(R_src.shape[0], 3, 3))

        # Apply the translation alignment to the source translations
        # aligned_T = (
        #     torch.bmm(
        #         R_src,
        #         align_t_T[..., None].repeat(R_src.shape[0], 1, 1)
        #     )[..., 0] + T_src * align_t_s
        # )

        # Apply the translation alignment to the source translations
        align_t_T_expanded = align_t_T[..., None].repeat(R_src.shape[0], 1, 1)
        transformed_T = torch.bmm(R_src, align_t_T_expanded)[..., 0]
        aligned_T = transformed_T + T_src * align_t_s

        if return_extri:
            extri = torch.cat([aligned_R, aligned_T.unsqueeze(-1)], dim=-1)
            return extri

        return aligned_R, aligned_T

    def _umeyama_alignment(self, source, target, with_scale=True):
        """Perform the Umeyama alignment to estimate the similarity
        transformation (scale, rotation, translation) that best maps the source
        to the target.

        Args:
            source (np.ndarray): Source point cloud of shape [N, 3].
            target (np.ndarray): Target point cloud of shape [N, 3].
            with_scale (bool): Whether to estimate scaling factor.

        Returns:
            T (np.ndarray): 4x4 transformation matrix.
        """
        assert (
            source.shape == target.shape
        ), "Source and target must have the same shape."

        # Number of points
        N = source.shape[0]

        # Compute centroids
        mean_src = source.mean(axis=0)
        mean_tgt = target.mean(axis=0)

        # Center the points
        src_centered = source - mean_src
        tgt_centered = target - mean_tgt

        # Compute covariance matrix
        covariance_matrix = np.dot(tgt_centered.T, src_centered) / N

        # Singular Value Decomposition
        U, D, Vt = np.linalg.svd(covariance_matrix)

        # Compute rotation
        R = np.dot(U, Vt)

        # Ensure a right-handed coordinate system
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = np.dot(U, Vt)

        if with_scale:
            # Compute scaling factor
            var_src = np.var(src_centered, axis=0).sum()
            scale = np.trace(np.dot(np.diag(D), np.eye(3))) / var_src
        else:
            scale = 1.0

        # Compute translation
        t = mean_tgt - scale * np.dot(R, mean_src)

        # Construct the transformation matrix
        T = np.identity(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t

        return T

    def _pt_registration(self):
        """Point cloud registration."""
        rotation, translation, scale, aligned_pt3, err = self.align_point_cloud(
            self.pt3d, self.gt_pt3d
        )
        self.pt3d = aligned_pt3
        # log avg err
        print(f"Average registration error: {err.mean()}")
        return rotation, translation, scale, aligned_pt3, err

    def _load_gt_pt3d(self):
        """Load the ground truth point cloud."""
        if self.model_path is None or not os.path.exists(self.model_path):
            print("No model path exists, skip loading the ground truth point cloud")
            return False
        else:
            self.gt_pt3d = get_all_points_on_model(self.model_path)
            # convert to tensor
            self.gt_pt3d = torch.tensor(self.gt_pt3d).float()
            return True

    def align_coordinates(self):
        """Align the predicted poses to the ground truth poses coordinates.

        Parameters:
            pred_poses (torch.Tensor): Predicted poses
            gt_poses (torch.Tensor): Ground truth poses

        Returns:
            torch.Tensor: Aligned predicted poses
        """
        assert self.gt_poses is not None, "Please set the ground truth poses"
        assert (
            self.pred_poses.shape[0] == self.gt_poses.shape[0]
        ), "The predicted and ground truth poses should have the same shape"
        assert (
            self.pred_poses is not None
        ), "No predicted poses, plz run reconstruction first"

        # convert to save device
        self.gt_poses = self.gt_poses.to(self.pred_poses.device)

        # check data shape
        if len(self.pred_poses.shape) == 2:
            self.pred_poses = self.pred_poses.unsqueeze(0)
            self.gt_poses = self.gt_poses.unsqueeze(0)

        # check extrinsics shape, we need Bx3x4 matrix
        if self.pred_poses.shape[1] == 4:
            self.pred_poses = self.pred_poses[:, :3, :]

        if self.gt_poses.shape[1] == 4:
            self.gt_poses = self.gt_poses[:, :3, :]

        align_t_R, align_t_T, align_t_s = self._align_camera_extrinsics(
            self.pred_poses, self.gt_poses
        )

        self.aligned_poses = self._apply_transformation(
            self.pred_poses, align_t_R, align_t_T, align_t_s
        )

        aligned_pt3d = align_t_s * torch.matmul(
            self.pt3d, align_t_R.squeeze(0).t()
        ) + align_t_T.squeeze(0)

        self.pt3d = aligned_pt3d

        return True

    def draw_3dbbox(self, img_list):
        assert self.pt3d is not None, "Please run the reconstruction first"
        assert self.pred_poses is not None, "Please run the reconstruction first"

        # get 3d bbox points(self.pt3d shape is Nx3)
        bbox3d = get_3d_bbox_from_pts(self.pt3d.cpu().numpy())

        # get the camera extrinsics
        extrinsics = self.pred_poses  # Bx3x4
        # get the camera intrinsics
        intrinsics = self.pred_intrinsics  # Bx3x3

        intrinsics = intrinsics.to(self.pred_poses.device)

        for idx, img_path in enumerate(img_list):
            img = cv2.imread(img_path)
            # reshape the image to 224x224
            # img = cv2.resize(img, (224, 224))
            bbox = reproj(
                intrinsics[idx].detach().cpu().numpy(),
                extrinsics[idx].detach().cpu().numpy(),
                bbox3d,
            )

            img, bbox = self._pad_image_and_bbox(img, bbox)

            img_with_bbox = draw_3d_box(img, bbox)

            # save img for debug
            # get current working directory
            cwd = os.getcwd()
            cv2.imwrite(f"{cwd}/bbox_{idx}.png", img_with_bbox)
            print(f"Save bbox_{idx}.png to {cwd}")

    def _pad_image_and_bbox(self, img, bbox):
        """Pads the image to ensure all bbox points are within the image
        boundaries.

        Adjusts bbox coordinates accordingly.
        :param img: Original image (H, W, C)
        :param bbox: 3D bbox corner points (8, 2)
        :return: Padded image and adjusted bbox
        """
        # Get the height, width, and channels of the image
        h, w, c = img.shape

        # Initialize padding values for top, bottom, left, and right
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

        # Iterate through the bbox points to check if they are outside the image boundaries
        for point in bbox:
            x, y = point
            if x < 0:
                pad_left = max(
                    pad_left, int(-x)
                )  # Update left padding if the point is outside
            if x >= w:
                pad_right = max(
                    pad_right, int(x - w + 1)
                )  # Update right padding if the point is outside
            if y < 0:
                pad_top = max(
                    pad_top, int(-y)
                )  # Update top padding if the point is outside
            if y >= h:
                pad_bottom = max(
                    pad_bottom, int(y - h + 1)
                )  # Update bottom padding if the point is outside

        # If no padding is required, return the original image and bbox
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return img, bbox

        # Apply padding to the image
        img_padded = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # Use black color for padding
        )

        # Adjust the bbox coordinates to account for the padding
        bbox_padded = bbox + np.array([pad_left, pad_top])

        return img_padded, bbox_padded

    def get_poses(self):
        return self.pred_poses

    def get_aligned_poses(self):
        return self.aligned_poses

    def get_intrinsics(self):
        return self.pred_intrinsics

    def get_pt3d(self):
        return self.pt3d

    def _check_gt_poses(self):
        if self.gt_poses is None:
            return False
        elif isinstance(self.gt_poses, torch.Tensor) or isinstance(
            self.gt_poses, np.ndarray
        ):
            return True
        else:
            # load the ground truth poses
            gt_poses = []
            for path in self.gt_poses:
                if not os.path.exists(path):
                    return False
                else:
                    pose = np.eye(4)
                    with open(path, "r") as f:
                        lines = f.readlines()
                        matrix_values = [
                            list(map(float, line.strip().split())) for line in lines
                        ]
                        assert (
                            len(matrix_values) == 3 or len(matrix_values) == 4
                        ) and all(
                            len(row) == 4 for row in matrix_values
                        ), "Pose file must contain a 3x4 or 4x4 matrix."

                        if len(matrix_values) == 4:
                            matrix_3x4 = np.array(matrix_values)[:3, :]
                        else:
                            matrix_3x4 = np.array(matrix_values)

                        R_mat = matrix_3x4[:, :3]
                        T_vec = matrix_3x4[:, 3]
                        pose[:3, :3] = R_mat
                        pose[:3, 3] = T_vec
                        gt_poses.append(pose)

            self.gt_poses = gt_poses
            # to tensor
            self.gt_poses = torch.tensor(self.gt_poses, dtype=torch.float32)

            return True

    def _to_object_coordinate(self, ply_path: str):
        """Convert the point cloud to the object coordinate system.

        Args:
            ply_path: Input PLY file path

        Returns:
            str: Path of the newly saved PLY file
        """
        import numpy as np
        import torch
        from plyfile import PlyData, PlyElement
        import os

        # Load the PLY file
        ply_data = PlyData.read(ply_path)
        vertex = ply_data["vertex"]

        # Convert the point cloud data to a numpy array
        points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T

        # Compute the centroid of the point cloud
        centroid = np.mean(points, axis=0)

        # Move the point cloud's center to the origin
        centered_points = points - centroid

        # Compute the covariance matrix and perform eigenvalue decomposition
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues in descending order and get the corresponding eigenvectors
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_indices]

        # Ensure a right-handed coordinate system
        if np.linalg.det(eigen_vectors) < 0:
            eigen_vectors[:, 2] = -eigen_vectors[:, 2]

        # Rotate the point cloud to align with the principal components
        rotated_points = centered_points @ eigen_vectors

        # World to Object transformation:
        # First translate by -centroid, then rotate by R^T
        # p_obj = R^T @ (p_world - centroid)
        # Or in matrix form: [R^T, -R^T@centroid; 0, 1] @ [p_world; 1]
        R = eigen_vectors  # numpy array

        # Convert to torch tensors for computation with pred_poses
        R_tensor = torch.tensor(
            R, dtype=torch.float32, device=self.pred_poses[0].device
        )
        centroid_tensor = torch.tensor(
            centroid, dtype=torch.float32, device=self.pred_poses[0].device
        )

        # Define Object2World transformation matrix (4x4)
        O2W = torch.eye(4, device=self.pred_poses[0].device)
        O2W[:3, :3] = R_tensor  # No transpose for object->world
        O2W[:3, 3] = centroid_tensor  # Translation component

        # Transform self.pred_poses (world2camera) to object2camera
        if hasattr(self, "pred_poses") and self.pred_poses is not None:
            for i in range(len(self.pred_poses)):
                # W2C @ p_world = camera coords
                # We want O2C @ p_object = camera coords
                # Since p_world = O2W @ p_object
                # W2C @ O2W @ p_object = camera coords
                # Therefore O2C = W2C @ O2W
                self.pred_poses[i] = self.pred_poses[i] @ O2W

        # Create new PLY data
        vertex_data = np.zeros(
            len(rotated_points),
            dtype=[(prop.name, prop.val_dtype) for prop in vertex.properties],
        )

        self.pt3d = torch.tensor(
            rotated_points, dtype=torch.float32, device=self.pred_poses[0].device
        )

        # Copy the rotated point cloud data to the new vertex data
        vertex_data["x"] = rotated_points[:, 0]
        vertex_data["y"] = rotated_points[:, 1]
        vertex_data["z"] = rotated_points[:, 2]

        # Copy other attributes (if any, such as color, etc.)
        for prop in vertex.properties:
            if prop.name not in ["x", "y", "z"]:
                vertex_data[prop.name] = vertex[prop.name]

        # Create a new PLY element
        ply_element = PlyElement.describe(vertex_data, "vertex")

        # Create a new PLY file
        new_ply_path = os.path.splitext(ply_path)[0] + "_object_coord.ply"
        PlyData([ply_element], text=True).write(new_ply_path)

        return new_ply_path

    def _self_pruning(
        self,
        ply_path: str,
        use_pred_pose: bool = False,
        enable_color_filter: bool = False,
    ):
        # this method load the org reconstructed point cloud and then:
        # reproject the point cloud to each image, and then points
        # that can be reprojected to all images are kept, otherwise are removed
        # this method can remove the background points

        # check gt pose
        if not self._check_gt_poses() and not use_pred_pose:
            print("No gt poses, skip self pruning")
            return ply_path
        else:
            self.gt_poses = self.gt_poses if not use_pred_pose else self.pred_poses
            self.intinsics = (
                self.intinsics if not use_pred_pose else self.pred_intrinsics
            )

        # load the point coud
        all_pt3ds, color = get_all_points_on_model(ply_path, True)
        # log init point cloud size
        print(f"Init point cloud size: {len(all_pt3ds)}")
        # if point == 0, raise error

        index = 0
        for K, pose, img in zip(self.intinsics, self.gt_poses, self.images):
            # reproject the point cloud to the image
            # get the camera extrinsics
            extrinsics = pose
            intrinsics = K

            if all_pt3ds.shape[0] < 3000:
                break

            # if tensor, convert to numpy
            if isinstance(extrinsics, torch.Tensor):
                extrinsics = extrinsics.detach().cpu().numpy()
            if isinstance(intrinsics, torch.Tensor):
                intrinsics = intrinsics.detach().cpu().numpy()

            proj_points = reproj(intrinsics, extrinsics, all_pt3ds)

            # check if the points are within the image boundaries
            # if not, remove the points
            # get the height and width of the image
            h, w = 224, 224
            # print(f"Image shape: {h}, {w}")
            # get the points within the image boundaries
            mask = (
                (proj_points[:, 0] >= 0)
                & (proj_points[:, 0] < w)
                & (proj_points[:, 1] >= 0)
                & (proj_points[:, 1] < h)
            )

            all_pt3ds = all_pt3ds[mask]
            color = color[mask]

            # hard filter, ensure the projected points's color is close to the image color
            # get the projected points
            proj_points = proj_points[mask]
            # get the color from the image
            proj_points = proj_points.astype(np.uint8)

            # get the color from the image
            # rgb
            # img shape : 3xHxW, proj_points shape: Nx2
            # img convert to numpy
            if enable_color_filter:
                img = img.detach().cpu().numpy()
                img_color = img[:, proj_points[:, 1], proj_points[:, 0]] * 255
                img_color = img_color.T
                # check the color difference
                color_diff = np.abs(img_color - color)
                # check the color difference
                color_diff = np.linalg.norm(color_diff, axis=1)
                # filter the points
                mask = color_diff < 160
                all_pt3ds = all_pt3ds[mask]
                color = color[mask]

            index += 1

        print(f"Pruned point cloud size: {len(all_pt3ds)}")
        # downsample the point cloud uniformly
        if len(all_pt3ds) > 10000:
            sampled_idx = np.random.choice(len(all_pt3ds), 3000, replace=False)
            all_pt3ds = all_pt3ds[sampled_idx]
            color = color[sampled_idx]

        # log pruned point cloud size
        print(f"Pruned point cloud size: {len(all_pt3ds)}")

        if len(all_pt3ds) == 0:
            raise ValueError("No points on the model")

        file_suffix = ply_path.split(".")[-1]
        file_suffix = "." + file_suffix

        mesh = trimesh.PointCloud(all_pt3ds)
        mesh.colors = color

        # trimesh sample surface
        # sampled_points, _ = trimesh.sample.sample_surface(mesh, count=len(all_pt3ds) // 2)

        mesh.export(f"{ply_path.replace(file_suffix, '_pruned' + file_suffix)}")

        # return new path
        return ply_path.replace(file_suffix, "_pruned" + file_suffix)
