import torch
import numpy as np
import cv2
from einops import rearrange


def recover_bb8_corners(bbox_feat, bbox_representation):
    """Recover 3D bounding box corners from feature predictions.

    Args:
        bbox_feat: Bounding box features, typically from network prediction
        bbox_representation: Type of representation ("voting", "conf_voting", or "heatmap")

    Returns:
        Tuple of (normalized_keypoints_2d, keypoints_2d)
    """
    # Convert all data to float32
    bbox_feat = bbox_feat.float()

    if bbox_representation == "voting" or bbox_representation == "conf_voting":
        B, T, H, W, _, _ = bbox_feat.shape

        # Create coordinate grids
        coords_x = (
            torch.arange(W).view(1, 1, 1, W).expand(B, T, H, W).to(bbox_feat.device)
        )
        coords_y = (
            torch.arange(H).view(1, 1, H, 1).expand(B, T, H, W).to(bbox_feat.device)
        )

        # Scale bbox_feat to real dimensions
        bbox_feat[..., :, 0] = bbox_feat[..., :, 0] * W  # dx
        bbox_feat[..., :, 1] = bbox_feat[..., :, 1] * H  # dy

        if bbox_representation == "voting":
            # Extract dx and dy (voting offsets)
            dx = bbox_feat[..., :, 0]  # [B, T, H, W, 8]
            dy = bbox_feat[..., :, 1]  # [B, T, H, W, 8]

            # Compute vote coordinates
            votes_x = coords_x.unsqueeze(-1) + dx  # [B, T, H, W, 8]
            votes_y = coords_y.unsqueeze(-1) + dy  # [B, T, H, W, 8]

            # Reshape votes for averaging
            votes_x = votes_x.view(B, T, -1, 8)  # [B, T, H*W, 8]
            votes_y = votes_y.view(B, T, -1, 8)  # [B, T, H*W, 8]

            # Compute mean of votes for each keypoint
            kpt_x_mean = votes_x.mean(dim=2)  # [B, T, 8]
            kpt_y_mean = votes_y.mean(dim=2)  # [B, T, 8]

        else:  # conf_voting
            # Extract offsets and confidence
            dx = bbox_feat[..., :, 0]  # [B, T, H, W, 8]
            dy = bbox_feat[..., :, 1]  # [B, T, H, W, 8]
            conf = bbox_feat[..., :, 2]  # [B, T, H, W, 8]

            # Compute vote coordinates
            votes_x = coords_x.unsqueeze(-1) + dx  # [B, T, H, W, 8]
            votes_y = coords_y.unsqueeze(-1) + dy  # [B, T, H, W, 8]

            # Reshape for weighted averaging
            votes_x = votes_x.view(B, T, -1, 8)  # [B, T, H*W, 8]
            votes_y = votes_y.view(B, T, -1, 8)  # [B, T, H*W, 8]
            conf = conf.view(B, T, -1, 8)  # [B, T, H*W, 8]

            # Compute weighted mean of votes for each keypoint
            weighted_sum_x = (votes_x * conf).sum(dim=2)  # [B, T, 8]
            weighted_sum_y = (votes_y * conf).sum(dim=2)  # [B, T, 8]
            conf_sum = conf.sum(dim=2) + 1e-7  # Add epsilon to avoid division by zero

            kpt_x_mean = weighted_sum_x / conf_sum  # [B, T, 8]
            kpt_y_mean = weighted_sum_y / conf_sum  # [B, T, 8]

    else:  # Heatmap representation
        B, T, H, W, _ = bbox_feat.shape  # bbox_feat is [B, T, H, W, 8]

        # Convert from [-1, 1] to [0, 1] range
        bbox_feat = (bbox_feat + 1) / 2

        # Permute to [B, T, 8, H, W] for easier processing
        heatmaps = bbox_feat.permute(0, 1, 4, 2, 3)  # [B, T, 8, H, W]
        heatmaps = heatmaps.view(B * T, 8, H * W)  # [B*T, 8, H*W]

        # Find top-k positions in each heatmap
        k = 20
        max_vals, idxs = torch.topk(heatmaps, k=k, dim=2)  # [B*T, 8, k]

        # Convert indices to x,y coordinates
        xs = idxs % W  # [B*T, 8, k]
        ys = idxs // W  # [B*T, 8, k]

        # Average the top-k positions
        kpt_x_mean = xs.float().mean(dim=2)  # [B*T, 8]
        kpt_y_mean = ys.float().mean(dim=2)  # [B*T, 8]

        # Reshape to [B, T, 8]
        kpt_x_mean = kpt_x_mean.view(B, T, 8)
        kpt_y_mean = kpt_y_mean.view(B, T, 8)

    # Stack the x,y coordinates to get 2D keypoints
    keypoints_2d = torch.stack([kpt_x_mean, kpt_y_mean], dim=3)  # [B, T, 8, 2]

    # Normalize keypoints to [-1, 1] range
    normalized_keypoints_2d = keypoints_2d / torch.tensor(
        [W, H], device=keypoints_2d.device
    ).view(1, 1, 1, 2)
    normalized_keypoints_2d = (normalized_keypoints_2d * 2) - 1

    return normalized_keypoints_2d, keypoints_2d


def recover_pose_from_bb8(bbox_feat, bbox_3d, K, bbox_representation):
    """Recover camera pose from predicted bounding box corners using PnP.

    Args:
        bbox_feat: Bounding box features
        bbox_3d: 3D bounding box coordinates
        K: Camera intrinsic parameters
        bbox_representation: Type of representation ("voting", "conf_voting", or "heatmap")

    Returns:
        Tuple of (poses, normalized_keypoints_2d)
    """
    B, T = bbox_feat.shape[0], bbox_feat.shape[1]
    normalized_keypoints_2d, keypoints_2d = recover_bb8_corners(
        bbox_feat, bbox_representation
    )

    # Convert all to float32 for numerical stability
    bbox_feat = bbox_feat.float()
    bbox_3d = bbox_3d.float()
    K = K.float()

    # Initialize output pose tensor
    poses = torch.zeros(B, T, 4, 4, device=bbox_feat.device)

    # Process each batch and time step
    for b in range(B):
        for t in range(T):
            # Get the 2D keypoints
            pts_2d = keypoints_2d[b, t]  # [8, 2]

            # Get the corresponding 3D keypoints
            pts_3d = bbox_3d[b, t]  # [8, 3]

            # Get the camera intrinsic matrix
            K_matrix = K[b, t] if len(K.shape) == 4 else K

            # Convert to numpy arrays for cv2.solvePnP
            pts_2d_np = pts_2d.cpu().numpy().astype(np.float32)
            pts_3d_np = pts_3d.cpu().numpy().astype(np.float32)
            K_np = K_matrix.cpu().numpy().astype(np.float32)
            dist_coeffs = None  # Assuming no lens distortion

            try:
                # Try RANSAC PnP first for robustness
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d_np,  # 3D points
                    pts_2d_np,  # 2D points
                    K_np,  # Camera intrinsics
                    dist_coeffs,  # Distortion coefficients
                    reprojectionError=1.0,  # Threshold for reprojection error
                    confidence=0.99,  # Confidence level for RANSAC
                    flags=cv2.SOLVEPNP_ITERATIVE,  # Use iterative method with RANSAC
                )

                # For testing, disable OpenCV RANSAC and force fallback
                success = False

                if not success:
                    # If RANSAC fails, fallback to a high-precision method without RANSAC
                    success, rvec, tvec = cv2.solvePnP(
                        pts_3d_np,
                        pts_2d_np,
                        K_np,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                if success:
                    # Convert rotation vector to rotation matrix
                    R_matrix, _ = cv2.Rodrigues(rvec)  # Convert rvec to rotation matrix
                    R_matrix = R_matrix.astype(np.float32)
                    tvec = tvec.reshape(3, 1).astype(np.float32)

                    # Form the pose matrix [R | t]
                    pose = np.hstack((R_matrix, tvec))  # [3, 4]

                    # Convert to torch tensor and store
                    poses[b, t, :3, :] = torch.from_numpy(pose).to(bbox_feat.device)
                    poses[b, t, 3, 3] = 1.0

            except Exception as e:
                # Handle exceptions (e.g., due to numerical issues)
                print(f"PnP failed due to exception: {e}")
                continue  # Pose remains zeros

    return poses, normalized_keypoints_2d


def recover_pose_from_dense_bb8(bbox_feat, bbox_3d, K, bbox_representation):
    """Recover camera pose from multiple predictions of bounding box corners
    using PnP.

    Args:
        bbox_feat: Multiple bounding box features
        bbox_3d: 3D bounding box coordinates
        K: Camera intrinsic parameters
        bbox_representation: Type of representation ("voting", "conf_voting", or "heatmap")

    Returns:
        Tuple of (poses, normalized_keypoints_2d)
    """
    # Handle multiple proposals
    B, N, T, H, W, _ = bbox_feat.shape

    # Convert to float32 for numerical stability
    bbox_feat = bbox_feat.float()
    bbox_3d = bbox_3d.float()
    K = K.float()

    # Repeat bbox_3d for each proposal
    bbox_3d = bbox_3d.unsqueeze(1).repeat(1, N, 1, 1, 1)

    # Initialize output tensors
    poses = torch.zeros(B, T, 4, 4, device=bbox_feat.device)

    # Collect all keypoints from N proposals
    normalized_keypoints_2d = []
    keypoints_2d = []

    # Process each proposal
    for i in range(N):
        # Get single proposal and process it
        current_feat = bbox_feat[:, i]  # B T H W 8
        norm_kpts, kpts = recover_bb8_corners(current_feat, bbox_representation)

        if len(normalized_keypoints_2d) == 0:
            normalized_keypoints_2d = norm_kpts
            keypoints_2d = kpts
        else:
            normalized_keypoints_2d = torch.cat(
                [normalized_keypoints_2d, norm_kpts], dim=1
            )
            keypoints_2d = torch.cat([keypoints_2d, kpts], dim=1)

    # Process each batch and time step
    for b in range(B):
        for t in range(T):
            # Collect all keypoints from N proposals
            pts_2d = keypoints_2d[b].reshape(-1, 2)  # [N*8, 2]
            pts_3d = bbox_3d[b, :, t].reshape(-1, 3)  # [N*8, 3]

            # Get camera intrinsic matrix
            K_matrix = K[b, t] if len(K.shape) == 4 else K

            # Convert to numpy arrays for cv2.solvePnP
            pts_2d_np = pts_2d.cpu().numpy().astype(np.float32)
            pts_3d_np = pts_3d.cpu().numpy().astype(np.float32)
            K_np = K_matrix.cpu().numpy().astype(np.float32)
            dist_coeffs = None  # Assuming no lens distortion

            try:
                # Try RANSAC with more points and higher threshold
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d_np,
                    pts_2d_np,
                    K_np,
                    dist_coeffs,
                    reprojectionError=2.0,  # Increased threshold due to multiple proposals
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    iterationsCount=1000,  # Increased iterations for more proposals
                )

                if not success:
                    # Fallback to regular PnP if RANSAC fails
                    success, rvec, tvec = cv2.solvePnP(
                        pts_3d_np,
                        pts_2d_np,
                        K_np,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                if success:
                    # Convert rotation vector to rotation matrix
                    R_matrix, _ = cv2.Rodrigues(rvec)
                    R_matrix = R_matrix.astype(np.float32)
                    tvec = tvec.reshape(3, 1).astype(np.float32)

                    # Form the pose matrix [R | t]
                    pose = np.hstack((R_matrix, tvec))

                    # Store the result
                    poses[b, t, :3, :] = torch.from_numpy(pose).to(bbox_feat.device)
                    poses[b, t, 3, 3] = 1.0

            except Exception as e:
                print(f"PnP failed due to exception: {e}")
                continue

    return poses, normalized_keypoints_2d
