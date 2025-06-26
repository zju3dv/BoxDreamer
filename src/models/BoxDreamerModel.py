import torch
import torch.nn as nn
from einops import rearrange

from src.models.modules.tracker.cotracker import CoTracker
from src.models.modules.encoder.dinov2 import DinoV2Wrapper
from src.models.modules.encoder.resnet import ResNetWrapper
from src.models.modules.backbone.betr import BETR
from src.utils.log import WARNING, INFO

from kornia.feature import LoFTR


from .utils.config_utils import validate_model_config, setup_camera_params
from .utils.camera_processing import make_camera_rays, encode_camera_as_vector
from .utils.data_processing import process_dense_input, normalize
from .utils.dense_processing import process_multi_round
from .utils.prediction_utils import process_prediction, calculate_bb8_projections


class BoxDreamer(nn.Module):
    """BoxDreamer model for predicting 3D object poses from multiple images."""

    def __init__(self, config):
        """Initialize the BoxDreamer model.

        Args:
            config: Dictionary containing model configuration
        """
        super().__init__()

        self.config = config
        self.module_configs = config["modules"]

        # Load and validate configuration
        self._load_config(self.module_configs)

        # Initialize model components
        self._initialize_modules(self.module_configs)

    def _load_config(self, module_configs):
        """Load and validate configuration parameters."""
        # Load basic settings
        self.use_matching = module_configs["use_matching"]
        self.use_tracking = module_configs["use_tracking"]
        self.use_keypoints = module_configs["use_keypoints"]
        self.use_rgb = module_configs["use_rgb"]
        self.use_pp = module_configs["use_pp"]
        self.regression_intri = module_configs["regression_intri"]
        self.roatation_type = module_configs["rotation_type"]
        self.coordinate = module_configs["coordinate"]
        self.pose_representation = module_configs["pose_representation"]
        self.bbox_representation = module_configs["bbox_representation"]
        self.image_size = module_configs["decoder"]["img_size"]
        self.patch_size = module_configs["decoder"]["patch_size"]
        self.patchify_rays = module_configs["patchify_rays"]

        # Validate configuration
        module_configs = validate_model_config(module_configs)
        self.bbox_representation = module_configs["bbox_representation"]

        # Load dense configuration
        self.dense_cfg = module_configs.get("dense_cfg", None)

        # Setup camera parameters
        module_configs, self.camera_dim, self.rotation_length = setup_camera_params(
            module_configs
        )
        self.module_configs = module_configs

    def _initialize_modules(self, module_configs):
        """Initialize model components based on configuration."""
        # Initialize tracker if needed
        if self.use_tracking:
            raise NotImplementedError("Tracking is not supported yet")
            from src.models.sources.cotracker.utils.visualizer import Visualizer

            self.tracker = CoTracker(**module_configs["tracker"])
            self.vis = Visualizer(
                save_dir="./debug_tracker",
                pad_value=120,
                linewidth=1,
                fps=1,
                show_first_frame=1,
            )
        else:
            self.tracker = None

        # Initialize matcher if needed
        if self.use_matching:
            self.matcher = LoFTR(pretrained="indoor")
        else:
            self.matcher = None

        # Initialize RGB encoder based on configuration
        if self.use_rgb:
            encoder_name = module_configs["encoder"]["name"]
            if encoder_name == "resnet":
                self.rgb_encoder = ResNetWrapper(**module_configs["encoder"]["resnet"])
            elif encoder_name == "dino":
                self.rgb_encoder = DinoV2Wrapper(**module_configs["encoder"]["dino"])
            elif encoder_name == "spa":
                raise NotImplementedError("SPA encoder is not supported yet")
            else:
                raise ValueError(f"Invalid encoder name: {encoder_name}")
        else:
            self.rgb_encoder = None

        # Initialize decoder
        self.decoder = BETR(**module_configs["decoder"])

    def forward(self, data):
        """Forward pass of the BoxDreamer model.

        Args:
            data: Dictionary containing input data

        Returns:
            Dictionary containing model predictions and intermediate results
        """
        # Extract input data
        (
            poses,
            frames,
            K,
            crop_params,
            image_masks,
            camera_mask,
        ) = self._extract_input_data(data)

        # Prepare camera representations
        (
            pose_feat,
            camera_rays,
            cameras,
            trans,
            rays_feat,
        ) = self._prepare_camera_representation(
            poses, K, crop_params, camera_mask, data
        )

        # Process RGB features if needed
        rgb_feature = self._process_rgb_features(frames)

        # Process inputs and get model predictions
        query_ret = self._process_inputs_and_predict(
            data, pose_feat, frames, camera_mask, rgb_feature, image_masks
        )

        # Update data
        (
            poses,
            frames,
            K,
            crop_params,
            image_masks,
            camera_mask,
        ) = self._extract_input_data(data)

        # Initialize prediction outputs
        pred_poses = poses.clone()
        pred_rays = rays_feat.clone() if rays_feat is not None else None

        # Update predictions based on pose representation
        self._update_predictions(data, query_ret, camera_mask, pred_rays)

        # Process final predictions
        if not self.training:
            pred_poses = self._process_evaluation(
                pred_poses,
                data,
                query_ret,
                camera_mask,
                cameras,
                rays_feat,
                crop_params,
                image_masks,
                trans,
                K,
            )
        elif self.pose_representation == "bb8":
            # For training with BB8, calculate projections
            calculate_bb8_projections(
                query_ret, data, camera_mask, self.bbox_representation
            )

        # Store final predictions
        data["pred_poses"] = pred_poses.clone()
        data["pred_intrinsics"] = data["intrinsics"]

        return data

    def _extract_input_data(self, data):
        """Extract and prepare input data from data dictionary."""
        poses = data["poses"].clone()
        frames = data["images"].clone()
        K = data["non_ndc_intrinsics"].clone()
        crop_params = data["crop_parameters"].clone()
        image_masks = data["image_masks"].clone()

        # Create camera mask based on query index
        query_idx = data["query_idx"].clone()

        if query_idx[0] != "none":
            camera_mask = torch.zeros(
                poses.shape[:2], dtype=torch.bool, device=poses.device
            )
            camera_mask[torch.arange(poses.shape[0]), query_idx] = True
        else:
            raise NotImplementedError("Query index must be specified")

        # Save camera mask to data
        data["camera_mask"] = camera_mask.clone()

        return poses, frames, K, crop_params, image_masks, camera_mask

    def _prepare_camera_representation(self, poses, K, crop_params, camera_mask, data):
        """Prepare camera representation based on configuration."""
        B = poses.shape[0]

        # Create masked pose for query frames
        masked_pose = poses.clone()
        masked_pose[camera_mask] = (
            torch.eye(4, device=poses.device).repeat(B, 1, 1).to(poses.dtype)
        )

        # Initialize return values
        camera_rays = cameras = trans = rays_feat = None

        # Create representation based on configuration
        if self.roatation_type == "ray":
            # Create camera rays for original and masked poses
            camera_rays, cameras, trans = make_camera_rays(
                poses,
                K,
                crop_params,
                self.image_size,
                self.patch_size,
                self.patchify_rays,
            )
            camera_rays, _, _ = make_camera_rays(
                masked_pose,
                K,
                crop_params,
                self.image_size,
                self.patch_size,
                self.patchify_rays,
            )

            # Convert to spatial representation
            rays_feat = rearrange(
                camera_rays.to_spatial().clone(),
                "(B T) C H W -> B T C H W",
                B=poses.shape[0],
            )
            rays_feat = rays_feat.to(poses.dtype)
            pose_feat = rays_feat
        elif self.roatation_type is not None:
            # Use vector representation
            pose_feat = encode_camera_as_vector(
                masked_pose,
                K,
                self.roatation_type,
                self.rotation_length,
                self.camera_dim,
                self.use_pp,
            )
        else:
            # Use BB8 representation
            pose_feat = data["bbox_feat"].clone() if "bbox_feat" in data else None

        return pose_feat, camera_rays, cameras, trans, rays_feat

    def _process_rgb_features(self, frames):
        """Process RGB features using the encoder if configured."""
        if not self.use_rgb:
            return None

        # Ensure encoder is on the correct device
        if frames.device != self.rgb_encoder.get_device():
            INFO("Moving RGB encoder to device of frames")
            self.rgb_encoder.to_device(frames.device)

        # Extract features using the encoder
        return self.rgb_encoder.predict(frames)

    def _process_inputs_and_predict(
        self, data, pose_feat, frames, camera_mask, rgb_feature, image_masks
    ):
        """Process inputs and generate predictions."""
        # Process dense configuration if enabled
        if self.dense_cfg is not None and self.dense_cfg.enable:
            # Filter and prepare dense input
            (
                data,
                pose_feat,
                frames,
                camera_mask,
                rgb_feature,
                image_masks,
            ) = process_dense_input(
                data,
                pose_feat,
                frames,
                camera_mask,
                rgb_feature,
                image_masks,
                self.dense_cfg,
            )

            # Process with multi-round for dense input
            if self.dense_cfg.multi_round:
                return process_multi_round(
                    data,
                    pose_feat,
                    frames,
                    camera_mask,
                    rgb_feature,
                    image_masks,
                    self.decoder,
                    self.dense_cfg,
                    self.bbox_representation,
                )
            else:
                # Standard processing for dense input
                return self.decoder(
                    pose_feat, frames, camera_mask, rgb_feature, normalize(image_masks)
                )
        else:
            # Standard processing for non-dense input
            return self.decoder(
                pose_feat, frames, camera_mask, rgb_feature, normalize(image_masks)
            )

    def _update_predictions(self, data, query_ret, camera_mask, pred_rays):
        """Update predictions based on pose representation."""
        if self.pose_representation == "plucker":
            # Update ray predictions
            pred_rays[camera_mask] = query_ret.clone().to(pred_rays.dtype)
            data["pred_camera_rays"] = pred_rays
        elif self.pose_representation == "bb8":
            # Update bbox predictions
            data["pred_bbox"] = data["bbox_feat"].clone()
            data["pred_bbox"][camera_mask] = query_ret
        else:
            raise NotImplementedError(
                f"Pose representation {self.pose_representation} is not supported"
            )

    def _process_evaluation(
        self,
        pred_poses,
        data,
        query_ret,
        camera_mask,
        cameras,
        rays,
        crop_params,
        image_masks,
        trans,
        K,
    ):
        """Process predictions for evaluation."""
        bbox_3d = data.get("bbox_3d", None)

        # Process based on pose representation
        return process_prediction(
            pred_poses,
            data,
            query_ret,
            camera_mask,
            self.pose_representation,
            cameras,
            rays,
            crop_params,
            image_masks,
            trans,
            K,
            bbox_3d,
            self.coordinate,
            self.image_size,
            self.patch_size,
            self.bbox_representation,
        )
