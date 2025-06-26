import torch
from torch import nn
from einops import rearrange

from .utils.pos_encodiong import get_2d_sincos_pos_embed
from .utils.blocks import SelfAttentionBlock
from src.models.sources.vggsfm.utils import *
from src.models.sources.vggsfm.models.modules import *


class BETR(nn.Module):
    """Box Estimation TRansformer.

    BETR is a transformer-based model for estimating object bounding
    boxes from image features.
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        **kwargs,
    ):
        super().__init__()

        # Configuration options
        self._initialize_config(d_model, nhead, kwargs)

        # Initialize attention blocks
        self._initialize_attention(
            num_decoder_layers, use_flash_attn=self._check_flash_attn()
        )

        # Initialize feature dimensions
        self._initialize_feature_dimensions()

        # Initialize embedding layers
        self._initialize_embeddings(kwargs)

    def _initialize_config(self, d_model, nhead, kwargs):
        """Initialize configuration parameters."""
        self.d_model = d_model
        self.nhead = nhead
        self.decoder_only = kwargs["decoder_only"]
        self.patch_size = kwargs["patch_size"]
        self.img_size = kwargs["img_size"]

        # Supervision flags
        self.nvs_supervision = kwargs.get("nvs_supervision", False)
        self.ray_supervision = kwargs.get("ray_supervision", False)
        self.use_mask = kwargs.get("use_mask", False)
        self.patchify_rays = kwargs.get("patchify_rays", False)

        # Representation types
        self.pose_representation = kwargs.get("pose_representation", "bb8")
        self.bbox_representation = kwargs.get("bbox_representation", "voting")

        # Feature configuration
        self.diff_emb = kwargs["diff_emb"]
        self.use_pretrained = kwargs["use_pretrained"]

        assert (
            self.nvs_supervision or self.ray_supervision
        ), "At least one supervision should be True"

    def _check_flash_attn(self):
        """Check if flash attention is available."""
        try:
            import flash_attn

            # print("Using flash attention")
            return True
        except:
            return False

    def _initialize_attention(self, num_decoder_layers, use_flash_attn=False):
        """Initialize attention blocks."""
        self.att_depth = num_decoder_layers
        self.attn = nn.Sequential(
            *[
                SelfAttentionBlock(
                    hidden_size=self.d_model,
                    num_heads=self.nhead,
                    mlp_ratio=4.0,
                    enable_flash_attn=use_flash_attn,
                    enable_sequence_parallelism=False,
                    qk_norm=True,
                )
                for i in range(self.att_depth)
            ]
        )

    def _initialize_feature_dimensions(self):
        """Initialize feature dimensions based on representation type."""
        self.cat_dim = 3  # Base RGB dimension

        # Add dimensions based on pose representation
        if self.pose_representation == "plucker":
            self.cat_dim += 6
        elif self.pose_representation == "bb8":
            if self.bbox_representation == "voting":
                self.cat_dim += 16
                self.box_dim = 16
            elif self.bbox_representation == "conf_voting":
                self.cat_dim += 24
                self.box_dim = 24
            elif self.bbox_representation == "heatmap":
                self.cat_dim += 8
                self.box_dim = 8
            else:
                raise NotImplementedError(
                    f"Not implemented bbox_representation: {self.bbox_representation}"
                )
        else:
            raise NotImplementedError(
                f"Not implemented pose_representation: {self.pose_representation}"
            )

    def _initialize_embeddings(self, kwargs):
        """Initialize embedding layers based on configuration."""
        # Initialize output projection layers
        self._initialize_projection_layers()

        # Initialize input embedding layers based on pretrained mode
        if self.use_pretrained:
            self._initialize_pretrained_embeddings()
        else:
            self._initialize_from_scratch_embeddings()

    def _initialize_projection_layers(self):
        """Initialize output projection layers."""
        if self.nvs_supervision:
            self.rgb_proj = nn.Linear(self.d_model, self.patch_size**2 * 3)

        if self.ray_supervision and self.pose_representation == "plucker":
            if self.patchify_rays:
                self.camera_ray_proj = nn.Linear(self.d_model, 6)
            else:
                self.camera_ray_proj = nn.Linear(self.d_model, self.patch_size**2 * 6)

        elif self.pose_representation == "bb8":
            if self.bbox_representation == "voting":
                self.bbox_proj = nn.Linear(
                    self.d_model, self.patch_size**2 * 16
                )  # 8 points * 2
            elif self.bbox_representation == "conf_voting":
                self.bbox_proj = nn.Linear(
                    self.d_model, self.patch_size**2 * 24
                )  # 8 points * (2 + 1) dx, dy, conf
            elif self.bbox_representation == "heatmap":
                self.bbox_proj = nn.Linear(
                    self.d_model, self.patch_size**2 * 8
                )  # 8 points * 1

    def _initialize_pretrained_embeddings(self):
        """Initialize embeddings for using pretrained features."""
        self.input_transform = Mlp(
            in_features=self.d_model, out_features=self.d_model, drop=0.1
        )  # pretrain rgb feature adapter
        self.norm = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1e-6)

        if self.pose_representation == "plucker":
            self.ray_emb = nn.Linear(6, self.d_model)
        elif self.pose_representation == "bb8":
            self.bbox_learnable_query = nn.Parameter(torch.zeros(1, self.d_model))
            if self.bbox_representation == "voting":
                self.bbox_emb = nn.Linear(self.patch_size**2 * 16, self.d_model)
            elif self.bbox_representation == "conf_voting":
                self.bbox_emb = nn.Linear(self.patch_size**2 * 24, self.d_model)
            elif self.bbox_representation == "heatmap":
                self.bbox_emb = nn.Linear(self.patch_size**2 * 8, self.d_model)
            else:
                raise NotImplementedError(
                    f"Not implemented bbox_representation: {self.bbox_representation}"
                )
        else:
            raise NotImplementedError(
                f"Not implemented pose_representation: {self.pose_representation}"
            )

    def _initialize_from_scratch_embeddings(self):
        """Initialize embeddings for training from scratch."""
        if self.pose_representation == "plucker":
            raise NotImplementedError("Not implemented pose_representation=plucker")
        elif self.pose_representation == "bb8":
            self.bbox_learnable_query = nn.Parameter(
                torch.zeros(1, self.patch_size**2 * (self.cat_dim - 3))
            )

        if not self.diff_emb:
            self.input_emb = nn.Linear(
                self.patch_size**2 * self.cat_dim, self.d_model
            )
        else:
            self.bbox_learnable_query = nn.Parameter(torch.zeros(1, self.d_model))
            self.input_ref_emb = nn.Linear(
                self.patch_size**2 * self.cat_dim, self.d_model
            )
            if self.ray_supervision:
                self.input_query_rgb_emb = nn.Linear(
                    self.patch_size**2 * self.cat_dim, self.d_model
                )
            if self.nvs_supervision:
                self.input_query_ray_emb = nn.Linear(
                    self.patch_size**2 * self.cat_dim, self.d_model
                )
            else:
                self.input_query_ray_emb = None

    def patchify(self, imgs, c):
        """Convert images to patches.

        Args:
            imgs: (N, c, H, W)
            c: Number of channels

        Returns:
            x: (N, L, patch_size**2 * c)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, c):
        """Convert patches back to images.

        Args:
            x: (N, L, patch_size**2 * c)
            c: Number of channels

        Returns:
            imgs: (N, c, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self, pose_feat, rgbs=None, masks=None, pretrain_rgb_feat=None, image_masks=None
    ):
        """Forward pass of the model.

        Args:
            pose_feat: Pose features in one of these formats:
                - camera_rays: (B, N, 6, H, W) for 'plucker' representation
                - bbox_feat: (B, N, bbox_dim, H, W) for 'bb8' representation
                  where bbox_dim depends on bbox_representation
            rgbs: RGB inputs of shape (B, N, 3, H, W)
            masks: Masks of shape (B, N)
            pretrain_rgb_feat: Optional pretrained RGB features
            image_masks: Optional image masks

        Returns:
            query_ret: Output features in the appropriate format for the pose representation
        """
        assert rgbs is not None, "rgbs input should not be None"
        B, N, C, H, W = rgbs.shape  # (B, N, C, H, W)
        assert (
            H == W == self.img_size
        ), f"H and W should be equal to img_size {self.img_size}, got {H}x{W}"

        # Process feature embeddings
        if pretrain_rgb_feat is not None:
            rgb_flat, pose_feat_flat = self._process_pretrained_features(
                pretrain_rgb_feat, pose_feat, B, N
            )
        else:
            rgb_flat, pose_feat_flat = self._process_raw_features(rgbs, pose_feat, B, N)

        # Initialize features and handle masked regions
        rgb_flat_init = rgb_flat.clone()
        pose_feat_flat_init = pose_feat_flat.clone()
        _, L, P, _ = rgb_flat.shape

        # Replace masked positions with learnable queries
        patch_grid_size = (self.img_size // self.patch_size) ** 2
        pose_feat_flat_init[masks] = self.bbox_learnable_query.expand(
            B, patch_grid_size, self.bbox_learnable_query.shape[-1]
        ).to(pose_feat_flat_init.dtype)

        # Generate combined features
        fuse_feat = self._generate_fused_features(
            rgb_flat_init, pose_feat_flat_init, masks, B, P
        )

        # Apply attention layers
        fuse_feat = rearrange(fuse_feat, "B N P C -> B (N P) C")
        fuse_feat = self.attn(fuse_feat)
        fuse_feat = rearrange(fuse_feat, "B (N P) C -> B N P C", B=B, P=P)

        # Extract query features for masked positions
        query_camera_feat = fuse_feat[masks].clone()

        # Generate output according to representation type
        query_ret = self._generate_output(query_camera_feat)

        return query_ret

    def _process_pretrained_features(self, pretrain_rgb_feat, pose_feat, B, N):
        """Process pretrained features."""
        # Process RGB features
        rgb_flat = pretrain_rgb_feat
        rgb_flat = rearrange(rgb_flat, "B N P C -> (B N) P C")
        rgb_flat = self.input_transform(rgb_flat)
        rgb_flat = self.norm(rgb_flat)
        rgb_flat = rearrange(rgb_flat, "(B N) P C -> B N P C", B=B, N=N)

        # Process pose features based on representation
        if self.pose_representation == "plucker":
            pose_feat_flat = rearrange(pose_feat, "B N C H W -> B N (H W) C")
            pose_feat_flat = self.ray_emb(pose_feat_flat)  # (B, N, L, d_model)
        elif self.pose_representation == "bb8":
            pose_feat_flat = rearrange(pose_feat, "B N C H W -> (B N) C H W")
            pose_feat_flat = self.patchify(
                pose_feat_flat, c=self.box_dim
            )  # (B*N, L, d_model)
            pose_feat_flat = rearrange(pose_feat_flat, "(B N) L C -> B N L C", B=B, N=N)
            pose_feat_flat = self.bbox_emb(pose_feat_flat)

        return rgb_flat, pose_feat_flat

    def _process_raw_features(self, rgbs, pose_feat, B, N):
        """Process raw RGB and pose features."""
        # Process RGB features
        rgb_flat = rearrange(rgbs, "B N C H W -> (B N) C H W")  # (B*N, C, H, W)
        rgb_flat = self.patchify(rgb_flat, c=3)  # (B*N, L, patch_size^2*3)
        rgb_flat = rearrange(rgb_flat, "(B N) L C -> B N L C", B=B, N=N)

        # Process pose features
        pose_feat_flat = rearrange(
            pose_feat, "B N C H W -> (B N) C H W"
        )  # (B*N, C, H, W)
        pose_feat_flat = self.patchify(
            pose_feat_flat, c=self.box_dim
        )  # (B*N, L, patch_size^2*box_dim)
        pose_feat_flat = rearrange(pose_feat_flat, "(B N) L C -> B N L C", B=B, N=N)

        return rgb_flat, pose_feat_flat

    def _generate_fused_features(self, rgb_flat, pose_feat_flat, masks, B, P):
        """Generate fused features from RGB and pose features."""
        # Calculate patch grid size
        patch_num = int(P**0.5)  # e.g., if P=196, patch_num=14

        # Generate 2D sine-cosine positional embeddings
        pos_emb = get_2d_sincos_pos_embed(
            self.d_model, grid_size=(patch_num, patch_num), device=rgb_flat.device
        ).permute(0, 2, 3, 1)[None]
        pos_emb = (
            pos_emb.reshape(1, 1, patch_num * patch_num, self.d_model)
            .to(rgb_flat.device)
            .to(rgb_flat.dtype)
        )

        # Combine RGB and pose features
        if self.use_pretrained:
            fuse_feat = pose_feat_flat + rgb_flat
        else:
            fuse_feat = torch.cat(
                [rgb_flat, pose_feat_flat], dim=-1
            )  # (B, N, P, concat_dim)
            if self.diff_emb:
                # Apply different embeddings for query and reference images
                query_feat = fuse_feat[masks]
                ref_feat = fuse_feat[~masks]
                query_feat = self.input_query_rgb_emb(query_feat)
                ref_feat = self.input_ref_emb(ref_feat)

                # Recombine with zero initialization
                fuse_feat = (
                    torch.zeros(
                        (
                            fuse_feat.shape[0],
                            fuse_feat.shape[1],
                            fuse_feat.shape[2],
                            self.d_model,
                        )
                    )
                    .to(fuse_feat.device)
                    .to(fuse_feat.dtype)
                )
                fuse_feat[masks] = query_feat
                fuse_feat[~masks] = ref_feat
            else:
                fuse_feat = self.input_emb(fuse_feat)

        # Add positional embeddings
        fuse_feat = fuse_feat + pos_emb  # (B, N, P, d_model)

        return fuse_feat

    def _generate_output(self, query_camera_feat):
        """Generate output based on pose representation."""
        if self.pose_representation == "plucker":
            query_camera_ray = self.camera_ray_proj(query_camera_feat)
            if self.patchify_rays:
                query_camera_ray = rearrange(
                    query_camera_ray,
                    "T (H W) C -> T C H W",
                    H=self.img_size // self.patch_size,
                    W=self.img_size // self.patch_size,
                )
            else:
                query_camera_ray = self.unpatchify(query_camera_ray, c=6)
            query_ret = query_camera_ray

        elif self.pose_representation == "bb8":
            query_bbox = self.bbox_proj(query_camera_feat)
            query_bbox = self.unpatchify(query_bbox, c=self.box_dim)
            query_ret = query_bbox

        elif self.pose_representation == "vector":
            raise NotImplementedError("Not implemented pose_representation=vector")

        else:
            raise NotImplementedError(
                f"Not implemented pose_representation: {self.pose_representation}"
            )

        # Apply sigmoid activation for heatmap representation
        if self.box_dim == 8:
            query_ret = torch.sigmoid(query_ret)
            # Convert to -1, 1 range for compatibility with other functions
            query_ret = 2 * query_ret - 1

        return query_ret
