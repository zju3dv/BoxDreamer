import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def make_gt_neighbor_score(poses, query_mask):
    """Calculate ground truth neighbor scores based on pose similarity.

    Args:
        poses: Camera poses, tensor of shape (B, T, 4, 4)
        query_mask: Boolean mask indicating query frames

    Returns:
        Tensor of scores for each reference frame
    """
    B, T = poses.shape[:2]

    # Extract query and reference poses based on the mask
    query_pose = poses[query_mask]  # (B 1) 4 4
    ref_poses = poses[~query_mask]  # (B T-1) 4 4

    # Translation distance calculation
    query_t = query_pose[:, :3, 3]  # (B, 3)
    ref_t = rearrange(
        ref_poses[:, :3, 3], "(B N) C -> B N C", B=B, N=T - 1
    )  # (B, T-1, 3)
    distance = torch.norm(query_t.unsqueeze(1) - ref_t, dim=-1)  # (B, T-1)

    # Rotation difference calculation
    query_rot = query_pose[:, :3, :3]  # (B, 3, 3)
    ref_rot = rearrange(
        ref_poses[:, :3, :3], "(B N) C D -> B N C D", B=B, N=T - 1
    )  # (B, T-1, 3, 3)
    rotation_diff = torch.matmul(
        query_rot.unsqueeze(1).transpose(-1, -2), ref_rot
    )  # (B, T-1, 3, 3)
    trace = rotation_diff.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # (B, T-1)
    rotation_degree = torch.acos(
        torch.clamp((trace - 1) / 2, min=-1.0, max=1.0)
    )  # (B, T-1)

    # Handle potential NaN values
    if torch.isnan(rotation_degree).any():
        rotation_degree = torch.zeros_like(rotation_degree)

    # Combine distance and rotation degree into a score
    distance_weight = 1.0
    rotation_weight = 1.0
    score = distance_weight * torch.exp(-distance) + rotation_weight * torch.exp(
        -rotation_degree
    )

    # Normalize the score to [0, 1]
    score_min, score_max = (
        score.min(dim=-1, keepdim=True)[0],
        score.max(dim=-1, keepdim=True)[0],
    )
    score = (score - score_min) / (score_max - score_min + 1e-8)

    return score.unsqueeze(-1)  # (B, T-1, 1)


def dino_matching(
    ref_features,
    query_features,
    ref_images,
    query_images,
    similarity_type="dot_product",
    topk=10,
    similarity_params=None,
):
    """Perform feature matching with DINO features to get neighbor view
    indices.

    Args:
        ref_features: Features from reference views
        query_features: Features from query view
        ref_images: Reference RGB images
        query_images: Query RGB image
        similarity_type: Method for computing similarity
        topk: Number of top matches to return
        similarity_params: Additional parameters

    Returns:
        Boolean mask of top-k matches
    """
    # Create simplified implementation
    if similarity_params is None:
        similarity_params = {}

    B, N, L, D = ref_features.shape

    # Create foreground masks based on image luminance
    def create_foreground_mask(images, threshold=0.05):
        if images.dim() == 5:
            images = images.view(-1, *images.shape[2:])

        # Calculate luminance
        luminance = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        fg_mask = (luminance > threshold).float()

        # Resize mask to feature resolution
        feature_size = int(np.sqrt(L))
        fg_mask = F.interpolate(
            fg_mask.unsqueeze(1), size=(feature_size, feature_size), mode="nearest"
        )

        return fg_mask.reshape(fg_mask.shape[0], -1)

    # Compute masked similarity
    def masked_similarity(query_feat, ref_feat, query_mask, ref_mask, method):
        # Apply masks to features
        query_mask = query_mask.unsqueeze(-1)
        ref_mask = ref_mask.unsqueeze(-1)
        query_feat_masked = query_feat * query_mask
        ref_feat_masked = ref_feat * ref_mask

        # Normalize masked features
        query_feat_norm = F.normalize(query_feat_masked, dim=-1)
        ref_feat_norm = F.normalize(ref_feat_masked, dim=-1)

        # Compute similarity based on method
        if method == "dot_product":
            sim = torch.bmm(query_feat_norm, ref_feat_norm.transpose(-2, -1))
            valid_mask = torch.bmm(query_mask, ref_mask.transpose(-2, -1))
            org_sim = sim.clone()
            sim = sim.masked_fill(valid_mask == 0, -1e4)
            return sim, org_sim
        # Other similarity methods would be implemented here
        else:
            return torch.bmm(query_feat_norm, ref_feat_norm.transpose(-2, -1)), None

    # Flatten and normalize features
    ref_features_flat = rearrange(ref_features, "b n l d -> (b n) l d")
    query_features_flat = rearrange(query_features, "b l d -> b 1 l d")
    query_features_expand = query_features_flat.expand(-1, N, -1, -1)
    query_features_flat = rearrange(query_features_expand, "b n l d -> (b n) l d")

    # Create foreground masks
    query_mask = create_foreground_mask(query_images)
    if ref_images.dim() == 5:
        ref_images_flat = ref_images.view(-1, *ref_images.shape[2:])
    ref_mask = create_foreground_mask(ref_images_flat)

    # Expand query mask to match reference shape
    query_mask_expand = query_mask.unsqueeze(1).expand(-1, N, -1)
    query_mask_flat = rearrange(query_mask_expand, "b n l -> (b n) l")

    # Compute similarity
    similarity, org_sim = masked_similarity(
        query_features_flat,
        ref_features_flat,
        query_mask_flat,
        ref_mask,
        method=similarity_type,
    )

    # Calculate mean similarity with mask consideration
    valid_similarity = similarity.masked_fill(similarity == -1e9, 0)
    valid_count = (similarity != -1e9).float().sum(dim=[1, 2])
    mean_similarity = valid_similarity.sum(dim=[1, 2]) / valid_count
    mean_similarity = rearrange(mean_similarity, "(b n) -> b n", b=B, n=N)

    # Handle NaN or Inf values
    mean_similarity = torch.nan_to_num(mean_similarity, 0.0, 0.0, 0.0)

    # Get top-k matches
    topk_values, topk_indices = torch.topk(mean_similarity, k=topk, dim=-1)
    topk_mask = torch.zeros_like(mean_similarity, dtype=torch.bool)
    for b in range(B):
        topk_mask[b, topk_indices[b]] = True

    return topk_mask
