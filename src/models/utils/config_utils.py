"""
Author: Yuanhong Yu
Date: 2025-03-17 18:49:14
LastEditTime: 2025-03-17 21:30:26
Description:

"""


def validate_model_config(config):
    """Validate BoxDreamer model configuration parameters.

    Args:
        config: Model configuration dictionary

    Returns:
        Validated and possibly modified configuration
    """
    # Make a copy to avoid modifying the original
    config = config.copy()

    # Validate pose and bbox representations
    assert config["pose_representation"] in ["plucker", "vector", "bb8"]
    assert config["bbox_representation"] in ["heatmap", "voting", "cornernet"]

    # For compatibility, treat and cornernet as heatmap
    if config["bbox_representation"] in ["cornernet"]:
        config["bbox_representation"] = "heatmap"

    assert config["coordinate"] in ["first_camera", "object"]

    # Validate encoder-specific requirements
    if config["use_rgb"] and config["encoder"]["name"] == "dino":
        assert (
            config["decoder"]["patch_size"] == 14
        ), "Dinov2 only supports patch size 14"
    if config["use_rgb"] and config["encoder"]["name"] == "spa":
        assert config["decoder"]["patch_size"] == 16, "SPA only supports patch size 16"

    # Validate ray settings
    assert (config["patchify_rays"] and config["use_rgb"]) or (
        not config["patchify_rays"] and not config["use_rgb"]
    ), "patchify_rays should be True when use_rgb is True"

    return config


def setup_camera_params(config):
    """Setup camera and rotation parameters based on configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Updated configuration with camera parameters
    """
    # Initialize parameters
    rotation_length = 0
    camera_dim = 0

    if config["rotation_type"] is not None:
        assert config["rotation_type"] in ["quat", "6d", "euler", "so3", "ray"]

        # Set rotation length based on type
        if config["rotation_type"] == "6d":
            rotation_length = 6
        elif config["rotation_type"] == "quat":
            rotation_length = 4
        else:
            rotation_length = 3

        # Calculate camera dimension
        if config["regression_intri"]:
            camera_dim = rotation_length + 3 + 1 + (2 if config["use_pp"] else 0)
        else:
            camera_dim = rotation_length + 3
    else:
        assert config["pose_representation"] == "bb8"

    # Update decoder configuration
    config["decoder"]["rotation_type"] = config["rotation_type"]
    config["decoder"]["camera_dim"] = camera_dim
    config["decoder"]["rotation_length"] = rotation_length
    config["decoder"]["use_pretrained"] = config["use_rgb"]
    config["decoder"]["patchify_rays"] = config["patchify_rays"]
    config["decoder"]["pose_representation"] = config["pose_representation"]
    config["decoder"]["bbox_representation"] = config["bbox_representation"]

    # Handle incompatible settings
    if config["use_rgb"] and config["decoder"]["diff_emb"]:
        from src.utils.log import WARNING

        WARNING("diff_emb is not supported when use_rgb is True (for now)")
        config["decoder"]["diff_emb"] = False

    return config, camera_dim, rotation_length
