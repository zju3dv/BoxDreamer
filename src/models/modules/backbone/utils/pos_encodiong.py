import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union


class PositionGetter1D(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, l, device):
        if not l in self.cache_positions:
            x = torch.arange(l, device=device)
            self.cache_positions[l] = x  # (w)
        pos = self.cache_positions[l].view(1, l).expand(b, -1).clone()
        return pos


# Position encoding for query image
class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        max_shape = tuple(max_shape)

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-math.log(10000.0) / d_model // 2)
        )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return (
            x + self.pe[:, :, : x.size(2), : x.size(3)]
        )  # , self.position_getter(x.size(0), x.size(2), x.size(3), x.device))


# Position encoding for 3D points
class KeypointEncoding_linear(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, inp_dim, feature_dim, layers: list, norm_method="batchnorm"):
        super().__init__()
        self.encoder = self.MLP([inp_dim] + layers + [feature_dim], norm_method)
        nn.init.constant_(self.encoder[-1].bias, 0.0)
        self.position_getter = PositionGetter1D()

    def forward(self, kpts, descriptors=None):
        """
        kpts: B*L*3 or B*L*4 or B*L*2
        descriptors: B*C*L
        """
        inputs = kpts  # B*L*2

        # B*C*L
        if descriptors is None:
            return self.encoder(inputs)
        else:
            return descriptors + self.encoder(inputs).transpose(
                2, 1
            )  # , self.position_getter(kpts.size(0), kpts.size(1), kpts.device)

    def MLP(self, channels: list, norm_method="batchnorm"):
        """Multi-layer perceptron"""
        n = len(channels)

        layers = []
        for i in range(1, n):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
            if i < n - 1:
                if norm_method == "batchnorm":
                    layers.append(nn.BatchNorm1d(channels[i]))
                elif norm_method == "layernorm":
                    layers.append(nn.LayerNorm(channels[i]))
                elif norm_method == "instancenorm":
                    layers.append(nn.InstanceNorm1d(channels[i]))
                else:
                    pass
                    # layers.append(nn.GroupNorm(channels[i], channels[i])) # group norm
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)


from pytorch3d.renderer.implicit import HarmonicEmbedding


class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    return_grid=False,
    device=None,
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(
        torch.tensor(embed_dim, device=device), grid
    )
    if return_grid:
        return (
            pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2),
            grid,
        )
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(0, 3, 1, 2)
