import os
from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from torch import einsum
from einops import rearrange, repeat

from inspect import isfunction
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except:
    flash_attn_qkvpacked_func, flash_attn_func = None, None


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# Copy from CLIP GitHub
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def modulate(x, shift, scale):
    # from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        # x = x.permute(1, 0, 2)
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # out = out.permute(1, 0, 2)
        return self.to_out(out)

class MultiheadAttentionFlashV2(nn.Module):
    def __init__(self, embed_dim, n_head, bias=False, shift_group=None):
        super().__init__()

        self.head_dim = embed_dim// n_head
        self.embed_dim = embed_dim 
        self.n_head = n_head
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.shift_group = shift_group
        # self.scale = dim_head ** -0.5

    def shift(self, qkv, bsz, q_len, group_size, num_heads, head_dim):
        qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        return qkv

    def forward(self, q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, need_weights=False, attn_mask=None):
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        h = self.n_head
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q, k, v))
        # print(q.dtype, k.dtype, v.dtype)
        if self.shift_group is not None:
             bsz, q_len, heads, head_dim = q.shape
             assert q_len % self.shift_group == 0
             q = shift(q, bsz, q_len, self.shift_group, self.n_head, self.head_dim)
             k = shift(k, bsz, q_len, self.shift_group, self.n_head, self.head_dim)
             v = shift(v, bsz, q_len, self.shift_group, self.n_head, self.head_dim)

        out = flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
        if self.shift_group is not None:
            out[:, :, h//2:] = attn_output[:, :, self.num_heads//2:].roll(h//2, dims=1)
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = out.permute(1, 0, 2)
        return (out,)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, modulate_feature_size: int = None, modulate_act_type: str = 'gelu', cross_att: bool = None, flash_v2: bool = None, shift_group: int = None):
        super().__init__()
        self.flash_v2 = flash_v2
        # print('vit flashv2', flash_v2)
        if self.flash_v2:
            self.attn = MultiheadAttentionFlashV2(d_model, n_head, shift_group=shift_group)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        if modulate_feature_size is not None:
            act_dict = {'gelu': QuickGELU,
                    'silu': nn.SiLU}
            self.modulation_fn = nn.Sequential(
                LayerNorm(modulate_feature_size),
                act_dict[modulate_act_type](),
                nn.Linear(modulate_feature_size, 3 * d_model, bias=True)
            )
            self.mlp_modulation_fn = nn.Sequential(
                LayerNorm(modulate_feature_size),
                act_dict[modulate_act_type](),
                nn.Linear(modulate_feature_size, 3 * d_model, bias=True)
            )
        else:
            self.modulation_fn = None
            self.mlp_modulation_fn = None

        self.cross_att = cross_att
        if self.cross_att:
            self.cross_att = CrossAttention(query_dim=d_model, context_dim=d_model,
                                    heads=n_head, dim_head=int(d_model//n_head), dropout=0)    
            self.ln_1_5 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
            length = x.shape[0]
            attn_mask = self.attn_mask[:length, :length]
        else:
            attn_mask = None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, modulation: torch.Tensor = None, context: torch.Tensor = None):
        y = self.ln_1(x)
        if self.modulation_fn is not None:
            shift, scale, gate = self.modulation_fn(modulation).chunk(3, dim=1)
            y = modulate(y, shift, scale)
        y = self.attention(y)
        # If we have modulation func for mlp as well, we will just use the gate for the attention
        if self.modulation_fn is not None and self.mlp_modulation_fn is not None:
            y = y * gate.unsqueeze(0)
        x = x + y
        if self.cross_att:
            y = self.cross_att(self.ln_1_5(x), context=context) 
            # print(y.mean().item())
            x = x + y

        y = self.ln_2(x)
        if self.mlp_modulation_fn is not None:
            shift, scale, gate = self.mlp_modulation_fn(modulation).chunk(3, dim=1)
            y = modulate(y, shift, scale)
        y = self.mlp(y)
        # For here we have two cases:
        # 1. If we have a modulation function for the MLP, we use it to modulate the output of the MLP
        # 2. If we don't have a modulation function for the MLP, we use the modulation function for the attention
        if self.modulation_fn is not None:
            y = y * gate.unsqueeze(0)
            
        x = x + y
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, modulate_feature_size: int = None, modulate_act_type: str = 'gelu', cross_att_layers: int = 0, return_all_layers=False, flash_v2=True, shift_group=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, modulate_feature_size=modulate_feature_size, modulate_act_type=modulate_act_type, cross_att = (_ + cross_att_layers)>=layers, flash_v2=flash_v2, shift_group=shift_group) for _ in range(layers)])

        self.grad_checkpointing = False
        self.return_all_layers = return_all_layers 
        self.flash_v2 = flash_v2

    def set_grad_checkpointing(self, flag=True):
        self.grad_checkpointing = flag

    def forward(self, x: torch.Tensor, modulation: torch.Tensor = None, context: torch.Tensor = None, additional_residuals = None):
        all_x = []
        if additional_residuals is not None:
            assert len(additional_residuals) == self.layers
        for res_i, module in enumerate(self.resblocks):
            if self.grad_checkpointing:
                # print("Grad checkpointing")
                x = checkpoint(module, x, modulation, context)
            else:
                x = module(x, modulation, context)
            if additional_residuals is not None:
                add_res = additional_residuals[res_i]
                x[:, :add_res.shape[1]] = x[:, :add_res.shape[1]] + add_res
            all_x.append(x)
        if self.return_all_layers:
            return all_x
        else:
            return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, weight: str =None,
            modulate_feature_size: int = None, modulate_act_type: str = 'gelu', cross_att_layers: int = 0, in_channels: int = 3, error_weight_init_mode='zero', add_zero_conv=False, return_all_layers=False, disable_post_ln=False, flash_v2=False, disable_dino=False, shift_group=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.add_zero_conv = add_zero_conv
        self.return_all_layers = return_all_layers
        self.disable_post_ln = disable_post_ln
        self.flash_v2 = flash_v2
        self.disable_dino = disable_dino
        if self.add_zero_conv:
            assert self.return_all_layers
            self.zero_convs = nn.ModuleList([ zero_module(nn.Conv1d(in_channels=width, out_channels=width, kernel_size=1, stride=1, bias=True)) for _ in range(layers)])

        self.transformer = Transformer(width, layers, heads, modulate_feature_size=modulate_feature_size, modulate_act_type=modulate_act_type, cross_att_layers=cross_att_layers, return_all_layers=return_all_layers, flash_v2=flash_v2, shift_group=shift_group)

        if not self.disable_post_ln:
            self.ln_post = LayerNorm(width)

        if weight is not None:
            if not self.disable_dino:
                if "clip" in weight:
                    raise NotImplementedError()
                elif weight.startswith("vit_b_16"):
                    load_timm_to_clip(self, config_name=weight, init_mode=error_weight_init_mode)
                elif weight.startswith("vit_b_8"):
                    load_timm_to_clip(self, config_name=weight, init_mode=error_weight_init_mode)
                else:
                    raise NotImplementedError()
            else:
                self.apply(_init_weights)

            # Init the weight and bias of modulation_fn to zero
            if modulate_feature_size != 0:
                for block in self.transformer.resblocks:
                    if block.modulation_fn is not None:
                        block.modulation_fn[2].weight.data.zero_()
                        block.modulation_fn[2].bias.data.zero_()
                        if block.mlp_modulation_fn is not None:
                            block.mlp_modulation_fn[2].weight.data.zero_()
                            block.mlp_modulation_fn[2].bias.data.zero_()
            for block in self.transformer.resblocks:
                if block.cross_att:
                    zero_module(block.cross_att.to_out)

    def set_grad_checkpointing(self, flag=True):
        self.transformer.set_grad_checkpointing(flag)

    def forward(self, x: torch.Tensor, modulation: torch.Tensor = None, context: torch.Tensor = None, additional_residuals=None, num_views=None, batch_size=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        if num_views is not None and batch_size is not None:
            x = x.reshape(batch_size, num_views * x.shape[1], x.shape[2])
        import ipdb;ipdb.set_trace()

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, modulation, context, additional_residuals=additional_residuals)

        if self.add_zero_conv:
            assert isinstance(x, (list, tuple))
            assert len(x) == len(self.zero_convs)
            new_x = []
            for sub_x, sub_zero_conv in zip(x, self.zero_convs):
                sub_x_out = sub_zero_conv(sub_x.permute(1, 2, 0))
                new_x.append(sub_x_out.permute(2, 0, 1))
            x = new_x

        if self.return_all_layers:
            assert isinstance(x, (list, tuple))
            return x

        if not self.disable_post_ln:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x)

        return x

    def extra_repr(self) -> str:
        return f"Positional embedding: {self.positional_embedding.shape}. Input Resolution: {self.input_resolution}"


class ViTEncoder(nn.Module):
    def __init__(self, latent_config, input_resolution, patch_size, layers, width, weight=None, **kwargs):
        super(ViTEncoder, self).__init__()
        self.vit = VisionTransformer(
            input_resolution=input_resolution,
            patch_size=patch_size,
            layers=layers,
            width=width,
            heads=width // 64,
            weight=weight,
        )

        d_model = width

        self.num_planes = latent_config.num_planes
        self.output_dim = latent_config.output_dim
        self.head = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(d_model)),
            ("fc1", nn.Linear(d_model, 2 * d_model)),
            ("gelu1", QuickGELU()),
            ("fc2", nn.Linear(2 * d_model, d_model)),
            ("gelu2", QuickGELU()),
            ("fc3", nn.Linear(d_model, self.num_planes * self.output_dim)),
        ]))

    def forward(self, images: torch.Tensor, modulation: torch.Tensor = None):
        """

        :param images: [b, 3, H, W]
        :return: [b, 3, dim, h, w]
        """
        batch_size, _, H, W = images.shape
        h, w = H // self.vit.patch_size, W // self.vit.patch_size

        features = self.vit(images, modulation)
        output = self.head(features)            # b, HW, 3*dim
        output = output.transpose(-1, -2)       # b, 3*dim, HW

        # Exclude CLS token, only use h x w
        output = output[..., 1:]

        # Reshape to [b, #planes, #outdim, h, w]
        output = output.reshape(batch_size, 3, self.output_dim, h, w).contiguous()

        return output


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic'):
    """
    Resize positional embeddings, implementation from google/simclr and open_clip.
    """
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return

    # Compute the grid size and extra tokens
    old_pos_len = state_dict["positional_embedding"].shape[0]
    old_grid_size = round((state_dict["positional_embedding"].shape[0]) ** 0.5)
    grid_size = round((model.positional_embedding.shape[0]) ** 0.5)
    if old_grid_size == grid_size:
        return
    extra_tokens = old_pos_len - (old_grid_size ** 2)

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed

    # Only interpolate the positional emb part, not the extra token part.
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, -1)[0]

    # Concatenate back the
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed


myname2timmname = {
    "vit_b_16_mae": None,
    "vit_b_16_in": "vit_base_patch16_224",
    "vit_b_16_in21k": 'vit_base_patch16_224_in21k',
    "vit_b_16_sam": 'vit_base_patch16_224_sam',
    "vit_b_16_dino": 'vit_base_patch16_224_dino',
    "vit_b_16_mill_in21k": 'vit_base_patch16_224_miil_in21k',
    "vit_b_16_mill": 'vit_base_patch16_224_miil',
    "vit_b_8_dino": 'vit_base_patch16_224_dino',
}


def load_timm_to_clip(module, config_name="vit_b_16_mae", init_mode='zero'):
    from torch import nn

    from clip.model import LayerNorm as CLIPLayerNorm
    from clip.model import QuickGELU

    from torch.nn import GELU
    from torch.nn import LayerNorm

    import json
    now_dir = os.path.abspath(os.path.dirname(__file__))
    timm2clip = json.load(open(f"{now_dir}/timm2clip_vit_b_16.json"))

    assert config_name in myname2timmname, f"The name {config_name} is not one of {list(myname2timmname.keys())}"
    try:
        timm_weight = torch.load(f"/sensei-fs/users/hatan/model/{config_name}.pth")["model"]
    except Exception as e:
        try:
            print(f"/input/yhxu/models/dino_weights/{config_name}.pth")
            timm_weight = torch.load(f"/input/yhxu/models/dino_weights/{config_name}.pth")["model"]
        except Exception as e:
            try:
                print(f"/home/yhxu/models/dino_weights/{config_name}.pth")
                timm_weight = torch.load(f"/home/yhxu/models/dino_weights/{config_name}.pth")["model"]
            except Exception as e:
                print("Please download weight with support/dump_timm_weights.py. \n"
                      "If using mae weight, please check https://github.com/facebookresearch/mae,"
                      "and download the weight as vit_b_16_mae.pth")
                assert False

    # Build model's state dict
    clipname2timmweight = {}
    for timm_key, clip_key in timm2clip.items():
        timm_value = timm_weight[timm_key]
        clipname2timmweight[clip_key[len("visual."):]] = timm_value.squeeze()

    # Resize positional embedding
    resize_pos_embed(clipname2timmweight, module)

    # Load weight to model.
    model_visual_keys = set(module.state_dict().keys())
    load_keys = set(clipname2timmweight.keys())
    # print(f"Load not in model: {load_keys - model_visual_keys}")
    # print(f"Model not in load: {model_visual_keys - load_keys}")
    # status = module.load_state_dict(clipname2timmweight, strict=False)
    try:
        status = module.load_state_dict(clipname2timmweight, strict=False)
    except:
        print('conv.weight has error!')
        if init_mode == 'zero':
            new_weight = torch.zeros_like(clipname2timmweight['conv1.weight'])
            new_weight = new_weight.repeat(1, 2, 1, 1)
            new_weight[:,:3] = clipname2timmweight['conv1.weight']
        elif init_mode == 'mean':
            new_weight = torch.zeros_like(clipname2timmweight['conv1.weight'])
            new_weight = new_weight.repeat(1, 3, 1, 1)
            new_weight = ((clipname2timmweight['conv1.weight']).repeat(1, 3, 1, 1))/3

        clipname2timmweight['conv1.weight'] = new_weight
        status = module.load_state_dict(clipname2timmweight, strict=False)

    # Since timm model has bias, we add it back here.
    module.conv1.bias = nn.Parameter(clipname2timmweight['conv1.bias'])

    # Reinit the visual weights that not covered by timm
    module.ln_pre.reset_parameters()

    def convert_clip_to_timm(module):
        """Copy from detectron2, frozen BN"""
        res = module
        if isinstance(module, CLIPLayerNorm):
            # Timm uses eps=1e-6 while CLIP uses eps=1e-5
            res = LayerNorm(module.normalized_shape, eps=1e-6, elementwise_affine=module.elementwise_affine)
            if module.elementwise_affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
        elif isinstance(module, QuickGELU):
            # Timm uses GELU while CLIP uses QuickGELU
            res = GELU()
        else:
            for name, child in module.named_children():
                new_child = convert_clip_to_timm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res

    return convert_clip_to_timm(module)
