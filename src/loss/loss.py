import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from .utils.perceptual import PerceptualLoss
from .utils.focal_loss import FocalLoss

class Loss(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.loss_funcs = self._initialize_loss_functions(cfg)

    def _initialize_loss_functions(self, cfg):
        loss_funcs = {}
        for loss_cfg in cfg.losses:
            loss_type = loss_cfg.type
            if loss_type == "mse":
                loss_funcs[loss_cfg.pred_key] = nn.MSELoss()
            elif loss_type == "cross_entropy":
                loss_funcs[loss_cfg.pred_key] = nn.CrossEntropyLoss()
            elif loss_type == "l1":
                loss_funcs[loss_cfg.pred_key] = nn.L1Loss()
            elif loss_type == "huber":
                loss_funcs[loss_cfg.pred_key] = nn.HuberLoss()
            elif loss_type == "bce":
                loss_funcs[loss_cfg.pred_key] = nn.BCEWithLogitsLoss()
            # Add more loss types as needed
            elif loss_type == "PerceptualLoss":
                loss_funcs[loss_cfg.pred_key] = PerceptualLoss()
            elif loss_type == "smooth_l1":
                loss_funcs[loss_cfg.pred_key] = nn.SmoothL1Loss()
            elif loss_type == "focal":
                loss_funcs[loss_cfg.pred_key] = FocalLoss()
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
        return loss_funcs

    def forward(self, data):
        total_loss = 0.0
        loss_details = {}
        for loss_cfg in self.cfg.losses:
            pred_key = loss_cfg.pred_key
            gt_key = loss_cfg.gt_key
            weight = loss_cfg.weight
            # todo : implement mask_key
            mask_key = loss_cfg.mask_key
            if pred_key in self.loss_funcs:
                loss_func = self.loss_funcs[pred_key]
                if mask_key is not None and data[mask_key] is not None:
                    assert mask_key in data, f"Mask key {mask_key} not found in data"
                    if weight[0] != 0.0:
                        mask_loss = loss_func(data[pred_key][data[mask_key]], data[gt_key][data[mask_key]]) * weight[0]
                    else:
                        mask_loss = 0.0
                    
                    if weight[1] != 0.0:
                        loss =  loss_func(data[pred_key][~data[mask_key]], data[gt_key][~data[mask_key]]) * weight[1] + mask_loss
                    else:
                        loss = mask_loss
                else:
                    mask_loss = None
                    loss = loss_func(data[pred_key], data[gt_key]) * weight
                
                total_loss += loss
                if pred_key in loss_details:
                    loss_details[pred_key] += loss.item()
                else:                    
                    loss_details[pred_key] = loss.item()
            else:
                raise ValueError(f"Input key not found in loss functions: {pred_key}")

        return total_loss, loss_details