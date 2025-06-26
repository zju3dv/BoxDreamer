import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22], weights=[1.0, 0.8, 0.5, 0.3]):
        """Initialize PerceptualLoss module.

        Args:
            layers (list): Layer indices to extract features from VGG16.
            weights (list): Weights for each layer's features.
        """
        super().__init__()

        # Load pretrained VGG16 model
        vgg = models.vgg16(pretrained=True).features
        vgg.eval()  # Set to evaluation mode
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        self.vgg = vgg

        self.layers = layers
        self.weights = weights
        assert len(self.layers) == len(
            self.weights
        ), "Number of layers and weights must match."

        # Define image normalization parameters (based on ImageNet)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred, gt):
        """
        Calculate perceptual loss between predicted image and ground truth image.

        Args:
            pred (torch.Tensor): Predicted image, shape (N, C, H, W).
            gt (torch.Tensor): Ground truth image, shape (N, C, H, W).

        Returns:
            torch.Tensor: Calculated perceptual loss.
        """
        # Normalize input
        # convert to same device
        if self.mean.device != pred.device:
            self.mean = self.mean.to(pred.device)
            self.std = self.std.to(pred.device)
            self.vgg = self.vgg.to(pred.device)

        pred = (pred - self.mean) / self.std
        gt = (gt - self.mean) / self.std

        # Initialize feature lists
        pred_features = []
        gt_features = []

        x_pred = pred
        x_gt = gt

        # Iterate over each layer of VGG16, extract features from specified layers
        for i, layer in enumerate(self.vgg):
            x_pred = layer(x_pred)
            x_gt = layer(x_gt)
            if i in self.layers:
                pred_features.append(x_pred)
                gt_features.append(x_gt)

        # Calculate weighted MSE loss for each layer's features
        loss = 0.0
        for feat_pred, feat_gt, weight in zip(pred_features, gt_features, self.weights):
            loss += weight * F.mse_loss(feat_pred, feat_gt)

        return loss
