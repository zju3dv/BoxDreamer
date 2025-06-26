# keypoints focal loss
# from cornerNet https://arxiv.org/pdf/1808.01244
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        """
        Focal Loss for keypoints detection as described in CornerNet paper.
        Args:
            alpha (float): Controls the contribution of easy negatives.
            beta (float): Controls the penalty for false positives near positives.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmaps of shape (B, C, H, W).
            gt (torch.Tensor): Ground truth heatmaps of shape (B, C, H, W).
        Returns:
            torch.Tensor: Scalar focal loss value.
        """
        # input range is [-1, 1] so we need to normalize it to [0, 1]
        pred = (pred + 1) / 2
        gt = (gt + 1) / 2
        # print(pred)
        # Ensure numerical stability by clamping predictions
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # Positive focal loss
        pos_mask = (gt == 1.0)  # Mask for positive locations
        pos_loss = torch.pow(1 - pred, self.alpha) * torch.log(pred + 1e-9) # 1e-9 is added to avoid log(0)
        pos_loss = pos_loss * pos_mask  # Apply the mask
        # pos_loss[torch.isnan(pos_loss)] = 0.0
        # pos_loss[torch.isinf(pos_loss)] = 0.0

        # Negative focal loss
        neg_mask = ~ pos_mask  # Mask for negative locations
        neg_weights = torch.pow(1 - gt, self.beta)  # Weight reduction around positives
        
        neg_loss = torch.pow(pred, self.alpha) * torch.log((1 - pred) + 1e-9)  # 1e-9 is added to avoid log(0)
        # if inf or nan occurs in neg_loss, set it to 0
        # neg_loss[torch.isnan(neg_loss)] = 0.0
        # neg_loss[torch.isinf(neg_loss)] = 0.0
        neg_loss = neg_loss * neg_mask * neg_weights  # Apply the mask and weights

        # Combine positive and negative loss
        num_pos = pos_mask.sum()  # Count of positive samples
        pos_loss_sum = pos_loss.sum()
        neg_loss_sum = neg_loss.sum()

        if num_pos > 0:
            loss = -(pos_loss_sum + neg_loss_sum) / num_pos
        else:
            loss = -neg_loss_sum
        
        return loss