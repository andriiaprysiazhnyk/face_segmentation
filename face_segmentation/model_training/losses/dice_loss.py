import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Implementation of mean soft-dice loss for semantic segmentation
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        """
        Args:
        y_pred: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        y_true: a tensor of shape [B, H, W].
        Returns:
        float: soft-iou loss.
        """
        y_pred_prob = torch.sigmoid(y_pred).unsqueeze(1)

        intersection = torch.sum(y_pred_prob * y_true, dim=(1, 2))
        dice_loss = ((2 * intersection + self.eps) / (
                torch.sum(y_pred_prob ** 2 + y_true ** 2, dim=(1, 2)) + self.eps))

        return 1 - dice_loss.mean()
