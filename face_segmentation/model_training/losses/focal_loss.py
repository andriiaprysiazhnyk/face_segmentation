import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Reduction type must be one of following: 'none' | 'mean' | 'sum'")

        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.cat((1 - y_pred, y_pred), dim=1) + self.eps
        target_one_hot = torch.eye(2, device=y_true.device)[y_true].permute(0, 3, 1, 2)

        weight = torch.pow(1. - y_pred, self.gamma)
        focal = -self.alpha * weight * torch.log(y_pred)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == "none":
            return loss_tmp
        elif self.reduction == "mean":
            return torch.mean(loss_tmp)
        elif self.reduction == "sum":
            return torch.sum(loss_tmp)
