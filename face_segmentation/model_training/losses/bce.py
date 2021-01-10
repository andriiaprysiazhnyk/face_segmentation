import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred.squeeze(1), y_true.float())
