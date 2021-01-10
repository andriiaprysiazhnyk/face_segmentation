def iou(y_pred, y_true):
    y_pred, y_true = y_pred.cpu(), y_true.cpu()
    y_pred = (y_pred.squeeze(1) > 0).long()

    intersection = (y_pred & y_true).sum(dim=[1, 2])
    union = (y_pred | y_true).sum(dim=[1, 2])
    eps = 1e-6
    return ((intersection + eps) / (union + eps)).mean().item()
