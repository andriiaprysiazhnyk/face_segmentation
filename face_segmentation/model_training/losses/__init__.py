from .bce import BCELoss
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss


def _get_single_loss(loss_config):
    loss_name = loss_config["name"]

    if loss_name == "bce":
        return BCELoss()
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name == "focal":
        return FocalLoss(alpha=loss_config["alpha"], gamma=loss_config.get("gamma", 2))
    else:
        raise ValueError(f"Loss [{loss_name}] not recognized.")


def get_loss_fn(losses_config):
    if isinstance(losses_config, dict):
        losses = [(_get_single_loss(losses_config), 1)]
    else:
        losses = [(_get_single_loss(loss_config), loss_config["weight"]) for loss_config in losses_config]

    def loss_fn(y_pred, y_true):
        loss = 0
        for criterion, weight in losses:
            loss += weight * criterion(y_pred, y_true)
        return loss

    return loss_fn
