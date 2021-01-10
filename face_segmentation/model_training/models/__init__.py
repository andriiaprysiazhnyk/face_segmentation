import torch

from .unet import UNet
from .deeplab import DeepLabV3


def get_network(model_config):
    """
    Create model form configuration
    Args:
        model_config (dict): dictionary of model config
    Return:
        model (torch.nn.Module): model created from config
    """
    arch = model_config["arch"]
    del model_config["arch"]

    if arch == "unet":
        return UNet(model_config["encoder"], model_config.get("encoder_depth", 5),
                    model_config.get("encoder_weights", None))
    elif arch == "deeplab_v3":
        return DeepLabV3(model_config["encoder"], model_config.get("encoder_depth", 5),
                         model_config.get("encoder_weights", None))
    else:
        raise ValueError(f"Model architecture {arch} not recognized.")


def load_network(model_path):
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model_type, weights = state_dict["type"], state_dict["weights"]
    del state_dict["type"], state_dict["weights"]

    if model_type == "unet":
        model = UNet(**state_dict)
    elif model_type == "deeplab_v3":
        model = DeepLabV3(**state_dict)
    else:
        raise TypeError(f"Unknown model type: {model_type}")

    model.load_state_dict(weights)
    return model
