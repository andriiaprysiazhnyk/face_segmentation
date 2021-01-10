from torch import nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    def __init__(self, encoder_name, encoder_depth, encoder_weights):
        super(UNet, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_depth = encoder_depth
        self.encoder_weights = encoder_weights

        self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                              encoder_depth=encoder_depth, classes=1, in_channels=3)

    def forward(self, x):
        return self.model(x)

    def get_params_groups(self):
        return (
            list(self.model.encoder.parameters()),
            list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())
        )

    def extended_state_dict(self):
        return {
            "type": "unet",
            "encoder_name": self.encoder_name,
            "encoder_depth": self.encoder_depth,
            "encoder_weights": self.encoder_weights,
            "weights": self.state_dict()
        }
