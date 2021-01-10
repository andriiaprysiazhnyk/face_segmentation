from torch import nn
import segmentation_models_pytorch as smp


class DeepLabV3(nn.Module):
    def __init__(self, encoder_name, encoder_depth, encoder_weights):
        super(DeepLabV3, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_depth = encoder_depth
        self.encoder_weights = encoder_weights

        self.model = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=encoder_weights,
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
            "type": "deeplab_v3",
            "encoder_name": self.encoder_name,
            "encoder_depth": self.encoder_depth,
            "encoder_weights": self.encoder_weights,
            "weights": self.state_dict()
        }
