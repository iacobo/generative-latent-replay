import torch
from torch import nn

from avalanche.models.base_model import BaseModel
from avalanche.models.mobilenetv1 import remove_sequential


class FrozenNet(nn.Module):
    """
    Wrapper for pytorch models which splits layers into latent
    and end features. Used for freezing latent features during
    Latent Replay and related continual learning methods (e.g. AR1).

    Warning: currently only imports modules from sequential of `model`
    i.e. bare functions will be skipped.
    """

    def __init__(
        self,
        model,
        latent_layer_num=0,
    ):
        super().__init__()
        remove_sequential(model, (all_layers := nn.ModuleList()))

        self.lat_features = nn.Sequential(*all_layers[:latent_layer_num])
        self.end_features = nn.Sequential(*all_layers[latent_layer_num:])

    def forward(self, raw_input, latent_input=None, return_lat_acts=False):

        if latent_input is None:
            lat_acts = self.lat_features(raw_input)
            full_acts = lat_acts
        else:
            with torch.no_grad():
                lat_acts = self.lat_features(raw_input)
            full_acts = torch.cat((lat_acts, latent_input), 0)

        logits = self.end_features(full_acts)

        if return_lat_acts:
            return logits, lat_acts
        else:
            return logits


def get_hidden_sizes(start_size=32, depth=4):
    sizes = [start_size] + [start_size * 2] * (depth - 2) + [start_size]
    return sizes


class SimpleCNN(nn.Module, BaseModel):
    """
    Convolutional Neural Network
    """

    def __init__(self, num_classes=10, hidden_size=32, hidden_layers=1, drop_rate=0.5):
        """Simple CNN implementation.

        Args:
            num_classes (int, optional): Output size. Defaults to 10.
            hidden_size (int, list, optional): Width of model. Defaults to 32.
            hidden_layers (int, optional): Depth of model. Defaults to 1.
        """
        super().__init__()

        self.features = nn.Sequential()

        for idx in range(hidden_layers):
            self.features.add_module(
                f"conv{idx}", nn.LazyConv2d(hidden_size, kernel_size=3, padding=1)
            )
            self.features.add_module(f"relu{idx}", nn.ReLU(inplace=False))

        self.features.add_module("MaxPool", nn.AdaptiveMaxPool2d(1))
        self.features.add_module("drop", nn.Dropout(p=drop_rate))
        self.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = self.features(x)
        return x


class SimpleMLP(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    """

    def __init__(self, num_classes=10, hidden_size=512, hidden_layers=1, drop_rate=0.5):
        """
        :param num_classes:   output size
        :param hidden_size:   hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate:     dropout rate. 0 to disable
        """
        super().__init__()

        self.features = nn.Sequential()  # nn.Flatten())

        for idx in range(hidden_layers):
            self.features.add_module(f"fc{idx}", nn.LazyLinear(hidden_size))
            self.features.add_module(f"relu{idx}", nn.ReLU(inplace=False))
            self.features.add_module(f"drop{idx}", nn.Dropout(p=drop_rate))

        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = x.contiguous()
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = self.features(x)
        return x
