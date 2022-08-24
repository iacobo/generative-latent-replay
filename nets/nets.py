################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is the definition od the Mid-caffenet high resolution in Pythorch
"""

import torch.nn as nn
import torch

from avalanche.models.base_model import BaseModel

try:
    from pytorchcv.models.mobilenet import DwsConvBlock
except Exception:
    from pytorchcv.models.common import DwsConvBlock


def remove_sequential(network, all_layers):

    for layer in network.children():
        # if sequential layer, apply recursively to layers in sequential layer
        if isinstance(layer, nn.Sequential):
            # print(layer)
            remove_sequential(layer, all_layers)
        else:  # if leaf node, add it to list
            # print(layer)
            all_layers.append(layer)


def remove_DwsConvBlock(cur_layers):

    all_layers = []  # nn.ModuleList()
    for layer in cur_layers:
        if isinstance(layer, DwsConvBlock):
            # print("helloooo: ", layer)
            for ch in layer.children():
                all_layers.append(ch)
        else:
            all_layers.append(layer)
    return all_layers


class FrozenNet(nn.Module):
    """
    Wrapper for pytorch models which splits layers into latent 
    and end features. Used for freezing latent features during 
    Latent Replay and related continual learning methods (e.g. AR1).
    """

    def __init__(
        self, model, latent_layer_num,
    ):
        super().__init__()

        all_layers = nn.ModuleList()
        remove_sequential(model, all_layers)
        # all_layers = remove_DwsConvBlock(all_layers)

        self.lat_features = nn.Sequential(*all_layers[:latent_layer_num])
        self.end_features = nn.Sequential(*all_layers[latent_layer_num:])

    def forward(self, x, latent_input=None, return_lat_acts=False):

        if latent_input is None:
            lat_acts = self.lat_features(x)
            full_acts = lat_acts
        else:
            with torch.no_grad():
                lat_acts = self.lat_features(x)
            full_acts = torch.cat((lat_acts, latent_input), 0)

        logits = self.end_features(full_acts)

        if return_lat_acts:
            return logits, lat_acts
        else:
            return logits


class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    **Example**::
        >>> from avalanche.models import SimpleMLP
        >>> n_classes = 10 # e.g. MNIST
        >>> model = SimpleMLP(num_classes=n_classes)
        >>> print(model) # View model details
    """

    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=512,
        hidden_layers=1,
        drop_rate=0.5,
    ):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()

        self.features = nn.Sequential()

        for idx in range(hidden_layers):
            cur_input_size = input_size if idx == 0 else hidden_size

            self.features.add_module(f"fc{idx}", nn.Linear(cur_input_size, hidden_size))
            self.features.add_module(f"relu{idx}", nn.ReLU(inplace=False))
            self.features.add_module(f"drop{idx}", nn.Dropout(p=drop_rate))

        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        x = x.contiguous()
        x = x.flatten(start_dim=1)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.flatten(start_dim=1)
        x = self.features(x)
        return x


if __name__ == "__main__":

    model = FrozenNet(pretrained=True)

    for name, param in model.named_parameters():
        print(name)
