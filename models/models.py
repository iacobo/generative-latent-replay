import torch
from torch import nn
from torch import distributions as D

from avalanche.models.base_model import BaseModel
from avalanche.models.mobilenetv1 import remove_sequential

from sklearn.mixture import GaussianMixture

import numpy as np


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


class SimpleCNN(nn.Module):
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

        self.features.add_module(nn.AdaptiveMaxPool2d(1))
        self.features.add_module(nn.Dropout(p=drop_rate))
        self.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    """

    def __init__(self, num_classes=10, hidden_size=512, hidden_layers=1, drop_rate=0.5):
        """
        :param num_classes: output size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
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


# Use PyTorch GMM
# https://pytorch.org/docs/stable/distributions.html#mixturesamefamily
class GMM(nn.Module):
    def __init__(self, n_components, dim, weights=None):
        """
        Initialises a GMM.

        Args:
            n_components (int): Number of components in GMM.
            dim (int):          Dimensionality of data to model.
        """
        super().__init__()
        self.dim = dim
        self.n_components = n_components

        # Initialise mixture weights to uniform
        if weights is None:
            self.weights = torch.ones(n_components)
        else:
            self.weights = nn.Parameter(weights)
            assert (
                self.weights.shape[0] == n_components
            ), "`weights` must be the same size as `n_components`"
        # Initialise normal mean/std's to random
        # JA: implement initialising with previous GMM's posteriors.

    def forward(self, n_components, dim):
        """
        Forward pass.
        """
        mix = D.Categorical(self.weights)
        comp = D.Independent(
            D.Normal(torch.randn(n_components, dim), torch.rand(n_components, dim)), 1
        )
        gmm = D.MixtureSameFamily(mix, comp)

        return gmm


class GMM_sk:
    def __init__(self, n_classes, cov_type="full"):
        """
        Initialises a GMM.

        Args:
            n_components (int): Number of components in GMM.
            cov_type (str):     Covariance type. One of "full", "diag", "tied", "spherical".
            dim (int):          Dimensionality of data to model.
        """
        super().__init__()
        self.cov_type = cov_type
        self.n_classes = n_classes

        self.estimator = GaussianMixture(
            n_components=self.n_classes,
            covariance_type=self.cov_type,
            max_iter=20,
            random_state=0,
        )

    def train(self, x, y):
        self.estimator.means_init = np.array(
            [x[y == i].mean(axis=0) for i in range(self.n_classes)]
        )
        self.estimator.fit(x)

    def forward(self, x, y):
        """
        Forward pass.
        """
        y_pred = self.estimator.predict(x)

        return y_pred

    def get_acc(self, y_pred, y_true):
        acc = np.mean(y_pred.ravel() == y_true.ravel()) * 100
        return acc

    def get_per_class_cluster(self, x):
        """
        Get per class cluster
        """

        # clusters = clusterer.fit(x)
        # n_clusters = len(clusters.centroids)
        # return clusters.centroids
        return None


if __name__ == "__main__":

    model = GMM(n_components=5, dim=2)
    print(model)
