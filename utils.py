import torch
from torch import nn, optim
from torch import distributions as D

import numpy as np
import matplotlib.pyplot as plt

from torchviz import make_dot


def get_device():
    """
    Returns:
        torch.device: Torch device. First GPU if available, else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def render_model(lat_mb_x, model, mb_x, train_exp_counter):
    """
    Renders graph of model.
    """
    make_dot(
        model(mb_x, latent_input=lat_mb_x, return_lat_acts=True),
        params=dict(
            list(model.named_parameters())
            + [("mb_x", mb_x)]
            + ([] if train_exp_counter == 0 else [("lat_mb_x", lat_mb_x)])
        ),
    ).render(f"torchviz_output_exp{train_exp_counter}", format="png")


def plot_random_example():
    """Plots random examples from each class / dist."""

    raise NotImplementedError


def plot_results(
    results,
    method_name,
    ax,
    n_experiences,
    metric="acc",
    mode="train",
    repeat_vals=False,
):

    results_clean = {"train": {"acc": [], "loss": []}, "test": {"acc": [], "loss": []}}
    loss_prefix = f"Loss_Stream/eval_phase/{mode}_stream/"
    acc_prefix = f"Top1_Acc_Stream/eval_phase/{mode}_stream/"

    results_clean[mode]["loss"] = [
        [result[f"{loss_prefix}Task{str(i).zfill(3)}"] for result in results]
        for i in range(n_experiences)
    ]
    results_clean[mode]["acc"] = [
        [result[f"{acc_prefix}Task{str(i).zfill(3)}"] for result in results]
        for i in range(n_experiences)
    ]

    res = results_clean[mode][metric]

    if repeat_vals:
        res = [list(np.repeat(val, repeat_vals)) for val in res]

    for i in range(n_experiences):
        ax.plot(res[i], label=f"Task {i}")

    ax.set_title(f"{method_name} {mode.capitalize()} {metric.capitalize()}")

    return results_clean


def plot_single_legend(fig):
    """
    For multiple subplots with shared labeled lines to plot,
    combines legends to remove redundancy.
    """
    labels_handles = {
        label: handle
        for ax in fig.axes
        for handle, label in zip(*ax.get_legend_handles_labels())
    }

    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=plt.gcf().transFigure,
    )


def train_gmm(x, n_epochs=4, lr=0.001, momentum=0.9):

    gmm = GMM()
    parameters = gmm.parameters()  # [weights, means, stdevs]
    optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)

    print("Fitting GMM")

    for i in range(n_epochs):
        optimizer.zero_grad()
        x = torch.randn(5000, 2)  # this can be an arbitrary x samples
        loss = -gmm.log_prob(x).mean()  # -densityflow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {i} | Loss: {loss}")

    return None


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
