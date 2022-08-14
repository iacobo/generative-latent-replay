import torch
from torch import nn
from torch import optim
import torch.distributions as D

from typing import NamedTuple, List, Callable
from torch import Tensor
from torch.nn import Module

import numpy as np
import matplotlib.pyplot as plt


class GMM(nn.Module):
    def __init__(self, n_components, dim, weights):
        """
        Initialises a GMM.

        Args:
            n_components (int): Number of components in GMM.
            dim (int):          Dimensionality of data to model.
        """
        super().__init__()
        self.n_components = n_components
        self.dim = dim
        self.weights = nn.Parameter(weights)
        # assert weights.shape[0] == n_components
        # Initialise mixture weights to uniform
        # Initialise normal mean/std's to random
        # JA: implement initialising with previous
        #     GMM's posteriors.

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


def plot_results(
    results,
    method_name,
    ax,
    n_experiences,
    metric="acc",
    mode="train",
    repeat_vals=True,
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
        res = [list(np.repeat(val, 2)) for val in res]

    for i in range(n_experiences):
        ax.plot(res[i], label=f"Task {i}")

    ax.set_title(f"{method_name} {mode.capitalize()} {metric.capitalize()}")

    return results_clean


def plot_single_legend(fig):
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


def plot_random_example():
    """Plots random examples from each class / dist.
    """

    return None


def train_gmm(n_epochs, x, lr=0.001, momentum=0.9):

    parameters = [weights, means, stdevs]
    optimizer1 = optim.SGD(parameters, lr=lr, momentum=momentum)
    gmm = GMM()

    for i in range(n_epochs):
        optimizer1.zero_grad()
        x = torch.randn(5000, 2)  # this can be an arbitrary x samples
        loss2 = -gmm.log_prob(x).mean()  # -densityflow.log_prob(inputs=x).mean()
        loss2.backward()
        optimizer1.step()

        print(i, loss2)

    return None


def get_device():
    """
    Returns:
        torch.device: Torch device. First GPU if available, else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# JA: copy of Avalanche funcs


class LayerAndParameter(NamedTuple):
    layer_name: str
    layer: Module
    parameter_name: str
    parameter: Tensor


def get_layers_and_params(model: Module, prefix="") -> List[LayerAndParameter]:
    result: List[LayerAndParameter] = []
    for param_name, param in model.named_parameters(recurse=False):
        result.append(LayerAndParameter(prefix[:-1], model, prefix + param_name, param))

    layer_name: str
    layer: Module
    for layer_name, layer in model.named_modules():
        if layer == model:
            continue

        layer_complete_name = prefix + layer_name + "."

        result += get_layers_and_params(layer, prefix=layer_complete_name)

    return result


def freeze_up_to(
    model: Module,
    freeze_until_layer: str = None,
    set_eval_mode: bool = True,
    set_requires_grad_false: bool = True,
    layer_filter: Callable[[LayerAndParameter], bool] = None,
    module_prefix: str = "",
):
    """
    A simple utility that can be used to freeze a model.
    :param model: The model.
    :param freeze_until_layer: If not None, the freezing algorithm will continue
        (proceeding from the input towards the output) until the specified layer
        is encountered. The given layer is excluded from the freezing procedure.
    :param set_eval_mode: If True, the frozen layers will be set in eval mode.
        Defaults to True.
    :param set_requires_grad_false: If True, the autograd engine will be
        disabled for frozen parameters. Defaults to True.
    :param layer_filter: A function that, given a :class:`LayerParameter`,
        returns `True` if the parameter must be frozen. If all parameters of
        a layer are frozen, then the layer will be set in eval mode (according
        to the `set_eval_mode` parameter. Defaults to None, which means that all
        parameters will be frozen.
    :param module_prefix: The model prefix. Do not use if non strictly
        necessary.
    :return:
    """

    frozen_layers = set()
    frozen_parameters = set()

    to_freeze_layers = dict()
    print("FREEZE_UP_TO")
    for param_def in get_layers_and_params(model, prefix=module_prefix):
        print(freeze_until_layer, param_def.layer_name, param_def.parameter_name)
        if (
            freeze_until_layer is not None
            and freeze_until_layer == param_def.layer_name
        ):
            break

        freeze_param = layer_filter is None or layer_filter(param_def)
        if freeze_param:
            if set_requires_grad_false:
                param_def.parameter.requires_grad = False
                frozen_parameters.add(param_def.parameter_name)

            if param_def.layer_name not in to_freeze_layers:
                to_freeze_layers[param_def.layer_name] = (True, param_def.layer)
        else:
            # Don't freeze this parameter -> do not set eval on the layer
            to_freeze_layers[param_def.layer_name] = (False, None)

    if set_eval_mode:
        for layer_name, layer_result in to_freeze_layers.items():
            if layer_result[0]:
                layer_result[1].eval()
                frozen_layers.add(layer_name)

    return frozen_layers, frozen_parameters
