import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchviz import make_dot
from pathlib import Path


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


def plot_results_old(
    results,
    method_name,
    ax,
    n_experiences,
    metric="acc",
    mode="train",
    repeat_vals=False,
):

    results_clean = {"train": {"acc": [], "loss": []}, "test": {"acc": [], "loss": []}}
    prefix = f"eval_phase/{mode}_stream/Task000/"

    if metric == "loss":
        results_clean[mode]["loss"] = [
            [result[f"Loss_Exp/{prefix}Exp{str(i).zfill(3)}"] for result in results]
            for i in range(n_experiences)
        ]

    elif metric == "acc":
        results_clean[mode]["acc"] = [
            [result[f"Top1_Acc_Exp/{prefix}Exp{str(i).zfill(3)}"] for result in results]
            for i in range(n_experiences)
        ]

    res = results_clean[mode][metric]

    if repeat_vals:
        res = [list(np.repeat(val, repeat_vals)) for val in res]

    for i in range(n_experiences):
        ax.plot(res[i], label=f"Task {i}")

    if metric == "acc":
        ax.set_title(method_name)

    return results_clean


def plot_results(
    method_name,
    ax,
    metric="acc",
    mode="train",
    repeat_vals=False,
):
    """
    Plots results from a single experiment.
    """
    results = pd.read_csv(f"log/{method_name}/eval_results.csv")
    results = results.groupby(["eval_exp", "training_exp"]).last().reset_index()

    n_experiences = len(results["eval_exp"].unique())

    results = [
        results[results["eval_exp"] == i][
            ["training_exp", "eval_accuracy", "eval_loss"]
        ].reset_index()
        for i in range(n_experiences)
    ]

    res = [res["eval_accuracy"] for res in results]
    if repeat_vals:
        res = [list(np.repeat(val, repeat_vals)) for val in res]

    for r in res:
        ax.plot(r, label=f"{method_name}")

    if metric == "acc":
        ax.set_title(method_name)

    return res


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
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        bbox_transform=plt.gcf().transFigure,
    )


def plot_multiple_results(mode="train", repeat_vals=10, loss=False):

    # Names of methods with results to plot.
    names = [f.name for f in Path("./log").iterdir() if f.is_dir()]

    fig, axes = plt.subplots(
        2,
        len(names),
        sharey="row",
        squeeze=False,
        figsize=(2 * len(names), 6),
    )

    for i, name in enumerate(names):
        plot_results(name, axes[0][i], "acc", mode, repeat_vals)
        if loss:
            plot_results(name, axes[1][i], "loss", mode, repeat_vals)

    plot_single_legend(fig)
    fig.axes[0].set_ylabel(f"{mode.capitalize()} Accuracy")
    if loss:
        fig.axes[1].set_ylabel(f"{mode.capitalize()} Loss")
    plt.xlabel("Epoch")
