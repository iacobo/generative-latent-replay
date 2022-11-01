import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchviz import make_dot
from pathlib import Path

# Plotting style
plt.style.use("seaborn-whitegrid")


def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# Model visualisation
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


# Data visualisation
def plot_random_example():
    """Plots random examples from each class / dist."""

    raise NotImplementedError


# Text results
def get_strategy_names():
    # Ordering of methods to plot.
    names = [f.name for f in Path("./results").iterdir() if f.is_dir()]
    if "Naive" in names:
        names = ["Naive"] + [name for name in names if name != "Naive"]
    return names


def get_results_df(method_name):

    results = pd.read_csv(f"results/{method_name}/eval_results.csv")
    results = results.groupby(["eval_exp", "training_exp"]).last().reset_index()

    n_experiences = len(results["eval_exp"].unique())

    results = [
        results[results["eval_exp"] == i][
            ["training_exp", "eval_accuracy", "eval_loss"]
        ].reset_index()
        for i in range(n_experiences)
    ]

    return results


def results_to_df(latex=False):
    """
    Args:
        results (dict): Dictionary of results from the experiment.

    Returns:
        pd.DataFrame: Results as a DataFrame.
    """

    strategy_names = get_strategy_names()

    results = [get_results_df(name) for name in strategy_names]

    final_avg_accs = [
        np.mean([task_res["eval_accuracy"].iloc[-1] for task_res in res])
        for res in results
    ]
    final_avg_loss = [
        np.mean([task_res["eval_loss"].iloc[-1] for task_res in res]) for res in results
    ]
    df = pd.DataFrame(
        {"Final Avg Acc": final_avg_accs, "Final Avg Loss": final_avg_loss},
        index=strategy_names,
    )

    df = df.style.highlight_max(axis=1, props="bfseries: ;")

    if latex:
        df = df.to_latex()

    return df


# Results plots
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
    results = get_results_df(method_name)
    long_name = {"acc": "accuracy", "loss": "loss"}

    res = [res[f"eval_{long_name[metric]}"] for res in results]

    if repeat_vals:
        res = [list(np.repeat(val, repeat_vals)) for val in res]

    for i, r in enumerate(res, start=1):
        ax.plot(r, label=f"Task {i}")

    if metric == "acc":
        ax.set_title(method_name)

    simpleaxis(ax)

    return res


def plot_multiple_results(mode="train", repeat_vals=10, loss=False):

    # Names of methods with results to plot.
    names = get_strategy_names()

    # Build figure
    if loss:
        height = 2
    else:
        height = 1
    fig, axes = plt.subplots(
        height,
        len(names),
        sharey="row",
        squeeze=False,
        figsize=(2 * len(names), height * 3),
    )

    # Plot results
    for i, name in enumerate(names):
        plot_results(name, axes[0][i], "acc", mode, repeat_vals)
        if loss:
            plot_results(name, axes[1][i], "loss", mode, repeat_vals)

    # Titles, labels etc.
    fig.supxlabel("Epoch")
    plot_single_legend(fig)
    fig.axes[0].set_ylabel(f"{mode.capitalize()} Accuracy")

    if loss:
        fig.axes[1].set_ylabel(f"{mode.capitalize()} Loss")


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
        # loc="lower center",
        bbox_to_anchor=(0.575, 0),
        bbox_transform=plt.gcf().transFigure,
    )
