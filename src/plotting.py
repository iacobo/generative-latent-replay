import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchviz import make_dot
from pathlib import Path
from avalanche.benchmarks.classic import RotatedMNIST, PermutedMNIST

# Plotting style
plt.style.use("seaborn-whitegrid")


def simpleaxis(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not grid:
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
def plot_random_example(n_examples, n_exp, experiment="PermutedMNIST"):
    """Plots random examples from each class / dist."""

    fig, ax = plt.subplots(n_examples, n_exp, figsize=(5, 5))

    # Dataset
    if experiment == "PermutedMNIST":
        experiences = PermutedMNIST(n_experiences=n_exp)
    elif experiment == "RotatedMNIST":
        experiences = RotatedMNIST(n_experiences=n_exp)

    train_stream = experiences.train_stream

    for i, train_exp in enumerate(train_stream, start=0):
        for j in range(n_examples):
            img = train_exp.dataset[j][0].numpy().squeeze()
            ax[j][i].imshow(img)
            # Remove gridlines and ticks
            ax[j][i].grid(False)
            ax[j][i].set_xticks([])
            ax[j][i].set_yticks([])

        ax[n_examples - 1][i].set_xlabel(f"Experience {i}")

    # Set labels and title
    fig.supxlabel("Experiences")
    fig.supylabel("Examples")
    fig.suptitle(f"Examples from each experience for {experiment}", fontsize=16)
    plt.tight_layout()
    plt.show()


# Text results
def get_strategy_names(experiment):
    # Ordering of methods to plot.
    names = [f.name for f in Path(f"./results/{experiment}").iterdir() if f.is_dir()]
    if "Naive" in names:
        names = ["Naive"] + [name for name in sorted(names) if name != "Naive"]
    return names


def get_results_df(method_name, experiment):

    results = pd.read_csv(f"results/{experiment}/{method_name}/eval_results.csv")
    results = results.groupby(["eval_exp", "training_exp"]).last().reset_index()

    n_experiences = len(results["eval_exp"].unique())

    results = [
        results[results["eval_exp"] == i][
            ["training_exp", "eval_accuracy", "eval_loss"]
        ].reset_index()
        for i in range(n_experiences)
    ]

    return results


def results_to_df(experiment="PermutedMNIST", latex=False, bold=False):
    """
    Args:
        results (dict): Dictionary of results from the experiment.

    Returns:
        pd.DataFrame: Results as a DataFrame.
    """

    strategy_names = get_strategy_names(experiment)

    results = [get_results_df(name, experiment) for name in strategy_names]

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

    if bold:
        df = df.style.highlight_max(axis=1, props="bfseries: ;")

    if latex:
        df = df.to_latex()

    return df


# Results plots
def plot_results(
    method_name, ax, experiment, metric="acc", mode="train", repeat_vals=False
):
    """
    Plots results from a single experiment.
    """
    results = get_results_df(method_name, experiment)
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


def plot_final_avg_results(experiment="RotatedMNIST_buffer_size"):

    fig, ax = plt.subplots(1, 2, figsize=(5, 3.25), dpi=300)
    fig.suptitle("Final average performance for GLR \n on Rotated MNIST")

    res = results_to_df(experiment=experiment)
    res.index = res.index.str.replace("GLR_", "").astype(int)
    res.sort_index(inplace=True)

    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1.2)

    # plt.xticks(res.index)

    for ax, metric, colour in zip(ax, ["Acc", "Loss"], ["C0", "C4"]):
        ax.plot(
            res[f"Final Avg {metric}"],
            linestyle="--",
            marker="o",
            color=colour,
            label=res.index,
        )

        ax.set_xlim(0, 32000)
        simpleaxis(ax, grid=True)
        # label x axis with buffer size
        ax.set_xlabel("Buffer Size")
        ax.set_ylabel(metric)

        # Annotate each point with x value
        for i in range(len(res)):
            percent = (100 * res.index[i]) // 60000
            x, y = res.index[i], res[f"Final Avg {metric}"].iloc[i]

            ax.annotate(
                f"{percent}%",
                xy=(x, y),
                xytext=(x, y + 0.05),
            )

            if i == 1:
                ax.plot(
                    x,
                    y,
                    markeredgecolor="orange",
                    fillstyle="none",
                    marker="o",
                    markersize=10,
                )

    plt.tight_layout()


def plot_multiple_results(
    mode="train", experiment="PermutedMNIST", repeat_vals=10, loss=False
):

    # Names of methods with results to plot.
    names = get_strategy_names(experiment)

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
        plot_results(name, axes[0][i], experiment, "acc", mode, repeat_vals)
        if loss:
            plot_results(name, axes[1][i], experiment, "loss", mode, repeat_vals)

    # Titles, labels etc.
    fig.supxlabel("Epoch")
    plot_single_legend(fig)
    fig.axes[0].set_ylabel(f"{mode.capitalize()} Accuracy")

    if loss:
        fig.axes[1].set_ylabel(f"{mode.capitalize()} Loss")

    fig.suptitle(f"{experiment.split('MNIST')[0]} MNIST", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/plots/{experiment}_{mode}.png", dpi=300)


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


if __name__ == "__main__":

    plot_random_example(4, 5)
    plot_random_example(4, 5, "RotatedMNIST")
