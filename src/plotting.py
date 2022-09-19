import numpy as np
import matplotlib.pyplot as plt

from torchviz import make_dot


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
    prefix = f"eval_phase/{mode}_stream/Task000/"

    results_clean[mode]["loss"] = [
        [result[f"Loss_Exp/{prefix}Exp{str(i).zfill(3)}"] for result in results]
        for i in range(n_experiences)
    ]
    results_clean[mode]["acc"] = [
        [result[f"Top1_Acc_Exp/{prefix}Exp{str(i).zfill(3)}"] for result in results]
        for i in range(n_experiences)
    ]

    res = results_clean[mode][metric]

    if repeat_vals:
        res = [list(np.repeat(val, repeat_vals)) for val in res]

    for i in range(n_experiences):
        ax.plot(res[i], label=f"Task {i}")

    ax.set_title(method_name)

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


def plot_multiple_results(
    results,
    titles,
    n_experiences,
    metric="acc",
    mode="train",
    repeat_vals=10,
):

    fig, axes = plt.subplots(1, len(titles), sharey="row")

    for i, (res, name) in enumerate(zip(results, titles)):
        plot_results(res, name, axes[i], n_experiences, metric, mode, repeat_vals)

    plot_single_legend(fig)
    fig.suptitle(f"{mode.capitalize()} {metric.capitalize()}")
