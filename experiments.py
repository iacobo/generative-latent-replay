import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Generative Latent Replay

        Experimental code to test generative latent replay on benchmark continual learning problems.

        i.e. normalising bottleneck representations and sampling from fitted GMM on latent space.
        """
    )
    return


@app.cell
def __():
    from pathlib import Path

    # ML imports
    import torch

    # Local imports
    from src import utils, plotting, models
    import main

    # Continual Learning strategies
    from avalanche.training import plugins
    return Path, main, models, plotting, plugins, torch, utils


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Setup
        """
    )
    return


@app.cell
def __(utils):
    # Reproducibility
    SEED = 43769
    utils.set_seed(SEED)

    # Reporting
    eval_rate = 1
    return SEED, eval_rate


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Problem definition
        """
    )
    return


@app.cell
def __(SEED, main, utils):
    # Number of tasks
    n_experiences = 5

    # Transform data to format expected by model
    transform = utils.get_transforms(resize=244, n_channels=3, normalise=True)

    # Load dataset
    experiment = "RotatedMNIST"
    experiences = main.get_experiences(experiment, n_experiences, transform, SEED)

    # Train and test streams
    train_stream = experiences.train_stream
    test_stream = experiences.test_stream
    return experiment, train_stream


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Hyperparameters
        """
    )
    return


@app.cell
def __(eval_rate, plugins, utils):
    # Replays
    replay_buffer_size = 6000

    # Define model skeleton
    model = "alexnet"

    # Frozen backbone
    if model == "alexnet":
        latent_layer_number = 16
    elif model == "mobilenet":
        latent_layer_number = 158

    # SGD hyperparams
    sgd_kwargs = {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,  # l2 regularization
    }

    strategy_kwargs = {
        "eval_every": 1,
        "train_epochs": 40,
        "train_mb_size": 64,
        "eval_mb_size": 128,
        "device": utils.get_device(),
        "plugins": [
            plugins.EarlyStoppingPlugin(
                patience=eval_rate,
                val_stream_name="train_stream/Task000",
                margin=0.003,  # metric
            )
        ],
    }
    return (
        latent_layer_number,
        model,
        replay_buffer_size,
        sgd_kwargs,
        strategy_kwargs,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Building base model
        """
    )
    return


@app.cell
def __(main, model):
    model_1 = main.get_model(model)
    return (model_1,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Loading Continual Learning strategies for experiments
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Training loop
        """
    )
    return


@app.cell
def __(
    Path,
    experiment,
    latent_layer_number,
    main,
    model_1,
    replay_buffer_size,
    sgd_kwargs,
    strategy_kwargs,
    train_stream,
    utils,
):
    strategy_name = 'Latent Replay'
    strategy = main.get_strategy(strategy_name, model_1, sgd_kwargs, strategy_kwargs, replay_buffer_size, latent_layer_number)
    for train_exp in train_stream:
        strategy.train(train_exp)
        strategy.eval(train_stream)
        utils.save_model(model_1, Path(f'results/{experiment}/{strategy_name}'), f'model_{train_exp.current_experience}.pt')
    utils.close_loggers(strategy)
    return


@app.cell
def __(models, torch):
    n_classes = 10
    n_samples = 60000
    dim = 9216
    sampler = models.GMM(n_classes=n_classes)
    x = torch.rand(n_samples, dim).detach().cpu().numpy()
    y = torch.randint(0, n_classes, (n_samples,)).detach().cpu().numpy()

    print(x.dtype, y.dtype)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Plotting
        """
    )
    return


@app.cell
def __(plotting):
    plotting.plot_multiple_results()
    plotting.plot_multiple_results(experiment="RotatedMNIST_archive")
    plotting.plot_final_avg_results()
    return


@app.cell
def __(plotting):
    # res_p = plotting.results_to_df().to_latex()
    # res_r = plotting.results_to_df(experiment='RotatedMNIST').to_latex()
    res_buff = plotting.results_to_df(experiment="RotatedMNIST_buffer_size").to_latex()
    print(res_buff)
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
