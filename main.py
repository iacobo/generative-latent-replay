import argparse
import numpy as np
from pathlib import Path

# ML imports
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import PermutedMNIST, RotatedMNIST

# Local imports
from src import utils, models

# Continual Learning strategies
from avalanche.training import Naive, Replay, EWC, plugins
from src.strategies import LatentReplay, GenerativeLatentReplay


# Helper functions
def get_model(model_name):
    if model_name == "alexnet":
        model = models.alexnet()
    elif model_name == "mobilenet":
        model = models.mobilenetv2()
    elif model_name == "mlp":
        model = models.SimpleMLP()
    elif model_name == "cnn":
        model = models.SimpleCNN()

    return model


def get_experiences(experiment, n_experiences, transform):
    if experiment == "PermutedMNIST":
        experiences = PermutedMNIST(
            n_experiences=n_experiences,
            train_transform=transform,
            eval_transform=transform,
            seed=args.SEED,
        )

    elif experiment == "RotatedMNIST":
        rotations = list(np.linspace(0, 360, n_experiences + 1, dtype=int))[:-1]
        experiences = RotatedMNIST(
            n_experiences=n_experiences,
            train_transform=transform,
            eval_transform=transform,
            seed=args.SEED,
            rotations_list=rotations,
        )

    else:
        raise ValueError("Experiment not implemented")

    return experiences


def get_strategy(
    strategy_name,
    model,
    experiment,
    sgd_kwargs,
    strategy_kwargs,
    replay_buffer_size,
    latent_layer_number,
):
    if strategy_name == "Latent Replay":
        strategy = LatentReplay(
            model=model,
            rm_sz=replay_buffer_size,
            latent_layer_num=latent_layer_number,
            evaluator=utils.get_eval_plugin(strategy_name, experiment),
            **strategy_kwargs,
            **sgd_kwargs,
        )

    elif strategy_name == "GLR":
        # Loading GLR model
        strategy = GenerativeLatentReplay(
            model=model,
            rm_sz=replay_buffer_size,
            latent_layer_num=latent_layer_number,
            evaluator=utils.get_eval_plugin(strategy_name, experiment),
            **strategy_kwargs,
            **sgd_kwargs,
        )

    elif strategy_name == "Naive":
        # Loading baseline (naive) model
        strategy = Naive(
            model=model,
            optimizer=SGD(model.parameters(), **sgd_kwargs),
            evaluator=utils.get_eval_plugin(strategy_name, experiment),
            **strategy_kwargs,
        )

    elif strategy_name == "Replay":
        # Loading benchmark (replay) model
        strategy = Replay(
            model=model,
            criterion=CrossEntropyLoss(),
            optimizer=SGD(model.parameters(), **sgd_kwargs),
            evaluator=utils.get_eval_plugin(strategy_name, experiment),
            **strategy_kwargs,
        )

    elif strategy_name == "EWC":
        # Loading benchmark (replay) model
        strategy = EWC(
            model=model,
            criterion=CrossEntropyLoss(),
            optimizer=SGD(model.parameters(), **sgd_kwargs),
            evaluator=utils.get_eval_plugin(strategy_name, experiment),
            ewc_lambda=1,
            **strategy_kwargs,
        )

    else:
        raise ValueError("Strategy not implemented")

    return strategy


def main(args):
    # Reproducibility
    utils.set_seed(args.SEED)

    # Reporting
    eval_rate = 1

    # Problem definition
    # Number of tasks
    n_experiences = 5
    # Transform data to format expected by model
    transform = utils.get_transforms(resize=244, n_channels=3, normalise=True)

    # Load dataset
    experiences = get_experiences(args.experiment, n_experiences, transform)
    # Train and test streams
    train_stream = experiences.train_stream
    test_stream = experiences.test_stream

    # Hyperparameters

    # Replays
    replay_buffer_size = args.buffer_size

    # Frozen backbone
    if args.latent_layer is None:
        if args.model == "alexnet":
            latent_layer_number = 16
        elif args.model == "mobilenet":
            latent_layer_number = 158
    else:
        latent_layer_number = args.latent_layer

    # SGD hyperparams
    sgd_kwargs = {
        "lr": 0.001,  # 0.1,  # 0.001
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

    # Building base model
    model = get_model(args.model)

    # Loading Continual Learning strategies for experiments
    # Training loop

    strategy = get_strategy(
        args.strategy,
        model,
        args.experiment,
        sgd_kwargs,
        strategy_kwargs,
        replay_buffer_size,
        latent_layer_number,
    )

    # rotations_list=[0, 60, 300],
    for train_exp in train_stream:
        strategy.train(train_exp, eval_streams=[train_exp])
        strategy.eval(train_stream)
        strategy.eval(test_stream)
        utils.save_model(
            strategy.model,
            Path(f"results/{args.experiment}/{args.strategy}"),
            f"model_{train_exp.current_experience}.pt",
        )

    # plotting.plot_multiple_results()
    # plotting.results_to_df()


if __name__ == "__main__":
    strats = ["Latent Replay", "GLR", "Naive", "Replay", "EWC"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=str,
        default="GLR",
        help="Strategy to use",
        choices=strats + ["all"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="alexnet",
        choices=["alexnet", "mobilenet", "efficientnet", "lenet", "mlp", "cnn"],
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="PermutedMNIST",
        choices=["PermutedMNIST", "RotatedMNIST"],
    )
    parser.add_argument("--SEED", type=int, default=43769)
    parser.add_argument("--latent_layer", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction)
    parser.add_argument("--buffer_size", type=int, default=6000)
    args = parser.parse_args()

    if args.strategy == "all":
        for strategy in strats:
            args.strategy = strategy
            main(args)
    else:
        main(args)
