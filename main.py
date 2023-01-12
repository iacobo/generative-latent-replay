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
    if args.experiment == "PermutedMNIST":
        experiences = PermutedMNIST(
            n_experiences=n_experiences,
            train_transform=transform,
            eval_transform=transform,
            seed=args.SEED,
        )
    elif args.experiment == "RotatedMNIST":

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

    # Train and test streams
    train_stream = experiences.train_stream
    test_stream = experiences.test_stream

    # Hyperparameters

    # Replays
    replay_buffer_size = 6000

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
        "eval_every": eval_rate,
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
    if args.model == "alexnet":
        model = models.alexnet()
    elif args.model == "mobilenet":
        model = models.mobilenetv2()
    elif args.model == "mlp":
        model = models.SimpleMLP()
    elif args.model == "cnn":
        model = models.SimpleCNN()

    # Loading Continual Learning strategies for experiments
    # Training loop

    if args.strategy == "Latent Replay":
        strategy = LatentReplay(
            model=model,
            rm_sz=replay_buffer_size,
            latent_layer_num=latent_layer_number,
            evaluator=utils.get_eval_plugin(args.strategy),
            **strategy_kwargs,
            **sgd_kwargs,
        )

    elif args.strategy == "GLR":
        # Loading GLR model
        strategy = GenerativeLatentReplay(
            model=model,
            rm_sz=replay_buffer_size,
            latent_layer_num=latent_layer_number,
            evaluator=utils.get_eval_plugin(args.strategy),
            **strategy_kwargs,
            **sgd_kwargs,
        )

    elif args.strategy == "Naive":
        # Loading baseline (naive) model
        strategy = Naive(
            model=model,
            optimizer=SGD(model.parameters(), **sgd_kwargs),
            evaluator=utils.get_eval_plugin(args.strategy),
            **strategy_kwargs,
        )

    elif args.strategy == "Replay":
        # Loading benchmark (replay) model
        strategy = Replay(
            model=model,
            criterion=CrossEntropyLoss(),
            optimizer=SGD(model.parameters(), **sgd_kwargs),
            evaluator=utils.get_eval_plugin(args.strategy),
            **strategy_kwargs,
        )

    elif args.strategy == "EWC":
        # Loading benchmark (replay) model
        strategy = EWC(
            model=model,
            criterion=CrossEntropyLoss(),
            optimizer=SGD(model.parameters(), **sgd_kwargs),
            evaluator=utils.get_eval_plugin(args.strategy),
            ewc_lambda=1,
            **strategy_kwargs,
        )

    else:
        raise ValueError("Strategy not implemented")

        # rotations_list=[0, 60, 300],
    for train_exp in train_stream:
        strategy.train(train_exp, eval_streams=[train_exp])
        strategy.eval(train_stream)
        strategy.eval(test_stream)
        utils.save_model(
            strategy.model,
            Path(f"results/{args.strategy}"),
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
        choices=["alexnet", "mobilenet", "efficientnet", "mlp", "cnn"],
    )
    parser.add_argument("--experiment", type=str, default="PermutedMNIST")
    parser.add_argument("--SEED", type=int, default=43769)
    parser.add_argument("--latent_layer", type=int, default=None)
    args = parser.parse_args()

    if args.strategy == "all":
        for strategy in strats:
            args.strategy = strategy
            main(args)
    else:
        main(args)
