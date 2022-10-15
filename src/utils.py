import random
import torch
import numpy as np
from torch import optim
from pathlib import Path

# from avalanche.benchmarks.utils import AvalancheSubset

from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    # forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    # timing_metrics,
    # cpu_usage_metrics,
    # confusion_matrix_metrics,
    # disk_usage_metrics,
)
from avalanche.logging import (
    # InteractiveLogger,
    TextLogger,
    # TensorboardLogger,
    # WandBLogger,
    CSVLogger,
)


def get_device():
    """
    Returns:
        torch.device: Torch device. First GPU if available, else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """
    Set seeds for reproducibility.
    """
    if seed is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_model(x, model, n_epochs=4, lr=0.001, momentum=0.9):

    gmm = model  # models.GMM()
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


def get_eval_plugin(strategy_name, csv=True, text=True):

    loggers = []
    base_path = Path("results")
    base_path.mkdir(exist_ok=True)

    if text:
        loggers.append(TextLogger(open(base_path / "log.txt", "a+")))
    if csv:
        loggers.append(CSVLogger(base_path / strategy_name))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        # timing_metrics(epoch=True, epoch_running=True),
        # cpu_usage_metrics(experience=True),
        # forgetting_metrics(experience=True, stream=True),
        # confusion_matrix_metrics(num_classes=experiences.n_classes, save_image=True,
        #                         stream=True),
        # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers,
    )

    return eval_plugin


def close_loggers(strategy):
    for logger in strategy.evaluator.loggers:
        try:
            logger.close()
        except AttributeError:
            pass
