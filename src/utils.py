import torch
import numpy as np
import random

from torch import optim
from pathlib import Path
from torchvision import transforms as T

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


def get_transforms(resize=False, to_tensor=True, n_channels=False, flatten=False):
    transforms = []
    # Resize image
    if resize:
        if isinstance(resize, int):
            resize = (resize, resize)
        transforms.append(T.Resize(resize))

    # Convert image to tensor
    if to_tensor:
        transforms.append(T.ToTensor())

    # Change 2d to greyscale 3d
    if n_channels == 1:
        transforms.append(T.Lambda(lambda x: x.unsqueeze(0)))

    # Change greyscale to n channel (3 = rgb)
    if n_channels > 1:
        transforms.append(T.Lambda(lambda x: x.repeat(n_channels, 1, 1)))

    # Flatten tensor
    if flatten:
        transforms.append(T.Lambda(torch.flatten))

    return T.Compose(transforms)
