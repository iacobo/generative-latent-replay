import torch
from torch import optim

import pandas as pd


def get_device():
    """
    Returns:
        torch.device: Torch device. First GPU if available, else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def results_to_df(strategy_names, results):
    """
    Args:
        results (dict): Dictionary of results from the experiment.

    Returns:
        pd.DataFrame: Results as a DataFrame.
    """

    final_avg_accs = [
        res[-1]["Top1_Acc_Stream/eval_phase/train_stream/Task000"] for res in results
    ]
    df = pd.DataFrame({"Strategy": strategy_names, "Final Avg Acc": final_avg_accs})

    return df


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
