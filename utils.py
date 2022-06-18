import torch
from torch import nn
from torch import optim
import torch.distributions as D


class GMM(nn.Module):
    def __init__(self, n_components, dim, weights):
        """
        Initialises a GMM.

        Args:
            n_components (int): Number of components in GMM.
            dim (int):          Dimensionality of data to model.
        """
        super().__init__()
        self.n_components = n_components
        self.dim = dim
        self.weights = nn.Parameter(weights)
        # assert weights.shape[0] == n_components
        # Initialise mixture weights to uniform
        # Initialise normal mean/std's to random
        # JA: implement initialising with previous
        #     GMM's posteriors.

    def forward(self, n_components, dim):
        """
        Forward pass.
        """
        mix = D.Categorical(self.weights)
        comp = D.Independent(
            D.Normal(torch.randn(n_components, dim), torch.rand(n_components, dim)), 1
        )
        gmm = D.MixtureSameFamily(mix, comp)

        return gmm


def train_gmm(n_epochs, x):

    parameters = [weights, means, stdevs]
    optimizer1 = optim.SGD(parameters, lr=0.001, momentum=0.9)
    gmm = GMM()

    for i in range(n_epochs):
        optimizer1.zero_grad()
        x = torch.randn(5000, 2)  # this can be an arbitrary x samples
        loss2 = -gmm.log_prob(x).mean()  # -densityflow.log_prob(inputs=x).mean()
        loss2.backward()
        optimizer1.step()

        print(i, loss2)

    return None
