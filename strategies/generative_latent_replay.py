from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.plugins.evaluation import default_evaluator

import utils
from models import FrozenNet
from strategies import LatentReplay


class GenerativeLatentReplay(LatentReplay):
    """Generative Latent Replay.

    This implementations allows for the use of Latent Replay to protect the
    lower level of the model from forgetting.
    """

    def __init__(
        self,
        model=None,
        criterion=None,
        lr: float = 0.001,
        momentum=0.9,
        l2=0.0005,
        train_epochs: int = 4,
        rm_sz: int = 1500,
        freeze_below_layer: str = "end_features.0",
        latent_layer_num: int = 19,
        generator="gmm",
        train_mb_size: int = 128,
        eval_mb_size: int = 128,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
    ):
        """
        Creates an instance of the LatentReplay strategy.

        :param criterion: The loss criterion to use. Defaults to None, in which
            case the cross entropy loss is used.
        :param lr: The learning rate (SGD optimizer).
        :param momentum: The momentum (SGD optimizer).
        :param l2: The L2 penalty used for weight decay.
        :param train_epochs: The number of training epochs. Defaults to 4.
        :param rm_sz: The size of the replay buffer. The replay buffer is shared
            across classes. Defaults to 1500.
        :param freeze_below_layer: A string describing the name of the layer
            to use while freezing the lower (nearest to the input) part of the
            model. The given layer is not frozen (exclusive).
        :param latent_layer_num: The number of the layer to use as the Latent
            Replay Layer. Usually this is the same of `freeze_below_layer`.
        :param train_mb_size: The train minibatch size. Defaults to 128.
        :param eval_mb_size: The eval minibatch size. Defaults to 128.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        """

        if plugins is None:
            plugins = []

        # Model setup
        model = FrozenNet(model=model, latent_layer_num=latent_layer_num)
        print(model)

        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2)

        if criterion is None:
            criterion = CrossEntropyLoss()

        self.lr = lr
        self.l2 = l2
        self.momentum = momentum
        self.generator = generator
        self.rm = None
        self.rm_sz = rm_sz
        self.freeze_below_layer = freeze_below_layer
        self.cur_acts: Optional[Tensor] = None
        self.cur_y: Optional[Tensor] = None
        self.replay_mb_size = 0

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

    def _after_training_exp(self, **kwargs):
        h = min(
            self.rm_sz // (self.clock.train_exp_counter + 1),
            self.cur_acts.size(0),
        )

        # Initialising replay buffer
        # Use PyTorch GMM
        # https://pytorch.org/docs/stable/distributions.html#mixturesamefamily
        if self.generator == "gmm":
            sampler = utils.GMM()
        elif self.generator == "kmeans":
            sampler = utils.KMeans()
        elif self.generator == "density":
            sampler = utils.DBSCAN()
        elif self.generator == "connectivity":
            sampler = utils.HAC()
        elif self.generator == "markov":
            sampler = utils.MarkovChain()
        else:
            raise NotImplementedError(f'Unknown generator "{self.generator}"')

        sampler = sampler.fit(self.cur_acts, self.cur_y)
        self.samplers.append(sampler)
        rm_add = sampler.sample(h)

        # replace patterns in random memory
        if self.clock.train_exp_counter == 0:
            self.rm = rm_add
        else:
            idxs_to_replace = torch.randperm(self.rm[0].size(0))[:h]

            self.rm[0][idxs_to_replace] = rm_add[0]
            self.rm[1][idxs_to_replace] = rm_add[1]

        self.cur_acts = None

        # Runs S.I. and CWR* plugin callbacks
        super()._after_training_exp(**kwargs)
