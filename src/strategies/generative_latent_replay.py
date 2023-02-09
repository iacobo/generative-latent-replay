from typing import Optional, List

import torch

from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.plugins.evaluation import default_evaluator

from src import models
from src.strategies import LatentReplay


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
        weight_decay=0.0005,
        train_epochs: int = 4,
        rm_sz: int = 1500,
        freeze_below_layer: str = "end_features.0",
        latent_layer_num: int = 19,
        generator="gmm",
        samplers=None,
        train_mb_size: int = 128,
        eval_mb_size: int = 128,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
        pretrained: bool = False,
    ):
        """
        Creates an instance of the LatentReplay strategy.

        :param criterion: The loss criterion to use. Defaults to None, in which
            case the cross entropy loss is used.
        :param lr: The learning rate (SGD optimizer).
        :param momentum: The momentum (SGD optimizer).
        :param weight_decay: The L2 penalty used for weight decay.
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

        self.generator = generator
        if samplers is None:
            self.samplers = []

        super().__init__(
            model=model,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            criterion=criterion,
            latent_layer_num=latent_layer_num,
            rm_sz=rm_sz,
            freeze_below_layer=freeze_below_layer,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            pretrained=pretrained,
        )

    def _after_training_exp(self, **kwargs):
        h = min(
            self.rm_sz // (self.clock.train_exp_counter + 1),
            self.cur_acts.size(0),
        )

        # Copy of data for training the generator
        x_copy = self.cur_acts.detach().clone()
        y_copy = self.cur_y.detach().clone()

        # Initialising replay buffer
        if self.generator == "gmm":
            n_classes = self.cur_y.unique().size(0)
            n_features = self.cur_acts.size(-1)
            means_init = torch.stack(
                [x_copy[y_copy == i].mean(dim=0) for i in range(n_classes)]
            ).unsqueeze(0)
            sampler = models.GMM(
                n_components=n_classes,
                n_features=n_features,
                covariance_type="diag",
                mu_init=means_init,
            )
        elif self.generator == "kmeans":
            sampler = models.KMeans()
        elif self.generator == "density":
            sampler = models.DBSCAN()
        elif self.generator == "connectivity":
            sampler = models.HAC()
        elif self.generator == "markov":
            sampler = models.MarkovChain()
        else:
            raise NotImplementedError(f'Unknown generator "{self.generator}"')

        print(f"Training generator {self.clock.train_exp_counter}...")
        sampler.fit(x_copy)
        print("Generator trained.")
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

        # Runs plugin callbacks
        super(LatentReplay, self)._after_training_exp(**kwargs)
