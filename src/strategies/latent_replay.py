from typing import Optional, List
import warnings

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
)
from avalanche.training.utils import freeze_up_to
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate

from src import utils
from src.models import FrozenNet


class LatentReplay(SupervisedTemplate):
    """Latent Replay.

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
        subsample_replays: bool = False,
        train_mb_size: int = 128,
        eval_mb_size: int = 128,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator(),
        eval_every=-1,
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
            model. The given layer is not frozen (exclusive). Please ensure this
            layer has a grad function. Defaults to "end_features.0".
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

        warnings.warn(
            "LatentReplay and GenerativeLatentReplay will only recognise "
            "modules defined in __init__. "
            "Modules defined in forward will be ignored."
        )

        if plugins is None:
            plugins = []

        # Model setup
        model = FrozenNet(model=model, latent_layer_num=latent_layer_num)

        optimizer = SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        if criterion is None:
            criterion = CrossEntropyLoss()

        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.rm = None
        self.rm_sz = rm_sz
        self.freeze_below_layer = freeze_below_layer
        self.cur_acts: Optional[Tensor] = None
        self.cur_y: Optional[Tensor] = None
        self.replay_mb_size = 0
        self.subsample_replays = subsample_replays

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

    def _before_training_exp(self, **kwargs):
        self.model.eval()
        self.model.end_features.train()

        # Freeze model backbone during subsequent experiences
        if self.clock.train_exp_counter > 0:
            frozen_layers, frozen_parameters = freeze_up_to(
                self.model, self.freeze_below_layer
            )
            print(f"Frozen layers:\n {frozen_layers}")

            # Adapt the model and optimizer
            self.optimizer = SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        # super()... will run S.I. and CWR* plugin callbacks
        super()._before_training_exp(**kwargs)

    def make_train_dataloader(
        self, num_workers=0, shuffle=True, pin_memory=True, **kwargs
    ):
        """
        Called after the dataset instantiation. Initialize the data loader.

        A "custom" dataloader is used: instead of using
        `self.train_mb_size` as the batch size, the data loader batch size will
        be computed ad `self.train_mb_size - latent_mb_size`. `latent_mb_size`
        is in turn computed as:

        `
        len(train_dataset) // ((len(train_dataset) + len(replay_buffer)
        // self.train_mb_size)
        `

        so that the number of iterations required to run an epoch on the current
        batch is equal to the number of iterations required to run an epoch
        on the replay buffer.

        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        """

        current_batch_mb_size = self.train_mb_size

        if self.clock.train_exp_counter > 0:
            train_patterns = len(self.adapted_dataset)

            if self.subsample_replays:
                current_batch_mb_size //= 2
            else:
                current_batch_mb_size = train_patterns // (
                    (train_patterns + self.rm_sz) // self.train_mb_size
                )

        current_batch_mb_size = max(1, current_batch_mb_size)
        self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)

        # Only supports SIT scenarios (no task labels).
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=current_batch_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )

    # JA: See here for implementing custom loss function:
    # https://github.com/ContinualAI/avalanche/pull/604

    def training_epoch(self, **kwargs):
        for mb_it, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            self.optimizer.zero_grad()

            # Grab y labels for the current minibatch
            cur_y = self.mb_y.detach().clone().cpu()

            if self.clock.train_exp_counter > 0:
                start = (self.replay_mb_size * mb_it) % self.rm[0].size(0)
                end = (self.replay_mb_size * (mb_it + 1)) % self.rm[0].size(0)

                lat_mb_x = self.rm[0][start:end].to(self.device)
                lat_mb_y = self.rm[1][start:end].to(self.device)

                # Set current y_labels to current minibatch plus replayed examples
                self.mbatch[1] = torch.cat((self.mb_y, lat_mb_y), 0)
            else:
                lat_mb_x = None

            # Forward pass. Here we are injecting latent patterns lat_mb_x.
            # lat_mb_x will be None for the very first batch (batch 0), which
            # means that lat_acts.shape[0] == self.mb_x[0].
            self._before_forward(**kwargs)

            # JA:
            if mb_it == 0:
                utils.render_model(
                    lat_mb_x, self.model, self.mb_x, self.clock.train_exp_counter
                )

            self.mb_output, lat_acts = self.model(
                self.mb_x, latent_input=lat_mb_x, return_lat_acts=True
            )

            if self.clock.train_exp_epochs == 0:
                # On the first epoch only: store latent activations. Those
                # activations will be used to update the replay buffer.
                lat_acts = lat_acts.detach().clone().cpu()

                if mb_it == 0:
                    self.cur_acts = lat_acts
                    self.cur_y = cur_y
                else:
                    self.cur_acts = torch.cat((self.cur_acts, lat_acts), 0)
                    self.cur_y = torch.cat((self.cur_y, cur_y), 0)

            self._after_forward(**kwargs)

            # Loss & Backward
            # We don't need to handle latent replay, as self.mb_y already
            # contains both current and replay labels.
            self.loss = self._criterion(self.mb_output, self.mb_y)
            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        h = min(
            self.rm_sz // (self.clock.train_exp_counter + 1),
            self.cur_acts.size(0),
        )

        # Initialising replay buffer
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]
        rm_add = [self.cur_acts[idxs_cur], self.cur_y[idxs_cur]]

        # replace patterns in random memory
        if self.clock.train_exp_counter == 0:
            self.rm = rm_add
        else:
            idxs_to_replace = torch.randperm(self.rm[0].size(0))[:h]

            self.rm[0][idxs_to_replace] = rm_add[0]
            self.rm[1][idxs_to_replace] = rm_add[1]

        self.cur_acts = None

        # Runs plugin callbacks
        super()._after_training_exp(**kwargs)
