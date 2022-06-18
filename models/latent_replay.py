import torch
from torch.utils.data import DataLoader

from avalanche.training import AR1


class LatentReplay(AR1):
    """AR1 with Latent Replay.

    This implementations allows for the use of both Synaptic Intelligence and
    Latent Replay to protect the lower level of the model from forgetting.

    While the original papers show how to use those two techniques in a mutual
    exclusive way, this implementation allows for the use of both of them
    concurrently. This behaviour is controlled by passing proper constructor
    arguments).
    """

    def make_train_dataloader(
        self, num_workers=0, shuffle=True, pin_memory=True, **kwargs
    ):
        """
        Called after the dataset instantiation. Initialize the data loader.

        For AR1 a "custom" dataloader is used: instead of using
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
            current_batch_mb_size = train_patterns // (
                (train_patterns + self.rm_sz) // self.train_mb_size
            )

        current_batch_mb_size = max(1, current_batch_mb_size)
        self.replay_mb_size = max(0, self.train_mb_size - current_batch_mb_size)

        # AR1 only supports SIT scenarios (no task labels).
        self.dataloader = DataLoader(
            self.adapted_dataset,
            num_workers=num_workers,
            batch_size=current_batch_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )

    # JA: See here for implementing custom loss function:
    # https://github.com/ContinualAI/avalanche/pull/604

    # JA: For sampled latent replays, define sampling
    # here (i.e. self.rm[0] = sampled x, self.rm[1] = sampled y)
    def training_epoch(self, **kwargs):
        for mb_it, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            if self.clock.train_exp_counter > 0:
                lat_mb_x = self.rm[0][
                    mb_it * self.replay_mb_size : (mb_it + 1) * self.replay_mb_size
                ]
                lat_mb_x = lat_mb_x.to(self.device)
                lat_mb_y = self.rm[1][
                    mb_it * self.replay_mb_size : (mb_it + 1) * self.replay_mb_size
                ]
                lat_mb_y = lat_mb_y.to(self.device)
                self.mbatch[1] = torch.cat((self.mb_y, lat_mb_y), 0)
            else:
                lat_mb_x = None

            # Forward pass. Here we are injecting latent patterns lat_mb_x.
            # lat_mb_x will be None for the very first batch (batch 0), which
            # means that lat_acts.shape[0] == self.mb_x[0].
            self._before_forward(**kwargs)
            self.mb_output, lat_acts = self.model(
                self.mb_x, latent_input=lat_mb_x, return_lat_acts=True
            )

            if self.clock.train_exp_epochs == 0:
                # On the first epoch only: store latent activations. Those
                # activations will be used to update the replay buffer.
                lat_acts = lat_acts.detach().clone().cpu()
                if mb_it == 0:
                    self.cur_acts = lat_acts
                else:
                    self.cur_acts = torch.cat((self.cur_acts, lat_acts), 0)
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
        """
        After training the model, fit new GMM on current batch of data
        and store GMM and associated frozen label generator (i.e. network head).
        
        Adds GMM and associated label generator to growing container of
        domain-specific samplers.
        """
        h = min(
            self.rm_sz // (self.clock.train_exp_counter + 1), self.cur_acts.size(0),
        )

        # JA: Initialising replay buffer
        # Use PyTorch GMM
        # https://pytorch.org/docs/stable/distributions.html#mixturesamefamily
        curr_data = self.experience.dataset
        idxs_cur = torch.randperm(self.cur_acts.size(0))[:h]
        rm_add_y = torch.tensor([curr_data.targets[idx_cur] for idx_cur in idxs_cur])

        rm_add = [self.cur_acts[idxs_cur], rm_add_y]

        # replace patterns in random memory
        if self.clock.train_exp_counter == 0:
            self.rm = rm_add
        else:
            idxs_2_replace = torch.randperm(self.rm[0].size(0))[:h]
            for j, idx in enumerate(idxs_2_replace):
                idx = int(idx)
                self.rm[0][idx] = rm_add[0][j]
                self.rm[1][idx] = rm_add[1][j]

        self.cur_acts = None

        # Runs S.I. and CWR* plugin callbacks
        super()._after_training_exp(**kwargs)
