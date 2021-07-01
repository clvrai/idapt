import gym.spaces
import numpy as np
import os
import pickle

from training.datasets import ExpertDataset
from training.utils.logger import logger
from training.utils.pytorch import random_crop


class OfflineDataset(ExpertDataset):
    """ Dataset class for Pretraining Data. """

    def __init__(
        self,
        subsample_interval=1,
        ac_space=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        use_low_level=False,
        max_size=20000,
    ):

        self.train = train  # training set or test set

        self._data = []
        self._ac_space = ac_space
        self._subsample_interval = subsample_interval
        self._use_low_level = use_low_level
        self._max_size = max_size

    def add_demos(self, demo):
        if "actions" in demo.keys() and len(demo["obs"]) != len(demo["actions"]) + 1:
            logger.error(
                "Mismatch in # of observations (%d) and actions (%d)",
                len(demo["obs"]),
                len(demo["actions"]),
            )
            return

        offset = np.random.randint(0, self._subsample_interval)

        if self._use_low_level:
            length = len(demo["low_level_actions"])
            for i in range(offset, length, self._subsample_interval):
                transition = {
                    "ob": demo["low_level_obs"][i],
                    "ob_next": demo["low_level_obs"][i + 1],
                }
                if isinstance(demo["low_level_actions"][i], dict):
                    transition["ac"] = demo["low_level_actions"][i]
                else:
                    transition["ac"] = gym.spaces.unflatten(
                        self._ac_space, demo["low_level_actions"][i]
                    )

                transition["done"] = 1 if i + 1 == length else 0

                self._data.append(transition)

            return

        length = len(demo["obs"]) - 1
        for i in range(offset, length, self._subsample_interval):
            transition = {
                "ob": demo["obs"][i],
            }
            if "obs_next" in demo:
                transition["ob_next"] = demo["obs_next"][i]
            else:
                transition["ob_next"] = demo["obs"][i + 1]

            if "actions" in demo.keys():
                if isinstance(demo["actions"][i], dict):
                    transition["ac"] = demo["actions"][i]
                else:
                    transition["ac"] = gym.spaces.unflatten(
                        self._ac_space, demo["actions"][i]
                    )
            if "rewards" in demo:
                transition["rew"] = demo["rewards"][i]
            if "dones" in demo:
                transition["done"] = int(demo["dones"][i])
            else:
                transition["done"] = 1 if i + 1 == length else 0

            if "state" in demo:
                transition["state"] = demo["state"][i]

            if "qpos" in demo and "qvel" in demo:
                transition["qpos"] = demo["qpos"][i]
                transition["qvel"] = demo["qvel"][i]

            if len(self._data) + len(transition) > self._max_size:
                # randomly delete overflow items from list
                over_capacity = len(self._data) + len(transition) - self._max_size
                del self._data[:: len(self._data) // over_capacity]
            self._data.append(transition)

        logger.warn(
            "Load demonstrations with %d states from RL", len(self._data),
        )

        return

    def add_demos_from_path(self, path):
        assert (
            path is not None
        ), "--demo_path should be set (e.g. demos/Sawyer_toy_table)"
        demo_files = self._get_demo_files(path)
        num_demos = 0

        # now load the picked numpy arrays
        for file_path in demo_files:
            with open(file_path, "rb") as f:
                demos = pickle.load(f)
                if not isinstance(demos, list):
                    demos = [demos]

                for demo in demos:
                    if (
                        "actions" in demo.keys()
                        and len(demo["obs"]) != len(demo["actions"]) + 1
                    ):
                        logger.error(
                            "Mismatch in # of observations (%d) and actions (%d) (%s)",
                            len(demo["obs"]),
                            len(demo["actions"]),
                            file_path,
                        )
                        continue

                    offset = np.random.randint(0, self._subsample_interval)
                    num_demos += 1

                    if self._use_low_level:
                        length = len(demo["low_level_actions"])
                        for i in range(offset, length, self._subsample_interval):
                            transition = {
                                "ob": demo["low_level_obs"][i],
                                "ob_next": demo["low_level_obs"][i + 1],
                            }
                            if isinstance(demo["low_level_actions"][i], dict):
                                transition["ac"] = demo["low_level_actions"][i]
                            else:
                                transition["ac"] = gym.spaces.unflatten(
                                    self._ac_space, demo["low_level_actions"][i]
                                )

                            transition["done"] = 1 if i + 1 == length else 0

                            self._data.append(transition)

                        continue

                    length = len(demo["obs"]) - 1
                    for i in range(offset, length, self._subsample_interval):
                        transition = {
                            "ob": demo["obs"][i],
                            "ob_next": demo["obs"][i + 1],
                        }
                        if "actions" in demo.keys():
                            if isinstance(demo["actions"][i], dict):
                                transition["ac"] = demo["actions"][i]
                            else:
                                transition["ac"] = gym.spaces.unflatten(
                                    self._ac_space, demo["actions"][i]
                                )
                        if "rewards" in demo:
                            transition["rew"] = demo["rewards"][i]
                        if "dones" in demo:
                            transition["done"] = int(demo["dones"][i])
                        else:
                            transition["done"] = 1 if i + 1 == length else 0

                        if "state" in demo:
                            transition["state"] = demo["state"][i]

                        if "qpos" in demo and "qvel" in demo:
                            transition["qpos"] = demo["qpos"][i]
                            transition["qvel"] = demo["qvel"][i]

                        if len(self._data) > self._max_size:
                            # only load max_size elements
                            break

                        self._data.append(transition)

        logger.warn(
            "Load %d demonstrations with %d states from %d files",
            num_demos,
            len(self._data),
            len(demo_files),
        )

    def clear(self):
        self._data = []
        logger.warn("Cleared Dataset")


class SeqFirstSampler(object):
    """ Samples transtions of size (sequence length, batch size, data dimension) for RNN policies """

    def __init__(self, seq_length, image_crop_size=84):
        self._seq_length = seq_length
        self._image_crop_size = image_crop_size

    def sample_func(self, episode_batch, batch_size_in_transitions):

        ### select only episodes of at least self._seq_length length
        valid_episode_idxs = [
            idx
            for idx, item in enumerate(episode_batch["ac"])
            if len(item) > self._seq_length
        ]

        batch_size = batch_size_in_transitions
        episode_idxs = np.random.choice(valid_episode_idxs, batch_size)

        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]) - self._seq_length)
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t : t + self._seq_length]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            episode_batch["ob_next"][episode_idx][t : t + self._seq_length]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]

        ## each transitions element is batch_size x seq_length x OrderedDict
        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0][0], dict):
                sub_keys = v[0][0].keys()

                try:
                    sequences = {
                        sub_key: [np.stack([v_[sub_key] for v_ in seq]) for seq in v]
                        for sub_key in sub_keys
                    }
                    new_transitions[k] = {
                        sub_key: np.stack(sequences[sub_key]).swapaxes(0, 1)
                        for sub_key in sub_keys
                    }
                except:
                    import ipdb

                    ipdb.set_trace()

            else:
                try:
                    sequences = [np.stack([v_ for v_ in seq]) for seq in v]
                    new_transitions[k] = np.stack(sequences).swapaxes(0, 1)
                except:
                    import ipdb

                    ipdb.set_trace()

        for k, v in new_transitions["ob"].items():
            if len(v.shape) in [4, 5]:
                S, B, C, H, W = v.shape
                new_transitions["ob"][k] = random_crop(
                    v.reshape((S * B, C, H, W)), self._image_crop_size
                ).reshape((S, B, C, self._image_crop_size, self._image_crop_size))

        for k, v in new_transitions["ob_next"].items():
            if len(v.shape) in [4, 5]:
                S, B, C, H, W = v.shape
                new_transitions["ob_next"][k] = random_crop(
                    v.reshape((S * B, C, H, W)), self._image_crop_size
                ).reshape((S, B, C, self._image_crop_size, self._image_crop_size))

        return new_transitions
