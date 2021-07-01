import numpy as np
from collections import defaultdict, namedtuple
from time import time

from training.utils.pytorch import random_crop


def make_buffer(shapes, buffer_size):
    buffer = {}
    for k, v in shapes.items():
        if isinstance(v, dict):
            buffer[k] = make_buffer(v, buffer_size)
        else:
            if len(v) >= 3:
                buffer[k] = np.empty((buffer_size, *v), dtype=np.uint8)
            else:
                buffer[k] = np.empty((buffer_size, *v), dtype=np.float32)
    return buffer


def add_rollout(buffer, rollout, idx: int):
    if isinstance(rollout, list):
        rollout = rollout[0]

    if isinstance(rollout, dict):
        for k in rollout.keys():
            add_rollout(buffer[k], rollout[k], idx)
    else:
        np.copyto(buffer[idx], rollout)


def get_batch(buffer: dict, idxs):
    batch = {}
    for k in buffer.keys():
        if isinstance(buffer[k], dict):
            batch[k] = get_batch(buffer[k], idxs)
        else:
            batch[k] = buffer[k][idxs]
    return batch


def augment_ob(batch, image_crop_size):
    for k, v in batch.items():
        if isinstance(batch[k], dict):
            augment_ob(batch[k], image_crop_size)
        elif len(batch[k].shape) > 3:
            batch[k] = random_crop(batch[k], image_crop_size)


class ReplayBufferPerStep(object):
    def __init__(
        self, shapes: dict, buffer_size: int, image_crop_size=84, absorbing_state=False
    ):
        self._capacity = buffer_size

        if absorbing_state:
            shapes["ob"]["absorbing_state"] = [1]
            shapes["ob_next"]["absorbing_state"] = [1]

        self._shapes = shapes
        self._keys = list(shapes.keys())
        self._image_crop_size = image_crop_size
        self._absorbing_state = absorbing_state

        self._buffer = make_buffer(shapes, buffer_size)
        self._idx = 0
        self._full = False

    def clear(self):
        self._idx = 0
        self._full = False

    # store the episode
    def store_episode(self, rollout):
        for k in self._keys:
            add_rollout(self._buffer[k], rollout[k], self._idx)

        self._idx = (self._idx + 1) % self._capacity
        self._full = self._full or self._idx == 0

    # sample the data from the replay buffer
    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self._capacity if self._full else self._idx, size=batch_size
        )
        batch = get_batch(self._buffer, idxs)

        # apply random crop to image
        augment_ob(batch, self._image_crop_size)

        return batch

    def state_dict(self):
        return {"buffer": self._buffer, "idx": self._idx, "full": self._full}

    def load_state_dict(self, state_dict):
        self._buffer = state_dict["buffer"]
        self._idx = state_dict["idx"]
        self._full = state_dict["full"]


class ReplayBuffer(object):
    def __init__(self, keys, buffer_size, sample_func):
        self._capacity = buffer_size
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self.clear()

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffer = defaultdict(list)

    # store transitions
    def store_episode(self, rollout):
        # @rollout can be any length of transitions
        for k in self._keys:
            if self._current_size < self._capacity:
                self._buffer[k].append(rollout[k])
            else:
                self._buffer[k][self._idx] = rollout[k]

        self._idx = (self._idx + 1) % self._capacity
        if self._current_size < self._capacity:
            self._current_size += 1

    # sample the data from the replay buffer
    def sample(self, batch_size):
        # sample transitions
        transitions = self._sample_func(self._buffer, batch_size)
        return transitions

    def state_dict(self):
        return self._buffer

    def load_state_dict(self, state_dict):
        self._buffer = state_dict
        self._current_size = len(self._buffer["ac"])


class ReplayBufferEpisode(object):
    def __init__(self, keys, buffer_size, sample_func):
        self._capacity = buffer_size
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self.clear()

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._new_episode = True
        self._buffer = defaultdict(list)

    # store the episode
    def store_episode(self, rollout):
        if self._new_episode:
            self._new_episode = False
            for k in self._keys:
                if self._current_size < self._capacity:
                    self._buffer[k].append(rollout[k])
                else:
                    self._buffer[k][self._idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffer[k][self._idx].extend(rollout[k])

        if rollout["done"][-1]:
            self._idx = (self._idx + 1) % self._capacity
            if self._current_size < self._capacity:
                self._current_size += 1
            self._new_episode = True

    # sample the data from the replay buffer
    def sample(self, batch_size):
        # sample transitions
        transitions = self._sample_func(self._buffer, batch_size)
        return transitions

    def state_dict(self):
        return self._buffer

    def load_state_dict(self, state_dict):
        self._buffer = state_dict
        self._current_size = len(self._buffer["ac"])


class RandomSampler(object):
    def __init__(self, image_crop_size=84):
        self._image_crop_size = image_crop_size

    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch["ac"])
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            episode_batch["ob_next"][episode_idx][t]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        for k, v in new_transitions["ob"].items():
            if len(v.shape) in [4, 5]:
                new_transitions["ob"][k] = random_crop(v, self._image_crop_size)

        for k, v in new_transitions["ob_next"].items():
            if len(v.shape) in [4, 5]:
                new_transitions["ob_next"][k] = random_crop(v, self._image_crop_size)

        return new_transitions


class SeqSamplerBasic(object):
    def __init__(self, seq_length, image_crop_size=84):
        self._seq_length = seq_length
        self._image_crop_size = image_crop_size

    def sample_func(self, episode_batch, batch_size_in_transitions):
        batch_size = batch_size_in_transitions

        done_locs = np.where(np.array(episode_batch["done"]).reshape(-1))[0]
        episode_start_loc = np.insert(done_locs+1, 0, 0)
        episode_len = np.diff(episode_start_loc)
        episode_len = np.append(episode_len, len(episode_batch["done"]) - episode_start_loc[-1])
        valid_episode_idxs = np.where(episode_len > self._seq_length)[0]

        episode_idxs = np.random.choice(valid_episode_idxs, batch_size)
        t_samples = [
            np.random.randint(episode_len[episode_idx] - self._seq_length)
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][
                    episode_start_loc[episode_idx] + t : episode_start_loc[episode_idx] + t + self._seq_length
                ]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]
    
        ## each transitions element is batch_size x seq_length x OrderedDict
        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0][0][0], dict):
                sub_keys = v[0][0][0].keys()

                try:
                    sequences = {
                        sub_key: [np.stack([v_[0][sub_key] for v_ in seq]) for seq in v]
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
                    sequences = [np.stack([v_[0] for v_ in seq]) for seq in v]
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


class SeqSampler(object):
    def __init__(self, seq_length, image_crop_size=84):
        self._seq_length = seq_length
        self._image_crop_size = image_crop_size

    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch["ac"])
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            episode_batch["ob_next"][episode_idx][t]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]

        # Create a key that stores the specified future fixed length of sequences, pad last states if necessary

        print(episode_idxs)
        print(t_samples)

        # List of dictionaries is created here..., flatten it out?
        transitions["following_sequences"] = [
            episode_batch["ob"][episode_idx][t : t + self._seq_length]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]

        # something's wrong here... should use index episode_idx to episode_batch, not transitions

        # # Pad last states
        # for episode_idx in episode_idxs:
        #     # curr_ep = episode_batch["ob"][episode_idx]
        #     # curr_ep.extend(curr_ep[-1:] * (self._seq_length - len(curr_ep)))
        #
        #     #all list should have 10 dictionaries now
        #     if isinstance(transitions["following_sequences"][episode_idx], dict):
        #         continue
        #     transitions["following_sequences"][episode_idx].extend(transitions["following_sequences"][episode_idx][-1:] * (self._seq_length - len(transitions["following_sequences"][episode_idx])))
        #
        #     #turn transitions["following_sequences"] to a dictionary
        #     fs_list = transitions["following_sequences"][episode_idx]
        #     container = {}
        #     container["ob"] = []
        #     for i in fs_list:
        #         container["ob"].extend(i["ob"])
        #     container["ob"] = np.array(container["ob"])
        #     transitions["following_sequences"][episode_idx] = container

        # Pad last states
        for i in range(len(transitions["following_sequences"])):
            # curr_ep = episode_batch["ob"][episode_idx]
            # curr_ep.extend(curr_ep[-1:] * (self._seq_length - len(curr_ep)))

            # all list should have 10 dictionaries now
            if isinstance(transitions["following_sequences"][i], dict):
                continue
            transitions["following_sequences"][i].extend(
                transitions["following_sequences"][i][-1:]
                * (self._seq_length - len(transitions["following_sequences"][i]))
            )

            # turn transitions["following_sequences"] to a dictionary
            fs_list = transitions["following_sequences"][i]
            container = {}
            container["ob"] = []
            for j in fs_list:
                container["ob"].extend(j["ob"])
            container["ob"] = np.array(container["ob"])
            transitions["following_sequences"][i] = container

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        for k, v in new_transitions["ob"].items():
            if len(v.shape) in [4, 5]:
                new_transitions["ob"][k] = random_crop(v, self._image_crop_size)

        for k, v in new_transitions["ob_next"].items():
            if len(v.shape) in [4, 5]:
                new_transitions["ob_next"][k] = random_crop(v, self._image_crop_size)

        return new_transitions
