import numpy as np
import torch
from collections import OrderedDict

from training.utils.normalizer import Normalizer
from training.utils.pytorch import to_tensor, center_crop


class BaseAgent(object):
    """ Base class for agents. """

    def __init__(self, config, ob_space):
        self._config = config

        self._ob_norm = Normalizer(
            ob_space, default_clip_range=config.clip_range, clip_obs=config.clip_obs
        )
        self._buffer = None

    def normalize(self, ob):
        """ Normalizes observations. """
        if self._config.ob_norm:
            return self._ob_norm.normalize(ob)
        return ob

    def act(self, ob, is_train=True):
        """ Returns action and the actor's activation given an observation @ob. """
        if hasattr(self, "_rl_agent"):
            return self._rl_agent.act(ob, is_train)

        ob = self.normalize(ob)

        ob = ob.copy()
        for k, v in ob.items():
            if self._config.encoder_type == "cnn" and len(v.shape) == 3:
                ob[k] = center_crop(v, self._config.encoder_image_size)
            else:
                ob[k] = np.expand_dims(ob[k], axis=0)

        with torch.no_grad():
            ob = to_tensor(ob, self._config.device)
            ac, activation, _, _ = self._actor.act(ob, deterministic=not is_train)

        for k in ac.keys():
            ac[k] = ac[k].cpu().numpy().squeeze(0)
            activation[k] = activation[k].cpu().numpy().squeeze(0)

        return ac, activation

    def update_normalizer(self, obs=None):
        """ Updates normalizers. """
        if self._config.ob_norm:
            if obs is None:
                for i in range(len(self._dataset)):
                    self._ob_norm.update(self._dataset[i]["ob"])
                self._ob_norm.recompute_stats()
            else:
                self._ob_norm.update(obs)
                self._ob_norm.recompute_stats()

    def store_episode(self, rollouts):
        """ Stores @rollouts to replay buffer. """
        raise NotImplementedError()

    def is_off_policy(self):
        return self._buffer is not None

    def set_buffer(self, buffer):
        self._buffer = buffer

    def replay_buffer(self):
        return self._buffer.state_dict()

    def load_replay_buffer(self, state_dict):
        self._buffer.load_state_dict(state_dict)

    def set_reward_function(self, predict_reward):
        self._predict_reward = predict_reward

    def sync_networks(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def _soft_update_target_network(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * source_param.data + tau * target_param.data
            )

    def _copy_target_network(self, target, source):
        self._soft_update_target_network(target, source, 0)


class BaseCrossEnvAgent(object):
    """Base class for cross env agents"""

    def __init__(self, config, source_ob_space, target_ob_space):
        self._config = config

        source_ob_norm = Normalizer(
            source_ob_space,
            default_clip_range=config.clip_range,
            clip_obs=config.clip_obs,
        )
        target_ob_norm = Normalizer(
            target_ob_space,
            default_clip_range=config.clip_range,
            clip_obs=config.clip_obs,
        )
        self._ob_norms = {"source": source_ob_norm, "target": target_ob_norm}
        self._buffers = None

    def normalize(self, ob, env_type):
        """ Normalizes observations in @env_type env"""
        if self._config.ob_norm:
            return self._ob_norms[env_type].normalize(ob)
        return ob

    def act(self, ob, env_type, is_train=True):
        """
        Returns action and the actor's activation given an observation @ob,
        and the type of env @env_type
        """
        raise NotImplementedError()

    def update_normalizer(self, obs, env_type):
        """ Updates normalizers. """
        if self._config.ob_norm:
            self._ob_norms[env_type].update(obs)
            self._ob_norms[env_type].recompute_stats()

    def store_episode(self, rollouts, env_type):
        """ Stores @rollouts to replay buffer. """
        raise NotImplementedError()

    def is_off_policy(self):
        return self._buffers is not None

    def set_buffer(self, buffer):
        self._buffer = buffer

    def replay_buffer(self):
        out = {}
        for k, v in self._buffers.items():
            out[k] = v.state_dict()
        return out

    def load_replay_buffer(self, state_dicts):
        for k, v in state_dicts.items():
            self._buffers[k].load_state_dict(v)
            self._buffers[k].load_state_dict(v)

    def set_reward_function(self, predict_reward):
        self._predict_reward = predict_reward

    def sync_networks(self):
        raise NotImplementedError()

    def train(self, env_type):
        raise NotImplementedError()

    def _soft_update_target_network(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * source_param.data + tau * target_param.data
            )

    def _copy_target_network(self, target, source):
        self._soft_update_target_network(target, source, 0)


class BaseMultiStageAgent(object):
    """Base class for Multi Stage agents"""

    def __init__(self, config, source_ob_space, target_ob_space):
        self._config = config

        source_ob_norm = Normalizer(
            source_ob_space,
            default_clip_range=config.clip_range,
            clip_obs=config.clip_obs,
        )
        target_ob_norm = Normalizer(
            target_ob_space,
            default_clip_range=config.clip_range,
            clip_obs=config.clip_obs,
        )
        self._ob_norms = {"source": source_ob_norm, "target": target_ob_norm}
        self._buffers = None

    def normalize(self, ob, env_type):
        """ Normalizes observations in @env_type env"""
        if self._config.ob_norm:
            return self._ob_norms[env_type].normalize(ob)
        return ob

    def act(self, ob, env_type, is_train=True, **kwargs):
        """
        Returns action and the actor's activation given an observation @ob,
        and the type of env @env_type
        """
        raise NotImplementedError()

    def update_normalizer(self, obs, env_type):
        """ Updates normalizers. """
        if self._config.ob_norm:
            self._ob_norms[env_type].update(obs)
            self._ob_norms[env_type].recompute_stats()

    def store_episode(self, rollouts, env_type):
        """ Stores @rollouts to replay buffer. """
        raise NotImplementedError()

    def is_off_policy(self):
        return self._buffers is not None

    def replay_buffer(self):
        out = {}
        for k, v in self._buffers.items():
            out[k] = v.state_dict()
        return out

    def load_replay_buffer(self, state_dicts):
        for k, v in state_dicts.items():
            self._buffers[k].load_state_dict(v)
            self._buffers[k].load_state_dict(v)

    def sync_networks(self):
        raise NotImplementedError()

    def train(self, env_type, curr_info):
        raise NotImplementedError()

    def decide_runner_type(self, curr_info):
        raise NotImplementedError()

    def decide_env_type(self, curr_info):
        raise NotImplementedError()

    def _soft_update_target_network(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - tau) * source_param.data + tau * target_param.data
            )

    def _copy_target_network(self, target, source):
        self._soft_update_target_network(target, source, 0)


class CrossEnvAgentWrapper:
    """a wrapper so that CrossEnvAgent can run in rollout runner"""

    def __init__(self, agent, env_type, **kwargs):
        self._agent = agent
        self._config = agent._config
        self._env_type = env_type
        self._action_args = kwargs

    def act(self, ob, is_train=True):
        return self._agent.act(
            ob, self._env_type, is_train=is_train, **self._action_args
        )
