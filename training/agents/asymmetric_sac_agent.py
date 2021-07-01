import copy
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from training.agents.base_agent import CrossEnvAgentWrapper
from training.agents.sac_agent import SACAgent
from training.datasets import (
    ReplayBuffer,
    ReplayBufferEpisode,
    RandomSampler,
    SeqFirstSampler, 
    ReplayBuffer, 
    SeqSamplerBasic
)
from training.networks.actor_critic import Actor, Critic
from training.networks.recurrent_actor_critic import RActor, RCritic
from training.utils.info_dict import Info
from training.utils.logger import logger
from training.utils.normalizer import Normalizer
from training.utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    compute_gradient_norm,
    compute_weight_norm,
    sync_networks,
    sync_grads,
    to_tensor,
    center_crop,
)


class ASACAgent(SACAgent):
    """
    Asymmetric SAC algorithm.  Modified SAC algorithm for image-based policies where critic input is ground truth state.
    """

    def __init__(self, config, ob_space, ac_space, env_ob_space):
        self._config = config

        self._ob_norm = Normalizer(
            ob_space, default_clip_range=config.clip_range, clip_obs=config.clip_obs
        )

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._env_ob_space = env_ob_space

        if config.target_entropy is not None:
            self._target_entropy = config.target_entropy
        else:
            self._target_entropy = -gym.spaces.flatdim(ac_space)
        self._log_alpha = torch.tensor(
            np.log(config.alpha_init_temperature),
            requires_grad=True,
            device=config.device,
        )

        # build up networks

        if config.encoder_type == "cnn":
            c_config = copy.copy(config)
            c_config.encoder_type = "mlp"
            self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
            self._critic = Critic(c_config, env_ob_space, ac_space)
            self._critic_target = Critic(c_config, env_ob_space, ac_space)
        else:
            self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
            self._critic = Critic(config, env_ob_space, ac_space)
            self._critic_target = Critic(config, env_ob_space, ac_space)

        self._network_cuda(config.device)
        self._copy_target_network(self._critic_target, self._critic)

        # optimizers
        self._alpha_optim = optim.Adam(
            [self._log_alpha], lr=config.alpha_lr, betas=(0.5, 0.999)
        )
        self._actor_optim = optim.Adam(
            self._actor.parameters(), lr=config.actor_lr, betas=(0.9, 0.999)
        )
        self._critic_optim = optim.Adam(
            self._critic.parameters(), lr=config.critic_lr, betas=(0.9, 0.999)
        )

        epochs = self._config.max_global_step
        lambda1 = lambda e: 0.1 if epochs == 0 else max((epochs - e) / epochs, 0.1)

        self._alpha_lr_scheduler = LambdaLR(self._alpha_optim, lr_lambda=lambda1)
        self._actor_lr_scheduler = LambdaLR(self._actor_optim, lr_lambda=lambda1)
        self._critic_lr_scheduler = LambdaLR(self._critic_optim, lr_lambda=lambda1)

        # per-episode replay buffer
        sampler = RandomSampler(image_crop_size=config.encoder_image_size)
        buffer_keys = ["ob", "ob_next", "ac", "done", "rew", "state", "state_next"]
        self._buffer = ReplayBuffer(
            buffer_keys, config.buffer_size, sampler.sample_func
        )

        self._update_iter = 0
        self._train_encoder = True

        self._log_creation()

    def _update_actor_and_alpha(self, o, state):
        info = Info()

        actions_target, _, log_pi, ent = self._actor.act(o, return_log_prob=True)

        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha.detach() * log_pi).mean()

        actor_loss = -torch.min(
            *self._critic(state, actions_target, detach_conv=True)
        ).mean()

        entropy = ent.mean()
        info["entropy_alpha"] = alpha.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_entropy"] = entropy.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        entropy = ent.mean()
        info["actor_entropy"] = entropy.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self._actor)
        self._actor_optim.step()

        # update alpha
        alpha_loss = -(alpha * (log_pi + self._target_entropy).detach()).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        return info

    def _update_critic(self, o, ac, rew, o_next, done, state, state_next):
        info = Info()

        # calculate the target Q value function
        with torch.no_grad():
            alpha = self._log_alpha.exp().detach()

            actions_next, _, log_pi_next, _ = self._actor.act(
                o_next, return_log_prob=True
            )
            q_next_value1, q_next_value2 = self._critic_target(state_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            target_q_value = (
                rew * self._config.reward_scale
                + (1 - done) * self._config.rl_discount_factor * q_next_value
            )

        # the q loss
        real_q_value1, real_q_value2 = self._critic(state, ac)

        critic1_loss = F.mse_loss(target_q_value, real_q_value1)
        critic2_loss = F.mse_loss(target_q_value, real_q_value2)
        critic_loss = critic1_loss + critic2_loss

        # update the critic
        self._critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self._critic)
        self._critic_optim.step()

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()
        info["min_target1_q"] = real_q_value1.min().cpu().item()
        info["min_target2_q"] = real_q_value2.min().cpu().item()
        info["target1_q"] = real_q_value1.mean().cpu().item()
        info["target2_q"] = real_q_value2.mean().cpu().item()
        info["critic1_loss"] = critic1_loss.cpu().item()
        info["critic2_loss"] = critic2_loss.cpu().item()

        return info

    def _update_network(self, transitions):
        info = Info()

        # pre-process observations
        o, o_next = transitions["ob"], transitions["ob_next"]
        o = self.normalize(o)
        o_next = self.normalize(o_next)

        bs = len(transitions["done"])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions["ac"])
        done = _to_tensor(transitions["done"]).reshape(bs, 1).float()
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        state = _to_tensor({"state": transitions["state"]})
        state_next = _to_tensor({"state": transitions["state_next"]})

        self._update_iter += 1

        critic_train_info = self._update_critic(
            o, ac, rew, o_next, done, state, state_next
        )
        info.add(critic_train_info)

        if self._update_iter % self._config.actor_update_freq == 0:
            actor_train_info = self._update_actor_and_alpha(o, state)
            info.add(actor_train_info)

        if self._update_iter % self._config.critic_target_update_freq == 0:
            for i, fc in enumerate(self._critic.fcs):
                self._soft_update_target_network(
                    self._critic_target.fcs[i],
                    fc,
                    self._config.critic_soft_update_weight,
                )

            self._soft_update_target_network(
                self._critic_target.encoder,
                self._critic.encoder,
                self._config.encoder_soft_update_weight,
            )

        return info.get_dict(only_scalar=True)

    ### Misc. ###

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating an Asymmetric SAC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info("The critic has %d parameters", count_parameters(self._critic))

    def set_encoder_requires_grad(self, req):
        self._train_encoder = req
        if self._config.encoder_type == "cnn":
            for param in self._actor.encoder.parameters():
                param.requires_grad = req

    def clear_buffer(self):
        self._buffer.clear()


class RecurrentASACAgent(SACAgent):
    """
    Recurrent Asymmetric SAC algorithm.
    """

    def __init__(
        self,
        config,
        ob_space,
        ac_space,
        env_ob_space,
        target_ob_space,
        target_ac_space,
        target_env_ob_space,
    ):
        self._config = config

        self._ob_norm = Normalizer(
            ob_space, default_clip_range=config.clip_range, clip_obs=config.clip_obs
        )

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._env_ob_space = env_ob_space

        if config.target_entropy is not None:
            self._target_entropy = config.target_entropy
        else:
            self._target_entropy = -gym.spaces.flatdim(ac_space)
        self._log_alpha = torch.tensor(
            np.log(config.alpha_init_temperature),
            requires_grad=True,
            device=config.device,
        )

        # build up networks

        if config.encoder_type == "cnn":
            c_config = copy.copy(config)
            c_config.encoder_type = "mlp"
            self._actor = RActor(config, ob_space, ac_space, config.tanh_policy)
            self._critic = RCritic(c_config, env_ob_space, ac_space=ac_space)
            self._critic_target = RCritic(c_config, env_ob_space, ac_space)

        else:
            self._actor = RActor(config, ob_space, ac_space, config.tanh_policy)
            self._critic = RCritic(
                config, env_ob_space, ac_space=ac_space, rnn=self._actor.rnn
            )
            self._critic_target = RCritic(config, env_ob_space, ac_space)

        self._network_cuda(config.device)
        self._copy_target_network(self._critic_target, self._critic)

        # optimizers
        self._alpha_optim = optim.Adam(
            [self._log_alpha], lr=config.alpha_lr, betas=(0.5, 0.999)
        )
        self._actor_optim = optim.Adam(
            self._actor.parameters(), lr=config.actor_lr, betas=(0.9, 0.999)
        )
        self._critic_optim = optim.Adam(
            self._critic.parameters(), lr=config.critic_lr, betas=(0.9, 0.999)
        )

        epochs = self._config.max_global_step
        lambda1 = lambda e: 0.1 if epochs == 0 else max((epochs - e) / epochs, 0.1)

        self._alpha_lr_scheduler = LambdaLR(self._alpha_optim, lr_lambda=lambda1)
        self._actor_lr_scheduler = LambdaLR(self._actor_optim, lr_lambda=lambda1)
        self._critic_lr_scheduler = LambdaLR(self._critic_optim, lr_lambda=lambda1)

        # per-episode replay buffer
        
        sampler = SeqSamplerBasic(
            seq_length=config.rnn_seq_length, image_crop_size=config.encoder_image_size
        )
        buffer_keys = [
            "ob",
            "ob_next",
            "ac",
            "done",
            "rew",
            "state",
            "state_next",
            "rnn_state_in",
            "rnn_state_out",
            "c_rnn_state_in",
            "ac_prev",
        ]
        self._buffer = ReplayBuffer(
            buffer_keys, config.buffer_size, sampler.sample_func
        )
        '''

        sampler = SeqFirstSampler(
            seq_length=config.rnn_seq_length, image_crop_size=config.encoder_image_size
        )
        buffer_keys = [
            "ob",
            "ob_next",
            "ac",
            "done",
            "rew",
            "state",
            "state_next",
            "rnn_state_in",
            "rnn_state_out",
            "c_rnn_state_in",
            "ac_prev",
        ]
        self._buffer = ReplayBufferEpisode(
            buffer_keys, config.buffer_size, sampler.sample_func
        )
        '''
        self._update_iter = 0
        self._train_encoder = True

        self.stages = ["policy_init"]
        self._grounding_step = 0

        self._log_creation()

    def act(self, ob, ac_prev, rnn_state_in, is_train=True):
        """ Returns action and the actor's activation given an observation, previous action, and input rnn state """

        ob = self.normalize(ob)

        ob = ob.copy()
        for k, v in ob.items():
            if self._config.encoder_type == "cnn" and len(v.shape) == 3:
                ob[k] = center_crop(v, self._config.encoder_image_size)
            else:
                ob[k] = np.expand_dims(ob[k], axis=0)

        if len(ac_prev.shape) == 1:
            ac_prev = ac_prev.copy()
            ac_prev = np.expand_dims(ac_prev, axis=0)
        rnn_state_in = rnn_state_in.copy()
        rnn_state_in = np.expand_dims(rnn_state_in, axis=1)

        with torch.no_grad():
            ob = to_tensor(ob, self._config.device)
            ac_prev = to_tensor(ac_prev, self._config.device)
            rnn_state_in = to_tensor(rnn_state_in, self._config.device)
            ac, activation, _, _, rnn_state = self._actor.act(
                ob, ac_prev, rnn_state_in, deterministic=not is_train
            )

        for k in ac.keys():
            ac[k] = ac[k].cpu().numpy().squeeze(0).squeeze(0)
            activation[k] = activation[k].cpu().numpy().squeeze(0).squeeze(0)

        rnn_state = rnn_state.cpu().numpy().squeeze(1)
        rnn_state_in = rnn_state_in.cpu().numpy().squeeze(1)

        return (
            ac,
            activation,
            {
                # "rnn_state_in": rnn_state_in,
                "rnn_state_out": rnn_state,
                # "ac_prev": ac_prev,
            },
        )

    def get_c_rnn_state(self, ob, ac_prev, rnn_state_in):
        ob = self.normalize(ob)

        ob = ob.copy()
        for k, v in ob.items():
            if self._config.encoder_type == "cnn" and len(v.shape) == 3:
                ob[k] = center_crop(v, self._config.encoder_image_size)
            else:
                ob[k] = np.expand_dims(ob[k], axis=0)

        if len(ac_prev.shape) == 1:
            ac_prev = ac_prev.copy()
            ac_prev = np.expand_dims(ac_prev, axis=0)
        rnn_state_in = rnn_state_in.copy()
        rnn_state_in = np.expand_dims(rnn_state_in, axis=1)

        ob_shape = []
        with torch.no_grad():
            ob = to_tensor(ob, self._config.device)
            ac_prev = to_tensor(ac_prev, self._config.device)
            rnn_state_in = to_tensor(rnn_state_in, self._config.device)

            for k, v in ob.items():
                if (self._config.encoder_type == "cnn" and len(v.shape) == 5) or (
                    self._config.encoder_type == "mlp" and len(v.shape) == 3
                ):
                    ob_shape = list(v.shape[:2])
                    ob[k] = ob[k].flatten(start_dim=0, end_dim=1)

            out = self._critic.encoder(ob)

            if len(ob_shape) > 0:
                out = out.reshape(ob_shape + [self._critic.encoder.output_dim])

            rnn_in = torch.cat((out, ac_prev), dim=-1)

            ### RNN input dim : (seq_len, batch, input=(50+ac_dim))

            if len(rnn_in.shape) == 2:
                rnn_in = torch.unsqueeze(rnn_in, dim=0)

            _, rnn_state = self._critic.rnn(rnn_in, rnn_state_in)
        return rnn_state.cpu().numpy().squeeze(1)

    def _update_actor_and_alpha(self, o, state, ac_prev, rnn_state_in):
        info = Info()

        actions_real, _, log_pi, ent, _ = self._actor.act(
            o, ac_prev, rnn_state_in, return_log_prob=True
        )

        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha.detach() * log_pi).mean()

        actor_loss = -torch.min(
            *self._critic(
                state,
                ac_prev,
                rnn_state_in,
                actions_real,
                detach_conv=True,
                detach_rnn=True,
            )
        ).mean()

        entropy = ent.mean()
        info["entropy_alpha"] = alpha.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_entropy"] = entropy.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        entropy = ent.mean()
        info["actor_entropy"] = entropy.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self._actor)
        if self._config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor.parameters(), self._config.max_grad_norm
            )
        self._actor_optim.step()

        # update alpha
        alpha_loss = -(alpha * (log_pi + self._target_entropy).detach()).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        if self._config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                [self._log_alpha], self._config.max_grad_norm
            )
        self._alpha_optim.step()

        return info

    def _update_critic(
        self, o, ac, rew, o_next, done, state, state_next, ac_prev, rnn_state_in
    ):
        info = Info()

        # calculate the target Q value function
        with torch.no_grad():
            alpha = self._log_alpha.exp().detach()

            actions_next, _, log_pi_next, _, _ = self._actor.act(
                o_next, ac_prev, rnn_state_in, return_log_prob=True
            )
            q_next_value1, q_next_value2 = self._critic_target(
                state_next, ac["ac"], rnn_state_in, actions_next, burn_in=self._config.burn_in
            )
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            target_q_value = (
                rew * self._config.reward_scale
                + (1 - done) * self._config.rl_discount_factor * q_next_value
            )

        # the q loss
        real_q_value1, real_q_value2 = self._critic(
            state, ac_prev, rnn_state_in, ac, burn_in=self._config.burn_in
        )

        critic1_loss = F.mse_loss(target_q_value, real_q_value1)
        critic2_loss = F.mse_loss(target_q_value, real_q_value2)
        critic_loss = critic1_loss + critic2_loss

        # update the critic
        self._critic_optim.zero_grad()
        critic_loss.backward()
        if self._config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._critic.parameters(), self._config.max_grad_norm
            )
        sync_grads(self._critic)
        self._critic_optim.step()

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()
        info["min_real1_q"] = real_q_value1.min().cpu().item()
        info["min_real2_q"] = real_q_value2.min().cpu().item()
        info["real1_q"] = real_q_value1.mean().cpu().item()
        info["real2_q"] = real_q_value2.mean().cpu().item()
        info["critic1_loss"] = critic1_loss.cpu().item()
        info["critic2_loss"] = critic2_loss.cpu().item()

        return info

    def _update_network(self, transitions):

        info = Info()

        # pre-process observations
        o, o_next = transitions["ob"], transitions["ob_next"]
        o = self.normalize(o)
        o_next = self.normalize(o_next)

        seq_len, bs = transitions["done"].shape  # (seq_len, batch_size, dim)
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        rnn_state_in = _to_tensor(transitions["rnn_state_in"][0].transpose((1, 0, 2)))
        c_rnn_state_in = _to_tensor(transitions["c_rnn_state_in"][0].transpose((1, 0, 2)))
        # rnn_state_out = _to_tensor(transitions["rnn_state_out"])
        ac_prev = _to_tensor(transitions["ac_prev"])
        ac = _to_tensor(transitions["ac"])
        done = _to_tensor(transitions["done"]).reshape(seq_len, bs, 1).float()
        rew = _to_tensor(transitions["rew"]).reshape(seq_len, bs, 1)

        state = _to_tensor(transitions["state"])
        state_next = _to_tensor(transitions["state_next"])

        self._update_iter += 1

        critic_train_info = self._update_critic(
            o, ac, rew, o_next, done, state, state_next, ac_prev, rnn_state_in
        )

        info.add(critic_train_info)

        if self._update_iter % self._config.actor_update_freq == 0:
            actor_train_info = self._update_actor_and_alpha(
                o, state, ac_prev, rnn_state_in
            )
            info.add(actor_train_info)

        if self._update_iter % self._config.critic_target_update_freq == 0:
            for i, fc in enumerate(self._critic.fcs):
                self._soft_update_target_network(
                    self._critic_target.fcs[i],
                    fc,
                    self._config.critic_soft_update_weight,
                )

            self._soft_update_target_network(
                self._critic_target.encoder,
                self._critic.encoder,
                self._config.encoder_soft_update_weight,
            )

            self._soft_update_target_network(
                self._critic_target.rnn,
                self._critic.rnn,
                self._config.critic_soft_update_weight,
            )

        return info.get_dict(only_scalar=True)

    ### Misc. ###

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a Recurrent Asymmetric SAC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info("The critic has %d parameters", count_parameters(self._critic))

    def run_in(self, env_type, **kwargs):
        return ASACAgentWrapper(self, env_type, **kwargs)

    def decide_runner_type(self, curr_info):
        return "off_policy"

    def decide_env_type(self, curr_info):
        return ["source"]

    def store_episode(self, rollouts, env, stage):
        super().store_episode(rollouts)

    def update_normalizer(self, obs, env, stage):
        super().update_normalizer(obs)

    def train(self, env, curr_info):
        return super().train()


class ASACAgentWrapper:
    def __init__(self, agent, env_type, **kwargs):
        self._agent = agent

    def act(self, ob, prev_ac, rnn_state_in, is_train=True):
        return self._agent.act(ob, prev_ac, rnn_state_in, is_train=is_train)
