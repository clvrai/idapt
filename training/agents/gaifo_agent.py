import copy
import numpy as np
import torch
import torch.autograd as autograd
import torch.distributions
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR, LambdaLR

from training.agents.base_agent import BaseAgent
from training.agents.ppo_agent import PPOAgent
from training.datasets import ReplayBuffer, RandomSampler
from training.datasets import ExpertDataset
from training.datasets import OfflineDataset
from training.networks.discriminator import Discriminator
from training.utils.general import join_spaces, join_dicts
from training.utils.info_dict import Info
from training.utils.logger import logger
from training.utils.mpi import mpi_average
from training.utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    sync_networks,
    sync_grads,
    to_tensor,
    obs2tensor,
)


class GAIfOAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space
        self._env_ob_space = env_ob_space
        self._disc_space = join_spaces(self._ob_space, self._env_ob_space)

        ### Define RL Agent and Disciminator Networks ###
        dconfig = copy.copy(config)
        dconfig.critic_ensemble = 1
        dconfig.actor_update_freq = 1
        dconfig.ob_norm = True
        dconfig.encoder_type = "mlp"
        dconfig.batch_size = config.gaifo_discriminator_batch_size
        dconfig.rl_discount_factor = config.gaifo_rl_discount_factor
        self._rl_agent = IfOPPOAgent(dconfig, ob_space, ac_space, env_ob_space)
        self._rl_agent.set_reward_function(self._predict_reward)

        dconfig.discriminator_mlp_dim = config.gaifo_discriminator_mlp_dim
        dconfig.discriminator_activation = config.gaifo_discriminator_activation

        self._discriminator = Discriminator(dconfig, self._disc_space)
        self._discriminator_loss = nn.BCEWithLogitsLoss()
        self._network_cuda(config.device)

        ### Define Optimizers ###
        self._discriminator_optim = optim.Adam(
            self._discriminator.parameters(),
            lr=config.gaifo_discriminator_lr,
            betas=(0.5, 0.999),
            weight_decay=config.gaifo_discriminator_weight_decay,
        )

        self._discriminator_lr_scheduler = StepLR(
            self._discriminator_optim,
            step_size=self._config.max_global_step // self._config.rollout_length // 5,
            gamma=1,
        )

        sampler = RandomSampler()
        self._buffer = ReplayBuffer(
            [
                "ob",
                "ob_next",
                "ac",
                "done",
                "rew",
                "ret",
                "adv",
                "ac_before_activation",
            ],
            config.rollout_length,
            sampler.sample_func,
        )

        self._rl_agent.set_buffer(self._buffer)

        self._dataset = OfflineDataset(
            self._config.demo_subsample_interval,
            self._ac_space,
            use_low_level=self._config.demo_low_level,
        )

        self._log_creation()
        self._curr_iter = 0

    def predict_reward(self, ob, ac, next_ob):
        """
        Calculates GARAT's IfO reward.
        """
        ob = self.normalize(ob)
        ob = to_tensor(ob, self._config.device)
        if next_ob["ob"].shape != self._ob_space.sample()["ob"].shape:
            next_ob = join_dicts(next_ob, {"ac": np.zeros_like(ac["ac"])})
        next_ob = self.normalize(next_ob)
        next_ob = to_tensor(next_ob, self._config.device)
        reward = self._predict_reward(ob, ac, next_ob)
        return reward.cpu().item()

    def _process_disc_in(self, ob, next_ob):
        """
        Generates discriminator input as (f_t, a_t, f_t+1) where each observation is feature-action pair, (f, a)
        """
        env_obs_size = self._env_ob_space["ob"].shape[-1]
        next_obs_size = next_ob["ob"].shape[-1]
        next_ob["ob"] = torch.split(
            next_ob["ob"], (env_obs_size, next_obs_size - env_obs_size), dim=-1
        )[0]
        disc_in = join_dicts(ob, next_ob)
        return disc_in

    def _predict_reward(self, ob, ac, next_ob):
        disc_in = self._process_disc_in(ob, next_ob)
        with torch.no_grad():
            ret = self._discriminator(disc_in)
            eps = 1e-10
            s = torch.sigmoid(ret)
            if self._config.gaifo_reward == "vanilla":
                reward = -(1 - s + eps).log()
            elif self._config.gaifo_reward == "gan":
                reward = (s + eps).log() - (1 - s + eps).log()
            elif self._config.gaifo_reward == "d":
                reward = ret
            elif self._config.gaifo_reward == "nonsatgan":
                reward = -(s + eps).log()
        return reward

    def store_episode(self, rollouts, env_type, stage):
        garat_rollouts = self.make_garat_state(rollouts, env_type)
        if env_type == "source":
            garat_rollouts["ac"] = garat_rollouts["AT_ac"]
            self._rl_agent.store_episode(garat_rollouts)
        elif env_type == "target":
            demos = {
                "obs": garat_rollouts["ob"] + [garat_rollouts["ob_next"][-1]],
                "actions": garat_rollouts["ac"],
                "rewards": garat_rollouts["rew"],
                "dones": garat_rollouts["done"],
                "obs_next": garat_rollouts["ob_next"],
            }
            self._set_demo(demos)

    def _set_demo(self, demos):
        self._dataset.add_demos(demos)
        self._data_loader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self._config.gaifo_discriminator_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self._data_iter = iter(self._data_loader)

    def make_garat_state(self, rollouts, env_type):
        """
        Generates GARAT input as concatenation of observation and action.
        """
        garat_rollouts = rollouts

        if env_type == "source":
            ac_key = "target_ac"
        elif env_type == "target":
            ac_key = "ac"

        garat_rollouts["ob"] = [
            join_dicts(ob, target_ac)
            for (ob, target_ac) in zip(rollouts["ob"], rollouts[ac_key])
        ]
        ob_last = join_dicts(
            rollouts["ob_next"][-1], {"ac": np.zeros_like(rollouts[ac_key][0]["ac"])}
        )
        garat_rollouts["ob_next"] = garat_rollouts["ob"][1:] + [ob_last]

        return garat_rollouts

    def train(self):
        train_info = Info()

        if self._curr_iter % self._config.gaifo_discriminator_update_freq == 0:
            num_batches = int(
                self._config.rollout_length
                // self._config.gaifo_discriminator_batch_size
                * self._config.gaifo_discriminator_update_freq
            )
            assert num_batches > 0
            for _ in range(num_batches):
                policy_data = self._buffer.sample(
                    self._config.gaifo_discriminator_batch_size
                )
                try:
                    expert_data = next(self._data_iter)
                except StopIteration:
                    self._data_iter = iter(self._data_loader)
                    expert_data = next(self._data_iter)

                _train_info = self._update_discriminator(policy_data, expert_data)
                train_info.add(_train_info)

            self._discriminator_lr_scheduler.step()

        if self._curr_iter % self._config.gaifo_agent_update_freq == 0:
            _train_info = self._rl_agent.train()
            train_info.add(_train_info)

        self._curr_iter += 1

        return train_info.get_dict(only_scalar=True)

    def _update_discriminator(self, policy_data, expert_data):
        info = Info()
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        p_o = policy_data["ob"]
        p_o = self.normalize(p_o)
        p_o = _to_tensor(p_o)

        p_o_next = policy_data["ob_next"]
        p_o_next = self.normalize(p_o_next)
        p_o_next = _to_tensor(p_o_next)

        p_in = self._process_disc_in(p_o, p_o_next)

        e_o = {k: v.numpy() for (k, v) in expert_data["ob"].items()}
        e_o = self.normalize(e_o)
        e_o = _to_tensor(e_o)

        e_o_next = {k: v.numpy() for (k, v) in expert_data["ob_next"].items()}
        e_o_next = self.normalize(e_o_next)
        e_o_next = _to_tensor(e_o_next)

        e_in = self._process_disc_in(e_o, e_o_next)

        p_logit = self._discriminator(p_in)
        e_logit = self._discriminator(e_in)

        p_output = torch.sigmoid(p_logit)
        e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(
            p_logit, torch.ones_like(p_logit).to(self._config.device)
        )
        e_loss = self._discriminator_loss(
            e_logit, torch.zeros_like(e_logit).to(self._config.device)
        )

        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean()
        entropy_loss = -self._config.gaifo_entropy_loss_coeff * entropy

        grad_pen = self._compute_grad_pen(p_in, e_in)
        grad_pen_loss = self._config.gaifo_grad_penalty_coeff * grad_pen

        gail_loss = p_loss + e_loss + entropy_loss + grad_pen_loss

        self._discriminator.zero_grad()
        gail_loss.backward()
        sync_grads(self._discriminator)
        self._discriminator_optim.step()

        info["gail_policy_output"] = p_output.mean().detach().cpu().item()
        info["gail_expert_output"] = e_output.mean().detach().cpu().item()
        info["gail_entropy"] = entropy.detach().cpu().item()
        info["gail_policy_loss"] = p_loss.detach().cpu().item()
        info["gail_expert_loss"] = e_loss.detach().cpu().item()
        info["gail_entropy_loss"] = entropy_loss.detach().cpu().item()
        info["gail_grad_pen"] = grad_pen.detach().cpu().item()
        info["gail_grad_loss"] = grad_pen_loss.detach().cpu().item()
        info["gail_loss"] = gail_loss.detach().cpu().item()

        return mpi_average(info.get_dict(only_scalar=True))

    def _compute_grad_pen(self, policy_ob, expert_ob):
        batch_size = self._config.gaifo_discriminator_batch_size
        alpha = torch.rand(batch_size, 1, device=self._config.device)

        def blend_dict(a, b, alpha):
            if isinstance(a, dict):
                return OrderedDict(
                    [(k, blend_dict(a[k], b[k], alpha)) for k in a.keys()]
                )
            elif isinstance(a, list):
                return [blend_dict(a[i], b[i], alpha) for i in range(len(a))]
            else:
                expanded_alpha = alpha.expand_as(a)
                ret = expanded_alpha * a + (1 - expanded_alpha) * b
                ret.requires_grad = True
                return ret

        interpolated_ob = blend_dict(policy_ob, expert_ob, alpha)
        inputs = list(interpolated_ob.values())

        interpolated_logit = self._discriminator(interpolated_ob)
        ones = torch.ones(interpolated_logit.size(), device=self._config.device)

        grad = autograd.grad(
            outputs=interpolated_logit,
            inputs=inputs,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    ### Misc. ###

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a GAIfO agent")
            logger.info(
                "The discriminator has %d parameters",
                count_parameters(self._discriminator),
            )

    def update_normalizer(self, obs):
        super().update_normalizer(obs)
        self._rl_agent.update_normalizer(obs)

    def state_dict(self):
        return {
            "rl_agent": self._rl_agent.state_dict(),
            "discriminator_state_dict": self._discriminator.state_dict(),
            "discriminator_optim_state_dict": self._discriminator_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "rl_agent" in ckpt:
            self._rl_agent.load_state_dict(ckpt["rl_agent"])
        else:
            self._rl_agent.load_state_dict(ckpt)
            self._network_cuda(self._config.device)
            return

        self._discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._discriminator_optim.load_state_dict(
            ckpt["discriminator_optim_state_dict"]
        )
        optimizer_cuda(self._discriminator_optim, self._config.device)

    def _network_cuda(self, device):
        self._discriminator.to(device)

    def sync_networks(self):
        self._rl_agent.sync_networks()
        sync_networks(self._discriminator)


class IfOPPOAgent(PPOAgent):
    """
    Modified PPO for GARAT IfO.  When computing advantage, use (ob, ac, ob_next) to predict IfO reward.
    """

    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space, ac_space, env_ob_space)

        self._actor_optim = optim.Adam(
            self._actor.parameters(), lr=config.ifo_actor_lr, betas=(0.5, 0.999)
        )
        self._critic_optim = optim.Adam(
            self._critic.parameters(), lr=config.ifo_critic_lr, betas=(0.5, 0.999)
        )

        self._actor_lr_scheduler = StepLR(
            self._actor_optim,
            step_size=self._config.max_global_step // self._config.rollout_length // 5,
            gamma=1,
        )
        self._critic_lr_scheduler = StepLR(
            self._critic_optim,
            step_size=self._config.max_global_step // self._config.rollout_length // 5,
            gamma=1,
        )

    def _compute_gae(self, rollouts):
        """Changes: pass next_ob into predict_reward"""
        T = len(rollouts["done"])
        ob = rollouts["ob"]
        ob = self.normalize(ob)
        ob = obs2tensor(ob, self._config.device)

        ob_last = rollouts["ob_next"][-1:]
        ob_last = self.normalize(ob_last)
        ob_last = obs2tensor(ob_last, self._config.device)
        done = rollouts["done"]
        rew = rollouts["rew"]

        vpred = self._critic(ob).detach().cpu().numpy()[:, 0]
        vpred_last = self._critic(ob_last).detach().cpu().numpy()[:, 0]
        vpred = np.append(vpred, vpred_last)
        assert len(vpred) == T + 1

        assert hasattr(self, "_predict_reward")
        if hasattr(self, "_predict_reward"):
            ac = rollouts["ac"]
            ob_next = rollouts["ob_next"]
            ob_next = self.normalize(ob_next)
            ob_next = obs2tensor(ob_next, self._config.device)

            rew_il = self._predict_reward(ob, ac, ob_next).cpu().numpy().squeeze()
            rew = (1 - self._config.gaifo_env_reward) * rew_il[
                :T
            ] + self._config.gaifo_env_reward * np.array(rew)
            assert rew.shape == (T,)

        adv = np.empty((T,), "float32")
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            delta = (
                rew[t]
                + self._config.rl_discount_factor * vpred[t + 1] * nonterminal
                - vpred[t]
            )
            adv[t] = lastgaelam = (
                delta
                + self._config.rl_discount_factor
                * self._config.gae_lambda
                * nonterminal
                * lastgaelam
            )

        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        if self._config.advantage_norm:
            rollouts["adv"] = ((adv - adv.mean()) / (adv.std() + 1e-5)).tolist()
        else:
            rollouts["adv"] = adv.tolist()

        rollouts["ret"] = ret.tolist()
