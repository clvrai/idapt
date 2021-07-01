import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from training.networks.distributions import (
    FixedCategorical,
    FixedNormal,
    Identity,
    MixedDistribution,
)
from training.networks.encoder import Encoder
from training.networks.basic_networks import RNN
from training.networks.utils import MLP, flatten_ac
from training.utils.general import join_dicts

""" Recurrent Actor, Critic models with LSTM layer for adaptive policies. """


class RActor(nn.Module):
    def __init__(self, config, ob_space, ac_space, tanh_policy, rnn=None, encoder=None):
        super().__init__()
        self._config = config
        self._ac_space = ac_space
        self._activation_fn = getattr(F, config.policy_activation)
        self._tanh = tanh_policy
        self._gaussian = config.gaussian_policy

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(config, ob_space)

        if rnn:
            self.rnn = rnn
        else:
            self.rnn = RNN(
                config, self.encoder.output_dim + np.prod(self._ac_space["ac"].shape)
            )

        self.fc = MLP(
            config,
            self.rnn.output_dim,
            config.policy_mlp_dim[-1],
            config.policy_mlp_dim[:-1],
        )

        self.fcs = nn.ModuleDict()
        self._dists = {}
        for k, v in ac_space.spaces.items():
            if isinstance(
                v, gym.spaces.Box
            ):  # and self._gaussian:  # for convenience to transfer bc policy
                self.fcs.update(
                    {
                        k: MLP(
                            config, config.policy_mlp_dim[-1], gym.spaces.flatdim(v) * 2
                        )
                    }
                )
            else:
                self.fcs.update(
                    {k: MLP(config, config.policy_mlp_dim[-1], gym.spaces.flatdim(v))}
                )

            if isinstance(v, gym.spaces.Box):
                if self._gaussian:
                    self._dists[k] = lambda m, s: FixedNormal(m, s)
                else:
                    self._dists[k] = lambda m, s: Identity(m)
            else:
                self._dists[k] = lambda m, s: FixedCategorical(logits=m)

    @property
    def info(self):
        return {}

    def forward(self, ob: dict, ac_prev: dict, rnn_state_in, detach_conv=False):
        ob_shape = []
        ob = ob.copy()

        for k, v in ob.items():
            if (self._config.encoder_type == "cnn" and len(v.shape) == 5) or (
                self._config.encoder_type == "mlp" and len(v.shape) == 3
            ):
                ob_shape = list(v.shape[:2])
                ob[k] = ob[k].flatten(start_dim=0, end_dim=1)

        out = self.encoder(ob, detach_conv=detach_conv)

        if len(ob_shape) > 0:
            out = out.reshape(ob_shape + [self.encoder.output_dim])

        rnn_in = torch.cat((out, ac_prev), dim=-1)

        ### RNN input dim : (seq_len, batch, input=(50+ac_dim))

        if len(rnn_in.shape) == 2:
            rnn_in = torch.unsqueeze(rnn_in, dim=0)

        out, rnn_state = self.rnn(rnn_in, rnn_state_in)
        out = self._activation_fn(self.fc(out))

        means, stds = OrderedDict(), OrderedDict()
        for k, v in self._ac_space.spaces.items():
            if isinstance(v, gym.spaces.Box):  # and self._gaussian:
                mean, log_std = self.fcs[k](out).chunk(2, dim=-1)
                log_std_min, log_std_max = (
                    self._config.log_std_min,
                    self._config.log_std_max,
                )
                log_std = torch.tanh(log_std)
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (
                    log_std + 1
                )
                std = log_std.exp()
            else:
                mean, std = self.fcs[k](out), None

            means[k] = mean
            stds[k] = std

        return means, stds, rnn_state

    def act(
        self,
        ob,
        ac_prev,
        rnn_state_in,
        deterministic=False,
        activations=None,
        return_log_prob=False,
        detach_conv=False,
    ):
        """ Samples action for rollout. """
        means, stds, rnn_state = self.forward(
            ob, ac_prev, rnn_state_in, detach_conv=detach_conv
        )

        dists = OrderedDict()
        for k in means.keys():
            dists[k] = self._dists[k](means[k], stds[k])

        actions = OrderedDict()
        mixed_dist = MixedDistribution(dists)
        if activations is None:
            if deterministic:
                activations = mixed_dist.mode()
            else:
                activations = mixed_dist.rsample()

        if return_log_prob:
            log_probs = mixed_dist.log_probs(activations)

        for k, v in self._ac_space.spaces.items():
            z = activations[k]
            if self._tanh and isinstance(v, gym.spaces.Box):
                action = torch.tanh(z)
                if return_log_prob:
                    # follow the Appendix C. Enforcing Action Bounds
                    log_det_jacobian = 2 * (np.log(2.0) - z - F.softplus(-2.0 * z)).sum(
                        dim=-1, keepdim=True
                    )
                    log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action

        if return_log_prob:
            log_probs = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
            entropy = mixed_dist.entropy()
        else:
            log_probs = None
            entropy = None

        return actions, activations, log_probs, entropy, rnn_state


class RCritic(nn.Module):
    def __init__(self, config, ob_space, ac_space=None, rnn=None, encoder=None):
        super().__init__()
        self._config = config

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(config, ob_space)

        input_dim = self.encoder.output_dim
        if ac_space is not None:
            input_dim += gym.spaces.flatdim(ac_space)

        if rnn:
            self.rnn = rnn
        else:
            self.rnn = RNN(
                config, self.encoder.output_dim + np.prod(ac_space["ac"].shape)
            )

        self.fcs = nn.ModuleList()

        for i in range(config.critic_ensemble):
            self.fcs.append(
                MLP(
                    config,
                    self.rnn.output_dim + np.prod(ac_space["ac"].shape),
                    1,
                    config.critic_mlp_dim,
                )
            )

    def forward(
        self, ob, ac_prev, rnn_state_in, ac=None, detach_conv=False, detach_rnn=False, burn_in=0
    ):

        ob_shape = []
        ob = ob.copy()
        # get new rnn state in if burn in
        if burn_in > 0:
            ob_burn_in = {}
            for k, v in ob.items():
                ob_shape_burn_in = list(v.shape[:2])
                ob_burn_in[k] = ob[k][:burn_in]
                ac_prev_burn_in = ac_prev[:burn_in]
                ob[k] = ob[k][burn_in:]
                ac_prev = ac_prev[burn_in:]
                ob_burn_in[k] = ob_burn_in[k].flatten(start_dim=0, end_dim=1)
            
            ob_shape_burn_in[0] = burn_in

            out_burn_in = self.encoder(ob_burn_in, detach_conv=detach_conv)

            if len(ob_shape_burn_in) > 0:
                out_burn_in = out_burn_in.reshape(ob_shape_burn_in + [self.encoder.output_dim])

            rnn_in_burn_in = torch.cat((out_burn_in, ac_prev_burn_in), dim=-1)

            with torch.no_grad():
                _, rnn_state_in = self.rnn(rnn_in_burn_in, rnn_state_in)

        for k, v in ob.items():
            if (self._config.encoder_type == "cnn" and len(v.shape) == 5) or (
                self._config.encoder_type == "mlp" and len(v.shape) == 3
            ):
                ob_shape = list(v.shape[:2])
                ob[k] = ob[k].flatten(start_dim=0, end_dim=1)

        out = self.encoder(ob, detach_conv=detach_conv)

        if len(ob_shape) > 0:
            out = out.reshape(ob_shape + [self.encoder.output_dim])

        rnn_in = torch.cat((out, ac_prev), dim=-1)

        ### RNN input dim : (seq_len, batch, input=(50+ac_dim))

        if len(rnn_in.shape) == 2:
            rnn_in = torch.unsqueeze(rnn_in, dim=0)

        if detach_rnn:
            with torch.no_grad():
                out, _ = self.rnn(rnn_in, rnn_state_in)
        else:
            out, _ = self.rnn(rnn_in, rnn_state_in)

        if ac is not None:
            #  only use the end of the sequence
            out = torch.cat([out[-1], flatten_ac(ac)[-1]], dim=-1)
        out = [fc(out) for fc in self.fcs]

        """
        if len(ob_shape) > 0:
            out = [o.reshape(ob_shape + [1]) for o in out]
        """

        if len(out) == 1:
            return out[0]
        return out
