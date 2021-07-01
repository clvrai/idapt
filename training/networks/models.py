"inverse and forward model"
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from collections import OrderedDict
from gym.spaces.box import Box
from gym.spaces.dict import Dict

from training.networks.generator import Generator1
from training.networks.encoder import GOTEncoder
from training.networks.utils import MLP
from training.utils.pytorch import to_tensor, random_crop, center_crop_images


class InverseModel(nn.Module):
    def __init__(self, config, encoder_dim, ac_space):
        super().__init__()
        self._config = config
        self._ac_space = ac_space
        self._activation_fn = getattr(F, config.policy_activation)
        input_dim = 2 * encoder_dim

        self.fc = MLP(
            config,
            input_dim,
            config.inverse_model_mlp_dim[-1],
            config.inverse_model_mlp_dim[:-1],
        )
        self.fcs = nn.ModuleDict()
        for k, v in ac_space.spaces.items():
            self.fcs.update(
                {
                    k: MLP(
                        config, config.inverse_model_mlp_dim[-1], gym.spaces.flatdim(v)
                    )
                }
            )

    def forward(self, feature, feature_next):
        feature = torch.cat([feature, feature_next], dim=-1)
        out = self._activation_fn(self.fc(feature))

        acs = OrderedDict()
        for k, v in self._ac_space.spaces.items():
            acs[k] = torch.tanh(self.fcs[k](out))
        return acs


class ForwardModel(nn.Module):
    def __init__(self, config, encoder_dim, ac_space):
        super().__init__()
        self._config = config
        input_dim = gym.spaces.flatdim(ac_space) + encoder_dim
        self.fc = MLP(config, input_dim, encoder_dim, config.forward_model_mlp_dim)

    def forward(self, feature, ac):
        if isinstance(ac, OrderedDict):
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            ac = torch.cat(ac, dim=-1)

        sa = torch.cat([feature, ac], dim=-1)
        out = self.fc(sa)
        return out


class InverseModelFromState(nn.Module):
    def __init__(self, config, ob_space, ac_space):
        super().__init__()
        self._config = config
        self._ac_space = ac_space
        self._activation_fn = getattr(F, config.policy_activation)
        input_dim = 2 * gym.spaces.flatdim(ob_space)

        self.fc = MLP(
            config,
            input_dim,
            config.inverse_model_mlp_dim[-1],
            config.inverse_model_mlp_dim[:-1],
        )
        self.fcs = nn.ModuleDict()
        for k, v in ac_space.spaces.items():
            self.fcs.update(
                {
                    k: MLP(
                        config, config.inverse_model_mlp_dim[-1], gym.spaces.flatdim(v)
                    )
                }
            )

    def forward(self, ob, ob_next):
        ob = list(ob.values())
        if len(ob[0].shape) == 1:
            ob = [x.unsqueeze(0) for x in ob]
        ob = torch.cat(ob, dim=-1)

        ob_next = list(ob_next.values())
        if len(ob_next[0].shape) == 1:
            ob_next = [x.unsqueeze(0) for x in ob_next]
        ob_next = torch.cat(ob_next, dim=-1)

        s_sn = torch.cat([ob, ob_next], dim=-1)

        out = self._activation_fn(self.fc(s_sn))

        acs = OrderedDict()
        for k, _ in self._ac_space.spaces.items():
            acs[k] = torch.tanh(self.fcs[k](out))
        return acs


class ForwardModelFromState(nn.Module):
    def __init__(self, config, ob_space, ac_space):
        super().__init__()
        self._config = config
        self._ob_space = ob_space
        input_dim = gym.spaces.flatdim(ac_space) + gym.spaces.flatdim(ob_space)
        self._activation_fn = getattr(F, config.policy_activation)
        self.fc = MLP(
            config,
            input_dim,
            config.forward_model_mlp_dim[-1],
            config.forward_model_mlp_dim[:-1],
        )
        self.fcs = nn.ModuleDict()
        for k, v in ob_space.spaces.items():
            self.fcs.update(
                {
                    k: MLP(
                        config, config.forward_model_mlp_dim[-1], gym.spaces.flatdim(v)
                    )
                }
            )

    def forward(self, ob, ac):
        if isinstance(ac, OrderedDict):
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            ac = torch.cat(ac, dim=-1)

        ob = list(ob.values())
        if len(ob[0].shape) == 1:
            ob = [x.unsqueeze(0) for x in ob]
        ob = torch.cat(ob, dim=-1)

        sa = torch.cat([ob, ac], dim=-1)
        out = self._activation_fn(self.fc(sa))
        ob_next = OrderedDict()
        for k, _ in self._ob_space.spaces.items():
            ob_next[k] = self.fcs[k](out)
        return ob_next


class GOT_Model(nn.Module):
    """GOT Model: CycleGAN with state reconstruction regularization loss."""

    def __init__(self, config, source_ob_space, target_ob_space, source_state_space):
        super().__init__()
        self._config = config

        self._encoder_type = config.encoder_type
        self._source_ob_space = source_ob_space
        self._target_ob_space = target_ob_space
        self._source_state_space = source_state_space

        self.output_dim = config.encoder_conv_output_dim
        self.predict_output_dim = gym.spaces.flatdim(self._source_state_space)

        ### Make Obs GAN ###
        self.G = {}
        self.D = {}
        self.G1 = Generator1(
            config, source_ob_space, source_ob_space, network_type="resnet"
        )
        self.D1 = Generator1(
            config, source_ob_space, source_ob_space, network_type="patchGAN_disc"
        )
        self.G["target"] = self.G1
        self.D["target"] = self.D1
        self.G2 = Generator1(
            config, source_ob_space, source_ob_space, network_type="resnet"
        )
        self.D2 = Generator1(
            config, source_ob_space, source_ob_space, network_type="patchGAN_disc"
        )
        self.G["source"] = self.G2
        self.D["source"] = self.D2

        if self._config.include_recon:
            ### Make Split Encoder ###
            domain_layers = config.encoder_domain_layers
            domain_kernel_size = config.encoder_kernel_size[:domain_layers]
            domain_stride = config.encoder_stride[:domain_layers]

            shared_kernel_size = config.encoder_kernel_size[domain_layers:]
            shared_stride = config.encoder_stride[domain_layers:]

            self.source_encoder = self.make_encoder(
                type=config.encoder_type,
                input_space=self._source_ob_space,
                conv_dim=config.encoder_conv_dim,
                image_size=config.encoder_image_size,
                kernel_size=domain_kernel_size,
                stride=domain_stride,
                conv_output_dim=None,
                flatten=False,
            )

            self.target_encoder = self.make_encoder(
                type=config.encoder_type,
                input_space=self._target_ob_space,
                conv_dim=config.encoder_conv_dim,
                image_size=config.encoder_image_size,
                kernel_size=domain_kernel_size,
                stride=domain_stride,
                conv_output_dim=None,
                flatten=False,
            )

            self.domain_layer_output_dim = self.source_encoder.output_dim

            self.shared_encoder = self.make_encoder(
                type=config.encoder_type,
                input_space=Dict(
                    {
                        "features": Box(
                            low=-np.inf, high=np.inf, shape=self.domain_layer_output_dim
                        )
                    }
                ),
                conv_dim=config.encoder_conv_dim,
                image_size=self.domain_layer_output_dim[-1],
                kernel_size=shared_kernel_size,
                stride=shared_stride,
                conv_output_dim=self.output_dim,
                flatten=True,
            )

            self.predict = MLP(
                config, self.output_dim, self.predict_output_dim, hid_dims=[256, 256]
            )

            self.domain_encoder = {
                "source": self.source_encoder,
                "target": self.target_encoder,
            }

        ### Define Losses
        self._recon_loss = nn.L1Loss()
        self._GAN_loss = nn.BCEWithLogitsLoss()
        self._cycle_loss = nn.L1Loss()

    def forward(self, ob, domain, detach_conv=False, predict_state=False):
        out = self.domain_encoder[domain](ob, detach_conv)
        out = OrderedDict({"features": out})

        out = self.shared_encoder(out, detach_conv, scale=False)

        assert len(out.shape) == 2

        if predict_state:
            pred = self.predict(out)
            return out, pred

        return out

    def generate(self, obs, target_domain="target"):
        obs = OrderedDict({"ob": (obs["ob"] / 255.0 * 2.0) - 1})

        out = self.G[target_domain](obs)
        obs_gen = OrderedDict({"ob": out})

        obs_gen["ob"] = (obs_gen["ob"] + 1) / 2.0 * 255.0
        return obs_gen

    def discriminate(self, obs, target_domain="target"):
        out = self.D[target_domain](obs)
        out = out.mean()
        return out

    ### Define Losses ###

    def state_recon_loss(self, obs, state, encoder_domain, obs_domain):
        obs = obs.copy()
        for k, v in obs.items():
            if self._config.encoder_type == "cnn" and len(v.shape) in [4, 5]:
                obs[k] = center_crop_images(v, self._config.encoder_image_size)
            else:
                obs[k] = np.expand_dims(obs[k], axis=0)
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        obs = _to_tensor(obs)

        if obs_domain == "source" and encoder_domain == "target":
            obs = self.generate(obs)

        feat, state_pred = self.forward(obs, encoder_domain, predict_state=True)

        state_target = torch.cat(list(state.values()), dim=-1)
        recon_loss = self._recon_loss(state_pred, state_target)

        return recon_loss

    def GAN_discriminator_loss(
        self, obs_target, obs_gen, target_domain="target", gradp=False
    ):
        obs_gen_target = OrderedDict({"ob": obs_gen["ob"].detach()})

        real_logit = self.discriminate(obs_target, target_domain)
        gen_logit = self.discriminate(obs_gen_target, target_domain)

        real_output = torch.sigmoid(real_logit)
        gen_output = torch.sigmoid(gen_logit)

        D_real_loss = self._GAN_loss(
            real_logit, 0.9 * torch.ones_like(real_logit).to(self._config.device)
        )
        D_gen_loss = self._GAN_loss(
            gen_logit, 0.1 * torch.ones_like(gen_logit).to(self._config.device)
        )

        D_loss = D_real_loss + D_gen_loss

        return D_loss, real_output, gen_output

    def GAN_cycle_loss(self, obs, obs_rec):
        cyc_loss = self._cycle_loss(obs["ob"], obs_rec["ob"])
        return cyc_loss

    def GAN_generator_loss(self, obs_gen, target_domain="target"):
        obs_gen_target = obs_gen

        gen_logit = self.discriminate(obs_gen_target, target_domain)

        gen_loss = self._GAN_loss(
            gen_logit, torch.ones_like(gen_logit).to(self._config.device)
        )

        G_loss = gen_loss

        return G_loss

    ### Network Helper Functions ###

    def make_encoder(
        self,
        type,
        input_space,
        conv_dim,
        image_size,
        kernel_size,
        stride,
        conv_output_dim,
        flatten,
    ):
        encoder_config = Namespace(
            encoder_conv_dim=conv_dim,
            encoder_image_size=image_size,
            encoder_kernel_size=kernel_size,
            encoder_stride=stride,
            encoder_conv_output_dim=conv_output_dim,
            encoder_type=type,
        )

        encoder = GOTEncoder(encoder_config, input_space, flatten=flatten)
        return encoder

    ### Misc. ###

    def set_source_networks_requires_grad(self, req):
        """
        Sets requires_grad for source_encoder, shared_encoder, and predict
        """
        if self._config.include_recon:
            for param in self.shared_encoder.parameters():
                param.requires_grad = req
            for param in self.domain_encoder["source"].parameters():
                param.requires_grad = req
            for param in self.predict.parameters():
                param.requires_grad = req

    def set_target_networks_requires_grad(self, req):
        if self._config.include_recon:
            for param in self.domain_encoder["target"].parameters():
                param.requires_grad = req

    def set_gen_networks_requires_grad(self, req):
        for key, G in self.G.items():
            for param in G.parameters():
                param.requires_grad = req

        if self._config.include_recon:
            for param in self.domain_encoder["target"].parameters():
                param.requires_grad = req

    def set_dis_networks_requires_grad(self, req):
        for key, D in self.D.items():
            for param in D.parameters():
                param.requires_grad = req

    def set_encoder_requires_grad(self, req):
        self.set_source_networks_requires_grad(req)
        self.set_gen_networks_requires_grad(req)
        self.set_dis_networks_requires_grad(req)
        self.set_target_networks_requires_grad(req)

    def copy_conv_weights_from(self, source):
        """ Tie convolutional layers """
        self.source_encoder.copy_conv_weights_from(source.source_encoder)
        self.target_encoder.copy_conv_weights_from(source.target_encoder)
        self.shared_encoder.copy_conv_weights_from(source.shared_encoder)
