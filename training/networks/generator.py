import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from gym.spaces.dict import Dict

from training.networks.basic_networks import (
    Identity,
    ResnetGenerator,
    NLayerDiscriminator,
    ResnetDCC,
)
from training.networks.utils import MLP, CNN


class Generator1(nn.Module):
    def __init__(
        self, config, input_space, output_space, network_type="mlp", hid_dims=[]
    ):
        super().__init__()
        self._config = config
        self.network_type = network_type

        input_dim = gym.spaces.flatdim(input_space)
        output_dim = gym.spaces.flatdim(output_space)

        if network_type == "mlp":
            self.net = MLP(config, input_dim, output_dim, hid_dims)
        elif network_type == "identity":
            self.net = Identity()
        elif network_type == "resnet":
            # import ipdb
            # ipdb.set_trace()
            # assert isinstance(input_space, Dict) and input_space.keys() == ["ob"]
            input_nc = output_nc = input_space["ob"].shape[0]
            self.net = ResnetGenerator(input_nc, output_nc, ngf=64, n_blocks=6)
        elif network_type == "resnet18dcc":
            input_nc = input_space["ob"].shape[0]
            self.net = ResnetDCC(config, input_nc, output_dim, hid_dims=[64, 32])
        elif network_type == "patchGAN_disc":
            # assert isinstance(input_space, Dict) and input_space.keys() == ["ob"]
            input_nc = input_space["ob"].shape[0]
            self.net = NLayerDiscriminator(input_nc, ndf=64, n_layers=3)

    def forward(self, ob):
        if isinstance(ob, OrderedDict):
            ob = list(ob.values())
            if len(ob[0].shape) == 1:
                ob = [x.unsqueeze(0) for x in ob]
            ob = torch.cat(ob, dim=-1)

        out = self.net(ob)
        return out


class Generator2(nn.Module):
    def __init__(
        self, config, ob_space, ac_space, output_space, network_type="mlp", hid_dims=[]
    ):
        super().__init__()
        self._config = config
        self.network_type = network_type
        self._ob_space = ob_space

        input_dim = gym.spaces.flatdim(ob_space) + gym.spaces.flatdim(ac_space)
        output_dim = gym.spaces.flatdim(output_space)

        if network_type == "mlp":
            self.net = MLP(config, input_dim, output_dim, hid_dims)

    def forward(self, ob, ac):
        if isinstance(ac, OrderedDict):
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            ac = torch.cat(ac, dim=-1)

        if isinstance(ob, OrderedDict):
            ob = list(ob.values())
            if len(ob[0].shape) == 1:
                ob = [x.unsqueeze(0) for x in ob]
            ob = torch.cat(ob, dim=-1)

        sa = torch.cat([ob, ac], dim=-1)
        out = self.net(sa)

        return out


class NLayerDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator, code reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
