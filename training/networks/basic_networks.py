import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import init
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from training.networks.utils import MLP, CNN, weight_init


class Identity(nn.Module):
    def forward(self, ob):
        return ob


class RNN(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size=config.rnn_dim)
        # self.embed = nn.Linear(input_dim, config.rnn_dim)
        self.output_dim = config.rnn_dim

    def forward(self, ob_ac, rnn_state):
        # ob_ac should be of shape (seq_len, batch, input_size: ob_dim + ac_dim)
        # rnn_state contains initial hidden, cell state of sequence (h_0, c_0), (2, batch_size, hidden_size)

        h_0, c_0 = torch.chunk(rnn_state, 2, dim=0)
        out, (h_n, c_n) = self.rnn(ob_ac, (h_0, c_0))
        rnn_state_out = torch.cat((h_n, c_n), dim=0)
        return out, rnn_state_out


class CNNLayer(nn.Module):
    def __init__(self, config, input_dim, flatten=True):
        super().__init__()
        self._flatten = flatten

        self.convs = nn.ModuleList()
        d_prev = input_dim
        d = config.encoder_conv_dim
        w = config.encoder_image_size
        for k, s in zip(config.encoder_kernel_size, config.encoder_stride):
            self.convs.append(nn.Conv2d(d_prev, d, int(k), int(s)))
            w = int(np.floor((w - (int(k) - 1) - 1) / int(s) + 1))
            d_prev = d

        print("Output of CNN (%d) = %d x %d x %d" % (w * w * d, w, w, d))

        if self._flatten:
            self.output_dim = config.encoder_conv_output_dim

            self.fc = nn.Linear(w * w * d, self.output_dim)
            self.ln = nn.LayerNorm(self.output_dim)

        if not self._flatten:
            self.output_dim = (d, w, w)

        self.apply(weight_init)

    def forward(self, ob, detach_conv=False):
        out = ob
        for conv in self.convs:
            out = F.relu(conv(out))

        if self._flatten:
            out = out.flatten(start_dim=1)

        if detach_conv:
            out = out.detach()

        if self._flatten:
            out = self.fc(out)
            out = self.ln(out)
            out = F.tanh(out)

        return out

    # from https://github.com/MishaLaskin/rad/blob/master/encoder.py
    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i, conv in enumerate(self.convs):
            assert type(source.convs[i]) == type(conv)
            conv.weight = source.convs[i].weight
            conv.bias = source.convs[i].bias


class ResNet18NChannels(ResNet):
    def __init__(self, inchannels,) -> None:
        super(ResNet18NChannels, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=256
        )

        self.conv1 = nn.Conv2d(
            inchannels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )


class ResnetDCC(nn.Module):
    def __init__(self, config, input_nc, output_dim, hid_dims=[]):
        super().__init__()

        self.resnet18 = ResNet18NChannels(input_nc)
        self.fc = MLP(config, 256, output_dim, hid_dims)

    def forward(self, input):
        input = (input / 255.0 * 2.0) - 1.0
        out = self.resnet18(input)
        out = self.fc(out)
        return out


class ResnetGenerator(nn.Module):
    """
    Resnet code reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.apply(init_weights(init_type="orthogonal"))

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """
    ResnetBlock code reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
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
        self.apply(init_weights(init_type="orthogonal"))

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def init_weights(init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization idapt: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization idapt [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    return init_func
