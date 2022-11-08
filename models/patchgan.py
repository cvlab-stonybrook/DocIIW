'''
Code modified from the Pix2Pix repository

'''
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.nn.utils import spectral_norm


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()
        if norm_layer is None:
            use_bias=True
            kw = 4
            padw = 1
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.01, False)
            ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    # norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.01, False)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.01, False)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias=False)]

            if use_sigmoid:
                sequence += [nn.Sigmoid()]

            self.model = nn.Sequential(*sequence)

        elif norm_layer=='spectral':
            use_bias=True
            kw = 4
            padw = 1
            sequence = [
                snconv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, False)
            ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [
                    snconv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    # norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, False)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [
                snconv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                # norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

            if use_sigmoid:
                sequence += [nn.Sigmoid()]

            self.model = nn.Sequential(*sequence)
        else: 
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            kw = 4
            padw = 1
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, False)
            ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, False)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

            if use_sigmoid:
                sequence += [nn.Sigmoid()]

            self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)