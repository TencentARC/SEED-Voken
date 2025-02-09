import functools
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import Tuple
import torch

from src.Open_MAGVIT2.modules.util import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

###
## Refer from Maskbit
##https://github.com/markweberdev/maskbit
###

class Conv2dSame(nn.Conv2d):
    """ Convolution wrapper for 2D convolutions using `SAME` padding."""
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """ Calculate padding such that the output has the same height/width when stride=1.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride size.
            d -> int: Dilation rate.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the convolution applying explicit `same` padding.

        Args:
            x -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)

class BlurBlock(nn.Module):
    def __init__(
        self,
        kernel: Tuple[int] = (1, 3, 3, 1)
    ):
        """ Initializes the blur block.

        Args:
            kernel -> Tuple[int]: The kernel size.
        """
        super().__init__()

        self.kernel_size = len(kernel)

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        """ Calculates the same padding for the BlurBlock.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride.

        Returns:
            pad -> int: The padding.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass.
        
        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            out -> torch.Tensor: The output tensor.
        """
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size, s=2)
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        weight = self.kernel.expand(ic, -1, -1, -1)

        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out
    
class NLayerDiscriminatorv2(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        num_stages: int = 3,
        activation_fn: str = "leaky_relu",
        blur_resample: bool = False,
        blur_kernel_size: int = 4
    ):
        """ Initializes the NLayerDiscriminatorv2.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            activation_fn -> str: The activation function.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (blur_kernel_size >= 3 and blur_kernel_size <= 5), "Blur kernel size must be in [3,5] when sampling]"

        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        if activation_fn == "leaky_relu":
            activation = functools.partial(nn.LeakyReLU, negative_slope=0.1)
        else:
            activation =  nn.SiLU

        self.block_in = nn.Sequential(
            Conv2dSame(
                num_channels,
                hidden_channels,
                kernel_size=init_kernel_size
            ),
            activation(),
        )

        BLUR_KERNEL_MAP = {
            3: (1,2,1),
            4: (1,3,3,1),
            5: (1,4,6,4,1),
        }

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            block = torch.nn.Sequential(
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                ),
                torch.nn.AvgPool2d(kernel_size=2, stride=2) if not blur_resample else BlurBlock(BLUR_KERNEL_MAP[blur_kernel_size]),
                torch.nn.GroupNorm(32, out_channels),
                activation(),
            )
            discriminator_blocks.append(block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)

        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        hidden_states = self.block_in(x)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return self.to_logits(hidden_states)

class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""
    # https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/losses/discriminator.py
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Not implemented.")
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=(kw, kw, kw),
                    stride=(2 if n == 1 else 1, 2, 2),
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw, kw), stride=1, padding=padw, bias=use_bias
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
