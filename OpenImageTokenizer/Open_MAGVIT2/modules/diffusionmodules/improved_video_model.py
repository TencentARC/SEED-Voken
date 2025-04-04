import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Callable, Any, List
from collections import OrderedDict
from torch.nn.modules.utils import _triple, _pair
from torch import Tensor

def swish(x):
    # swish
    return x*torch.sigmoid(x)

class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        *,
        kernel_size: Union[int, Tuple[int, int, int]],
        causal: bool,
        padding: Union[int, Tuple[int, int, int]] = 0,
        stride: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        **kwargs: Any
    ):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        if causal:
            padding = 0
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        time_pad = kernel_size[0] - 1 if causal else kernel_size[0] // 2
        
        if causal:
            self.padding = (
                width_pad,
                width_pad,
                height_pad,
                height_pad,
                time_pad,
                0
            )
        else:
            self.padding(
                width_pad,
                width_pad,
                height_pad,
                height_pad,
                time_pad,
                time_pad
            )

        self.conv_1 = nn.Conv3d(in_planes, out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        groups=groups,
                        bias=bias,
                        **kwargs)
        
    def forward(self, x):
        """
        CasuelConv3D
        """
        x = F.pad(x, self.padding, mode="constant")
        x = self.conv_1(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, 
                 in_filters,
                 out_filters,
                 use_conv_shortcut = False,
                 use_agn = False,
                 ) -> None:
        super().__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn: ## agn is GroupNorm likewise skip it if has agn before
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = ConvBlock3D(in_filters, out_filters, kernel_size=(3, 3, 3), causal=True, padding=1, bias=False)
        self.conv2 = ConvBlock3D(out_filters, out_filters, kernel_size=(3, 3, 3), causal=True, padding=1, bias=False)
        

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = ConvBlock3D(in_filters, out_filters, kernel_size=(3, 3, 3), causal=True, padding=1, bias=False)
            else:
                self.nin_shortcut = ConvBlock3D(in_filters, out_filters, kernel_size=(1, 1, 1), causal=True, padding=0, bias=False)
    

    def forward(self, x, **kwargs):
        residual = x

        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return x + residual
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False,
                ):
        super().__init__()

        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution

        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        
        self.conv_in = ConvBlock3D(in_channels,
                                   ch,
                                   kernel_size=(3, 3, 3),
                                   causal = True,
                                   padding=1,
                                   bias=False
        )

        ## construct the model
        self.down = nn.ModuleList()

        in_ch_mult = (1,)+tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level] #[1, 1, 2, 2, 4]
            block_out = ch*ch_mult[i_level] #[1, 2, 2, 4]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                if ch_mult[i_level] == 1: #downsampling
                    down.downsample = ConvBlock3D(block_out, block_out, kernel_size=(3, 3, 3), causal=True, stride=(1, 2, 2), padding=1)
                else:
                    down.downsample = ConvBlock3D(block_out, block_out, kernel_size=(3, 3, 3), causal=True, stride=(2, 2, 2), padding=1)

            self.down.append(down)
        
        ### mid
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        ### end
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = ConvBlock3D(block_out, z_channels, kernel_size=(1, 1, 1), causal=True)

    def forward(self, x):

        ## down
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            
            if i_level <  self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
        
        ## mid 
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
        

        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False,) -> None:
        super().__init__()

        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch*ch_mult[self.num_blocks-1]

        self.conv_in = ConvBlock3D(
            z_channels, block_in, kernel_size=(3, 3, 3), causal=True, padding=1, bias=True
        )

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        self.up = nn.ModuleList()

        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                if ch_mult[i_level -1] == 1: #last upsampling
                    up.upsample = Upsampler(block_in, block_size=(1, 2, 2))
                else:
                    up.upsample = Upsampler(block_in, block_size=(2, 2, 2))
            self.up.insert(0, up)
        
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)

        self.conv_out = ConvBlock3D(block_in, out_ch, kernel_size=(3, 3, 3), causal=True, padding=1)
    
    def forward(self, z):
        
        style = z.clone() #for adaptive groupnorm

        z = self.conv_in(z)

        ## mid
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        
        ## upsample
        for i_level in reversed(range(self.num_blocks)):
            ### pass in each resblock first adaGN
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)

        return z

def depth_to_space3d(x: torch.Tensor, block_size: List) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions"
        )
    c, t, h, w = x.shape[-4:]

    time_block, h_block, w_block = block_size
    s = time_block * h_block * w_block
    if c % s != 0:
        raise ValueError(
            f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels"
        )

    outer_dims = x.shape[:-4]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, time_block, h_block, w_block, c // s, t, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 4, 5, 1, 6, 2, 7, 3)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, t * time_block, h * h_block,
                            w * w_block)

    return x

class Upsampler(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        
        dim_out = dim
        for block_si in block_size:
            dim_out = dim_out * block_si
        
        self.block_size = block_size
        self.conv1 = ConvBlock3D(dim, dim_out, kernel_size=(3, 3, 3), causal=True, padding=1)
        self.depth2space = depth_to_space3d
    
    def forward(self, x):
        """
        input_video: [B C T H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, self.block_size)
        if self.block_size[0] > 1:
            ### drop the first s-1 frames
            if out.shape[2] > 2: #video input
                out = torch.concat([out[:, :, [0], ...], out[:, :, 2:, ...]], dim=2)
            else:
                out = out[:, :, [0], ...] #only take the first frame
        return out

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps
    
    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)
       
        x = self.gn(x)
        x = scale * x + bias

        return x

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps
    
    def forward(self, x, quantizer):
        """
        input: [B C T H W]
        """
        B, C, _, _, _ = x.shape
        ### calcuate var for scale
        scale = rearrange(quantizer, "b c t h w -> b c (t h w)")
        scale = scale.var(dim=-1) + self.eps #not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c t h w -> b c (t h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1, 1)
       
        x = self.gn(x)
        x = scale * x + bias

        return x

if __name__ == "__main__":
    x = torch.randn(size = (2, 3, 17, 128, 128))
    encoder = Encoder(ch=128, in_channels=3, num_res_blocks=4, z_channels=18, out_ch=3, resolution=128)
    decoder = Decoder(out_ch=3, z_channels=18, num_res_blocks=4, ch=128, in_channels=3, resolution=128)
    z = encoder(x)
    out = decoder(z)