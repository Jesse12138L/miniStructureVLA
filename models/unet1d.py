"""1D U-Net used as the diffusion denoiser.

Takes the noisy action sequence (B, horizon, action_dim) plus a conditioning vector and
predicts the noise that was added to it. The conditioning vector is the timestep
embedding concatenated with the observation features, and it is injected into every
residual block as FiLM (one scale and one bias per channel).

Compared with the reference diffusion_policy (conditional_unet1d.py +
conv1d_components.py) the structure matches, except that einops is replaced by permute
and the channel transitions are driven straight from `down_dims`.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """The standard transformer-style sinusoidal encoding: an integer timestep becomes
    a vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # (B,) -> (B, dim)
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device)
                          * -(math.log(10000) / (half - 1)))
        args = t[:, None].float() * freqs[None]  # (B,1) * (1,half) -> (B,half)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish. The basic unit the whole network is built from."""

    def __init__(self, in_ch, out_ch, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            # padding = kernel_size // 2 leaves the sequence length unchanged
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1dBlocks with a residual connection, and the conditioning vector
    injected between them as FiLM."""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size, n_groups=8):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel_size, n_groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel_size, n_groups)
        # emits twice the output channels; split into scale and bias
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        # a 1x1 conv only when the channel count changes, so the residual lines up
        self.residual_conv = (nn.Conv1d(in_ch, out_ch, 1)
                              if in_ch != out_ch else nn.Identity())

    def forward(self, x, cond):  # x: (B, C, T), cond: (B, cond_dim)
        out = self.block1(x)
        # chunk(2, dim=-1) splits the last axis in half: first half scale, second bias,
        # each of shape (B, out_ch)
        scale, bias = self.cond_encoder(cond).chunk(2, dim=-1)
        # [..., None] makes them (B, out_ch, 1), which then broadcasts over time
        out = out * scale[..., None] + bias[..., None]
        out = self.block2(out)
        return out + self.residual_conv(x)


class Downsample1d(nn.Module):
    """Halve the sequence length: a stride-2 conv, with padding 1 to keep it exact."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Double the sequence length.

    A transposed conv with kernel 4, stride 2, padding 1. The kernel is 4 rather than 3
    because that is what makes the output exactly twice the input, mirroring
    Downsample1d - with kernel 3 the lengths would not line up with the skip tensors.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    """Three stages: down (residual blocks + downsample) -> mid (residual blocks) ->
    up (concatenate skip + upsample).

    The tensor it exposes is (B, horizon, input_dim); convolutions want channels first,
    so internally it becomes (B, input_dim, horizon) and is turned back at the end.

    With horizon=16 and three down stages the length goes 16 -> 8 -> 4 (the last down
    stage does not downsample), the mid blocks run at length 4, and the up path brings
    it back 4 -> 8 -> 16.
    """

    def __init__(self, input_dim, global_cond_dim, down_dims=(64, 128, 256),
                 step_embed_dim=128, kernel_size=5, n_groups=8):
        super().__init__()
        # channel count per stage: [action_dim] + down_dims, e.g. [4, 256, 512, 1024]
        dims = [input_dim] + list(down_dims)
        # consecutive pairs: (4,256), (256,512), (512,1024)
        pairs = list(zip(dims[:-1], dims[1:]))

        # the timestep is widened and narrowed again, like a small MLP over the encoding
        self.step_encoder = nn.Sequential(
            SinusoidalPosEmb(step_embed_dim),
            nn.Linear(step_embed_dim, step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(step_embed_dim * 4, step_embed_dim),
        )
        # what every residual block is conditioned on: timestep + observation features
        cond_dim = step_embed_dim + global_cond_dim

        def block(in_ch, out_ch):
            return ConditionalResidualBlock1D(in_ch, out_ch, cond_dim,
                                              kernel_size, n_groups)

        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(pairs):
            is_last = (i == len(pairs) - 1)
            self.down.append(nn.ModuleList([
                block(a, b),                                     # change width from a to b
                block(b, b),                                     # process again at that width
                nn.Identity() if is_last else Downsample1d(b),   # last stage keeps its length
            ]))

        mid_dim = dims[-1]
        self.mid = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])

        # each up stage concatenates the skip first (doubling the channels), then
        # upsamples - mirroring the two real downsamples on the way down
        self.up = nn.ModuleList()
        for a, b in reversed(pairs[1:]):
            self.up.append(nn.ModuleList([
                block(b * 2, a),    # catch the skip: 2b channels, squeezed back to a
                block(a, a),        # process again at that width
                Upsample1d(a),      # upsample, one-to-one with the down path
            ]))

        # the up path ends at down_dims[0] channels; a 1x1 conv maps back to actions
        start_dim = down_dims[0]
        self.final = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond):
        """sample: (B, horizon, input_dim) -> predicted noise, same shape"""
        # to channels-first. permute returns a non-contiguous view, which Conv1d accepts
        # fine - add .contiguous() here if you ever need to .view() the result.
        x = sample.permute(0, 2, 1)
        cond = torch.cat([self.step_encoder(timestep), global_cond], dim=-1)

        # one skip connection per down stage, pushed shallow to deep
        skips = []
        for block1, block2, down in self.down:
            x = block1(x, cond)
            x = block2(x, cond)
            skips.append(x)
            x = down(x)

        for block in self.mid:
            x = block(x, cond)

        for block1, block2, up in self.up:
            # skips is a LIFO stack: the down path pushed shallow to deep, so pop()
            # hands back the deepest pending skip - the one this up stage matches
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = up(x)

        return self.final(x).permute(0, 2, 1)
