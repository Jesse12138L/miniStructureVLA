"""1D U-Net used as the diffusion denoiser.

Takes the noisy action sequence (B, horizon, action_dim) plus a conditioning vector,
and predicts the noise that was added to it. 条件向量 = 时间步嵌入与观测特征拼接而成，
送进每个残差块做 FiLM 调制（每通道一个 scale 和一个 bias）。

对比原版 diffusion_policy 的 conditional_unet1d.py + conv1d_components.py：结构一致，
只是把 einops 换成 permute，并且去掉了原版里那个恒为假的 is_last 判断（等价于每段都
上采样）。
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """标准 transformer 式正弦位置编码，把整数时间步变成向量。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # (B,) -> (B, dim)
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device)
                          * -(math.log(10000) / (half - 1)))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish。整套网络的基本单元。"""

    def __init__(self, in_ch, out_ch, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """两个 Conv1dBlock 加残差连接，条件向量以 FiLM 方式注入两者之间。"""

    def __init__(self, in_ch, out_ch, cond_dim, kernel_size, n_groups=8):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch, kernel_size, n_groups)
        self.block2 = Conv1dBlock(out_ch, out_ch, kernel_size, n_groups)
        # 输出两倍通道数，拆成 scale 和 bias
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual_conv = (nn.Conv1d(in_ch, out_ch, 1)
                              if in_ch != out_ch else nn.Identity())

    def forward(self, x, cond):  # x: (B, C, T)，cond: (B, cond_dim)
        out = self.block1(x)
        scale, bias = self.cond_encoder(cond).chunk(2, dim=-1)  # 各 (B, out_ch)
        out = out * scale[..., None] + bias[..., None]           # 沿时间轴广播
        out = self.block2(out)
        return out + self.residual_conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    """三段结构：down（残差块 + 下采样）→ mid（残差块）→ up（拼 skip + 上采样）。

    对外张量是 (B, horizon, input_dim)；卷积要求通道优先，所以内部转成
    (B, input_dim, horizon)，最后再转回来。
    """

    def __init__(self, input_dim, global_cond_dim, down_dims=(64, 128, 256),
                 step_embed_dim=128, kernel_size=5, n_groups=8):
        super().__init__()
        # 列表拼接： [4] + [256, 512, 1024]  →  [4, 256, 512, 1024]
        dims = [input_dim] + list(down_dims)
        pairs = list(zip(dims[:-1], dims[1:]))

        self.step_encoder = nn.Sequential(
            SinusoidalPosEmb(step_embed_dim),
            nn.Linear(step_embed_dim, step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(step_embed_dim * 4, step_embed_dim),
        )
        cond_dim = step_embed_dim + global_cond_dim

        def block(in_ch, out_ch):
            return ConditionalResidualBlock1D(in_ch, out_ch, cond_dim,
                                              kernel_size, n_groups)

        # 最后一段不再下采样
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(pairs):
            is_last = (i == len(pairs) - 1)
            self.down.append(nn.ModuleList([
                block(a, b),                                     # 把通道数从 a 改成 b
                block(b, b),                                     # 同宽度再加工一次
                nn.Identity() if is_last else Downsample1d(b),   # 最后一段不下采样
            ]))

        mid_dim = dims[-1]
        self.mid = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])

        # 每段先拼接 skip（通道数翻倍）再上采样，与 down 的两段下采样对称
        self.up = nn.ModuleList()
        for a, b in reversed(pairs[1:]):
            self.up.append(nn.ModuleList([
                block(b * 2, a),    # 先接住 skip：通道翻倍成 2b，再压回 a
                block(a, a),        # 同宽度再加工一次
                Upsample1d(a),      # 上采样，与 down 的下采样一一对称
            ]))

        # up 路径最后一段输出的是 down_dims[0] 个通道，最后再用 1×1 卷积投回动作维度
        start_dim = down_dims[0]
        self.final = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond):
        """sample: (B, horizon, input_dim) -> 同样形状的预测噪声"""
        x = sample.permute(0, 2, 1)
        cond = torch.cat([self.step_encoder(timestep), global_cond], dim=-1)

        skips = []
        for block1, block2, down in self.down:
            x = block1(x, cond)
            x = block2(x, cond)
            skips.append(x)
            x = down(x)

        for block in self.mid:
            x = block(x, cond)

        for block1, block2, up in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = up(x)

        return self.final(x).permute(0, 2, 1)
