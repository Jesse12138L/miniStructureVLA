"""Diffusion Policy.

Predicts a whole action sequence by denoising it from pure noise. 训练时给真实动作加噪、
让 U-Net 预测加了什么噪声；推理时从纯噪声出发反复去噪，得到一段动作。

结构和 real-stanford/diffusion_policy 一致：ConditionalUnet1D + DDPM + 全局条件。
数值对齐它的配置——cosine beta 表、T=100、clip_sample、epsilon 预测、fixed_small 方差。
没有引入 diffusers，调度器是手写的。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import StateEncoderMLP, TextEncoderTinyGRU
from .unet1d import ConditionalUnet1D


class DiffusionSchedule(nn.Module):
    """DDPM 的噪声表和采样更新。

    原版配置里虽然写了 beta_start/beta_end，但选了 squaredcos_cap_v2 的表之后这两个值是
    被忽略的——cosine 表由 alpha_bar 函数直接算出来，所以这里不接这两个参数。

    做成 nn.Module 是为了让噪声表跟着模型一起搬到 GPU（注册成 buffer 才会被 .to() 搬运）。
    """

    # clip_sample_range 原版是 1.0，因为原版用 LinearNormalizer 把动作按「范围」压到
    # [-1, 1]，x0 本来就在盒子里。本项目沿用 z-score，归一化后的动作在 ±3 附近，再按
    # ±1 裁会把每一维都削掉一截：实测预测的 std 只有真值的 0.5~0.6 倍，L1 是关掉裁剪
    # 时的 4 倍。取 4.0 相当于不裁（数据最远到 ±3.1），裁剪的原意是「别跑出数据范围」，
    # 这个值就够。
    def __init__(self, num_train_timesteps=100, max_beta=0.999, clip_sample_range=4.0):
        super().__init__()
        self.T = num_train_timesteps
        self.clip = clip_sample_range

        def alpha_bar(t):  # 加0.008避避免 t=0 附近 β小到几乎为0。cos->,t=0,alpha = 1,t = 1,alpha = 0
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = [min(1 - alpha_bar((i + 1) / self.T) / alpha_bar(i / self.T), max_beta)
                 for i in range(self.T)]#把连续曲线离散化成 100 个 β
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - self.betas, dim=0))

    def add_noise(self, x0, noise, t):  # t: (B,) 每样本一个时间步    #前向
        ab = self.alpha_bar[t].view(-1, 1, 1)  # (B,1,1)，沿 horizon 和 action 维广播
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def step(self, eps_pred, t, x_t):  # t: 标量 int
        """一步反向去噪：先从 x_t 估出 x0 并裁剪，再算后验均值。"""
        ab = self.alpha_bar[t]
        x0 = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
        x0 = x0.clamp(-self.clip, self.clip)
        if t == 0:
            return x0  # 最后一步没有更早的时间步，直接输出估出的 x0
        ab_prev = self.alpha_bar[t - 1]
        var = (1 - ab_prev) / (1 - ab) * self.betas[t]  # variance_type=fixed_small
        # 后验均值 = sqrt(ab_prev)*x0 + sqrt(1 - ab_prev - var)*eps。
        # eps 项必须减掉 var：这部分方差是由下面那支独立噪声 sqrt(var)*z 承担的。
        # 写成 sqrt(1 - ab_prev) 就等于在正确方差之外又多注入了一份 var 的独立噪声。
        mean = ab_prev.sqrt() * x0 + (1 - ab_prev - var).clamp(min=0).sqrt() * eps_pred
        return mean + var.sqrt() * torch.randn_like(x_t)


class SpatialSoftmax(nn.Module):
    """把特征图压成一组关键点的期望坐标，而不是平均掉整个空间。

    对每个关键点算一张单通道热力图，在空间维度做 softmax，再求 x/y 的期望。
    输出 (B, num_kp*2)。和全局平均池化的区别在于：它能表达"东西在画面哪个位置"。
    """

    def __init__(self, in_ch, num_kp, grid):
        super().__init__()
        self.heatmap = nn.Conv2d(in_ch, num_kp, kernel_size=1)
        coords = torch.linspace(-1, 1, grid)
        self.register_buffer("grid_x", coords.repeat(grid))            # 展平后 x 变化最快
        self.register_buffer("grid_y", coords.repeat_interleave(grid))

    def forward(self, x):  # (B, C, g, g)
        h = torch.softmax(self.heatmap(x).flatten(2), dim=-1)  # (B, K, g*g)
        return torch.cat([(h * self.grid_x).sum(-1),
                          (h * self.grid_y).sum(-1)], dim=-1)  # (B, 2K)


class DiffusionPolicy(nn.Module):
    def __init__(self, vocab_size, state_dim, action_dim, chunk_size=16,
                 n_action_steps=8, d_model=128, img_size=128, num_kp=32,
                 down_dims=(64, 128, 256), step_embed_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = chunk_size
        self.n_action_steps = n_action_steps

        # 图像：复用 MLP/ACT 那三层卷积，末端换成 spatial softmax
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.spatial_softmax = SpatialSoftmax(128, num_kp, grid=img_size // 8)
        self.txt_encoder = TextEncoderTinyGRU(vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim, d_model=d_model)

        obs_dim = num_kp * 2 + 2 * d_model
        self.unet = ConditionalUnet1D(action_dim, obs_dim, down_dims, step_embed_dim)
        self.schedule = DiffusionSchedule()

    def _encode_obs(self, image, text_ids, state):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return torch.cat([self.spatial_softmax(x),
                          self.txt_encoder(text_ids),
                          self.state_encoder(state)], dim=-1)

    def loss(self, image, text_ids, state, chunk, is_pad):
        """
        chunk:  (B, horizon, action_dim)，episode 末端补零
        is_pad: (B, horizon) bool

        目标是**噪声**而不是动作本身（epsilon 预测）。补零位置也一起加噪，但被掩码排除在
        损失之外——原版这里不掩码、直接把补零当目标学，本项目沿用已有的掩码做法。
        """
        cond = self._encode_obs(image, text_ids, state)
        noise = torch.randn_like(chunk)
        t = torch.randint(0, self.schedule.T, (chunk.size(0),), device=chunk.device)
        eps_pred = self.unet(self.schedule.add_noise(chunk, noise, t), t, cond)

        valid = ~is_pad.unsqueeze(-1)
        err = (eps_pred - noise) ** 2 * valid
        return err.sum() / (valid.sum() * self.action_dim).clamp(min=1)

    @torch.no_grad()
    def act(self, image, text_ids, state):
        """从纯噪声出发跑完整采样。只返回前 n_action_steps 步（原版 16/8 两段式）。"""
        self.eval()
        cond = self._encode_obs(image, text_ids, state)
        x = torch.randn(cond.size(0), self.horizon, self.action_dim, device=cond.device)
        for t in reversed(range(self.schedule.T)):
            eps = self.unet(x, torch.full((cond.size(0),), t, device=cond.device), cond)
            x = self.schedule.step(eps, t, x)
        return x[:, :self.n_action_steps]
