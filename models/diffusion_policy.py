"""Diffusion Policy.

Predicts a whole action sequence by denoising it from pure noise. Training adds noise
to real actions and has the U-Net predict what was added; inference starts from pure
noise and denoises step by step to get an action sequence.

The structure follows real-stanford/diffusion_policy: ConditionalUnet1D + DDPM + global
conditioning. The numbers match its config too - cosine beta table, T=100, clip_sample,
epsilon prediction, fixed_small variance. `diffusers` is not used; the scheduler here
is hand-written so the whole thing stays readable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import StateEncoderMLP, TextEncoderTinyGRU
from .losses import masked_mse_loss
from .unet1d import ConditionalUnet1D


class DiffusionSchedule(nn.Module):
    """The DDPM noise table and the sampling update.

    The reference config lists beta_start/beta_end, but with the squaredcos_cap_v2
    schedule those two values are ignored - the cosine table comes straight out of the
    alpha_bar function - so they are not parameters here.

    It is an nn.Module so the noise tables ride along when the model moves to the GPU:
    only buffers registered on a module are carried by `.to()`.
    """

    # clip_sample_range is 1.0 in the reference implementation, because there the
    # LinearNormalizer squashes actions *by range* into [-1, 1], so x0 is inside that
    # box by construction. This project uses z-score normalisation instead, and the
    # normalised actions sit around +-3, so clipping to +-1 shaves a slice off every
    # dimension: measured, the predicted std came out 0.5-0.6x the true std and the L1
    # was 4x worse than with clipping off. 4.0 is effectively no clipping (the data
    # reaches +-3.1). The point of clipping is "do not leave the data range", and this
    # value achieves that.
    def __init__(self, num_train_timesteps=100, max_beta=0.999, clip_sample_range=4.0):
        super().__init__()
        self.T = num_train_timesteps
        self.clip = clip_sample_range

        def alpha_bar(t):
            """Cosine schedule: signal retained at time t, with t normalised to [0, 1].

            The +0.008 offset keeps beta from being ~0 near t=0. The curve runs from
            alpha_bar(0) = 1 (clean) to alpha_bar(1) = 0 (pure noise).
            """
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        # Discretise the continuous curve into T betas: each beta is the fraction of the
        # remaining signal that alpha_bar loses over one step, capped at max_beta.
        betas = [min(1 - alpha_bar((i + 1) / self.T) / alpha_bar(i / self.T), max_beta)
                 for i in range(self.T)]
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        # alpha_bar[t] = product of (1 - beta) up to t, i.e. the signal retained at t
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - self.betas, dim=0))

    def add_noise(self, x0, noise, t):
        """Forward process: the noisy sample at step t.

        t: (B,) one timestep per sample
        """
        ab = self.alpha_bar[t].view(-1, 1, 1)  # (B,1,1) so it broadcasts over horizon and action dims
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def step(self, eps_pred, t, x_t):
        """One reverse step: estimate x0 from x_t, clip it, then take a posterior step.

        t: a plain int, shared by the whole batch
        """
        ab = self.alpha_bar[t]
        # invert the forward equation to get x0 back out of the predicted noise
        x0 = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
        x0 = x0.clamp(-self.clip, self.clip)
        if t == 0:
            return x0  # no earlier timestep to step to; the estimate is the answer
        ab_prev = self.alpha_bar[t - 1]
        var = (1 - ab_prev) / (1 - ab) * self.betas[t]  # variance_type=fixed_small
        # posterior mean = sqrt(ab_prev)*x0 + sqrt(1 - ab_prev - var)*eps.
        # The eps term has to have var subtracted from it: that share of the variance is
        # carried by the independent noise sqrt(var)*z added below. Writing
        # sqrt(1 - ab_prev) instead injects an extra var of independent noise on top of
        # the correct amount.
        mean = ab_prev.sqrt() * x0 + (1 - ab_prev - var).clamp(min=0).sqrt() * eps_pred
        return mean + var.sqrt() * torch.randn_like(x_t)


class SpatialSoftmax(nn.Module):
    """Compress a feature map into the expected (x, y) of a few keypoints, instead of
    averaging the whole spatial extent away.

    Each keypoint gets a one-channel heatmap, softmaxed over the spatial axis, and the
    expected x/y coordinates are read off it. Output is (B, num_kp*2). The difference
    from global average pooling is that this can express *where* in the frame something
    is - which is exactly what a spatial task needs.
    """

    def __init__(self, in_ch, num_kp, grid):
        super().__init__()
        self.heatmap = nn.Conv2d(in_ch, num_kp, kernel_size=1)
        coords = torch.linspace(-1, 1, grid)
        # Both buffers line up with the flattened (grid, grid) heatmap: for flat index
        # i, x = i % grid and y = i // grid. Swapping repeat for repeat_interleave
        # transposes the coordinate grid, which is a silent bug rather than a crash.
        self.register_buffer("grid_x", coords.repeat(grid))             # tiled -> coords[i % grid]
        self.register_buffer("grid_y", coords.repeat_interleave(grid))  # each x grid -> coords[i // grid]

    def forward(self, x):  # (B, C, g, g)
        h = torch.softmax(self.heatmap(x).flatten(2), dim=-1)  # (B, K, g*g)
        # expected coordinate per keypoint, then concatenate x and y
        return torch.cat([(h * self.grid_x).sum(-1),
                          (h * self.grid_y).sum(-1)], dim=-1)  # (B, 2K)


class DiffusionPolicy(nn.Module):
    """Denoise a whole action sequence, conditioned on the current observation.

    `horizon` is how many future actions are predicted at once; `n_action_steps` is how
    many of them are actually executed before re-planning (the reference's 16/8 split).
    """

    def __init__(self, vocab_size, state_dim, action_dim, chunk_size=16,
                 n_action_steps=8, d_model=128, img_size=128, num_kp=32,
                 down_dims=(64, 128, 256), step_embed_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = chunk_size
        self.n_action_steps = n_action_steps

        # Image: the same three convs as the MLP/ACT policies, but the tail is a spatial
        # softmax rather than a flatten.
        # [SHARED CONV TRUNK] - the same three convs as encoders / act_policy
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.spatial_softmax = SpatialSoftmax(128, num_kp, grid=img_size // 8)
        self.txt_encoder = TextEncoderTinyGRU(vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim, d_model=d_model)

        # the conditioning vector the U-Net sees: image keypoints + text + state
        obs_dim = num_kp * 2 + 2 * d_model
        self.unet = ConditionalUnet1D(action_dim, obs_dim, down_dims, step_embed_dim)
        self.schedule = DiffusionSchedule()

    def _encode_obs(self, image, text_ids, state):
        """Observation -> the (B, obs_dim) global conditioning vector."""
        # [SHARED CONV TRUNK] forward pass, matching models/encoders.py
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return torch.cat([self.spatial_softmax(x),
                          self.txt_encoder(text_ids),
                          self.state_encoder(state)], dim=-1)

    def loss(self, image, text_ids, state, chunk, is_pad):
        """
        chunk:  (B, horizon, action_dim), zero-padded at the end of an episode
        is_pad: (B, horizon) bool, True on that padding

        The target is the *noise*, not the action itself (epsilon prediction). The
        padding is noised along with everything else but masked out of the loss - the
        reference implementation does not mask here and simply learns the zeros as a
        target; this project keeps the masking the other two policies use.
        """
        cond = self._encode_obs(image, text_ids, state)
        noise = torch.randn_like(chunk)
        t = torch.randint(0, self.schedule.T, (chunk.size(0),), device=chunk.device)
        eps_pred = self.unet(self.schedule.add_noise(chunk, noise, t), t, cond)
        return masked_mse_loss(eps_pred, noise, is_pad)

    @torch.no_grad()
    def act(self, image, text_ids, state):
        """Run the full sampling loop from pure noise.

        Only the first n_action_steps come back (the reference's 16/8 split): the caller
        executes those, then re-plans from a fresh observation.
        """
        self.eval()
        cond = self._encode_obs(image, text_ids, state)
        x = torch.randn(cond.size(0), self.horizon, self.action_dim, device=cond.device)
        for t in reversed(range(self.schedule.T)):
            eps = self.unet(x, torch.full((cond.size(0),), t, device=cond.device), cond)
            x = self.schedule.step(eps, t, x)
        return x[:, :self.n_action_steps]
