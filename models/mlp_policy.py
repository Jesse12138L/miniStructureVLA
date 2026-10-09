"""Plain MLP policy.

Encodes the image, instruction and robot state, concatenates the three tokens, and
turns them into an action with one MLP - no sampling loop, no attention. It is the
simplest thing that can consume the same encoders as the other policies, which is
what makes it a baseline worth comparing against.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageEncoderTinyCNNSpatial, StateEncoderMLP, TextEncoderTinyGRU


class MLPPolicy(nn.Module):
    def __init__(self, vocab_size, state_dim, action_dim, chunk_size=1, d_model=128,
                 img_size=64, hidden=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.img_encoder = ImageEncoderTinyCNNSpatial(d_model, img_size)
        self.txt_encoder = TextEncoderTinyGRU(vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim, d_model=d_model)
        # LayerNorm after the first block keeps the fused embedding at a stable scale
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk_size * action_dim),
        )

    def _embed(self, image, text_ids, state):
        return torch.cat([self.img_encoder(image),
                          self.txt_encoder(text_ids),
                          self.state_encoder(state)], dim=-1)

    def loss(self, image, text_ids, state, chunk, is_pad):
        """
        chunk:  (B, chunk_size, action_dim), zero-padded at the end of an episode
        is_pad: (B, chunk_size) bool, True on that padding

        L1 rather than MSE because the gripper dimension is effectively bimodal - it
        is open or closed, never in between - and the mean of two modes is a value
        the robot is never commanded to hold.
        """
        pred = self.head(self._embed(image, text_ids, state))
        pred = pred.view(-1, self.chunk_size, self.action_dim)
        valid = ~is_pad.unsqueeze(-1)
        err = F.l1_loss(pred, chunk, reduction='none') * valid
        return err.sum() / (valid.sum() * self.action_dim).clamp(min=1)

    @torch.no_grad()
    def act(self, image, text_ids, state):
        self.eval()
        pred = self.head(self._embed(image, text_ids, state))
        return pred.view(-1, self.chunk_size, self.action_dim)
