"""Encoders for images, text, and states."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderTinyCNNSpatial(nn.Module):
    """Three stride-2 convolutions, then the feature map is flattened so each spatial
    location keeps its own slot in the output vector instead of being averaged away.

    The 1x1 conv first shrinks the channels to keep `proj` affordable - flattening all
    128 channels would need ~100x more weights. Output is out_channels * (img_size/8)^2
    values, i.e. 128 at img_size=64 and 512 at 128.
    """

    def __init__(self, d_model=128, img_size=64, out_channels=2):
        super().__init__()
        if img_size % 8 != 0:
            raise ValueError(f"img_size {img_size} must be divisible by 8 (three "
                             f"stride-2 convolutions)")
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.reduce = nn.Conv2d(128, out_channels, kernel_size=1)
        feature = img_size // 8
        self.proj = nn.Linear(out_channels * feature * feature, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.reduce(x)      # (B, C, S, S) - 1x1, no spatial mixing
        x = x.flatten(1)        # (B, C*S*S)
        x = self.proj(x)
        x = self.ln(x)
        return x  # (B, d_model)


class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        x = self.embed(token_ids)  # (B, T, d_word)
        _, h_last = self.gru(x)
        x = h_last[0]  # (B, d_model)
        x = self.ln(x)
        return x


class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.net(s)
        x = self.ln(x)
        return x
