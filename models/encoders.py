"""Encoders for images, text, and states.

All three policies (MLP / ACT / Diffusion) start from the same place: turn each
modality into one `d_model` vector with a small, readable network, then let the policy
decide how to fuse them.

This file is the canonical definition of the shared convolutional trunk - see the
[SHARED CONV TRUNK] block below.
"""

import torch.nn as nn
import torch.nn.functional as F

# === SHARED CONV TRUNK - canonical definition ===============================
# All three policies run an identical image trunk before they diverge:
#
#     input (B, 3, S, S)
#       conv1     3 -> 32    k=5 s=2 p=2   ReLU    -> (B,  32, S/2, S/2)
#       conv2    32 -> 64    k=3 s=2 p=1   ReLU    -> (B,  64, S/4, S/4)
#       conv3    64 -> 128   k=3 s=2 p=1   ReLU    -> (B, 128, S/8, S/8)
#
# It is copy-pasted rather than shared on purpose: pulling it into one module would
# rename each policy's conv1/conv2/conv3 parameters and invalidate the trained
# checkpoints in checkpoints/. The copies are:
#
#     models/encoders.py          ImageEncoderTinyCNNSpatial  (below)
#     models/act_policy.py        ACTPolicy.__init__ / _memory
#     models/diffusion_policy.py  DiffusionPolicy.__init__ / _encode_obs
#
# Edit all three together. Each site is tagged with  # [SHARED CONV TRUNK]
# ============================================================================


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
        # [SHARED CONV TRUNK] - the same three convs as act_policy / diffusion_policy
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.reduce = nn.Conv2d(128, out_channels, kernel_size=1)
        feature = img_size // 8
        self.proj = nn.Linear(out_channels * feature * feature, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # [SHARED CONV TRUNK] forward pass, matching the two policies
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.reduce(x)      # (B, C, S, S) - 1x1, no spatial mixing
        x = x.flatten(1)        # (B, C*S*S); unlike mean-pooling, position survives
        x = self.proj(x)
        x = self.ln(x)
        return x  # (B, d_model)


class TextEncoderTinyGRU(nn.Module):
    """Instruction text -> one (B, d_model) token.

    Word embedding, then a single-layer GRU, then LayerNorm. The last hidden state is
    the summary, so the instruction length does not have to be fixed. A GRU rather
    than a mean over word vectors because word order carries meaning here ("push the
    puck to the goal" is not "push the goal to the puck").
    """

    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        x = self.embed(token_ids)  # (B, T, d_word)
        # A GRU returns (per-step outputs, final hidden state). Only the second is
        # wanted, so the per-step outputs go to `_`.
        _, h_last = self.gru(x)
        x = h_last[0]  # h_last is (num_layers=1, B, d_model); [0] drops the layer axis
        x = self.ln(x)
        return x


class StateEncoderMLP(nn.Module):
    """Robot state -> one (B, d_model) token.

    A two-layer MLP with LayerNorm. `state_dim` is the full 39-dim Meta-World vector
    the environment reports; which of its fields the policy actually gets to see is
    decided by utils/state_masking.py, before the data reaches the model. The z-score
    normalisation is applied in scripts/train.py.
    """

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
