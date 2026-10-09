"""Masked regression losses for action chunks.

Every policy predicts a fixed-length chunk of future actions. Near the end of an
episode the chunk would run past the last step, so it is zero-padded and `is_pad`
marks the padding. All three policies need the same "average the error, but ignore
the padding" arithmetic, so it lives here once instead of in each of them.
"""

import torch.nn.functional as F


def _masked_mean(err, is_pad):
    """Average a per-element error over the valid (non-padding) entries.

    err:    (B, T, D) per-element error
    is_pad: (B, T) bool, True on the padding rows to ignore

    The denominator counts valid *elements*, not valid steps, hence the `* D`.
    `.clamp(min=1)` keeps an all-padding chunk from dividing by zero and returning
    NaN - only reachable with a zero-length episode, but a NaN loss is a miserable
    thing to debug.
    """
    valid = ~is_pad.unsqueeze(-1)   # (B, T, 1); broadcasts against the D axis
    denom = (valid.sum() * err.size(-1)).clamp(min=1)
    return (err * valid).sum() / denom


def masked_l1_loss(pred, target, is_pad):
    """Masked mean absolute error.

    pred, target: (B, T, D)
    is_pad:       (B, T) bool, True marks padding

    L1 rather than MSE for the MLP and ACT heads: the gripper dimension is
    effectively bimodal - open or closed, never in between - and the mean of two
    modes is a command the robot is never asked to hold.
    """
    return _masked_mean(F.l1_loss(pred, target, reduction="none"), is_pad)


def masked_mse_loss(pred, target, is_pad):
    """Masked mean squared error.

    Same shapes and mask semantics as masked_l1_loss. The diffusion head uses this
    because its target is the sampled noise, not the action.
    """
    return _masked_mean((pred - target) ** 2, is_pad)
