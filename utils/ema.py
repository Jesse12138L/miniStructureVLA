"""Exponential moving average of the model weights.

Common in diffusion training: evaluation uses this smoothed copy of the weights, which
gives steadier numbers than whatever the weights happen to be at the end of training.
The decay schedule is copied from the reference diffusion_policy - decay starts near 0
(a straight copy) and climbs towards max_value.
"""

import copy

import torch


class EMA:
    """Keeps a shadow copy of a model and folds the live weights into it each step."""

    def __init__(self, model, power=0.75, max_value=0.9999):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.power = power
        self.max_value = max_value
        self.step_count = 0

    def step(self, model):
        """Fold the current weights of `model` into the shadow copy.

        decay ramps up as (1 + step_count) ** -power: the first steps are dominated by
        the live weights, and the average gets stickier over time.

        Note this averages `parameters()` only, not `buffers()`, so any running
        statistics stored as buffers would not be smoothed.
        """
        decay = min(1 - (1 + self.step_count) ** -self.power, self.max_value)
        self.step_count += 1
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                # decay * old shadow + (1 - decay) * live weights, updated in place.
                # The reference writes this as the one-liner
                # ema_p.mul_(decay).add_(p, alpha=1 - decay); split out here because
                # the in-place ops and `alpha=` are the part worth seeing.
                ema_p.mul_(decay)                 # shadow * decay
                ema_p.add_(p, alpha=1 - decay)    # plus (1 - decay) * live weights

    def state_dict(self):
        """The shadow weights - this is what gets checkpointed and evaluated."""
        return self.model.state_dict()
