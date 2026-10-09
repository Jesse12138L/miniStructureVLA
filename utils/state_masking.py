"""Masking of privileged state fields in Meta-World observations.

Meta-World packs its 39-dim observation as

    [0:18]   current frame: [0:3] end-effector pos, [3] gripper, [4:18] object poses
    [18:36]  previous frame, same layout (the observation is frame-stacked)
    [36:39]  goal position

The end-effector and gripper readings are proprioception a real robot has for free.
The object poses and goal position are privileged: they hand the policy the whole task
geometry, so it can solve the task without looking at the image. Zeroing both leaves 8
of 39 dims.

Both scripts/collect_data.py and scripts/test.py call this, so the slices live here
rather than in either one - a disagreement would train and evaluate the policy on
different inputs, silently, since masking does not change the shape.
"""

import numpy as np

PRIVILEGED_STATE_SLICES = (
    slice(4, 18),    # current frame: object poses, two slots of (position + quaternion)
    slice(22, 36),   # previous frame: the same block
    slice(36, 39),   # goal position
)


EXPECTED_STATE_DIM = 39


def mask_privileged_state(state):
    """Zero the object poses and the goal position, leaving only proprioception."""
    state = np.array(state, dtype=np.float32, copy=True)
    if state.shape[-1] != EXPECTED_STATE_DIM:
        # The slices are positional, so a differently shaped observation would be
        # zeroed in the wrong places - and silently, since nothing downstream checks.
        raise ValueError(
            f"mask_privileged_state expects the {EXPECTED_STATE_DIM}-dim Meta-World "
            f"observation, got {state.shape[-1]} dims; the slice indices would zero "
            f"the wrong fields"
        )
    for sl in PRIVILEGED_STATE_SLICES:
        state[sl] = 0.0
    return state
