"""Standalone viewer: watch a Meta-World MT1 expert policy drive a task.

This is not part of the training or evaluation pipeline - it exists so you can see what
a task looks like before collecting data from it. Run it directly to open a MuJoCo
window:

    python -m envs.metaworld_mt1

It loops until you press Ctrl-C, resetting each time the task succeeds. Change
ENV_NAME below to look at a different task.
"""

import time

import gymnasium as gym
# imported for its side effect: it registers the 'Meta-World/MT1' gym id
import metaworld
from metaworld.policies import ENV_POLICY_MAP

ENV_NAME = 'bin-picking-v3'
# ['corner', 'corner2', 'corner3', 'corner4', 'topview', 'behindGripper', 'gripperPOV']
# Note the collected data uses 'corner3' (see the README); this default is the upstream
# script's, so switch it if you want the view the policies were actually trained on.
CAMERA_NAME = 'corner2'
SEED = 42

# All 50 Meta-World MT1 tasks, grouped by what the arm has to do. Every one of them has
# a scripted expert, and every one uses the same 39-dim observation and 4-dim action, so
# any of these can be collected with scripts/collect_data.py and trained on without
# touching a line of model code. Just point ENV_NAME above at one of them.
TASKS = {
    "push / slide / sweep": [
        "push-v3", "push-back-v3", "push-wall-v3",
        "plate-slide-v3", "plate-slide-side-v3", "plate-slide-back-v3",
        "plate-slide-back-side-v3", "sweep-v3", "sweep-into-v3",
    ],
    "pick / place / insert": [
        "pick-place-v3", "pick-place-wall-v3", "pick-out-of-hole-v3", "bin-picking-v3",
        "shelf-place-v3", "peg-insert-side-v3", "peg-unplug-side-v3", "hand-insert-v3",
        "box-close-v3", "disassemble-v3", "assembly-v3", "hammer-v3",
        "basketball-v3", "soccer-v3",
    ],
    "drawers / doors / windows": [
        "drawer-open-v3", "drawer-close-v3",
        "door-open-v3", "door-close-v3", "door-lock-v3", "door-unlock-v3",
        "window-open-v3", "window-close-v3",
    ],
    "handles / levers / sticks": [
        "handle-press-v3", "handle-press-side-v3",
        "handle-pull-v3", "handle-pull-side-v3",
        "lever-pull-v3", "stick-pull-v3", "stick-push-v3",
    ],
    "buttons / dials / faucets": [
        "button-press-v3", "button-press-topdown-v3",
        "button-press-wall-v3", "button-press-topdown-wall-v3",
        "coffee-button-v3", "coffee-pull-v3", "coffee-push-v3",
        "dial-turn-v3", "faucet-open-v3", "faucet-close-v3",
    ],
    "reach (nothing to move)": [
        "reach-v3", "reach-wall-v3",
    ],
}

# The tasks that need several coordinated contact stages instead of one push, which makes
# them the interesting failures. Bin-picking needs grasp, carry and release in sequence;
# the insert and assembly tasks need the grasp to line up precisely. Worth watching here
# before concluding that a low success rate is a bug in the policy.
HARD_TASKS = [
    "bin-picking-v3", "assembly-v3", "disassemble-v3", "hammer-v3",
    "stick-pull-v3", "peg-insert-side-v3", "hand-insert-v3", "shelf-place-v3",
    "box-close-v3", "door-lock-v3", "coffee-pull-v3", "pick-out-of-hole-v3",
]


def main():
    env = gym.make(
        'Meta-World/MT1',
        env_name=ENV_NAME,
        seed=SEED,
        # 'human' opens a window and draws on every step, so no explicit render() call
        # is needed in the loop below
        render_mode='human',
        camera_name=CAMERA_NAME,
    )

    obs, _ = env.reset()
    policy = ENV_POLICY_MAP[ENV_NAME]()

    try:
        while True:
            action = policy.get_action(obs)
            obs, _, _, _, info = env.step(action)
            time.sleep(1 / 60)  # pace it at roughly 60 fps so the window is watchable

            # Meta-World signals success through info, not by terminating the episode
            if int(info['success']) == 1:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
