"""Standalone viewer: watch a Meta-World MT1 expert policy drive a task.

This is not part of the training or evaluation pipeline - it exists so you can see what
a task looks like before collecting data from it. Run it directly to open a MuJoCo
window:

    python -m envs.metaworld_mt1

It loops until you press Ctrl-C, resetting each time the task succeeds.
"""

import time

import gymnasium as gym
# imported for its side effect: it registers the 'Meta-World/MT1' gym id
import metaworld
from metaworld.policies import ENV_POLICY_MAP


def main():
    seed = 42
    # ['corner', 'corner2', 'corner3', 'corner4', 'topview', 'behindGripper', 'gripperPOV']
    camera_name = 'corner2'
    env_name = 'bin-picking-v3'
    env = gym.make(
        'Meta-World/MT1',
        env_name=env_name,
        seed=seed,
        # 'human' opens a window and draws on every step, so no explicit render() call
        # is needed in the loop below
        render_mode='human',
        camera_name=camera_name,
    )

    obs, _ = env.reset()
    policy = ENV_POLICY_MAP[env_name]()

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
