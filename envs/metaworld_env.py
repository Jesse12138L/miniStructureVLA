import gymnasium as gym
import numpy as np
# imported for its side effect: it registers the 'Meta-World/MT1' gym id
import metaworld


class MetaWorldMT1Wrapper:
    """A thin adapter that turns a Meta-World MT1 env into (image, state) pairs.

        reset(seed=None)  -> (image, state, info)
        step(action)      -> (image, state, reward, done, info)
        close()

    Three things worth knowing before using it:

    * `done` reflects only the env's truncate/terminate flags. Meta-World reports task
      success through info["success"] and does *not* terminate the episode when the task
      succeeds, so waiting on `done` alone runs every episode to the step limit. Both
      callers (scripts/collect_data.py, scripts/test.py) add the success check
      themselves.
    * Constructing the wrapper already resets the env once and renders a frame, to
      measure state_dim / action_dim / obs_shape. Construction has side effects.
    * The state comes back raw and unmasked. Which fields the policy may see is decided
      by utils/state_masking.py at the call sites.
    """

    def __init__(self, env_name='push-v3', seed=42, render_mode='rgb_array', camera_name='topview'):
        self.env = gym.make(
            'Meta-World/MT1',
            env_name=env_name,
            seed=seed,
            render_mode=render_mode,
            camera_name=camera_name
        )
        self.render_mode = render_mode

        # one reset and one render purely to read the shapes off the env
        obs, _ = self.env.reset()
        self.state_dim = self._extract_state(obs).shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.obs_shape = self._get_image().shape

    def _extract_state(self, obs):
        """Turn whatever the env calls an observation into a flat float32 vector.

        MT1 hands back a plain array, which is the branch actually taken here. The dict
        branches are kept for Meta-World variants that split the observation into
        robot_state / object_state.
        """
        if isinstance(obs, dict):
            if "observation" in obs:
                state = obs["observation"]
            elif "robot_state" in obs or "object_state" in obs:
                state_parts = []
                if "robot_state" in obs:
                    state_parts.append(obs["robot_state"])
                if "object_state" in obs:
                    state_parts.append(obs["object_state"])
                state = np.concatenate(state_parts, axis=-1)
            else:
                raise KeyError(
                    f"No suitable state keys in observation dict. "
                    f"Available keys: {list(obs.keys())}"
                )
        else:
            state = obs
        return np.asarray(state, dtype=np.float32)

    def _get_image(self):
        """Render one frame as uint8 HWC.

        render() comes out vertically flipped, so it is flipped back here;
        scripts/collect_data.py applies the same flip. A disagreement between the two
        would train and evaluate the policy on mirrored images, silently.
        """
        img = np.flipud(self.env.render()).astype(np.uint8)
        return img

    def reset(self, seed=None):
        """Returns (image, state, info). `seed=None` means "do not reseed"."""
        obs, info = self.env.reset(seed=seed)
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, info

    def step(self, action):
        """Returns (image, state, reward, done, info).

        The raw gym tuple is (obs, reward, truncate, terminate, info); the two done
        flags are fused into a single `done` here, which is why the order looks
        different. See the class docstring about `done` not covering success.
        """
        obs, reward, truncate, terminate, info = self.env.step(action)
        done = truncate or terminate
        state = self._extract_state(obs)
        image = self._get_image()
        return image, state, reward, done, info

    def close(self):
        self.env.close()
