"""Collect demonstration data from Meta-World MT1 environments using expert policies"""

import os
import argparse
import time
import numpy as np
import gymnasium as gym
import metaworld
import imageio.v2 as imageio
from metaworld.policies import ENV_POLICY_MAP
from utils.tokenizer import SimpleTokenizer
from utils.state_masking import mask_privileged_state

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="bin-picking-v3")
    parser.add_argument("--camera-name", type=str, default="corner3",
                        help="Meta-World camera: corner, corner2, corner3, corner4, topview, behindGripper, gripperPOV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100,
                        help="Peak RAM is roughly twice the raw image bytes, so it "
                             "depends on episode length: 100 push episodes is about "
                             "8 GB, 100 bin-picking episodes about 12 GB")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-path", type=str, default="data/metaworld_bc.npz")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional sleep between steps for visualization (seconds)")
    parser.add_argument("--instruction", type=str, default="push the object to the goal",
                        help="Fixed instruction for all episodes")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True,
                        help="Record the first --video-episodes episodes to MP4 "
                             "(on by default; disable with --no-save-video)")
    parser.add_argument("--video-dir", type=str, default="videos_collect",
                        help="Directory for recorded videos (used with --save-video). "
                             "Kept separate from scripts/test.py's outputs")
    parser.add_argument("--video-episodes", type=int, default=5,
                        help="How many of the first episodes to record")
    return parser.parse_args()


def extract_state(obs):
    """Flatten the Meta-World observation and zero the privileged fields."""
    state = np.asarray(obs, dtype=np.float32).ravel()
    return mask_privileged_state(state)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    env = gym.make(
        "Meta-World/MT1",
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array", # gives images
        camera_name=args.camera_name,
    )

    obs, info = env.reset(seed=args.seed)
    policy = ENV_POLICY_MAP[args.env_name]()

    images = []
    states = []
    actions = []
    texts = []

    # fixed instruction for this dataset
    instruction = args.instruction

    # index one past the last step of each episode. The ACT policy samples an action
    # chunk per step and that chunk must not run past the end of its episode, which
    # the flat arrays alone cannot tell you.
    episode_ends = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        # record only the first few episodes, to keep the frames list small
        recording = args.save_video and ep < args.video_episodes
        frames = []

        while not done and steps < args.max_steps:
            # The expert returns p * error with no upper bound, and the env clips it to
            # [-1, 1] before applying it. Clip here too so the stored target is the
            # action that was actually executed.
            action_raw = np.asarray(policy.get_action(obs), dtype=np.float32)
            action = np.clip(action_raw, env.action_space.low, env.action_space.high)

            # log current transition; env.render() comes out vertically flipped, so
            # flip it back (envs/metaworld_env.py applies the same flip for test.py)
            img = np.flipud(env.render()) # (H, W, 3) uint8
            state = extract_state(obs) # (state_dim,)

            # copy once and share that array between the dataset and the video frames
            frame = np.ascontiguousarray(img)
            images.append(frame)
            states.append(state.copy())
            actions.append(np.asarray(action, dtype=np.float32).copy())
            texts.append(instruction)

            if recording:
                frames.append(frame)

            # step env
            obs, reward, truncate, terminate, info = env.step(action)
            done = bool(truncate or terminate) or (int(info.get("success", 0)) == 1)
            steps += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {ep+1}/{args.episodes} finished after {steps} steps, success={int(info.get('success', 0))}")

        episode_ends.append(len(actions))

        if recording:
            video_path = os.path.join(args.video_dir, f"collect_{args.env_name}_ep{ep+1:03d}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"Saved video to {video_path}")

    env.close()

    # stack arrays
    images = np.stack(images, axis=0)   # (N, H, W, 3)
    states = np.stack(states, axis=0)   # (N, state_dim)
    actions = np.stack(actions, axis=0) # (N, action_dim)

    # tokenize instructions
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)
    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)
    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(text_ids_list):
        text_ids[i, :len(seq)] = np.array(seq, dtype=np.int64)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=tokenizer.vocab,
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
    )

    print("Saved Meta-World push dataset to", args.output_path)
    print("  images:", images.shape)
    print("  states:", states.shape)
    print("  actions:", actions.shape)
    print("  text_ids:", text_ids.shape)


if __name__ == "__main__":
    main()
