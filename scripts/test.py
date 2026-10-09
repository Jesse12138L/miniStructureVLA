"""Evaluate a trained policy in a Meta-World MT1 environment.

Run it as a module, from the repo root, against a checkpoint from scripts/train.py:

    python -m scripts.test --checkpoint checkpoints/act_bp.pt --episodes 5

The checkpoint carries the config it was trained with, so no config file is needed here.
"""

import os
import argparse
import torch
import cv2
import imageio.v2 as imageio

from envs.metaworld_env import MetaWorldMT1Wrapper
from models.build import build_policy
from utils.chunking import action_queue_from_chunk
from utils.tokenizer import SimpleTokenizer
from utils.state_masking import mask_privileged_state


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy in Meta-World MT1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/act_bp.pt")
    parser.add_argument("--env-name", type=str, default="bin-picking-v3")
    parser.add_argument("--camera-name", type=str, default="corner3",
                        help="must match the camera the training data was collected with")
    parser.add_argument("--instruction", type=str, default="push the object to the goal",
                        help="must match the instruction the training data was collected "
                             "with; the tokenizer and the policy were both fit to it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda",
                        help="falls back to cpu when cuda is not available")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True,
                        help="record the first few episodes; pass --no-save-video to skip")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--video-per-outcome", type=int, default=3,
                        help="keep at most this many videos of each outcome")
    return parser.parse_args()


def load_policy(checkpoint_path: str, device: torch.device):
    """Rebuild the policy from the config the checkpoint carries, so this needs no
    config file. Returns (model, tokenizer, resize_to, norm)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "config" not in ckpt:
        raise SystemExit(f"{checkpoint_path} predates the config format; "
                         f"retrain it with scripts/train.py")

    cfg, vocab = ckpt["config"], ckpt["vocab"]
    model = build_policy(cfg, max(vocab.values()) + 1, ckpt["state_dim"], ckpt["action_dim"])
    model = model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm = ({k: torch.tensor(v, device=device) for k, v in ckpt["norm_stats"].items()}
            if ckpt["norm_stats"] else None)
    print(f"[test] policy={cfg['policy']}  resize_to={cfg['data']['resize_to']}  "
          f"chunk={cfg['data']['chunk_size']}")
    return model, SimpleTokenizer(vocab=vocab), cfg["data"]["resize_to"], norm


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load model + tokenizer
    print(f"[test] Loading checkpoint from {args.checkpoint}")
    model, tokenizer, resize_to, norm = load_policy(args.checkpoint, device)

    # encode instruction
    instr_tokens = tokenizer.encode(args.instruction)
    text_ids = torch.tensor(instr_tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, T_text)

    # environment
    env = MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )

    print(f"[test] Meta-World MT1 env: {args.env_name} (camera: {args.camera_name})")
    print(f"[test] state_dim={env.state_dim}, action_dim={env.action_dim}, obs_shape={env.obs_shape}")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    # evaluation
    n_success = 0
    saved_videos = {"success": 0, "fail": 0}
    for ep in range(args.episodes):
        img, state, info = env.reset()
        step = 0
        ep_reward = 0.0
        ep_success = 0

        frames = [img.copy()]

        # ACT predicts a chunk; plan once and feed the actions to the env one at a
        # time, re-planning when the queue runs dry. A single-step policy puts one
        # action in the queue, so the loop below is the same either way.
        action_queue = []

        done = False
        while not done and step < args.max_steps:
            if not action_queue:
                # resize to the resolution the policy was trained on; `img` stays
                # full-resolution for the video frames
                obs_img = img
                if obs_img.shape[0] != resize_to or obs_img.shape[1] != resize_to:
                    obs_img = cv2.resize(obs_img, (resize_to, resize_to))

                img_t = torch.from_numpy(obs_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1, 3, H, W)

                obs_state = mask_privileged_state(state)
                state_t = torch.from_numpy(obs_state).float().unsqueeze(0)  # (1, state_dim)

                img_t = img_t.to(device)
                state_t = state_t.to(device)
                # normalise after the move: the stats live on the model's device
                if norm is not None:
                    state_t = (state_t - norm["state_mean"]) / norm["state_std"]

                # inference
                with torch.no_grad():
                    action_t = model.act(img_t, text_ids, state_t)
                chunk = action_t.squeeze(0).cpu().numpy()   # (chunk, action_dim)
                if norm is not None:
                    chunk = chunk * norm["action_std"].cpu().numpy() \
                                  + norm["action_mean"].cpu().numpy()
                action_queue = action_queue_from_chunk(chunk)

            action_np = action_queue.pop(0)

            # step environment
            img, state, reward, done, info = env.step(action_np)
            ep_reward += reward
            step += 1

            # Meta-World signals task completion via info["success"], not via the env's
            # terminate flag, so fold it into the loop condition. This mirrors the done
            # check in scripts/collect_data.py.
            if int(info.get("success", 0)) == 1:
                ep_success = 1
                done = True

            frames.append(img.copy())

        n_success += ep_success
        print(
            f"[test] Episode {ep+1}/{args.episodes}: "
            f"reward={ep_reward:.3f}, steps={step}, success={ep_success}"
        )

        # save video, but only the first few of each outcome so a long run does not
        # dump one file per episode
        outcome = "success" if ep_success else "fail"
        if args.save_video and saved_videos[outcome] < args.video_per_outcome:
            saved_videos[outcome] += 1
            video_path = os.path.join(
                args.video_dir, f"{args.env_name}_ep{ep+1:03d}_{outcome}.mp4")
            with imageio.get_writer(video_path, fps=20) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[test] Saved video to {video_path} "
                  f"({outcome} {saved_videos[outcome]}/{args.video_per_outcome})")

    print(
        f"[test] Success rate: {n_success}/{args.episodes} "
        f"= {100.0 * n_success / args.episodes:.1f}%"
    )

    env.close()
    print("[test] Done.")


if __name__ == "__main__":
    main()
