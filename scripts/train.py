"""Train a policy from a YAML config in configs/.

The config defines the experiment; the CLI only carries two run-level overrides. The
whole config goes into the checkpoint, so scripts/test.py can rebuild the model
without needing the config file.
"""

import argparse
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from models.build import build_policy
from utils.config import load
from utils.ema import EMA


class TrainingDataset(Dataset):
    """Yields (img, state, action_chunk, is_pad, text_ids).

    With chunk_size=1 the chunk holds one action and is_pad is all False, which is
    exactly the old single-step behaviour. Near the end of an episode the chunk is
    zero-padded and is_pad marks the padding; a chunk never runs past its episode.
    """

    def __init__(self, path, resize_to, chunk_size=1):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]             # (N, H, W, 3)
        self.states = data["states"]             # (N, state_dim)
        self.actions = data["actions"]           # (N, action_dim)
        self.text_ids = data["text_ids"]         # (N, T_text)
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.resize_to = resize_to
        self.chunk_size = chunk_size

        ends = data["episode_ends"] if "episode_ends" in data else self._infer_ends()
        self.episode_ends = ends
        # per-step lookup: the exclusive end of the episode each step belongs to
        self.ep_end_of = np.repeat(ends, np.diff(np.append([0], ends)))

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def _infer_ends(self):
        """Datasets without episode_ends. The observation is frame-stacked and the env
        seeds _prev_obs with the current frame on reset, so each episode's first step
        has obs[0:4] == obs[18:22]."""
        starts = np.flatnonzero(np.all(self.states[:, 0:4] == self.states[:, 18:22], axis=1))
        print(f"[train] no 'episode_ends' in the dataset; recovered {len(starts)} episodes")
        return np.append(starts[1:], len(self.states))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.cv2 is not None and img.shape[0] != self.resize_to:
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        stop = min(idx + self.chunk_size, self.ep_end_of[idx])
        chunk = self.actions[idx:stop]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[len(chunk):] = True
        if len(chunk) < self.chunk_size:
            chunk = np.concatenate(
                [chunk, np.zeros((self.chunk_size - len(chunk), chunk.shape[1]), chunk.dtype)])

        return (img,
                torch.from_numpy(self.states[idx]).float(),
                torch.from_numpy(chunk).float(),
                torch.from_numpy(is_pad),
                torch.from_numpy(self.text_ids[idx]).long())


def norm_stats(dataset, indices, device):
    """z-score statistics from the training split only.

    Masked state dims are constant zero, so their std is 0 and dividing by it would
    give NaN, hence the floor."""
    s = dataset.states[indices].astype(np.float64)
    a = dataset.actions[indices].astype(np.float64)
    return {
        "state_mean": torch.tensor(s.mean(0), dtype=torch.float32, device=device),
        "state_std": torch.tensor(np.maximum(s.std(0), 1e-6), dtype=torch.float32, device=device),
        "action_mean": torch.tensor(a.mean(0), dtype=torch.float32, device=device),
        "action_std": torch.tensor(np.maximum(a.std(0), 1e-6), dtype=torch.float32, device=device),
    }


def batch_loss(model, cfg, batch, device, norm):
    """Move a batch to the device, normalise it, and run the policy's loss."""
    img, state, chunk, is_pad, text_ids = [t.to(device) for t in batch]
    if norm is not None:
        state = (state - norm["state_mean"]) / norm["state_std"]
        chunk = (chunk - norm["action_mean"]) / norm["action_std"]

    # every policy takes the same (chunk, is_pad) contract
    return model.loss(img, text_ids, state, chunk, is_pad)


def evaluate(model, loader, cfg, device, norm):
    """在验证集上跑一遍平均损失。"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            total += batch_loss(model, cfg, batch, device, norm).item() * batch[0].size(0)
    return total / max(len(loader.dataset), 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override the config")
    parser.add_argument("--device", type=str, default=None, help="override the config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["train"]["device"] = args.device

    data, train_cfg = cfg["data"], cfg["train"]
    os.makedirs(os.path.dirname(data["save_path"]) or ".", exist_ok=True)
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    dataset = TrainingDataset(data["dataset_path"], data["resize_to"], data["chunk_size"])
    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    # 按「整集」划分，而不是按「单步」划分。
    # 同一次演示里相邻两帧几乎一模一样，按步划分会把 t 和 t+1 分到两边：
    # 每集的结束下标（排他性），如ends = [104, 255, 360, 465, 565, 688, 788, 910, 1033，...]  共 100 个
    ends = np.asarray(dataset.episode_ends)   
    # ends[:-1]丢掉最后一个 → [104, 255, 360, 465, ..., 10333]
    # [[0], ends[:-1]]前面加一个 [0] → [[0], [104, 255, 360, ...]]
    # np.concatenate(...)拼成一个数组
    starts = np.concatenate([[0], ends[:-1]])
    n_val_eps = max(1, int(round(len(ends) * train_cfg["val_split"])))  # 验证集占几「集」，至少 1 集
    val_eps = set(torch.randperm(len(ends), generator=torch.Generator().manual_seed(0))
                  [:n_val_eps].tolist())      # 固定种子抽出 n_val_eps 个集号；
                                              # 用独立的 Generator，避免污染全局随机状态

    train_idx, val_idx = [], [] # 装训练集和验证集的下标（两个独立的列表）
    for ep, (start, end) in enumerate(zip(starts, ends)):
        # 整集登记：这一集被抽中就整集进验证集，否则整集进训练集
        (val_idx if ep in val_eps else train_idx).extend(range(int(start), int(end)))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=train_cfg["batch_size"], shuffle=True) # 训练要打乱顺序
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=train_cfg["batch_size"]) # 验证只求和，不必打乱

    norm = norm_stats(dataset, train_idx, device) if data["normalize"] else None

    model = build_policy(cfg, vocab_size, state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    # 影子权重：验证和选模型都用它，原版 diffusion policy 就是这么做的
    ema = EMA(model, power=train_cfg["ema_power"]) if train_cfg.get("use_ema") else None
    print(f"policy={cfg['policy']}  params={sum(p.numel() for p in model.parameters()):,}  "
          f"train={len(train_idx)}  val={len(val_idx)}  ema={ema is not None}  device={device}")

    best_val, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            loss = batch_loss(model, cfg, batch, device, norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.step(model)
            train_total += loss.item() * batch[0].size(0)

        eval_model = ema.model if ema is not None else model
        val_loss = evaluate(eval_model, val_loader, cfg, device, norm)
        # keep the best epoch rather than the last: closed-loop success is noisy
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(eval_model.state_dict())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            line = (f"epoch {epoch+1:4d}/{train_cfg['epochs']}  "
                    f"train={train_total / max(len(train_idx), 1):.4f}  "
                    f"val={val_loss:.4f}  best={best_val:.4f}@{best_epoch}")
            if ema is not None:  # 顺便看 EMA 到底有没有用
                line += f"  (raw val={evaluate(model, val_loader, cfg, device, norm):.4f})"
            print(line)

    torch.save(
        {
            # 存的是 EMA 权重，scripts/test.py 直接拿它评测
            "model_state_dict": best_state,
            "raw_state_dict": model.state_dict(),
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "config": cfg,
            "norm_stats": {k: v.cpu().tolist() for k, v in norm.items()} if norm else {},
        },
        data["save_path"],
    )
    print(f"Saved checkpoint: {data['save_path']}  (best val {best_val:.4f} @ epoch {best_epoch})")


if __name__ == "__main__":
    main()
