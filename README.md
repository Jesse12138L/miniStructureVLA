# miniStructureVLA

A minimal, beginner-friendly **Vision-Language-Action** model: image, text instruction
and robot state in; continuous robot actions out. Educational, not state of the art —
a few hundred lines you can read in one sitting.

Derived from [keivalya/mini-vla](https://github.com/keivalya/mini-vla) (MIT, © 2025
Keivalya Pandya). See [LICENSE](LICENSE).

> 中文版见 [README.zh-CN.md](README.zh-CN.md).

## What it does

```
image ──► CNN encoder  ─┐
text  ──► GRU encoder  ─┼──► fuse ──► action head ──► 4-D continuous action
state ──► MLP encoder  ─┘
```

Everything before the action head is shared, so the head is the only variable between
the three policies. It is chosen by YAML config:

| action head | how it produces actions | params |
|---|---|---|
| **MLP** | one MLP over the three tokens; a single action, re-planned every step | 413k |
| **ACT** | a transformer decoder with learnable queries emits 20 actions at once; a VAE shapes the latent, zeroed at inference | 1.31M |
| **Diffusion** | a hand-written DDPM denoises an action sequence from noise; predicts 16 steps, executes 8 | 67.6M |

The task is Meta-World MT1, a simulated Sawyer arm. The default dataset is
`bin-picking-v3`: pick a block out of one bin and drop it in another.

## Install

```bash
conda create -n mini-vla python=3.10 && conda activate mini-vla
pip install -r requirements.txt
```

## Run

Run every script as a module, from the repo root — the `from models…` imports need it.

```bash
# 1. collect demonstrations with Meta-World's scripted expert
python -m scripts.collect_data --env-name bin-picking-v3 --camera-name corner3 \
  --episodes 100 --max-steps 200 --output-path data/bp_masked.npz

# 2. train. The config defines the run and is stored inside the checkpoint,
#    so evaluation needs no config file.
python -m scripts.train --config configs/mlp_bin_picking.yaml

# 3. evaluate in the environment
python -m scripts.test --checkpoint checkpoints/mlp_bp.pt --episodes 50 --max-steps 250
```

`data/` and `checkpoints/` are gitignored, so a fresh clone has neither — collect and
train before evaluating.

## Observation

Meta-World's MT1 observation is 39 dims, frame-stacked:

| slice | meaning |
|---|---|
| `[0:3]` / `[3]` | end-effector position / gripper opening |
| `[4:18]` | object poses — two slots of (position, quaternion) |
| `[18:36]` | the same block, for the previous frame |
| `[36:39]` | goal position |

`utils/state_masking.py` zeroes the privileged fields — object poses and the goal, 31 of
the 39 dims — leaving the 8 dims a real robot gets for free from its own sensors. That is
what forces the vision pathway to do the work, and `collect_data.py` and `test.py` share
the one function so they cannot disagree about it.

## Layout

```
configs/    one YAML per experiment
envs/       MT1 wrapper, plus a viewer that lists all 50 tasks by category
models/     encoders, the three action heads, the U-Net, the losses
scripts/    collect_data.py · train.py · test.py
utils/      state masking, chunking, tokenizer, EMA, config loader
```

## Things to try

- **Change the head, not the code.** `configs/*.yaml` is the whole experiment surface.
- **Change the task.** All 50 MT1 tasks share the same observation and action layout, so
  switching is a `--env-name` and a new dataset. `envs/metaworld_mt1.py` lists them;
  `python -m envs.metaworld_mt1` opens a window to watch one.
- **Make the language mean something.** The instruction here is a single fixed string, so
  the text encoder can be dropped without changing anything. Giving it real work to do
  needs several tasks in one dataset, or a benchmark where one scene admits different
  correct instructions.
