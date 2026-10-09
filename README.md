# miniStructureVLA

A minimal, beginner-friendly **Vision-Language-Action** model: it takes a camera image,
a text instruction and the robot's own state, and produces continuous robot actions.

This is a teaching repo, not a research artifact. The model is ~350 lines of code — 817
with the comments and docstrings — and it trains on a laptop GPU in well under an hour.

> **Attribution.** Derived from [keivalya/mini-vla](https://github.com/keivalya/mini-vla)
> (MIT, © 2025 Keivalya Pandya).

## What it does

All three policies share the same shape, so the interesting part is what happens after
the encoders:

```
image ──► CNN encoder  ─┐
text  ──► GRU encoder  ─┼──► fuse ──► action head ──► 4-D continuous action
state ──► MLP encoder  ─┘
```

| action head | how it produces actions | params |
|---|---|---|
| **MLP** | concatenate the three tokens, push through one MLP, get one action. Re-plans every step. | 413k |
| **ACT** | a transformer decoder with learnable queries emits a chunk of 20 actions at once; a small VAE shapes the latent, zeroed at inference. | 1.31M |
| **Diffusion** | a hand-written DDPM denoises a whole action sequence from pure noise, conditioned on the observation. Predicts 16 steps, executes 8. | 67.6M |

The task is Meta-World MT1 — a simulated Sawyer arm. The default dataset is
`bin-picking-v3`: pick a block out of one bin and drop it in another.

## Install

```bash
conda create --name mini-vla python=3.10
conda activate mini-vla
git clone https://github.com/Jesse12138L/miniStructureVLA.git
cd miniStructureVLA
pip install -r requirements.txt
```

## Run it

Run every script as a module (`python -m scripts.…`) from the repo root. The scripts
import `from models…`, `from envs…` and `from utils…`, and those only resolve when the
repo root is on `sys.path` — `python scripts/train.py` fails with
`ModuleNotFoundError: No module named 'models'`.

### 1. Collect demonstrations

The expert is Meta-World's own scripted controller, so this is fast (a few minutes) and
needs no human teleoperation.

```bash
python -m scripts.collect_data \
  --env-name bin-picking-v3 \
  --camera-name corner3 \
  --episodes 100 \
  --max-steps 200 \
  --output-path data/bp_masked.npz
```

The scripted expert finishes a bin-picking episode in 100–120 steps, so `--max-steps 200`
leaves headroom. If you raise it, collection gets slower but the demonstrations stay
complete. Expect ~12 GB of RAM and a ~3.6 GB output file for 100 episodes at the default
480×480 render size — the script holds every frame in memory before writing once.

### 2. Train

```bash
python -m scripts.train --config configs/mlp_bin_picking.yaml
python -m scripts.train --config configs/act_bin_picking.yaml
python -m scripts.train --config configs/diffusion_bin_picking.yaml
```

The config defines the experiment; the whole config is stored inside the checkpoint, so
evaluation needs no config file. `--epochs` and `--device` override the config.

### 3. Evaluate

```bash
python -m scripts.test \
  --checkpoint checkpoints/act_bp.pt \
  --env-name bin-picking-v3 \
  --camera-name corner3 \
  --instruction "push the object to the goal" \
  --episodes 50 \
  --max-steps 250
```

> [!NOTE]
> `data/` and `checkpoints/` are gitignored, so a fresh clone ships with neither. You
> have to collect the data and train before you can evaluate — the dataset is ~3.6 GB
> and no checkpoints are committed.

## The observation

Meta-World's MT1 observation is a flat 39-dim vector, frame-stacked (current frame then
previous frame):

| slice | meaning |
|---|---|
| `[0:3]` | end-effector position |
| `[3]` | gripper opening |
| `[4:18]` | object poses — two slots of (position, quaternion) |
| `[18:36]` | the same three fields for the previous frame |
| `[36:39]` | goal position |

`utils/state_masking.py` zeroes the **privileged** fields — object poses and the goal
position, 31 of the 39 dims — leaving only the 8 dims of end-effector and gripper state
that a real robot gets for free from its own sensors.

This matters because it is what forces the vision pathway to do real work: with the
object's coordinates handed to it, a policy can solve the task without looking at the
image. `scripts/collect_data.py` and `scripts/test.py` both call the same function, so
they cannot silently disagree about which fields are masked.

Images are rendered at `camera_name`, flipped vertically (`env.render()` comes out
upside down), resized to `data.resize_to` (128 in all three configs) and scaled to
`[0, 1]`.

## Project layout

```
configs/          one YAML per experiment; defines the whole run
envs/
  metaworld_env.py    thin wrapper over MT1: returns (image, state), not raw obs
  metaworld_mt1.py    viewer for watching an expert drive a task; lists all 50 tasks
models/
  encoders.py         image CNN / text GRU / state MLP — the canonical conv trunk
  mlp_policy.py       action head 1
  act_policy.py       action head 2
  diffusion_policy.py action head 3, plus the hand-written DDPM schedule
  unet1d.py           the 1-D U-Net denoiser used by the diffusion head
  losses.py           masked L1 / MSE over the valid (non-padding) action steps
  build.py            config -> policy
scripts/
  collect_data.py     roll out the expert, save a .npz
  train.py            train from a config, checkpoint the best validation epoch
  test.py             evaluate a checkpoint in the environment, save videos
utils/
  state_masking.py    which observation fields are privileged
  chunking.py         padding / unwrapping action chunks
  tokenizer.py        whitespace word-level tokenizer
  ema.py              exponential moving average of the weights
  config.py           YAML loader
```

Run `python -m envs.metaworld_mt1` to open a window and watch the expert policy before
you collect from it.

## Things to try next

Roughly in order of difficulty:

- **Change the head, not the code.** Everything except the action head is shared, so
  `configs/*.yaml` is the whole experiment surface. Try `chunk_size: 1` for ACT, or
  `kl_weight: 0.0` to isolate the VAE's contribution.
- **Run the masking ablation yourself.** Set `PRIVILEGED_STATE_SLICES = ()` in
  `utils/state_masking.py` and retrain. Handing the policy the object's coordinates is a
  direct test of whether the vision pathway is load-bearing.
- **Change the task.** All 50 Meta-World MT1 tasks share the same 39-dim observation and
  4-dim action, so switching task means changing `--env-name` and pointing a config at the
  new dataset — no model code moves. `envs/metaworld_mt1.py` lists all 50 by category and
  flags the multi-stage ones; run `python -m envs.metaworld_mt1` to watch one before
  collecting from it.
- **Swap the vision encoder** for CLIP or SigLIP features, which is where a VLA usually
  gets its language grounding from.
- **Make the language mean something.** The instruction in this dataset is a single
  fixed string, so the text encoder can be removed without changing the policy's score
  at all — the project is a VLA in shape, but the L is currently decorative. Giving the
  instruction real work to do needs either several tasks in one dataset or a benchmark
  where the same scene admits different correct instructions (`libero_object` is the
  usual choice).

## License

MIT — see [LICENSE](LICENSE). Original work © 2025 Keivalya Pandya.
