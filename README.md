# mini-VLA

A minimal, beginner-friendly **Vision-Language-Action** model: it takes a camera image,
a text instruction and the robot's own state, and produces continuous robot actions.

This is a teaching repo, not a research artifact. The model is ~350 lines of code — 817
with the comments and docstrings — and it trains on a laptop GPU in well under an hour.

> **Attribution.** This project is derived from
> [keivalya/mini-vla](https://github.com/keivalya/mini-vla) (MIT, © 2025 Keivalya
> Pandya), which built the original single-head VLA-diffusion policy along with the blog
> series explaining it. This fork replaces the upstream `models/fusion.py` +
> `models/diffusion_head.py` + `models/vla_diffusion_policy.py` with three
> interchangeable action heads selected by YAML config, and adds the MLP / ACT /
> Diffusion comparison in [Results](#results). The MIT licence and the upstream
> copyright are retained in [LICENSE](LICENSE).

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

> [!IMPORTANT]
> **Run every script as a module (`python -m scripts.…`), from the repo root.**
> The scripts import `from models…`, `from envs…` and `from utils…`, and those only
> resolve when the repo root is on `sys.path`. Running `python scripts/train.py` fails
> with `ModuleNotFoundError: No module named 'models'`.

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

## Results

`bin-picking-v3`, camera `corner3`, 100 episodes / 11.5k steps, 150 epochs, 50
evaluation episodes on an RTX 3060 Laptop (6 GB):

| | MLP | ACT | Diffusion |
|---|---|---|---|
| params | 413k | 1.31M | 67.6M |
| train (150 ep) | 501 s | 1445 s | 1800 s |
| evaluate (50 ep) | 24 s | 18 s | 611 s |
| **success rate** | **96%** (48/50) | 64% (32/50) | 52% (26/50) |
| first-action L1 vs expert | 0.006 | 0.047 | 0.111 |

> [!NOTE]
> The L1 row is only good for ranking, not as an absolute error. The MLP and ACT columns
> are measured against the real expert actions in the dataset; the diffusion column could
> not be reproduced that way and was measured against the MLP's closed-loop actions
> instead. Both references give the same ordering — under the old clip setting, diffusion
> scored 0.407 and 0.456 against the two — which is what makes the comparison usable.

**The success rate tracks the first-action error and nothing else.** Parameter count and
training time are uncorrelated with it — 413k parameters beat 67.6M, and 8 minutes of
training beat 30. The scripted expert is a proportional feedback controller, so the
demonstrations are single-valued and memoryless; a single-step regressor fits that
almost exactly, while ACT's chunked open-loop execution and the diffusion head's
stochastic sampling both work against it at this data scale.

> [!NOTE]
> `checkpoints/mlp_bp.pt` is not in the repo, so the 96% column cannot be reproduced
> from a committed checkpoint. Retrain it with `configs/mlp_bin_picking.yaml` (~8
> minutes) to reproduce it. Closed-loop success rates on 50 episodes carry roughly a
> ±6 point confidence interval, so treat small gaps as noise.

### A bug worth reading about: `clip_sample_range`

The diffusion policy originally scored **0%** while its validation loss looked fine.
The cause was a line copied from the reference implementation without its assumption:

`real-stanford/diffusion_policy` normalises actions *by range* into `[-1, 1]`, so its
`clip_sample_range=1.0` is self-consistent. This project uses z-score normalisation
instead, which puts the actions near ±3 — and clipping to ±1 at every denoising step
drags the whole trajectory inward. The symptom was subtle: the predicted action std came
out 0.5–0.6× the true std, uniformly across all four dimensions, with no crash and no
shape error. Raising the clip to `4.0` (effectively off — the data reaches ±3.1) took
success from **0% to 52%**, with no retraining, since it is purely an inference-time
change.

The lesson generalises: **a constant copied from a reference implementation carries that
implementation's normalisation assumptions with it.**

## Project layout

```
configs/          one YAML per experiment; defines the whole run
envs/
  metaworld_env.py    thin wrapper over MT1: returns (image, state), not raw obs
  metaworld_mt1.py    standalone viewer for watching an expert drive a task
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
- **Add a task.** Any of Meta-World's 50 MT1 tasks has a scripted expert with the same
  39-dim observation and 4-dim action, so a new task is a new `--env-name` plus a
  config. `python -m envs.metaworld_mt1` will show you one before you commit to it.
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
