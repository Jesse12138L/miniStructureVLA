"""Build a policy from a config dict.

Used by both scripts/train.py and scripts/test.py, so the model a checkpoint is rebuilt
from at evaluation is necessarily the one that was trained.
"""

from .act_policy import ACTPolicy
from .diffusion_policy import DiffusionPolicy
from .mlp_policy import MLPPolicy


def build_policy(cfg, vocab_size, state_dim, action_dim):
    """Instantiate the policy named by cfg["policy"] (still on the CPU).

    vocab_size, state_dim and action_dim come from the dataset while training and from
    the checkpoint while evaluating - never from the config, which is why they are
    arguments here rather than keys.

    Config schema (see configs/*.yaml). Only the sub-block matching `policy` is read:

        policy: "mlp" | "act" | "diffusion"

        data:
          resize_to: int          # images are resized to this square size
          chunk_size: int         # actions predicted per forward pass
        model:
          d_model: int
          mlp:       { hidden: int }
          act:       { n_heads, n_enc_layers, dim_feedforward, latent_dim, kl_weight }
          diffusion: { n_action_steps, num_kp, down_dims, step_embed_dim }

    Note that `chunk_size` means slightly different things per policy: the number of
    actions executed per plan for MLP and ACT, and the prediction horizon for diffusion
    (which then executes only the first n_action_steps of it).

    There is no schema validation - a missing key raises a plain KeyError naming it.
    """
    model, resize_to = cfg["model"], cfg["data"]["resize_to"]

    if cfg["policy"] == "diffusion":
        diff = model["diffusion"]
        return DiffusionPolicy(
            vocab_size=vocab_size,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=cfg["data"]["chunk_size"],
            n_action_steps=diff["n_action_steps"],
            img_size=resize_to,
            d_model=model["d_model"],
            num_kp=diff["num_kp"],
            down_dims=diff["down_dims"],
            step_embed_dim=diff["step_embed_dim"],
        )

    if cfg["policy"] == "mlp":
        return MLPPolicy(
            vocab_size=vocab_size,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=cfg["data"]["chunk_size"],
            img_size=resize_to,
            d_model=model["d_model"],
            hidden=model["mlp"]["hidden"],
        )

    if cfg["policy"] == "act":
        act = model["act"]
        return ACTPolicy(
            vocab_size=vocab_size,
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=cfg["data"]["chunk_size"],
            img_size=resize_to,
            d_model=model["d_model"],
            n_heads=act["n_heads"],
            n_enc_layers=act["n_enc_layers"],
            dim_feedforward=act["dim_feedforward"],
            latent_dim=act["latent_dim"],
            kl_weight=act["kl_weight"],
        )

    raise ValueError(
        f"unknown policy {cfg['policy']!r}; expected 'mlp', 'act' or 'diffusion'")
