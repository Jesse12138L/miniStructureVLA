"""Building and unwrapping action chunks.

A chunk is a fixed-length window of future actions. It never runs past the end of its
episode: the tail is zero-padded and `is_pad` marks the padding. Training builds
chunks out of the dataset, evaluation unwraps whatever the policy returns into one
action per environment step. Both live here so the two scripts agree on the contract.
"""

import numpy as np


def episode_end_per_step(ends):
    """Expand exclusive episode-end indices into a per-step lookup table.

    ends:    (n_episodes,) exclusive end index of each episode, e.g. [104, 255, ...]
    returns: (ends[-1],) where every step of episode k holds ends[k]

    The per-episode lengths are diff([0] + ends); repeating each end once per step of
    its episode turns that into a table you can index by step.
    """
    lengths = np.diff(np.append([0], ends))
    return np.repeat(ends, lengths)


def padded_action_chunk(actions, idx, chunk_size, episode_end):
    """Slice the action chunk starting at step `idx`, zero-padding the tail.

    actions:     (N, action_dim)
    episode_end: exclusive end of the episode that contains idx
    returns:     chunk (chunk_size, action_dim), is_pad (chunk_size,) bool

    Stopping at episode_end is what keeps a chunk from crossing into the next episode
    and teaching the policy that one demonstration runs straight into another.
    """
    stop = min(idx + chunk_size, episode_end)
    chunk = actions[idx:stop]
    is_pad = np.zeros(chunk_size, dtype=bool)
    is_pad[len(chunk):] = True
    if len(chunk) < chunk_size:
        chunk = np.concatenate(
            [chunk, np.zeros((chunk_size - len(chunk), chunk.shape[1]), chunk.dtype)])
    return chunk, is_pad


def action_queue_from_chunk(chunk):
    """Split a policy's output into single actions, one per environment step.

    chunk:   (chunk_size, action_dim), or a bare (action_dim,) from a single-step
             policy
    returns: a list the caller pops from, re-planning when it runs dry

    A single-step policy yields a one-element queue, so the evaluation loop reads the
    same whether or not the policy predicts chunks.
    """
    return list(chunk if chunk.ndim > 1 else chunk[None])
