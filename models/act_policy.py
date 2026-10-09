"""ACT (Action Chunking with Transformers) policy.

Predicts a chunk of future actions from one observation: a transformer decoder whose
queries are a learnable matrix produces `chunk_size` outputs, and one shared action
head turns them into actions. A small VAE also encodes the demonstration into a style
latent, which is zeroed at inference as in the original paper.

Two deliberate differences from ACT (Zhao et al. 2023):
  - the learnable queries are the decoder's input, where the original feeds zeros and
    adds the queries inside attention instead. With one decoder layer the two are
    nearly the same, and the original's version starts from a degenerate step (its
    self-attention values are all zero).
  - the VAE reads out its summary from the state token rather than a dedicated CLS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import StateEncoderMLP, TextEncoderTinyGRU
from .losses import masked_l1_loss


def make_encoder(d_model, n_heads, dim_feedforward, dropout, n_layers):
    """A standard transformer encoder, used both for the policy's observation encoder
    and for the VAE."""
    layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout,
                                       batch_first=True)
    return nn.TransformerEncoder(layer, n_layers)


class ACTPolicy(nn.Module):
    def __init__(self, vocab_size, state_dim, action_dim, chunk_size=20, d_model=128,
                 n_heads=4, n_enc_layers=2, dim_feedforward=512, latent_dim=16,
                 kl_weight=10.0, img_size=128, dropout=0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Image tokens: the same convolutions as the MLP policy, but each spatial
        # location stays its own token instead of being flattened into one vector.
        grid = img_size // 8
        # [SHARED CONV TRUNK] - the same three convs as encoders / diffusion_policy
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.img_proj = nn.Conv2d(128, d_model, kernel_size=1)
        self.img_pos = nn.Parameter(torch.zeros(1, grid * grid, d_model))

        self.txt_encoder = TextEncoderTinyGRU(vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim, d_model=d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Encoder sees [latent, state, text, image tokens...].
        self.encoder_pos = nn.Parameter(torch.zeros(1, 3, d_model))
        self.encoder = make_encoder(d_model, n_heads, dim_feedforward, dropout, n_enc_layers)

        # Decoder: the learnable queries are its input, one shared head reads them out.
        self.queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward,
                                                   dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 1)
        self.action_head = nn.Linear(d_model, action_dim)

        # VAE over [state, actions...]; attention lets the state token summarise the
        # demonstration, and mu/logvar are read off it.
        self.vae_state = nn.Linear(state_dim, d_model)
        self.vae_action = nn.Linear(action_dim, d_model)
        self.vae_pos = nn.Parameter(torch.zeros(1, chunk_size + 1, d_model))
        self.vae_encoder = make_encoder(d_model, n_heads, dim_feedforward, dropout, n_enc_layers)
        self.vae_head = nn.Linear(d_model, latent_dim * 2)

    def _memory(self, image, text_ids, state, latent):
        """Build the encoder's token sequence: [latent, state, text, image tokens...]."""
        # [SHARED CONV TRUNK] forward pass, matching models/encoders.py
        img = F.relu(self.conv1(image))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))
        # (B, d_model, g, g) -> (B, g*g, d_model): one token per spatial location
        img = self.img_proj(img).flatten(2).transpose(1, 2) + self.img_pos
        # The three non-image tokens get their own learned positions; the image tokens
        # already carry img_pos, added above.
        extra = torch.stack([latent, self.state_encoder(state),
                             self.txt_encoder(text_ids)], dim=1) + self.encoder_pos

        return self.encoder(torch.cat([extra, img], dim=1))

    def _decode(self, memory):
        """Expand the learnable queries to the batch size and read out actions."""
        # expand broadcasts without copying - every batch item starts from the same
        # learned queries
        queries = self.queries.expand(memory.size(0), -1, -1)
        return self.action_head(self.decoder(queries, memory))

    def loss(self, image, text_ids, state, chunk, is_pad):
        """
        chunk:  (B, chunk_size, action_dim), zero-padded at the end of an episode
        is_pad: (B, chunk_size) bool, True on that padding
        """
        tokens = torch.cat([self.vae_state(state).unsqueeze(1),
                            self.vae_action(chunk)], dim=1) + self.vae_pos
        # The token sequence is [state, action_1 ... action_N] but is_pad only covers
        # the actions, so a False column is prepended to leave the state token
        # unmasked. Inside `src_key_padding_mask`, True means "ignore this position".
        mask = torch.cat([torch.zeros_like(is_pad[:, :1]), is_pad], dim=1)
        enc = self.vae_encoder(tokens, src_key_padding_mask=mask)  # (B, chunk+1, d_model)
        summary = enc[:, 0]  # attention lets the state token summarise the whole demo
        mu, logvar = self.vae_head(summary).chunk(2, dim=-1)  # each (B, latent_dim)

        # reparameterisation: sample z ~ N(mu, sigma^2) so gradients reach mu/logvar
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        pred = self._decode(self._memory(image, text_ids, state, self.latent_proj(latent)))

        l1 = masked_l1_loss(pred, chunk, is_pad)
        # The KL term pulls the latent towards a standard normal, which is what makes
        # sampling z = 0 at inference stay in distribution.
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + self.kl_weight * kl

    @torch.no_grad()
    def act(self, image, text_ids, state):
        """Inference uses z = 0; the VAE encoder only exists to shape the latent."""
        self.eval()
        zero = torch.zeros(image.size(0), self.latent_dim, device=image.device)
        return self._decode(self._memory(image, text_ids, state, self.latent_proj(zero)))
