"""Terminal-state predictor (Phase 1 skeleton).

Inputs:
  - prefix tokens: (B, K, max_objects, 13) — K frames of object-list features
  - prefix mask:   (B, K, max_objects)
  - actions:       (B, K-1) action-id ints (optional, can be ignored at first)
  - env_marker:    (B,) string identifier (optional fast-path conditioning)

Output:
  - predicted terminal token distribution: per-token (color logits, geometric mean+var)
  - per-token mask logit (whether this token slot is "real" or "absent" at terminal)

Architecture (skeleton):
  - per-token MLP encodes the 13-feature vector to D-dim.
  - per-frame transformer encoder (permutation-invariant within a frame; attention over tokens).
  - inter-frame transformer encoder (attention over frames; uses learned per-frame position embedding).
  - terminal-prediction head: queries the encoded sequence with a learned set of
    "terminal slot" queries, decodes each into (color logits, geometric features).

This file intentionally stops at the model definition. Training loop, loss
functions (cross-entropy + flow-matching), and InfoNCE auxiliary land in
scripts/train_predictor.py during Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_FEATURE_DIM = 13  # mirror models.object_list. ObjectToken.to_vector() output dim.


@dataclass
class PredictorConfig:
    feature_dim: int = _FEATURE_DIM
    embed_dim: int = 256
    n_heads: int = 8
    n_token_layers: int = 2     # within-frame attention layers
    n_frame_layers: int = 4     # cross-frame attention layers
    max_objects: int = 128
    max_frames: int = 32
    n_terminal_slots: int = 128 # number of "terminal token" queries
    n_color_classes: int = 256  # cross-env color vocabulary cap


class TokenEncoder(nn.Module):
    """13-feature object token → embed_dim."""

    def __init__(self, feature_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, max_objects, feature_dim)
        return self.mlp(x)


class TerminalPredictor(nn.Module):
    """Predicts the terminal-frame object list given a prefix of frames.

    Skeleton — forward returns a structured output dict; training script wires
    losses on top.
    """

    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_enc = TokenEncoder(cfg.feature_dim, cfg.embed_dim)
        self.frame_pos_emb = nn.Embedding(cfg.max_frames, cfg.embed_dim)
        self.token_pos_emb = nn.Embedding(cfg.max_objects, cfg.embed_dim)

        # Within-frame transformer (permutation-invariant — no position emb on
        # tokens within a frame; geometric features carry the position info).
        within_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.within_frame_enc = nn.TransformerEncoder(within_layer, num_layers=cfg.n_token_layers)

        # Cross-frame transformer (frames have order; use learned frame position emb).
        cross_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.cross_frame_enc = nn.TransformerEncoder(cross_layer, num_layers=cfg.n_frame_layers)

        # Learned queries for terminal slots.
        self.terminal_queries = nn.Parameter(
            torch.randn(cfg.n_terminal_slots, cfg.embed_dim) * 0.02
        )

        # Decoder layer: cross-attention from terminal queries to encoded prefix.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.terminal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Heads
        self.color_head = nn.Linear(cfg.embed_dim, cfg.n_color_classes)
        self.geom_head = nn.Linear(cfg.embed_dim, cfg.feature_dim - 1)  # everything except color
        self.exists_head = nn.Linear(cfg.embed_dim, 1)  # binary: this slot is occupied

    def encode_prefix(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode prefix into a (B, K * max_objects, D) sequence usable by the decoder.

        Within-frame attention happens per frame (no cross-frame leakage).
        Then frame position embedding is added and cross-frame attention happens.
        """
        B, K, M, F = tokens.shape
        assert F == self.cfg.feature_dim
        # Encode tokens
        h = self.token_enc(tokens)  # (B, K, M, D)
        # Within-frame attention (flatten B*K -> apply per-frame, reshape back).
        h = h.reshape(B * K, M, -1)
        within_mask = (mask.reshape(B * K, M) < 0.5)  # True = pad (ignore)
        h = self.within_frame_enc(h, src_key_padding_mask=within_mask)
        h = h.reshape(B, K, M, -1)
        # Add frame position embedding to each token's embedding
        frame_idx = torch.arange(K, device=tokens.device)
        h = h + self.frame_pos_emb(frame_idx)[None, :, None, :]
        # Flatten across K and M for cross-frame attention
        h = h.reshape(B, K * M, -1)
        cross_mask = (mask.reshape(B, K * M) < 0.5)
        h = self.cross_frame_enc(h, src_key_padding_mask=cross_mask)
        return h, cross_mask

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        B = tokens.shape[0]
        memory, memory_mask = self.encode_prefix(tokens, mask)
        queries = self.terminal_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.terminal_decoder(
            tgt=queries, memory=memory, memory_key_padding_mask=memory_mask
        )  # (B, n_terminal_slots, D)
        return {
            "color_logits": self.color_head(decoded),       # (B, n_slots, n_color_classes)
            "geom": self.geom_head(decoded),                 # (B, n_slots, F-1)
            "exists_logits": self.exists_head(decoded).squeeze(-1),  # (B, n_slots)
        }
