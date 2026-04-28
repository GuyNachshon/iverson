"""Terminal-state predictor (Phase 1).

Inputs:
  - prefix tokens: (B, K, max_objects, 13) — K frames of object-list features.
  - prefix mask:   (B, K, max_objects)
  - feature_mask:  (13,) bool — which features the model is allowed to see.
                   Used for the invariance-feature-only ablation per the
                   Phase 0c amendment.

Output:
  Per-slot per-feature logit dicts. Heads are all categorical:
    - color_id_logits   (B, n_slots, 256)
    - color_rank_logits (B, n_slots, 32)
    - log_size_logits   (B, n_slots, 32)         (HL-Gauss target)
    - bbox_*_logits     (B, n_slots, 32) x 4     (HL-Gauss target)
    - cx_logits, cy_logits  (B, n_slots, 32)     (HL-Gauss target)
    - aspect_logits     (B, n_slots, 32)         (HL-Gauss target)
    - log_neighbors_logits (B, n_slots, 16)      (HL-Gauss target)
    - is_singleton_logit, touches_edge_logit  (B, n_slots) — BCE binary
    - exists_logits     (B, n_slots) — BCE binary

Architecture:
  - per-token MLP encodes 13-feature vector to D-dim.
  - within-frame transformer (permutation-invariant; geometry features
    carry within-frame position info).
  - cross-frame transformer with learned per-frame position embedding.
  - learned terminal-slot queries cross-attend to encoded prefix.

Per the AlphaFold 2 / Trajectory Transformer / "Stop Regressing" pattern,
all continuous outputs are predicted as classification over bins with
HL-Gauss soft targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .discretize import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_BIN_SPECS,
    F_COLOR_ID,
    F_COLOR_RANK,
    F_LOG_SIZE,
    F_LOG_NEIGHBORS,
    F_ASPECT,
    F_BBOX_XMIN,
    F_BBOX_YMIN,
    F_BBOX_XMAX,
    F_BBOX_YMAX,
    F_CX,
    F_CY,
    F_IS_SINGLETON,
    F_TOUCHES_EDGE,
    N_BINS_LOG_SIZE,
    N_BINS_BBOX,
    N_BINS_CENTROID,
    N_BINS_ASPECT,
    N_BINS_NEIGHBORS,
    N_COLOR_ID,
    N_COLOR_RANK,
)


_FEATURE_DIM = 13


@dataclass
class PredictorConfig:
    feature_dim: int = _FEATURE_DIM
    embed_dim: int = 256
    n_heads: int = 8
    n_token_layers: int = 2
    n_frame_layers: int = 4
    max_objects: int = 128
    max_frames: int = 32
    n_terminal_slots: int = 128


class TokenEncoder(nn.Module):
    """13-feature object token → embed_dim. Supports per-feature masking."""

    def __init__(self, feature_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor,
                feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (..., feature_dim). feature_mask: (feature_dim,) — 1=visible, 0=hidden.
        if feature_mask is not None:
            assert feature_mask.shape == (self.feature_dim,), \
                f"feature_mask must be shape ({self.feature_dim},), got {feature_mask.shape}"
            x = x * feature_mask.to(x.dtype).to(x.device)
        return self.mlp(x)


class TerminalPredictor(nn.Module):
    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        self.token_enc = TokenEncoder(cfg.feature_dim, D)
        self.frame_pos_emb = nn.Embedding(cfg.max_frames, D)

        within_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.n_heads, dim_feedforward=D * 4,
            batch_first=True, activation="gelu",
        )
        # enable_nested_tensor=False: MPS doesn't implement
        # _nested_tensor_from_mask_left_aligned. Disabling this fast path
        # keeps the implementation portable across MPS / CUDA / CPU.
        self.within_frame_enc = nn.TransformerEncoder(
            within_layer, num_layers=cfg.n_token_layers, enable_nested_tensor=False,
        )

        cross_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.n_heads, dim_feedforward=D * 4,
            batch_first=True, activation="gelu",
        )
        self.cross_frame_enc = nn.TransformerEncoder(
            cross_layer, num_layers=cfg.n_frame_layers, enable_nested_tensor=False,
        )

        self.terminal_queries = nn.Parameter(torch.randn(cfg.n_terminal_slots, D) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=cfg.n_heads, dim_feedforward=D * 4,
            batch_first=True, activation="gelu",
        )
        self.terminal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Per-feature heads. All linear from embed_dim to bin count.
        self.head_color_id = nn.Linear(D, N_COLOR_ID)
        self.head_color_rank = nn.Linear(D, N_COLOR_RANK)
        self.head_log_size = nn.Linear(D, N_BINS_LOG_SIZE)
        self.head_bbox_xmin = nn.Linear(D, N_BINS_BBOX)
        self.head_bbox_ymin = nn.Linear(D, N_BINS_BBOX)
        self.head_bbox_xmax = nn.Linear(D, N_BINS_BBOX)
        self.head_bbox_ymax = nn.Linear(D, N_BINS_BBOX)
        self.head_cx = nn.Linear(D, N_BINS_CENTROID)
        self.head_cy = nn.Linear(D, N_BINS_CENTROID)
        self.head_aspect = nn.Linear(D, N_BINS_ASPECT)
        self.head_log_neighbors = nn.Linear(D, N_BINS_NEIGHBORS)
        self.head_is_singleton = nn.Linear(D, 1)
        self.head_touches_edge = nn.Linear(D, 1)
        self.head_exists = nn.Linear(D, 1)

    def encode_prefix(self, tokens: torch.Tensor, mask: torch.Tensor,
                      feature_mask: torch.Tensor | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        B, K, M, F = tokens.shape
        assert F == self.cfg.feature_dim
        h = self.token_enc(tokens, feature_mask=feature_mask)  # (B,K,M,D)
        h = h.reshape(B * K, M, -1)
        within_pad = (mask.reshape(B * K, M) < 0.5)
        h = self.within_frame_enc(h, src_key_padding_mask=within_pad)
        h = h.reshape(B, K, M, -1)
        frame_idx = torch.arange(K, device=tokens.device)
        h = h + self.frame_pos_emb(frame_idx)[None, :, None, :]
        h = h.reshape(B, K * M, -1)
        cross_pad = (mask.reshape(B, K * M) < 0.5)
        h = self.cross_frame_enc(h, src_key_padding_mask=cross_pad)
        return h, cross_pad

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                feature_mask: torch.Tensor | None = None) -> dict:
        B = tokens.shape[0]
        memory, memory_pad = self.encode_prefix(tokens, mask, feature_mask=feature_mask)
        queries = self.terminal_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.terminal_decoder(
            tgt=queries, memory=memory, memory_key_padding_mask=memory_pad,
        )  # (B, n_slots, D)
        return {
            "color_id_logits":   self.head_color_id(decoded),
            "color_rank_logits": self.head_color_rank(decoded),
            "log_size_logits":   self.head_log_size(decoded),
            "bbox_xmin_logits":  self.head_bbox_xmin(decoded),
            "bbox_ymin_logits":  self.head_bbox_ymin(decoded),
            "bbox_xmax_logits":  self.head_bbox_xmax(decoded),
            "bbox_ymax_logits":  self.head_bbox_ymax(decoded),
            "cx_logits":         self.head_cx(decoded),
            "cy_logits":         self.head_cy(decoded),
            "aspect_logits":     self.head_aspect(decoded),
            "log_neighbors_logits": self.head_log_neighbors(decoded),
            "is_singleton_logit":   self.head_is_singleton(decoded).squeeze(-1),
            "touches_edge_logit":   self.head_touches_edge(decoded).squeeze(-1),
            "exists_logits":     self.head_exists(decoded).squeeze(-1),
        }


# Feature mask presets per the Phase 0c amendment.
def feature_mask_full(device: torch.device) -> torch.Tensor:
    return torch.ones(_FEATURE_DIM, device=device)


def feature_mask_invariant(device: torch.device) -> torch.Tensor:
    """Hide env-correlated absolute features (color_id, raw bbox coords).
    Keep: color_rank (per-env normalized), log_size (relative scale),
    centroids (normalized to [0,1]), aspect (ratio), structural booleans,
    log_neighbors.
    """
    m = torch.ones(_FEATURE_DIM, device=device)
    m[F_COLOR_ID] = 0.0
    m[F_BBOX_XMIN] = 0.0
    m[F_BBOX_YMIN] = 0.0
    m[F_BBOX_XMAX] = 0.0
    m[F_BBOX_YMAX] = 0.0
    return m
