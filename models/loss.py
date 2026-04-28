"""Terminal-state predictor loss.

The loss aggregates per-feature classifications:
  - color_id, color_rank: hard-target cross-entropy.
  - log_size, bbox, centroid, aspect, log_neighbors: HL-Gauss soft target CE.
  - is_singleton, touches_edge: BCE (binary).
  - exists: BCE on the existence flag.

CRITICAL: per-feature losses on color/geom/binary features are gated by the
TARGET exists mask. Padded slots in the target should not produce gradients
on the per-feature heads; only the exists head learns "this slot is empty."
Without this gate, the model's per-feature heads pour gradient into noise on
~half of slots, dominating early training.

Slot assignment: we don't try to match predicted slots to target slots
(Hungarian matching). Instead we use the simplest workable assignment:
position-based — slot i predicts target object i in the (size-sorted)
target frame. The model's terminal-slot queries learn slot-specific
specializations. Permutation invariance within the prefix is preserved by
the within-frame transformer. (Better assignment strategies — e.g., a
DETR-style bipartite matcher — are deferred to a follow-up if simple
position assignment underperforms.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .discretize import (
    F_ASPECT,
    F_BBOX_XMAX,
    F_BBOX_XMIN,
    F_BBOX_YMAX,
    F_BBOX_YMIN,
    F_COLOR_ID,
    F_COLOR_RANK,
    F_CX,
    F_CY,
    F_IS_SINGLETON,
    F_LOG_NEIGHBORS,
    F_LOG_SIZE,
    F_TOUCHES_EDGE,
    N_BINS_ASPECT,
    N_BINS_BBOX,
    N_BINS_CENTROID,
    N_BINS_LOG_SIZE,
    N_BINS_NEIGHBORS,
    N_COLOR_ID,
    N_COLOR_RANK,
    hard_bin_target,
    hl_gauss_target,
)


def _soft_ce(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    """Soft-target cross-entropy. logits and target both (N, n_bins).
    Returns (N,) per-sample loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_target * log_probs).sum(-1)


def _hard_ce(logits: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """Standard CE. logits (N, n_bins), target_idx (N,). Returns (N,)."""
    return F.cross_entropy(logits, target_idx, reduction="none")


def predictor_loss(
    out: dict,
    target_tokens: torch.Tensor,   # (B, M, 13)
    target_mask: torch.Tensor,     # (B, M) — 1 = real object, 0 = padding
    loss_weights: torch.Tensor | None = None,  # (B,) per-sample weight
    geom_loss_weight: float = 1.0,
    color_loss_weight: float = 1.0,
    binary_loss_weight: float = 0.5,
    exists_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute the predictor loss + per-component diagnostics.

    The per-feature losses are summed across features and gated by the
    target_mask so padded slots don't contribute. The exists head is trained
    on ALL slots (including padded), so it can learn to predict "this slot
    is empty."
    """
    B, M, _ = target_tokens.shape

    # Flatten (B, M) → (B*M,) for per-slot losses.
    flat = target_tokens.reshape(B * M, -1)
    mask_flat = target_mask.reshape(B * M)

    # ---- exists (all slots) ----
    exists_logits = out["exists_logits"].reshape(B * M)
    exists_target = mask_flat
    exists_loss = F.binary_cross_entropy_with_logits(
        exists_logits, exists_target, reduction="none"
    )

    # Per-sample weights (one per (B,) item, broadcast over slots).
    if loss_weights is not None:
        slot_weights = loss_weights.unsqueeze(1).expand(B, M).reshape(B * M)
    else:
        slot_weights = torch.ones_like(exists_loss)

    exists_loss_mean = (exists_loss * slot_weights).mean()

    # ---- per-feature losses, gated by mask_flat ----
    valid = mask_flat > 0.5
    n_valid = valid.sum().clamp_min(1).float()

    def _gated_mean(per_slot_loss: torch.Tensor) -> torch.Tensor:
        """Mean over valid slots only, weighted by slot_weights."""
        if not per_slot_loss.shape == valid.shape:
            raise RuntimeError(f"shape mismatch: {per_slot_loss.shape} vs {valid.shape}")
        return ((per_slot_loss * valid * slot_weights).sum() / n_valid).clamp_min(0.0)

    # color_id (hard CE, integer target).
    color_id_target = flat[:, F_COLOR_ID].long().clamp(0, N_COLOR_ID - 1)
    loss_color_id = _gated_mean(_hard_ce(out["color_id_logits"].reshape(-1, N_COLOR_ID), color_id_target))

    # color_rank (hard CE).
    color_rank_target = flat[:, F_COLOR_RANK].long().clamp(0, N_COLOR_RANK - 1)
    loss_color_rank = _gated_mean(_hard_ce(out["color_rank_logits"].reshape(-1, N_COLOR_RANK), color_rank_target))

    # Continuous features → HL-Gauss soft CE.
    def _hlg_loss(logits_key: str, feat_idx: int, n_bins: int, lo: float, hi: float
                   ) -> torch.Tensor:
        soft = hl_gauss_target(flat[:, feat_idx], n_bins, lo, hi)
        return _gated_mean(_soft_ce(out[logits_key].reshape(-1, n_bins), soft))

    loss_log_size = _hlg_loss("log_size_logits", F_LOG_SIZE, N_BINS_LOG_SIZE, 0.0, 5.5)
    loss_bbox_xmin = _hlg_loss("bbox_xmin_logits", F_BBOX_XMIN, N_BINS_BBOX, 0.0, 1.0)
    loss_bbox_ymin = _hlg_loss("bbox_ymin_logits", F_BBOX_YMIN, N_BINS_BBOX, 0.0, 1.0)
    loss_bbox_xmax = _hlg_loss("bbox_xmax_logits", F_BBOX_XMAX, N_BINS_BBOX, 0.0, 1.0)
    loss_bbox_ymax = _hlg_loss("bbox_ymax_logits", F_BBOX_YMAX, N_BINS_BBOX, 0.0, 1.0)
    loss_cx = _hlg_loss("cx_logits", F_CX, N_BINS_CENTROID, 0.0, 1.0)
    loss_cy = _hlg_loss("cy_logits", F_CY, N_BINS_CENTROID, 0.0, 1.0)
    loss_aspect = _hlg_loss("aspect_logits", F_ASPECT, N_BINS_ASPECT, 0.0, 12.0)
    loss_log_neighbors = _hlg_loss("log_neighbors_logits", F_LOG_NEIGHBORS, N_BINS_NEIGHBORS, 0.0, 4.0)

    # Binary features (BCE, gated).
    is_singleton_logit = out["is_singleton_logit"].reshape(-1)
    is_singleton_target = flat[:, F_IS_SINGLETON]
    loss_is_singleton = _gated_mean(F.binary_cross_entropy_with_logits(
        is_singleton_logit, is_singleton_target, reduction="none"
    ))

    touches_edge_logit = out["touches_edge_logit"].reshape(-1)
    touches_edge_target = flat[:, F_TOUCHES_EDGE]
    loss_touches_edge = _gated_mean(F.binary_cross_entropy_with_logits(
        touches_edge_logit, touches_edge_target, reduction="none"
    ))

    # Aggregate.
    color_loss = (loss_color_id + loss_color_rank) * color_loss_weight
    geom_loss = (loss_log_size + loss_bbox_xmin + loss_bbox_ymin
                 + loss_bbox_xmax + loss_bbox_ymax + loss_cx + loss_cy
                 + loss_aspect + loss_log_neighbors) * geom_loss_weight
    binary_loss = (loss_is_singleton + loss_touches_edge) * binary_loss_weight
    exists_loss_total = exists_loss_mean * exists_loss_weight

    total = color_loss + geom_loss + binary_loss + exists_loss_total

    diag = {
        "loss_color_id": float(loss_color_id),
        "loss_color_rank": float(loss_color_rank),
        "loss_log_size": float(loss_log_size),
        "loss_bbox": float(loss_bbox_xmin + loss_bbox_ymin + loss_bbox_xmax + loss_bbox_ymax) / 4.0,
        "loss_cx": float(loss_cx),
        "loss_cy": float(loss_cy),
        "loss_aspect": float(loss_aspect),
        "loss_log_neighbors": float(loss_log_neighbors),
        "loss_is_singleton": float(loss_is_singleton),
        "loss_touches_edge": float(loss_touches_edge),
        "loss_exists": float(exists_loss_mean),
        "n_valid_slots": float(n_valid),
        "total": float(total),
    }
    return total, diag
