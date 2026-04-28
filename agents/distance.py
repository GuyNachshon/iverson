"""Distance between an actual frame and a predicted terminal distribution.

For each predicted slot with exists_prob > threshold, the model has predicted
a distribution over (color, centroid, aspect, ...). For each real object in
the actual frame, we want to find the predicted slot it best matches and
sum match-quality.

Two algorithms:
  - greedy: for each actual object, greedily pick the best unmatched predicted
    slot. O(n_actual * n_pred). Fast but suboptimal.
  - hungarian: optimal bipartite matching via scipy.optimize. O(n^3) but
    typically n < 100, fine.

The score is the sum of per-pair "feature distances" plus penalties for
unmatched-actual objects (model failed to predict them) and unmatched-
predicted objects (model spurious predictions).

This is what the agent uses to score candidate successor states: each
successor produces a distance, and we pick the action whose successor
minimizes distance to the predicted terminal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.discretize import expected_value_decode
from models.object_list import Frame


@dataclass
class TerminalPrediction:
    """Decoded predicted-terminal output, ready for distance computation.

    All arrays have shape (n_slots,) or (n_slots, n_features). Filtered by
    exists_prob threshold so callers don't carry around full 128-slot arrays.
    """
    n_active: int
    color_id: np.ndarray       # (n_active,) — argmax color id (raw)
    color_rank: np.ndarray     # (n_active,) — argmax color_rank
    cx: np.ndarray             # (n_active,) — expected centroid x in [0,1]
    cy: np.ndarray             # (n_active,)
    aspect: np.ndarray         # (n_active,)
    log_size: np.ndarray       # (n_active,)
    is_singleton: np.ndarray   # (n_active,) bool
    touches_edge: np.ndarray   # (n_active,) bool
    exists_prob: np.ndarray    # (n_active,) — sigmoid prob, kept for confidence-weighted match


def decode_predictor_output(out: dict, exists_threshold: float = 0.5
                              ) -> TerminalPrediction:
    """Convert raw predictor output (per-slot logit dicts) to a TerminalPrediction.

    `out` is the dict returned by TerminalPredictor.forward(); we expect a
    single sample (B=1) or take batch index 0.
    """
    if out["color_id_logits"].dim() == 3:
        # B,n_slots,...; take batch 0
        out = {k: v[0] for k, v in out.items()}

    exists_prob = torch.sigmoid(out["exists_logits"]).cpu().numpy()
    active = exists_prob > exists_threshold
    if not active.any():
        return TerminalPrediction(
            n_active=0,
            color_id=np.zeros(0, dtype=np.int64),
            color_rank=np.zeros(0, dtype=np.int64),
            cx=np.zeros(0), cy=np.zeros(0),
            aspect=np.zeros(0), log_size=np.zeros(0),
            is_singleton=np.zeros(0, dtype=bool),
            touches_edge=np.zeros(0, dtype=bool),
            exists_prob=np.zeros(0),
        )

    color_id = out["color_id_logits"].argmax(-1).cpu().numpy()
    color_rank = out["color_rank_logits"].argmax(-1).cpu().numpy()
    cx = expected_value_decode(out["cx_logits"], 0.0, 1.0).cpu().numpy()
    cy = expected_value_decode(out["cy_logits"], 0.0, 1.0).cpu().numpy()
    aspect = expected_value_decode(out["aspect_logits"], 0.0, 12.0).cpu().numpy()
    log_size = expected_value_decode(out["log_size_logits"], 0.0, 5.5).cpu().numpy()
    is_singleton = (torch.sigmoid(out["is_singleton_logit"]).cpu().numpy() > 0.5)
    touches_edge = (torch.sigmoid(out["touches_edge_logit"]).cpu().numpy() > 0.5)

    return TerminalPrediction(
        n_active=int(active.sum()),
        color_id=color_id[active].astype(np.int64),
        color_rank=color_rank[active].astype(np.int64),
        cx=cx[active], cy=cy[active],
        aspect=aspect[active], log_size=log_size[active],
        is_singleton=is_singleton[active],
        touches_edge=touches_edge[active],
        exists_prob=exists_prob[active],
    )


def _pair_distance(pred: TerminalPrediction, frame: Frame, i: int, j: int
                    ) -> float:
    """Distance between predicted slot i and frame object j."""
    obj = frame.objects[j]
    # Centroid distance (L2 in normalized coords).
    d_cent = ((pred.cx[i] - obj.centroid_norm[0]) ** 2
              + (pred.cy[i] - obj.centroid_norm[1]) ** 2) ** 0.5
    # Color rank mismatch (cheap, generalizable).
    d_color = 0.0 if pred.color_rank[i] == obj.color_rank else 1.0
    # Aspect mismatch (log-ratio).
    d_asp = abs(pred.aspect[i] - obj.aspect) / max(obj.aspect, 0.5)
    # Size mismatch in log space.
    d_size = abs(pred.log_size[i] - np.log1p(obj.size))
    return float(d_cent + 0.5 * d_color + 0.3 * d_asp + 0.2 * d_size)


def _greedy_match(pred: TerminalPrediction, frame: Frame
                    ) -> tuple[float, list[tuple[int, int, float]]]:
    if pred.n_active == 0 and not frame.objects:
        return 0.0, []
    if pred.n_active == 0:
        # No predictions but real objects exist — penalize each.
        return float(len(frame.objects) * 1.5), []
    if not frame.objects:
        # Predictions exist but no real objects — penalize each.
        return float(pred.n_active * 1.5), []

    n_p, n_f = pred.n_active, len(frame.objects)
    pairs = []
    for i in range(n_p):
        for j in range(n_f):
            pairs.append((i, j, _pair_distance(pred, frame, i, j)))
    pairs.sort(key=lambda x: x[2])

    matched_pred = set()
    matched_frame = set()
    matched_pairs: list[tuple[int, int, float]] = []
    total = 0.0
    for i, j, d in pairs:
        if i in matched_pred or j in matched_frame:
            continue
        matched_pred.add(i)
        matched_frame.add(j)
        matched_pairs.append((i, j, d))
        total += d

    # Penalize unmatched
    total += 1.5 * (n_p - len(matched_pred)) + 1.5 * (n_f - len(matched_frame))
    return float(total), matched_pairs


def _hungarian_match(pred: TerminalPrediction, frame: Frame
                       ) -> tuple[float, list[tuple[int, int, float]]]:
    if pred.n_active == 0 and not frame.objects:
        return 0.0, []
    if pred.n_active == 0:
        return float(len(frame.objects) * 1.5), []
    if not frame.objects:
        return float(pred.n_active * 1.5), []
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_match(pred, frame)

    n_p, n_f = pred.n_active, len(frame.objects)
    cost = np.zeros((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_f):
            cost[i, j] = _pair_distance(pred, frame, i, j)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pairs = [(int(i), int(j), float(cost[i, j]))
                       for i, j in zip(row_ind, col_ind)]
    total = float(cost[row_ind, col_ind].sum())
    # Unmatched
    unmatched_pred = n_p - len(row_ind)
    unmatched_frame = n_f - len(col_ind)
    total += 1.5 * unmatched_pred + 1.5 * unmatched_frame
    return total, matched_pairs


def distance_to_predicted_terminal(
    pred: TerminalPrediction,
    frame: Frame,
    method: str = "hungarian",
) -> tuple[float, list[tuple[int, int, float]]]:
    """Score `frame` against the predicted terminal `pred`. Lower = better match.

    Returns (total_distance, matched_pairs) where matched_pairs is
    [(pred_slot_idx, frame_obj_idx, pair_distance), ...] for diagnostics.
    """
    if method == "hungarian":
        return _hungarian_match(pred, frame)
    if method == "greedy":
        return _greedy_match(pred, frame)
    raise ValueError(f"unknown match method: {method}")
