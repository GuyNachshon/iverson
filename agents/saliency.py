"""Saliency: identify the most action-worthy objects in a frame.

The saliency score answers: "if I were going to interact with one object,
which would I pick?" Used to:
  - Pick a click target for ACTION6 (complex action with x,y).
  - Bias action scoring toward salient objects.

Saliency components (combined linearly):
  - **Color rarity**: rare colors are more salient. Low color_rank = rare.
  - **Edge proximity**: edge-touching objects (UI buttons) get a bonus.
  - **Smallness**: small objects often interactable; bonus.
  - **Distinctive aspect**: very tall/wide objects stand out.

Tunable weights — these are guesses; first-pass v3.5 won't tune them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arcengine import GameAction

from models.object_list import Frame, ObjectToken


@dataclass
class SaliencyConfig:
    color_rarity_weight: float = 1.0
    edge_proximity_weight: float = 1.5
    smallness_weight: float = 0.5
    aspect_distinctness_weight: float = 0.3
    min_size_for_click: int = 1
    max_size_for_click: int = 256  # avoid clicking on the background patch
    n_top_for_action: int = 8


def _smallness_score(size: int, max_size: int = 256) -> float:
    if size <= 0:
        return 0.0
    return float(np.clip(1.0 - size / max_size, 0.0, 1.0))


def _aspect_distinctness(aspect: float) -> float:
    """How much aspect deviates from 1.0 (square)."""
    return float(min(abs(aspect - 1.0), 5.0) / 5.0)


def _saliency_per_object(obj: ObjectToken, n_objects_total: int) -> float:
    """Combined saliency score for one object."""
    cfg = SaliencyConfig()
    color_rarity = 0.0 if n_objects_total <= 1 else (obj.color_rank / max(n_objects_total - 1, 1))
    edge = 1.0 if obj.touches_edge else 0.0
    small = _smallness_score(obj.size)
    asp = _aspect_distinctness(obj.aspect)
    return float(
        cfg.color_rarity_weight * color_rarity
        + cfg.edge_proximity_weight * edge
        + cfg.smallness_weight * small
        + cfg.aspect_distinctness_weight * asp
    )


def score_objects(frame: Frame) -> np.ndarray:
    if not frame.objects:
        return np.zeros(0, dtype=np.float32)
    n = len(frame.objects)
    return np.asarray([_saliency_per_object(o, n) for o in frame.objects],
                       dtype=np.float32)


def best_click_target(frame: Frame, exclude_idxs: set[int] | None = None
                       ) -> tuple[int, int]:
    """Return (x, y) on a 64x64 grid for ACTION6 click.

    Picks the highest-saliency object (excluding `exclude_idxs`) that is
    within the size range we'd interact with. Falls back to (32, 32) if no
    valid object exists.
    """
    cfg = SaliencyConfig()
    if not frame.objects:
        return 32, 32
    scores = score_objects(frame)
    exclude = exclude_idxs or set()
    H, W = frame.grid_shape
    candidates = []
    for i, obj in enumerate(frame.objects):
        if i in exclude:
            continue
        if obj.size < cfg.min_size_for_click or obj.size > cfg.max_size_for_click:
            continue
        candidates.append((i, scores[i]))
    if not candidates:
        return 32, 32
    candidates.sort(key=lambda x: -x[1])
    best_idx = candidates[0][0]
    obj = frame.objects[best_idx]
    cx_norm = obj.centroid_norm[0]
    cy_norm = obj.centroid_norm[1]
    # Convert normalized [0,1] over (H,W) → integer (x,y) in [0,63].
    # The grid native coord is (col, row) → (x, y).
    x = int(round(cx_norm * (W - 1)))
    y = int(round(cy_norm * (H - 1)))
    x = max(0, min(63, x))
    y = max(0, min(63, y))
    return x, y


def score_action(frame: Frame, action_id: int) -> float:
    """Per-action saliency score (action-uniform with a bonus for ACTION6).

    Simple actions (1-5, 7) get a constant score; ACTION6 gets a bonus
    proportional to the best object's saliency.
    """
    if action_id == GameAction.ACTION6.value:
        if not frame.objects:
            return 0.0
        scores = score_objects(frame)
        # Top-k mean of saliencies — "is there something worth clicking?"
        cfg = SaliencyConfig()
        topk = sorted(scores, reverse=True)[:cfg.n_top_for_action]
        return float(np.mean(topk)) if topk else 0.0
    return 0.0  # simple actions tie at the saliency layer
