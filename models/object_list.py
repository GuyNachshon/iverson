"""Shared object-list representation across environments.

A grid (or rendered observation) is decomposed into a list of objects, each a
typed token of geometric features. Designed so the *same* schema describes
ARC-AGI-3 grids, MiniGrid symbolic grids, Sokoban boards, etc. — the
terminal-state predictor learns environment-invariant goal structure by
seeing the same geometric tokens across many surfaces.

Token schema (all fields per object):
  - color_id (int)             — raw color/type id from the source env (env-specific)
  - color_rank (int)           — rank by frequency (0 = most frequent / "background")
  - size (int)                 — number of cells in the component
  - bbox (x_min, y_min, x_max, y_max)
  - centroid_norm (cx, cy)     — centroid coords normalized to [0,1] over grid
  - aspect (float)             — bbox width / max(bbox height, 1)
  - is_singleton (bool)        — size == 1
  - touches_edge (bool)        — bbox touches grid boundary
  - touches_others (int)       — count of distinct neighboring (different-color) objects
  - env_marker (str)           — env identifier (e.g., "arc_agi_3", "minigrid")

A `Frame` is the full set of objects from one observation, plus metadata.

Note on slot attention (revisit later):
  Connected-components-by-color is a deterministic, lossless decomposition for
  grid worlds. For envs with smooth visual features (Doom RGB, Mario sprites)
  it'll be inadequate and we'd add a slot-attention head trained on Tier C
  rendered observations. Until then, every Tier A+D env is grid/symbolic and
  this approach is sufficient.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ObjectToken:
    color_id: int
    color_rank: int
    size: int
    bbox: tuple[int, int, int, int]   # x_min, y_min, x_max, y_max (inclusive)
    centroid_norm: tuple[float, float]
    aspect: float
    is_singleton: bool
    touches_edge: bool
    touches_others: int
    env_marker: str

    def to_vector(self) -> np.ndarray:
        """Compact float vector representation for model consumption.

        Layout (12 floats):
          [color_id, color_rank, log_size,
           x_min_n, y_min_n, x_max_n, y_max_n,
           cx, cy, aspect, is_singleton, touches_edge, touches_others_log]
        """
        x_min, y_min, x_max, y_max = self.bbox
        # bbox normalization happens at Frame.to_array level (need grid size)
        return np.array([
            self.color_id,
            self.color_rank,
            np.log1p(self.size),
            x_min, y_min, x_max, y_max,
            self.centroid_norm[0], self.centroid_norm[1],
            self.aspect,
            float(self.is_singleton),
            float(self.touches_edge),
            np.log1p(self.touches_others),
        ], dtype=np.float32)


@dataclass
class Frame:
    """Object-list view of one observation. Includes raw shape so coords decode."""

    objects: list[ObjectToken]
    grid_shape: tuple[int, int]
    env_marker: str
    raw: Optional[np.ndarray] = field(default=None, repr=False)  # original grid for debugging

    def __len__(self) -> int:
        return len(self.objects)

    def to_array(self, max_objects: int = 64, pad_value: float = 0.0) -> np.ndarray:
        """Pack objects into a fixed-size (max_objects, D) array with mask.

        Returns (tokens, mask) where mask[i]=1 if object i is real, 0 if pad.
        """
        D = 13  # must match ObjectToken.to_vector
        out = np.full((max_objects, D), pad_value, dtype=np.float32)
        mask = np.zeros((max_objects,), dtype=np.float32)
        H, W = self.grid_shape
        for i, obj in enumerate(self.objects[:max_objects]):
            v = obj.to_vector()
            # bbox normalization
            v[3] = v[3] / max(W - 1, 1)
            v[4] = v[4] / max(H - 1, 1)
            v[5] = v[5] / max(W - 1, 1)
            v[6] = v[6] / max(H - 1, 1)
            out[i] = v
            mask[i] = 1.0
        return out, mask


# ---------------------------------------------------------------------------
# Connected-components decomposition
# ---------------------------------------------------------------------------

def _flood_fill_components(grid: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Per-color connected components. Returns (label grid, num_components).

    Label 0 reserved for "no component" (shouldn't happen — every cell has a color).
    Each non-zero label corresponds to one (color, component) pair.

    connectivity: 4 (orthogonal only) or 8 (orthogonal + diagonal).
    """
    H, W = grid.shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_label = 1

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    for y in range(H):
        for x in range(W):
            if labels[y, x] != 0:
                continue
            color = int(grid[y, x])
            # BFS
            stack = [(y, x)]
            labels[y, x] = next_label
            while stack:
                cy, cx = stack.pop()
                for dy, dx in offsets:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] == 0 and int(grid[ny, nx]) == color:
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
            next_label += 1

    return labels, next_label - 1


def grid_to_objects(
    grid: np.ndarray,
    env_marker: str,
    connectivity: int = 4,
    drop_background: bool = True,
    min_size: int = 1,
) -> Frame:
    """Decompose a 2D integer-color grid into an object list.

    Args:
        grid: 2D int array, cell values are color/type ids.
        env_marker: identifier string for downstream env-conditional features.
        connectivity: 4 (default) or 8 for component adjacency.
        drop_background: if True, drop the most-frequent-color components (treat
            as background). Most ARC-AGI-3 games have a uniform background.
        min_size: minimum component size to keep.

    Returns:
        Frame with objects sorted by size descending, then by bbox (y, x).
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {grid.shape}")
    H, W = grid.shape

    # Color frequency (for color_rank and background detection)
    color_counts = Counter(int(c) for c in grid.flatten())
    sorted_colors = [c for c, _ in color_counts.most_common()]
    color_to_rank = {c: i for i, c in enumerate(sorted_colors)}
    background_color = sorted_colors[0] if sorted_colors else None

    labels, num = _flood_fill_components(grid, connectivity=connectivity)

    # Build per-component metadata
    objects: list[ObjectToken] = []
    for label in range(1, num + 1):
        ys, xs = np.where(labels == label)
        if len(ys) < min_size:
            continue
        color = int(grid[ys[0], xs[0]])
        if drop_background and color == background_color:
            continue

        x_min, y_min = int(xs.min()), int(ys.min())
        x_max, y_max = int(xs.max()), int(ys.max())
        size = int(len(ys))
        cx = float(xs.mean()) / max(W - 1, 1)
        cy = float(ys.mean()) / max(H - 1, 1)
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        aspect = bbox_w / max(bbox_h, 1)
        is_singleton = size == 1
        touches_edge = (x_min == 0 or y_min == 0 or x_max == W - 1 or y_max == H - 1)
        # Count distinct neighboring (different-color) labels
        neighbor_labels: set = set()
        for y, x in zip(ys, xs):
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] != label:
                    neighbor_labels.add(int(labels[ny, nx]))
        touches_others = len(neighbor_labels)

        objects.append(ObjectToken(
            color_id=color,
            color_rank=color_to_rank.get(color, -1),
            size=size,
            bbox=(x_min, y_min, x_max, y_max),
            centroid_norm=(cx, cy),
            aspect=aspect,
            is_singleton=is_singleton,
            touches_edge=touches_edge,
            touches_others=touches_others,
            env_marker=env_marker,
        ))

    objects.sort(key=lambda o: (-o.size, o.bbox[1], o.bbox[0]))

    return Frame(objects=objects, grid_shape=(H, W), env_marker=env_marker, raw=grid)


# ---------------------------------------------------------------------------
# Round-trip reconstruction (sanity check, not actually used for prediction)
# ---------------------------------------------------------------------------

def objects_to_grid(frame: Frame, background_color: int = 0) -> np.ndarray:
    """Reconstruct an approximate grid from an object list (bbox fill).

    Lossy: doesn't preserve component shape, only bbox + color. Used for
    sanity checking that the decomposition captures the right things.
    """
    H, W = frame.grid_shape
    out = np.full((H, W), background_color, dtype=np.int64)
    # Render in size-descending order so smaller objects overlay larger ones
    for obj in sorted(frame.objects, key=lambda o: -o.size):
        x_min, y_min, x_max, y_max = obj.bbox
        out[y_min:y_max + 1, x_min:x_max + 1] = obj.color_id
    return out
