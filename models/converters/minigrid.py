"""MiniGrid observation → object-list Frame.

MiniGrid envs yield observations of shape (H, W, 3) where channels encode
(object_type, color, state). We compose object_type and color into a single
"color id" so the same connected-components decomposition applies.

We don't depend on the `minigrid` package here — accept either:
  - a numpy array of shape (H, W, 3) with the standard MiniGrid encoding, or
  - a numpy array of shape (H, W) with pre-encoded ids.

The MiniGrid `OBJECT_TO_IDX` mapping (from minigrid.core.constants):
  {'unseen': 0, 'empty': 1, 'wall': 2, 'floor': 3, 'door': 4, 'key': 5,
   'ball': 6, 'box': 7, 'goal': 8, 'lava': 9, 'agent': 10}
"""
from __future__ import annotations

import numpy as np

from ..object_list import Frame, grid_to_objects

ENV_MARKER = "minigrid"

# Synthetic id = object_type * 16 + color (color is 0..5 in MiniGrid: red, green,
# blue, purple, yellow, grey). This packs both into a single int while keeping
# them recoverable. 11 types * 6 colors = 66 distinct ids.


def _compose_ids(obs: np.ndarray) -> np.ndarray:
    """obs: (H, W, 3) → (H, W) int. We use type * 16 + color."""
    if obs.ndim == 2:
        return obs.astype(np.int64)
    if obs.ndim != 3 or obs.shape[-1] != 3:
        raise ValueError(f"expected (H,W,3) or (H,W), got {obs.shape}")
    obj_type = obs[..., 0].astype(np.int64)
    color = obs[..., 1].astype(np.int64)
    return obj_type * 16 + color


def minigrid_to_frame(
    obs: np.ndarray,
    connectivity: int = 4,
    drop_background: bool = True,
) -> Frame:
    """Convert a MiniGrid observation to a shared Frame.

    `obs` may be either (H,W,3) raw MiniGrid encoding or (H,W) pre-encoded ids.
    """
    grid = _compose_ids(np.asarray(obs))
    return grid_to_objects(
        grid,
        env_marker=ENV_MARKER,
        connectivity=connectivity,
        drop_background=drop_background,
    )
