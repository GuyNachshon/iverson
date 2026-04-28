"""Sokoban room_state grid → object-list Frame.

room_state encoding (gym-sokoban):
  0 = outside-wall (we treat as a single-color background)
  1 = floor
  2 = target
  3 = box-on-target
  4 = box
  5 = player
"""
from __future__ import annotations

import numpy as np

from ..object_list import Frame, grid_to_objects

ENV_MARKER = "sokoban"


def sokoban_to_frame(room_state: np.ndarray) -> Frame:
    grid = np.asarray(room_state, dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError(f"expected 2D room_state, got {grid.shape}")
    return grid_to_objects(grid, env_marker=ENV_MARKER, drop_background=True)
