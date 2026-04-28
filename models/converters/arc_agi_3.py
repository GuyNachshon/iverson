"""ARC-AGI-3 FrameData → object-list Frame.

The grid is `frame.frame[0]` (or `FrameDataRaw.frame[0]` from the LocalEnvironmentWrapper).
Cell values are 0–15 color ids on a max-64x64 grid.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..object_list import Frame, grid_to_objects

ENV_MARKER = "arc_agi_3"


def arc_agi_3_to_frame(
    arc_frame: Any,
    connectivity: int = 4,
    drop_background: bool = True,
) -> Frame:
    """Convert an ARC-AGI-3 frame (FrameData or FrameDataRaw) to a shared Frame.

    Accepts either `FrameData` (frame is list[list[list[int]]]) or
    `FrameDataRaw` (frame is list[np.ndarray]).
    """
    raw_frame = arc_frame.frame
    if not raw_frame:
        return Frame(objects=[], grid_shape=(0, 0), env_marker=ENV_MARKER)
    grid = np.asarray(raw_frame[0], dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError(f"expected 2D grid from ARC-AGI-3, got {grid.shape}")
    return grid_to_objects(
        grid,
        env_marker=ENV_MARKER,
        connectivity=connectivity,
        drop_background=drop_background,
    )
