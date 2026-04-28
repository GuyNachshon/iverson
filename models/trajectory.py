"""Shared on-disk trajectory format for the Tier 1 corpus.

A trajectory is a sequence of observations + actions ending in a terminal state.
For terminal-state-prediction training we need: the prefix (observations and
actions up to step K) and the terminal frame (final observation when the
episode ended successfully). We store the full sequence so we can sample
arbitrary prefix lengths during training.

On-disk format: parquet, one row per trajectory. Lists of object-list frames
serialize as nested arrow types. Per-frame data is a list of (max_objects, 13)
float arrays plus masks; we store these flattened with shape metadata so they
round-trip cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .object_list import Frame, ObjectToken


@dataclass
class Trajectory:
    env_marker: str
    run_id: str
    frames: list[Frame]
    actions: list[int]
    rewards: list[float]
    success: bool
    metadata: dict = field(default_factory=dict)

    @property
    def terminal_frame(self) -> Optional[Frame]:
        return self.frames[-1] if self.frames else None

    def __len__(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

# We pack each Frame as flat arrays + grid_shape. ObjectTokens are reduced to
# their .to_vector() output (13 floats); we lose env_marker per-token (it's on
# the Frame), and we lose nothing else. On load we reconstruct ObjectTokens
# from the vector, which is sufficient for training (we use vectors directly).

_MAX_OBJECTS_DEFAULT = 64
_FEATURE_DIM = 13


def _frame_to_packed(frame: Frame, max_objects: int = _MAX_OBJECTS_DEFAULT) -> dict:
    tokens, mask = frame.to_array(max_objects=max_objects)
    return {
        "tokens": tokens.astype(np.float32).tobytes(),
        "mask": mask.astype(np.float32).tobytes(),
        "n_objects": int(mask.sum()),
        "grid_h": int(frame.grid_shape[0]),
        "grid_w": int(frame.grid_shape[1]),
    }


def _packed_to_arrays(packed: dict, max_objects: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.frombuffer(packed["tokens"], dtype=np.float32).reshape(max_objects, _FEATURE_DIM).copy()
    mask = np.frombuffer(packed["mask"], dtype=np.float32).reshape(max_objects).copy()
    return tokens, mask


def write_trajectories(
    trajectories: Iterable[Trajectory],
    path: str | Path,
    max_objects: int = _MAX_OBJECTS_DEFAULT,
) -> int:
    """Write trajectories to a parquet file. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    count = 0
    for t in trajectories:
        packed_frames = [_frame_to_packed(f, max_objects=max_objects) for f in t.frames]
        rows.append({
            "env_marker": t.env_marker,
            "run_id": t.run_id,
            "success": t.success,
            "n_frames": len(t.frames),
            "n_actions": len(t.actions),
            "actions": t.actions,
            "rewards": [float(r) for r in t.rewards],
            "frame_tokens": [pf["tokens"] for pf in packed_frames],
            "frame_masks": [pf["mask"] for pf in packed_frames],
            "frame_n_objects": [pf["n_objects"] for pf in packed_frames],
            "frame_grid_h": [pf["grid_h"] for pf in packed_frames],
            "frame_grid_w": [pf["grid_w"] for pf in packed_frames],
            "metadata": str(t.metadata) if t.metadata else "",
            "max_objects": max_objects,
        })
        count += 1

    if not rows:
        return 0

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    return count


def read_trajectories(path: str | Path) -> list[dict]:
    """Read raw rows from a parquet file. Each row is a dict with packed frames.

    Use `unpack_frames` to reconstruct numpy arrays per row.
    """
    path = Path(path)
    table = pq.read_table(path)
    return table.to_pylist()


def unpack_frames(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct (frames_tokens [T, max_objects, D], masks [T, max_objects]).

    Returns numpy arrays ready for model input.
    """
    max_objects = int(row["max_objects"])
    n_frames = int(row["n_frames"])
    tokens_list = []
    masks_list = []
    for i in range(n_frames):
        tok, m = _packed_to_arrays(
            {"tokens": row["frame_tokens"][i], "mask": row["frame_masks"][i]},
            max_objects=max_objects,
        )
        tokens_list.append(tok)
        masks_list.append(m)
    if not tokens_list:
        return (np.zeros((0, max_objects, _FEATURE_DIM), dtype=np.float32),
                np.zeros((0, max_objects), dtype=np.float32))
    return np.stack(tokens_list), np.stack(masks_list)


# ---------------------------------------------------------------------------
# Reconstruction (for inspection only, not used in training)
# ---------------------------------------------------------------------------

def reconstruct_frame_from_tokens(
    tokens: np.ndarray,
    mask: np.ndarray,
    grid_shape: tuple[int, int],
    env_marker: str,
) -> Frame:
    """Inverse of Frame.to_array — useful for inspection scripts.

    Note: token vectors have lossy compression on size (log1p), so the
    reconstructed Frame's token sizes will differ slightly. Only used for
    debugging / round-trip tests, not training.
    """
    H, W = grid_shape
    objects: list[ObjectToken] = []
    for i in range(tokens.shape[0]):
        if mask[i] < 0.5:
            continue
        v = tokens[i]
        # bbox stored normalized — denormalize
        x_min = int(round(v[3] * max(W - 1, 1)))
        y_min = int(round(v[4] * max(H - 1, 1)))
        x_max = int(round(v[5] * max(W - 1, 1)))
        y_max = int(round(v[6] * max(H - 1, 1)))
        objects.append(ObjectToken(
            color_id=int(v[0]),
            color_rank=int(v[1]),
            size=int(round(np.expm1(v[2]))),
            bbox=(x_min, y_min, x_max, y_max),
            centroid_norm=(float(v[7]), float(v[8])),
            aspect=float(v[9]),
            is_singleton=bool(v[10] > 0.5),
            touches_edge=bool(v[11] > 0.5),
            touches_others=int(round(np.expm1(v[12]))),
            env_marker=env_marker,
        ))
    return Frame(objects=objects, grid_shape=grid_shape, env_marker=env_marker)
