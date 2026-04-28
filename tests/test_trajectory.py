"""Round-trip tests for trajectory serialization."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from models.object_list import grid_to_objects
from models.trajectory import (
    Trajectory,
    read_trajectories,
    reconstruct_frame_from_tokens,
    unpack_frames,
    write_trajectories,
)


def _make_frame(seed: int):
    rng = np.random.default_rng(seed)
    g = np.zeros((6, 6), dtype=np.int64)
    # Place 1-4 colored cells deterministically
    n = rng.integers(1, 5)
    for _ in range(n):
        y = int(rng.integers(0, 6))
        x = int(rng.integers(0, 6))
        c = int(rng.integers(1, 5))
        g[y, x] = c
    return grid_to_objects(g, env_marker="test")


def test_round_trip_single_trajectory() -> None:
    frames = [_make_frame(i) for i in range(5)]
    actions = [0, 1, 2, 3]
    rewards = [0.0, 0.0, 0.0, 1.0]
    t = Trajectory(
        env_marker="test",
        run_id="run_0",
        frames=frames,
        actions=actions,
        rewards=rewards,
        success=True,
        metadata={"seed": 0},
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "trajs.parquet"
        n = write_trajectories([t], path)
        assert n == 1
        rows = read_trajectories(path)
        assert len(rows) == 1
        row = rows[0]
        assert row["env_marker"] == "test"
        assert row["run_id"] == "run_0"
        assert row["success"] is True
        assert row["n_frames"] == 5
        assert row["actions"] == [0, 1, 2, 3]
        # Unpack arrays
        tok, mask = unpack_frames(row)
        from models.trajectory import _MAX_OBJECTS_DEFAULT
        assert tok.shape == (5, _MAX_OBJECTS_DEFAULT, 13)
        assert mask.shape == (5, _MAX_OBJECTS_DEFAULT)
        # Per-frame n_objects matches what mask says
        for i in range(5):
            assert int(mask[i].sum()) == row["frame_n_objects"][i]


def test_multiple_trajectories() -> None:
    trajectories = []
    for j in range(3):
        frames = [_make_frame(j * 10 + i) for i in range(4)]
        trajectories.append(Trajectory(
            env_marker="test",
            run_id=f"run_{j}",
            frames=frames,
            actions=[0, 1, 2],
            rewards=[0.0, 0.0, 1.0],
            success=True,
        ))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "trajs.parquet"
        write_trajectories(trajectories, path)
        rows = read_trajectories(path)
        assert len(rows) == 3
        run_ids = {r["run_id"] for r in rows}
        assert run_ids == {"run_0", "run_1", "run_2"}


def test_reconstruct_frame_lossy_but_close() -> None:
    g = np.zeros((10, 10), dtype=np.int64)
    g[2:5, 3:6] = 4    # 3x3 block, size 9
    g[7, 7] = 9        # singleton
    f = grid_to_objects(g, env_marker="test")
    tok, mask = f.to_array(max_objects=16)
    f_rec = reconstruct_frame_from_tokens(tok, mask, (10, 10), "test")
    # Same number of objects, same colors, similar sizes (log1p round-trip).
    assert len(f_rec.objects) == len(f.objects)
    for orig, rec in zip(f.objects, f_rec.objects):
        assert orig.color_id == rec.color_id
        assert abs(orig.size - rec.size) <= 1  # off-by-1 from log1p rounding
        assert orig.is_singleton == rec.is_singleton
        assert orig.touches_edge == rec.touches_edge
