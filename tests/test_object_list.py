"""Tests for the shared object-list representation."""
from __future__ import annotations

import numpy as np

from models.object_list import grid_to_objects, objects_to_grid


def test_empty_grid() -> None:
    g = np.zeros((4, 4), dtype=np.int64)
    f = grid_to_objects(g, env_marker="test")
    assert f.grid_shape == (4, 4)
    # Single uniform color — dropped as background.
    assert len(f.objects) == 0


def test_single_object() -> None:
    g = np.zeros((5, 5), dtype=np.int64)
    g[2, 2] = 7
    f = grid_to_objects(g, env_marker="test")
    assert len(f.objects) == 1
    o = f.objects[0]
    assert o.color_id == 7
    assert o.size == 1
    assert o.is_singleton
    assert o.bbox == (2, 2, 2, 2)
    assert o.touches_edge is False


def test_two_objects_different_colors() -> None:
    g = np.zeros((5, 5), dtype=np.int64)
    g[0, 0] = 3       # 1-cell object, touches edge
    g[2:4, 2:4] = 5   # 2x2 block, doesn't touch edge
    f = grid_to_objects(g, env_marker="test")
    assert len(f.objects) == 2
    # Sorted by size descending
    assert f.objects[0].color_id == 5
    assert f.objects[0].size == 4
    assert f.objects[0].bbox == (2, 2, 3, 3)
    assert f.objects[1].color_id == 3
    assert f.objects[1].size == 1
    assert f.objects[1].touches_edge is True


def test_two_components_same_color() -> None:
    g = np.zeros((5, 5), dtype=np.int64)
    g[0, 0] = 7
    g[4, 4] = 7
    f = grid_to_objects(g, env_marker="test")
    # Two separate components even though same color
    assert len(f.objects) == 2
    assert all(o.color_id == 7 for o in f.objects)


def test_8_connectivity() -> None:
    # Diagonal cells: with 4-conn they're separate components, with 8-conn they merge.
    g = np.zeros((4, 4), dtype=np.int64)
    g[1, 1] = 5
    g[2, 2] = 5
    f4 = grid_to_objects(g, env_marker="test", connectivity=4)
    f8 = grid_to_objects(g, env_marker="test", connectivity=8)
    assert len(f4.objects) == 2
    assert len(f8.objects) == 1


def test_drop_background_disabled() -> None:
    g = np.zeros((3, 3), dtype=np.int64)
    g[1, 1] = 1
    # Without drop_background we keep all components, including the bg ring.
    f = grid_to_objects(g, env_marker="test", drop_background=False)
    assert len(f.objects) == 2


def test_to_array_padding() -> None:
    g = np.zeros((5, 5), dtype=np.int64)
    g[0, 0] = 1
    g[4, 4] = 2
    f = grid_to_objects(g, env_marker="test")
    tokens, mask = f.to_array(max_objects=8)
    assert tokens.shape == (8, 13)
    assert mask.shape == (8,)
    assert mask[:2].sum() == 2.0
    assert mask[2:].sum() == 0.0


def test_frame_raw_is_copied_not_referenced() -> None:
    """Regression: Frame.raw must hold a copy, not a reference to the caller's grid.

    Bug history: collect_sudoku_one mutated `grid` in place and stored snapshots
    via Frame; they all ended up showing the final state because Frame.raw shared
    memory with the caller's mutating array.
    """
    g = np.zeros((4, 4), dtype=np.int64)
    g[1, 1] = 3
    f1 = grid_to_objects(g, env_marker="test")
    g[1, 1] = 7  # mutate caller's array
    g[2, 2] = 9
    f2 = grid_to_objects(g, env_marker="test")
    # f1.raw must reflect the original state, not the mutated one.
    assert f1.raw[1, 1] == 3
    assert f1.raw[2, 2] == 0
    assert f2.raw[1, 1] == 7
    assert f2.raw[2, 2] == 9


def test_round_trip_simple() -> None:
    # Round-trip via objects_to_grid is bbox-fill, lossy in shape but should
    # preserve color in the bbox region.
    g = np.zeros((5, 5), dtype=np.int64)
    g[1:3, 1:3] = 7
    f = grid_to_objects(g, env_marker="test")
    recon = objects_to_grid(f, background_color=0)
    assert (recon == g).all()  # 2x2 block is its own bbox, lossless here
