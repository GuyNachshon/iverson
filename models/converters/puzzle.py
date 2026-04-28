"""Puzzle observation → object-list Frame for scripted-solver puzzles.

Sudoku, Nonogram, and 15-puzzle all reduce to a 2D integer grid where each
cell value encodes (state, value). We use distinct color_id ranges to keep
puzzle types separable in the corpus:
  - Sudoku: 0 = empty, 1..9 = digits  (10 ids)
  - Nonogram: 0 = unknown, 1 = empty, 2 = filled, 3 = pinned-clue  (4 ids)
  - 15-puzzle: 0 = blank, 1..15 = tile values  (16 ids)
"""
from __future__ import annotations

import numpy as np

from ..object_list import Frame, grid_to_objects


def sudoku_to_frame(grid: np.ndarray) -> Frame:
    return grid_to_objects(np.asarray(grid, dtype=np.int64),
                            env_marker="sudoku", drop_background=True)


def nonogram_to_frame(grid: np.ndarray) -> Frame:
    return grid_to_objects(np.asarray(grid, dtype=np.int64),
                            env_marker="nonogram", drop_background=True)


def fifteen_puzzle_to_frame(grid: np.ndarray) -> Frame:
    return grid_to_objects(np.asarray(grid, dtype=np.int64),
                            env_marker="fifteen", drop_background=True)
