"""Per-env converters: env-specific observation → shared object-list Frame."""
from .arc_agi_3 import arc_agi_3_to_frame
from .minigrid import minigrid_to_frame
from .puzzle import (
    fifteen_puzzle_to_frame,
    nonogram_to_frame,
    sudoku_to_frame,
)
from .sokoban import sokoban_to_frame

__all__ = [
    "arc_agi_3_to_frame",
    "fifteen_puzzle_to_frame",
    "minigrid_to_frame",
    "nonogram_to_frame",
    "sokoban_to_frame",
    "sudoku_to_frame",
]
