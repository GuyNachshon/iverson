"""Scripted-solver puzzle collectors: Sudoku, Nonogram, 15-puzzle.

Each generates a starting state and a forward-only solving trajectory. Frames
are captured *between* moves so the predictor sees gradual progress toward
the terminal state.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Iterator, Optional

import numpy as np

from models.converters import (
    fifteen_puzzle_to_frame,
    nonogram_to_frame,
    sudoku_to_frame,
)
from models.trajectory import Trajectory

logger = logging.getLogger(__name__)


# ============================================================================
# Sudoku
# ============================================================================

def _sudoku_valid(grid: np.ndarray, row: int, col: int, val: int) -> bool:
    if val in grid[row]:
        return False
    if val in grid[:, col]:
        return False
    br, bc = (row // 3) * 3, (col // 3) * 3
    if val in grid[br:br + 3, bc:bc + 3]:
        return False
    return True


def _sudoku_candidates(grid: np.ndarray, row: int, col: int) -> list[int]:
    return [v for v in range(1, 10) if _sudoku_valid(grid, row, col, v)]


def _sudoku_solve_inplace(grid: np.ndarray, rng: random.Random) -> bool:
    """Backtracking solver. Returns True if solved."""
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            if grid[y, x] == 0:
                cands = _sudoku_candidates(grid, y, x)
                rng.shuffle(cands)
                for v in cands:
                    grid[y, x] = v
                    if _sudoku_solve_inplace(grid, rng):
                        return True
                    grid[y, x] = 0
                return False
    return True


def _generate_sudoku(seed: int, num_clues: int = 32) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Generate a (puzzle, solution) pair. Returns None on failure."""
    rng = random.Random(seed)
    solution = np.zeros((9, 9), dtype=np.int64)
    if not _sudoku_solve_inplace(solution, rng):
        return None
    puzzle = solution.copy()
    cells = [(y, x) for y in range(9) for x in range(9)]
    rng.shuffle(cells)
    for y, x in cells[: 81 - num_clues]:
        puzzle[y, x] = 0
    return puzzle, solution


def _sudoku_solve_trace(puzzle: np.ndarray) -> Optional[list[tuple[int, int, int]]]:
    """Forward-only solver: at each step, find the cell with fewest candidates and place it.

    If a cell has 1 candidate (forced), use that. If multiple, pick lowest.
    Returns list of (row, col, value) moves, or None if a contradiction arises.
    """
    grid = puzzle.copy()
    moves: list[tuple[int, int, int]] = []
    while True:
        empty = [(y, x) for y in range(9) for x in range(9) if grid[y, x] == 0]
        if not empty:
            return moves
        # Find cell with fewest candidates
        best = None
        best_cands: list[int] = []
        for y, x in empty:
            cands = _sudoku_candidates(grid, y, x)
            if not cands:
                return None
            if best is None or len(cands) < len(best_cands):
                best = (y, x)
                best_cands = cands
                if len(cands) == 1:
                    break
        if best is None:
            return None
        y, x = best
        # Take the smallest candidate (deterministic forward solve)
        v = best_cands[0]
        grid[y, x] = v
        moves.append((y, x, v))


def collect_sudoku_one(seed: int, num_clues: int = 35) -> Optional[Trajectory]:
    gen = _generate_sudoku(seed, num_clues=num_clues)
    if gen is None:
        return None
    puzzle, _solution = gen

    moves = _sudoku_solve_trace(puzzle)
    if moves is None:
        return None

    grid = puzzle.copy()
    frames = [sudoku_to_frame(grid)]
    actions: list[int] = []
    rewards: list[float] = []
    for (y, x, v) in moves:
        grid[y, x] = v
        # Encode action as a single int: y * 81 + x * 9 + (v - 1) ∈ [0, 729)
        actions.append(int(y * 81 + x * 9 + (v - 1)))
        rewards.append(0.0)
        frames.append(sudoku_to_frame(grid))
    rewards[-1] = 1.0  # terminal reward

    return Trajectory(
        env_marker="sudoku",
        run_id=f"sudoku-{num_clues}clues::seed{seed}",
        frames=frames,
        actions=actions,
        rewards=rewards,
        success=True,
        metadata={"seed": seed, "num_clues": num_clues, "moves": len(moves)},
    )


# ============================================================================
# 15-puzzle
# ============================================================================

# Goal state: 1..15 then 0 (blank) at bottom-right
_FIFTEEN_GOAL = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 0]],
    dtype=np.int64,
)


def _scramble_fifteen(rng: random.Random, n_moves: int) -> np.ndarray:
    grid = _FIFTEEN_GOAL.copy()
    by, bx = 3, 3
    for _ in range(n_moves):
        moves = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = by + dy, bx + dx
            if 0 <= ny < 4 and 0 <= nx < 4:
                moves.append((ny, nx))
        ny, nx = rng.choice(moves)
        grid[by, bx], grid[ny, nx] = grid[ny, nx], grid[by, bx]
        by, bx = ny, nx
    return grid


def _manhattan(grid: np.ndarray) -> int:
    total = 0
    for y in range(4):
        for x in range(4):
            v = int(grid[y, x])
            if v == 0:
                continue
            ty, tx = (v - 1) // 4, (v - 1) % 4
            total += abs(y - ty) + abs(x - tx)
    return total


def _solve_fifteen(start: np.ndarray, max_states: int = 200_000) -> Optional[list[tuple[int, int]]]:
    """A* with Manhattan heuristic. Returns list of blank-cell positions visited."""
    start_t = tuple(start.flatten().tolist())
    goal_t = tuple(_FIFTEEN_GOAL.flatten().tolist())
    if start_t == goal_t:
        return []
    # Find blank
    by, bx = next((y, x) for y in range(4) for x in range(4) if start[y, x] == 0)

    # Priority queue (heapq) keyed by (f, counter)
    import heapq
    counter = 0
    init_h = _manhattan(start)
    heap: list = [(init_h, counter, 0, start_t, by, bx, [])]
    visited: dict = {start_t: 0}
    while heap and len(visited) < max_states:
        f, _, g, state_t, by, bx, path = heapq.heappop(heap)
        if state_t == goal_t:
            return path
        state = np.array(state_t, dtype=np.int64).reshape(4, 4)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = by + dy, bx + dx
            if not (0 <= ny < 4 and 0 <= nx < 4):
                continue
            new_state = state.copy()
            new_state[by, bx], new_state[ny, nx] = new_state[ny, nx], new_state[by, bx]
            new_t = tuple(new_state.flatten().tolist())
            new_g = g + 1
            if new_t in visited and visited[new_t] <= new_g:
                continue
            visited[new_t] = new_g
            new_h = _manhattan(new_state)
            counter += 1
            heapq.heappush(heap, (new_g + new_h, counter, new_g, new_t, ny, nx, path + [(ny, nx)]))
    return None


def collect_fifteen_one(seed: int, scramble_moves: int = 20) -> Optional[Trajectory]:
    rng = random.Random(seed)
    start = _scramble_fifteen(rng, scramble_moves)
    path = _solve_fifteen(start)
    if path is None:
        return None
    grid = start.copy()
    by, bx = next((y, x) for y in range(4) for x in range(4) if grid[y, x] == 0)
    frames = [fifteen_puzzle_to_frame(grid)]
    actions: list[int] = []
    rewards: list[float] = []
    for (ny, nx) in path:
        # Direction encoding: 0=up, 1=down, 2=left, 3=right (where blank moves)
        dy, dx = ny - by, nx - bx
        action = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}[(dy, dx)]
        grid[by, bx], grid[ny, nx] = grid[ny, nx], grid[by, bx]
        by, bx = ny, nx
        actions.append(action)
        rewards.append(0.0)
        frames.append(fifteen_puzzle_to_frame(grid))
    if rewards:
        rewards[-1] = 1.0

    return Trajectory(
        env_marker="fifteen",
        run_id=f"fifteen-s{scramble_moves}::seed{seed}",
        frames=frames,
        actions=actions,
        rewards=rewards,
        success=True,
        metadata={"seed": seed, "scramble_moves": scramble_moves, "solve_moves": len(path)},
    )


# ============================================================================
# Nonogram
# ============================================================================

def _nonogram_clues(grid: np.ndarray) -> tuple[list[list[int]], list[list[int]]]:
    """Compute row and column clues for a 0/1 grid."""
    H, W = grid.shape
    rows: list[list[int]] = []
    for y in range(H):
        runs = []
        cur = 0
        for x in range(W):
            if grid[y, x] == 1:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        rows.append(runs or [0])
    cols: list[list[int]] = []
    for x in range(W):
        runs = []
        cur = 0
        for y in range(H):
            if grid[y, x] == 1:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        cols.append(runs or [0])
    return rows, cols


def _line_possibilities(line_len: int, clues: list[int]) -> list[tuple[int, ...]]:
    """All valid 0/1 fills for a row/col matching the clues."""
    if clues == [0]:
        return [tuple([0] * line_len)]
    total = sum(clues) + len(clues) - 1
    if total > line_len:
        return []
    results = []

    def fill(idx: int, pos: int, current: list[int]) -> None:
        if idx == len(clues):
            results.append(tuple(current + [0] * (line_len - len(current))))
            return
        run = clues[idx]
        max_start = line_len - sum(clues[idx:]) - (len(clues) - idx - 1)
        for start in range(pos, max_start + 1):
            new = current + [0] * (start - pos) + [1] * run
            if idx < len(clues) - 1:
                new = new + [0]
            fill(idx + 1, start + run + (1 if idx < len(clues) - 1 else 0), new)

    fill(0, 0, [])
    return results


def _solve_nonogram(rows_clues: list[list[int]], cols_clues: list[list[int]],
                    H: int, W: int, max_iters: int = 1000) -> Optional[list[tuple[int, int, int]]]:
    """Constraint-propagation solver: at each step, find a cell forced by all line possibilities.

    Returns list of (row, col, value) cell-fill moves, or None if can't finish via propagation alone.
    Cell values: 0 = empty, 1 = filled.
    """
    grid = np.full((H, W), -1, dtype=np.int64)  # -1 = unknown
    moves: list[tuple[int, int, int]] = []

    # Per-line possibility caches
    row_poss = [_line_possibilities(W, c) for c in rows_clues]
    col_poss = [_line_possibilities(H, c) for c in cols_clues]
    if any(not p for p in row_poss + col_poss):
        return None

    for _ in range(max_iters):
        progress = False
        # Filter possibilities by current grid state
        for y in range(H):
            row_poss[y] = [p for p in row_poss[y]
                            if all(grid[y, x] == -1 or grid[y, x] == p[x] for x in range(W))]
        for x in range(W):
            col_poss[x] = [p for p in col_poss[x]
                            if all(grid[y, x] == -1 or grid[y, x] == p[y] for y in range(H))]
        if any(not p for p in row_poss + col_poss):
            return None
        # Find forced cells
        for y in range(H):
            for x in range(W):
                if grid[y, x] != -1:
                    continue
                row_vals = {p[x] for p in row_poss[y]}
                col_vals = {p[y] for p in col_poss[x]}
                forced = row_vals & col_vals
                if len(forced) == 1:
                    v = forced.pop()
                    grid[y, x] = v
                    moves.append((y, x, v))
                    progress = True
        if not progress:
            break
    if (grid == -1).any():
        return None
    return moves


def _generate_nonogram(seed: int, H: int, W: int, fill_prob: float = 0.5
                       ) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    rng = np.random.default_rng(seed)
    grid = (rng.random((H, W)) < fill_prob).astype(np.int64)
    rows, cols = _nonogram_clues(grid)
    return grid, rows, cols


def collect_nonogram_one(seed: int, H: int = 5, W: int = 5) -> Optional[Trajectory]:
    target, rows_clues, cols_clues = _generate_nonogram(seed, H, W)
    moves = _solve_nonogram(rows_clues, cols_clues, H, W)
    if moves is None:
        return None
    grid = np.zeros((H, W), dtype=np.int64)  # start with all-unknown rendered as 0
    # Use distinct color ids: 0 = unknown/empty, 1 = empty-confirmed, 2 = filled
    frames = [nonogram_to_frame(grid)]
    actions: list[int] = []
    rewards: list[float] = []
    for (y, x, v) in moves:
        # Map: v=0 (empty cell) -> color 1, v=1 (filled cell) -> color 2
        grid[y, x] = 1 if v == 0 else 2
        actions.append(int(y * W + x) + (v * H * W))
        rewards.append(0.0)
        frames.append(nonogram_to_frame(grid))
    if rewards:
        rewards[-1] = 1.0

    return Trajectory(
        env_marker="nonogram",
        run_id=f"nonogram-{H}x{W}::seed{seed}",
        frames=frames,
        actions=actions,
        rewards=rewards,
        success=True,
        metadata={"seed": seed, "H": H, "W": W, "moves": len(moves)},
    )


# ============================================================================
# Drivers
# ============================================================================

def collect_sudoku(n: int, seed_offset: int = 0,
                   num_clues_choices: tuple[int, ...] = (35, 38, 40)
                   ) -> Iterator[Trajectory]:
    successes = 0
    attempts = 0
    while successes < n and attempts < 4 * n:
        seed = seed_offset + attempts
        nc = num_clues_choices[attempts % len(num_clues_choices)]
        attempts += 1
        traj = collect_sudoku_one(seed, num_clues=nc)
        if traj is None:
            continue
        successes += 1
        yield traj
    logger.info(f"  sudoku: {successes}/{attempts}")


def collect_fifteen(n: int, seed_offset: int = 0,
                    scramble_choices: tuple[int, ...] = (10, 15, 20, 25)
                    ) -> Iterator[Trajectory]:
    successes = 0
    attempts = 0
    while successes < n and attempts < 4 * n:
        seed = seed_offset + attempts
        scramble = scramble_choices[attempts % len(scramble_choices)]
        attempts += 1
        traj = collect_fifteen_one(seed, scramble_moves=scramble)
        if traj is None:
            continue
        successes += 1
        yield traj
    logger.info(f"  fifteen: {successes}/{attempts}")


def collect_nonogram(n: int, seed_offset: int = 0,
                     size_choices: tuple[tuple[int, int], ...] = ((5, 5), (5, 7), (7, 7))
                     ) -> Iterator[Trajectory]:
    successes = 0
    attempts = 0
    while successes < n and attempts < 4 * n:
        seed = seed_offset + attempts
        H, W = size_choices[attempts % len(size_choices)]
        attempts += 1
        traj = collect_nonogram_one(seed, H=H, W=W)
        if traj is None:
            continue
        successes += 1
        yield traj
    logger.info(f"  nonogram: {successes}/{attempts}")
