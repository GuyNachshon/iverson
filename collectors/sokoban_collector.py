"""Collect Sokoban trajectories via BFS solver.

We solve each generated room with state-space BFS over (player_pos,
frozenset(box_positions)) and replay the optimal sequence through gym-sokoban
to capture per-step room_state observations.

Sokoban room_state encoding:
  0 = outside-wall, 1 = floor, 2 = target, 3 = box-on-target, 4 = box, 5 = player

The 8-action variant (5..8 are walks, 1..4 are pushes) is what gym-sokoban
exposes. The BFS uses 4 directional moves; gym-sokoban's "push" semantics
work: trying to walk into a box pushes it if the cell behind is empty.

Tradeoff: BFS over even 7x7 with 2 boxes can have ~10^4 states. Cap the
search at a reasonable node budget (50k) so unsolvable rooms don't hang.
"""
from __future__ import annotations

import logging
import warnings
from collections import deque
from typing import Iterator, Optional

import numpy as np

# gym 0.26 references np.bool8 which NumPy 2.0 removed. Restore the alias before
# importing gym so its passive env checker doesn't crash.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_  # type: ignore[attr-defined]

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")
warnings.filterwarnings("ignore", message="Gym has been unmaintained")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

import gym  # noqa: E402
import gym_sokoban  # noqa: E402, F401  (registers envs)

from models.converters import sokoban_to_frame  # noqa: E402
from models.trajectory import Trajectory  # noqa: E402

logger = logging.getLogger(__name__)


# gym-sokoban actions: 1=up, 2=down, 3=left, 4=right (push-or-walk depending on
# whether a box is in front).
_DIRS = {
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}


def _extract_state(room_state: np.ndarray, room_fixed: np.ndarray) -> tuple[tuple[int, int], frozenset]:
    """Extract (player_position, set_of_box_positions) from room_state."""
    # room_state values: 1=floor, 2=target (empty), 3=box-on-target, 4=box, 5=player.
    # room_fixed values: 1=floor, 2=target.
    boxes = set()
    H, W = room_state.shape
    px, py = -1, -1
    for y in range(H):
        for x in range(W):
            v = int(room_state[y, x])
            if v == 5:
                px, py = x, y
            elif v == 4 or v == 3:
                boxes.add((x, y))
    return ((px, py), frozenset(boxes))


def _is_walkable(room_fixed: np.ndarray, x: int, y: int) -> bool:
    H, W = room_fixed.shape
    if not (0 <= x < W and 0 <= y < H):
        return False
    return int(room_fixed[y, x]) in (1, 2)  # floor or target


def _is_solved(boxes: frozenset, targets: set) -> bool:
    return boxes == targets


def _solve(room_state: np.ndarray, room_fixed: np.ndarray,
           max_nodes: int = 50_000) -> Optional[list[int]]:
    """BFS for the shortest action sequence solving this Sokoban room.

    Returns list of actions (1..4) or None if unsolvable / timed out.
    """
    targets = set()
    H, W = room_fixed.shape
    for y in range(H):
        for x in range(W):
            if int(room_fixed[y, x]) == 2:
                targets.add((x, y))

    start_player, start_boxes = _extract_state(room_state, room_fixed)
    if _is_solved(start_boxes, targets):
        return []

    visited: set[tuple[tuple[int, int], frozenset]] = set()
    visited.add((start_player, start_boxes))
    queue: deque = deque([(start_player, start_boxes, [])])
    nodes = 0
    while queue:
        nodes += 1
        if nodes > max_nodes:
            return None
        (px, py), boxes, path = queue.popleft()
        for action, (dy, dx) in _DIRS.items():
            nx, ny = px + dx, py + dy
            if not _is_walkable(room_fixed, nx, ny):
                continue
            new_boxes = boxes
            if (nx, ny) in boxes:
                # Push: cell beyond box must be walkable AND not contain another box.
                bx, by = nx + dx, ny + dy
                if not _is_walkable(room_fixed, bx, by):
                    continue
                if (bx, by) in boxes:
                    continue
                new_boxes = (boxes - {(nx, ny)}) | {(bx, by)}
            new_player = (nx, ny)
            key = (new_player, new_boxes)
            if key in visited:
                continue
            visited.add(key)
            new_path = path + [action]
            if _is_solved(new_boxes, targets):
                return new_path
            queue.append((new_player, new_boxes, new_path))
    return None


def collect_one(env_id: str, seed: int, max_solve_nodes: int = 50_000) -> Optional[Trajectory]:
    """Generate a Sokoban room, solve it, replay the solution capturing frames."""
    env = gym.make(env_id)
    unw = env.unwrapped
    unw.seed(seed)
    env.reset()

    room_state = unw.room_state.copy()
    room_fixed = unw.room_fixed.copy()

    actions = _solve(room_state, room_fixed, max_nodes=max_solve_nodes)
    if actions is None:
        return None

    # Replay through env to capture per-step room_state. Use unwrapped to
    # bypass gym 0.26's passive env checker (broken on numpy 2.0).
    frames = [sokoban_to_frame(unw.room_state.copy())]
    rewards: list[float] = []
    for action in actions:
        result = unw.step(action)
        if len(result) == 4:
            _, r, _term, _info = result
        else:
            _, r, _term, _trunc, _info = result
        frames.append(sokoban_to_frame(unw.room_state.copy()))
        rewards.append(float(r))

    return Trajectory(
        env_marker="sokoban",
        run_id=f"{env_id}::seed{seed}",
        frames=frames,
        actions=actions,
        rewards=rewards,
        success=True,
        metadata={"env_id": env_id, "seed": seed,
                  "dim_room": list(unw.dim_room),
                  "num_boxes": int(unw.num_boxes)},
    )


DEFAULT_ENVS = [
    "Sokoban-small-v0",       # 7x7, 2 boxes
    "Sokoban-small-v1",       # variant
    "Sokoban-v0",             # default 10x10, 4 boxes (harder)
]


def collect(
    env_ids: Optional[list[str]] = None,
    n_per_env: int = 100,
    seed_offset: int = 0,
    max_solve_nodes: int = 50_000,
) -> Iterator[Trajectory]:
    env_ids = env_ids or DEFAULT_ENVS
    for env_id in env_ids:
        successes = 0
        attempts = 0
        # Cap attempts to 4x target since some seeds will time out
        while successes < n_per_env and attempts < 4 * n_per_env:
            seed = seed_offset + attempts
            attempts += 1
            try:
                traj = collect_one(env_id, seed=seed, max_solve_nodes=max_solve_nodes)
            except Exception as e:
                logger.debug(f"collect_one crashed for {env_id} seed={seed}: {e!r}")
                continue
            if traj is None:
                continue
            successes += 1
            yield traj
        logger.info(f"  {env_id}: {successes}/{attempts} solved")
