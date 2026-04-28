"""Collect MiniGrid / BabyAI trajectories using BabyAIBot.

Strategy:
  - For each (env_id, seed) we run the BabyAIBot until termination.
  - Discard non-success trajectories (timeout, lava death, etc.).
  - Encode each observation as a shared object-list Frame via the minigrid
    converter, then bundle as a Trajectory.

This produces optimal-or-near-optimal trajectories. For our terminal-state
prediction objective, optimality is fine — the model learns terminals, not
policies. We accept the bias toward short trajectories.

Note on the obs encoding:
  Default MiniGrid obs is partial (agent's 7x7 view). We use FullyObsWrapper
  so the symbolic grid is the full env state. This matches what we want
  because terminal-state prediction needs the *world* terminal, not a partial
  view of it.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

# Suppress pygame's pkg_resources deprecation noise.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")

import gymnasium as gym  # noqa: E402
import minigrid  # noqa: E402, F401  (registers envs)
from minigrid.utils.baby_ai_bot import BabyAIBot  # noqa: E402
from minigrid.wrappers import FullyObsWrapper  # noqa: E402

from models.converters import minigrid_to_frame  # noqa: E402
from models.trajectory import Trajectory  # noqa: E402

logger = logging.getLogger(__name__)


# Curated env list. Each env_id is a BabyAI mission type that the BabyAIBot
# can solve reliably. We diversify across goal types: navigation, retrieval,
# door+key, sequencing.
DEFAULT_ENVS: list[str] = [
    "BabyAI-GoToObj-v0",        # navigate to a specific object
    "BabyAI-GoToLocal-v0",      # navigate (local view)
    "BabyAI-PickupLoc-v0",      # pick up an object at location
    "BabyAI-OpenDoor-v0",       # open a specific door
    "BabyAI-OpenRedDoor-v0",    # color-conditioned door open
    "BabyAI-PutNextLocal-v0",   # place an object next to another
    "BabyAI-Synth-v0",          # mixed synthetic missions
    "BabyAI-GoToSeq-v0",        # navigate to a sequence of objects
]


@dataclass
class CollectorStats:
    attempts: int = 0
    successes: int = 0
    bot_failures: int = 0
    timeouts: int = 0
    truncations: int = 0
    avg_length: float = 0.0


def _encode_full_obs(env: gym.Env) -> np.ndarray:
    """Return the (H, W, 3) symbolic grid for the current env state.

    Uses the unwrapped grid.encode() directly so we don't have to wrap+step
    the env through FullyObsWrapper for every encoding.
    """
    base = env.unwrapped
    full_grid = base.grid.encode()  # (H, W, 3) — type, color, state
    # Place agent token in the agent's cell
    ax, ay = base.agent_pos
    full_grid[ay, ax] = np.array([10, 0, base.agent_dir], dtype=np.int64)  # type=10 (agent)
    return full_grid.astype(np.int64)


def collect_one(
    env_id: str,
    seed: int,
    max_steps: int = 64,
) -> Optional[Trajectory]:
    """Run one episode with BabyAIBot. Returns a Trajectory on success, None on failure."""
    env = gym.make(env_id)
    try:
        obs, _ = env.reset(seed=seed)
    except Exception as e:
        logger.debug(f"reset failed for {env_id} seed={seed}: {e!r}")
        return None

    try:
        bot = BabyAIBot(env.unwrapped)
    except Exception as e:
        logger.debug(f"bot init failed for {env_id} seed={seed}: {e!r}")
        return None

    frames = [minigrid_to_frame(_encode_full_obs(env))]
    actions: list[int] = []
    rewards: list[float] = []

    success = False
    for step in range(max_steps):
        try:
            action = bot.replan()
        except Exception as e:
            logger.debug(f"bot.replan crashed at step {step}: {e!r}")
            break
        if action is None:
            break
        action = int(action)
        try:
            obs, r, term, trunc, info = env.step(action)
        except Exception as e:
            logger.debug(f"env.step crashed: {e!r}")
            break
        actions.append(action)
        rewards.append(float(r))
        frames.append(minigrid_to_frame(_encode_full_obs(env)))
        if term:
            # In MiniGrid, term=True with reward>0 means success.
            success = bool(r > 0)
            break
        if trunc:
            break

    if not success:
        return None

    return Trajectory(
        env_marker="minigrid",
        run_id=f"{env_id}::seed{seed}",
        frames=frames,
        actions=actions,
        rewards=rewards,
        success=True,
        metadata={"env_id": env_id, "seed": seed, "mission": str(obs.get("mission", ""))},
    )


def collect(
    env_ids: Optional[list[str]] = None,
    n_per_env: int = 100,
    max_steps: int = 64,
    seed_offset: int = 0,
) -> Iterator[Trajectory]:
    """Yield successful trajectories. Iterates across envs round-robin per seed.

    For each (env, seed) we attempt collection; if the bot fails we just skip
    and try the next seed. Returns successful Trajectories only.
    """
    env_ids = env_ids or DEFAULT_ENVS
    stats = CollectorStats()
    lengths: list[int] = []
    for env_id in env_ids:
        env_stats = {"attempts": 0, "successes": 0}
        # Iterate seeds until we hit n_per_env successes (cap at 4× attempts)
        attempt = 0
        while env_stats["successes"] < n_per_env and attempt < 4 * n_per_env:
            seed = seed_offset + attempt
            attempt += 1
            stats.attempts += 1
            env_stats["attempts"] += 1
            traj = collect_one(env_id, seed=seed, max_steps=max_steps)
            if traj is None:
                stats.bot_failures += 1
                continue
            stats.successes += 1
            env_stats["successes"] += 1
            lengths.append(len(traj))
            yield traj
        logger.info(
            f"  {env_id}: {env_stats['successes']}/{env_stats['attempts']} successes"
        )

    if lengths:
        stats.avg_length = sum(lengths) / len(lengths)
    logger.info(
        f"Collected {stats.successes}/{stats.attempts} ({stats.bot_failures} bot failures), "
        f"avg length {stats.avg_length:.1f}"
    )
