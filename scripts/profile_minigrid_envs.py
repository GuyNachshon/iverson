"""Profile per-env collection speed/reliability for BabyAI envs.

For each candidate env, attempt to collect N successful trajectories with a
node/step cap. Print one line per env so we can see which are fast.

Usage:
    uv run python -u -m scripts.profile_minigrid_envs --per-env 5 --timeout-sec 30
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collectors.minigrid_collector import collect_one  # noqa: E402

DEFAULT_CANDIDATES = [
    "BabyAI-GoToObjMaze-v0",
    "BabyAI-GoToObjMazeS4-v0",
    "BabyAI-GoToObjMazeS5-v0",
    "BabyAI-GoToRedBall-v0",
    "BabyAI-GoToRedBlueBall-v0",
    "BabyAI-OneRoomS8-v0",
    "BabyAI-OneRoomS12-v0",
    "BabyAI-OpenDoorColor-v0",
    "BabyAI-OpenRedBlueDoors-v0",
    "BabyAI-OpenTwoDoors-v0",
    "BabyAI-KeyCorridorS3R2-v0",
    "BabyAI-KeyCorridorS3R3-v0",
    "BabyAI-PutNextS5N1-v0",
    "BabyAI-PutNextS6N3-v0",
    "BabyAI-UnlockLocal-v0",
    "BabyAI-UnlockPickup-v0",
    "BabyAI-UnlockToUnlock-v0",
    "BabyAI-PickupDist-v0",
    "BabyAI-SynthLoc-v0",
    "BabyAI-SynthSeq-v0",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-env", type=int, default=5)
    parser.add_argument("--max-attempts-mult", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--timeout-sec", type=int, default=60,
                        help="abandon the env after this wall-time")
    parser.add_argument("--seed-offset", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print(f"# {len(DEFAULT_CANDIDATES)} envs, per-env={args.per_env}, "
          f"timeout-sec={args.timeout_sec}", flush=True)
    for env_id in DEFAULT_CANDIDATES:
        start = time.time()
        successes = 0
        attempts = 0
        seed = args.seed_offset
        while (successes < args.per_env
               and attempts < args.max_attempts_mult * args.per_env
               and (time.time() - start) < args.timeout_sec):
            try:
                t = collect_one(env_id, seed=seed, max_steps=args.max_steps)
            except Exception:
                t = None
            attempts += 1
            seed += 1
            if t is not None:
                successes += 1
        elapsed = time.time() - start
        print(f"{env_id:38s}  {successes}/{attempts}  {elapsed:5.1f}s "
              f"({successes / max(elapsed, 0.001):.1f} traj/s)",
              flush=True)


if __name__ == "__main__":
    main()
