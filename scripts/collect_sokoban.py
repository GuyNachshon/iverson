"""Collect Sokoban trajectories via BFS solver and write to parquet."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collectors.sokoban_collector import DEFAULT_ENVS, collect  # noqa: E402
from models.trajectory import write_trajectories  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-env", type=int, default=100)
    parser.add_argument("--out", default="data/sokoban.parquet")
    parser.add_argument("--envs", default=None,
                        help="comma-separated env_ids (defaults to all)")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--max-solve-nodes", type=int, default=100_000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    env_ids = args.envs.split(",") if args.envs else DEFAULT_ENVS
    print(f"# envs: {env_ids}")
    print(f"# target: {args.per_env} per env  =>  {args.per_env * len(env_ids)} total")

    start = time.time()
    trajs = list(collect(
        env_ids=env_ids,
        n_per_env=args.per_env,
        seed_offset=args.seed_offset,
        max_solve_nodes=args.max_solve_nodes,
    ))
    elapsed = time.time() - start

    out = Path(args.out)
    n = write_trajectories(trajs, out)
    print(f"# wrote {n} trajectories to {out} in {elapsed:.1f}s "
          f"({n / max(elapsed, 0.001):.1f} traj/s)")
    print(f"# file size: {out.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
