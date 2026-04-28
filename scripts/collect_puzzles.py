"""Collect Sudoku, 15-puzzle, and Nonogram trajectories. Writes one parquet per type."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collectors.puzzle_collector import (  # noqa: E402
    collect_fifteen,
    collect_nonogram,
    collect_sudoku,
)
from models.trajectory import write_trajectories  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=300,
                        help="successful trajectories per puzzle type")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--types", default="sudoku,fifteen,nonogram",
                        help="comma-separated puzzle types")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    types = args.types.split(",")

    for puzzle_type in types:
        print(f"\n# === {puzzle_type} ===")
        start = time.time()
        if puzzle_type == "sudoku":
            trajs = list(collect_sudoku(args.per_type, seed_offset=args.seed_offset))
        elif puzzle_type == "fifteen":
            trajs = list(collect_fifteen(args.per_type, seed_offset=args.seed_offset))
        elif puzzle_type == "nonogram":
            trajs = list(collect_nonogram(args.per_type, seed_offset=args.seed_offset))
        else:
            print(f"  unknown type: {puzzle_type}")
            continue
        elapsed = time.time() - start
        out = out_dir / f"{puzzle_type}.parquet"
        n = write_trajectories(trajs, out)
        print(f"  wrote {n} -> {out} in {elapsed:.1f}s ({n/max(elapsed,0.001):.1f} traj/s)")
        print(f"  size: {out.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
