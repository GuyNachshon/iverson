"""Audit a trajectory corpus on disk. Prints per-env stats + diversity metrics.

Usage:
    uv run python -m scripts.audit_corpus data/minigrid.parquet
    uv run python -m scripts.audit_corpus data/*.parquet
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.trajectory import read_trajectories, unpack_frames  # noqa: E402


def _env_id_from_run(run_id: str) -> str:
    return run_id.split("::", 1)[0]


def audit(paths: list[Path]) -> None:
    rows = []
    for p in paths:
        rows.extend(read_trajectories(p))

    print(f"# total trajectories: {len(rows)}")
    if not rows:
        return

    by_env_marker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_env_marker[r["env_marker"]].append(r)

    for marker, group in by_env_marker.items():
        print(f"\n## env_marker: {marker}  ({len(group)} trajectories)")
        # per-env_id breakdown
        by_env_id = defaultdict(list)
        for r in group:
            by_env_id[_env_id_from_run(r["run_id"])].append(r)
        for env_id, trajs in sorted(by_env_id.items()):
            n_frames = [t["n_frames"] for t in trajs]
            n_actions = [t["n_actions"] for t in trajs]
            successes = sum(1 for t in trajs if t["success"])
            print(
                f"  {env_id:35s} n={len(trajs):4d}  success={successes/len(trajs):.2%}  "
                f"frames={mean(n_frames):.1f}±{stdev(n_frames) if len(n_frames) > 1 else 0:.1f}  "
                f"actions={mean(n_actions):.1f}"
            )

        # Object-list stats across the env_marker
        all_objects_per_frame: list[int] = []
        all_terminal_objects: list[int] = []
        all_color_ids: Counter = Counter()
        terminal_color_id_lists: list[set] = []
        all_centroids: list[tuple[float, float]] = []
        for r in group:
            tokens, mask = unpack_frames(r)
            for t in range(tokens.shape[0]):
                n = int(mask[t].sum())
                all_objects_per_frame.append(n)
                for k in range(n):
                    cid = int(tokens[t, k, 0])
                    all_color_ids[cid] += 1
                    all_centroids.append((float(tokens[t, k, 7]), float(tokens[t, k, 8])))
            # terminal frame stats
            if tokens.shape[0] > 0:
                final_t = tokens.shape[0] - 1
                n_term = int(mask[final_t].sum())
                all_terminal_objects.append(n_term)
                term_colors = {int(tokens[final_t, k, 0]) for k in range(n_term)}
                terminal_color_id_lists.append(term_colors)

        print(f"\n  objects per frame:    median={median(all_objects_per_frame)}, "
              f"mean={mean(all_objects_per_frame):.1f}, "
              f"max={max(all_objects_per_frame)}")
        print(f"  objects in terminal:  median={median(all_terminal_objects)}, "
              f"mean={mean(all_terminal_objects):.1f}")
        print(f"  distinct color_ids:   {len(all_color_ids)}  "
              f"top 5 = {all_color_ids.most_common(5)}")

        # Terminal-state diversity proxy: how many distinct color_id-sets?
        unique_terminal_sets = len({frozenset(s) for s in terminal_color_id_lists})
        print(f"  distinct terminal color-id sets: {unique_terminal_sets} / {len(terminal_color_id_lists)}")

        # Better terminal-diversity metric: signature = (sorted color_ids, sorted
        # rounded-centroid-positions). Captures positional diversity that color
        # sets miss (Sokoban, Sudoku, 15-puzzle all have identical color-id sets
        # at terminal but vastly different positions).
        terminal_pos_signatures: list[tuple] = []
        for r in group:
            tokens, mask = unpack_frames(r)
            if tokens.shape[0] == 0:
                continue
            final_t = tokens.shape[0] - 1
            n = int(mask[final_t].sum())
            sig = []
            for k in range(n):
                cid = int(tokens[final_t, k, 0])
                cx = round(float(tokens[final_t, k, 7]), 2)
                cy = round(float(tokens[final_t, k, 8]), 2)
                sig.append((cid, cx, cy))
            terminal_pos_signatures.append(tuple(sorted(sig)))
        unique_pos_sigs = len(set(terminal_pos_signatures))
        print(f"  distinct terminal pos signatures: {unique_pos_sigs} / {len(terminal_pos_signatures)}")

        # Centroid distribution
        cents = np.asarray(all_centroids)
        if cents.size:
            print(f"  centroid x range: [{cents[:, 0].min():.2f}, {cents[:, 0].max():.2f}]  "
                  f"y range: [{cents[:, 1].min():.2f}, {cents[:, 1].max():.2f}]")


def cross_env_summary(paths: list[Path]) -> None:
    """Print cross-env totals + diversity per env_marker on one line each."""
    rows = []
    for p in paths:
        rows.extend(read_trajectories(p))
    by_marker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_marker[r["env_marker"]].append(r)
    print("\n## Cross-env summary")
    total = 0
    for marker, group in sorted(by_marker.items()):
        n_unique_terms = 0
        terms = set()
        for r in group:
            tokens, mask = unpack_frames(r)
            if tokens.shape[0] == 0:
                continue
            final_t = tokens.shape[0] - 1
            n = int(mask[final_t].sum())
            sig = []
            for k in range(n):
                cid = int(tokens[final_t, k, 0])
                cx = round(float(tokens[final_t, k, 7]), 2)
                cy = round(float(tokens[final_t, k, 8]), 2)
                sig.append((cid, cx, cy))
            terms.add(tuple(sorted(sig)))
        n_unique_terms = len(terms)
        total += len(group)
        print(f"  {marker:12s} n={len(group):5d}  unique_terminals={n_unique_terms:5d}  "
              f"diversity={n_unique_terms/max(len(group),1):.0%}")
    print(f"  TOTAL trajectories: {total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="parquet files to audit")
    parser.add_argument("--summary-only", action="store_true",
                        help="skip per-env detail, print only cross-env summary")
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths]
    if not args.summary_only:
        audit(paths)
    cross_env_summary(paths)


if __name__ == "__main__":
    main()
