"""Run an agent against all available local environments and print results.

Usage:
    uv run python -m scripts.run_baseline                # random baseline, all games
    uv run python -m scripts.run_baseline --agent v25    # iverson v2.5 (TBD)
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from arc_agi import OperationMode

from agents.base import GameResult, make_arcade, run_agent
from agents.random_baseline import RandomBaseline


def build_agent(name: str, game_id: str, baseline_actions: list[int]) -> Any:
    if name == "random":
        return RandomBaseline(game_id=game_id, baseline_actions=baseline_actions, seed=0)
    raise ValueError(f"unknown agent: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="random", choices=["random"])
    parser.add_argument("--game", default=None, help="prefix filter, comma-separated")
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--mode", default="OFFLINE", choices=["OFFLINE", "ONLINE", "COMPETITION"])
    parser.add_argument("--quiet", action="store_true", help="suppress arc_agi info logs")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("arc_agi").setLevel(logging.WARNING)
        logging.getLogger("arcengine").setLevel(logging.WARNING)

    arc = make_arcade(OperationMode[args.mode])
    envs = arc.available_environments
    print(f"# {len(envs)} environments available")

    if args.game:
        prefixes = args.game.split(",")
        envs = [e for e in envs if any(e.game_id.startswith(p) for p in prefixes)]

    results: list[GameResult] = []
    for info in envs:
        env = arc.make(info.game_id)
        if env is None:
            print(f"  ! could not make env {info.game_id}")
            continue
        agent = build_agent(args.agent, info.game_id, list(info.baseline_actions or []))
        try:
            result = run_agent(agent, env, max_actions=args.max_actions)
        except Exception as e:
            print(f"  ! {info.game_id} crashed: {e!r}")
            continue
        results.append(result)
        score = result.weighted_game_score()
        print(
            f"  {info.game_id:40s} levels={result.levels_completed}/{len(info.baseline_actions or [])} "
            f"actions={result.actions_taken:4d} apl={result.actions_per_level} "
            f"state={result.final_state.value:12s} "
            f"baseline={info.baseline_actions} score={score:.4f} ({result.seconds:.1f}s)"
        )

    if results:
        avg = sum(r.weighted_game_score() for r in results) / len(results)
        print(f"\n# avg per-game weighted score: {avg:.3f}  (n={len(results)})")
        summary = {
            "agent": args.agent,
            "mode": args.mode,
            "n_games": len(results),
            "avg_score": avg,
            "results": [
                {
                    "game_id": r.game_id,
                    "levels_completed": r.levels_completed,
                    "actions_taken": r.actions_taken,
                    "actions_per_level": r.actions_per_level,
                    "final_state": r.final_state.value,
                    "seconds": r.seconds,
                    "score": r.weighted_game_score(),
                    "per_level_score": r.per_level_score(),
                    "baseline_actions": r.baseline_actions,
                }
                for r in results
            ],
        }
        with open(f"logs/baseline_{args.agent}_{args.mode.lower()}.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
