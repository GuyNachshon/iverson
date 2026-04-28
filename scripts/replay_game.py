"""Replay a game with a chosen agent and export an MP4 of the trajectory.

Captures (frame, action, click_target) at each step. Renders frames to RGB
via arc_agi.rendering.frame_to_rgb_array, overlays the action label and
click marker (for ACTION6), and writes an MP4 via matplotlib's animation.

Usage:
    uv run python -m scripts.replay_game --agent v35 --game bt33 --out replays/bt33_v35.mp4
    uv run python -m scripts.replay_game --agent v35 --game r11l --mode ONLINE --out replays/r11l_v35.mp4
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.")
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

from arc_agi import Arcade, OperationMode  # noqa: E402
from arc_agi.rendering import frame_to_rgb_array  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.animation as anim  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from agents.iverson_v35 import IversonV35  # noqa: E402
from agents.iverson_v25 import IversonV25  # noqa: E402
from agents.random_baseline import RandomBaseline  # noqa: E402


def build_agent(name: str, game_id: str, baseline_actions: list[int]) -> Any:
    if name == "v35":
        return IversonV35(game_id=game_id, baseline_actions=baseline_actions, seed=0)
    if name == "v25":
        return IversonV25(game_id=game_id, baseline_actions=baseline_actions, seed=0)
    if name == "random":
        return RandomBaseline(game_id=game_id, baseline_actions=baseline_actions, seed=0)
    raise ValueError(f"unknown agent: {name}")


def find_game_id(arc: Arcade, prefix: str) -> str:
    for info in arc.available_environments:
        if info.game_id.startswith(prefix):
            return info.game_id
    raise ValueError(f"no game starting with {prefix!r} in {len(arc.available_environments)} envs")


def run_and_capture(arc: Arcade, game_id: str, agent_name: str, max_actions: int = 200) -> dict:
    """Run an agent on a game, capturing frames + actions per step.

    Wraps env.step to record (raw_obs, action_taken, click_xy) per call so
    we can replay the trajectory visually.
    """
    info = next(i for i in arc.available_environments if i.game_id == game_id)
    env = arc.make(game_id)
    if env is None:
        raise RuntimeError(f"failed to make env {game_id}")
    agent = build_agent(agent_name, game_id, list(info.baseline_actions or []))

    captured = {
        "game_id": game_id,
        "baselines": list(info.baseline_actions or []),
        "frames_raw": [env.observation_space],
        "actions": [],
        "states": [env.observation_space.state.value],
        "levels": [env.observation_space.levels_completed],
    }

    # Wrap env.step to capture every transition.
    original_step = env.step

    def capturing_step(action, data=None, reasoning=None):
        raw = original_step(action, data=data or {}, reasoning=reasoning or {})
        click_x = data.get("x") if (data and action.is_complex()) else None
        click_y = data.get("y") if (data and action.is_complex()) else None
        captured["actions"].append({
            "step": len(captured["actions"]),
            "action_id": action.value,
            "action_name": action.name,
            "click_x": click_x,
            "click_y": click_y,
        })
        captured["frames_raw"].append(raw)
        captured["states"].append(raw.state.value)
        captured["levels"].append(raw.levels_completed)
        return raw

    env.step = capturing_step  # type: ignore[method-assign]

    # Use the official runner.
    from agents.base import run_agent  # local import to avoid top-level cycle
    result = run_agent(agent, env, max_actions=max_actions)
    captured["n_frames"] = len(captured["frames_raw"])
    captured["final_levels_completed"] = result.levels_completed
    captured["actions_per_level"] = list(result.actions_per_level)
    captured["final_state"] = result.final_state.value
    return captured


def render_to_mp4(captured: dict, out_path: Path, fps: int = 8, scale: int = 4) -> None:
    """Render the captured trajectory to MP4 with action overlays."""
    frames_rgb: list[np.ndarray] = []
    for raw in captured["frames_raw"]:
        # raw.frame is list of np arrays, each (H, W) — render the first.
        if not raw.frame:
            continue
        rgb = frame_to_rgb_array(steps=0, frame=raw.frame[0], scale=scale)
        frames_rgb.append(rgb)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#202020")
    ax.set_facecolor("#202020")
    ax.axis("off")
    im = ax.imshow(frames_rgb[0])
    title = ax.text(
        0.5, 1.02, "", transform=ax.transAxes, ha="center", va="bottom",
        color="white", fontsize=12, family="monospace",
    )
    click_marker = ax.add_patch(patches.Circle((0, 0), 8, color="red",
                                                  fill=False, linewidth=2, alpha=0.0))

    H, W = frames_rgb[0].shape[:2]

    def update(i: int):
        im.set_array(frames_rgb[i])
        if i == 0:
            title.set_text(f"{captured['game_id']}  step 0/{len(frames_rgb)-1}  "
                            f"state={captured['states'][0]}  level={captured['levels'][0]}")
            click_marker.set_alpha(0.0)
        else:
            a = captured["actions"][i - 1]
            label = f"step {a['step']+1}/{len(frames_rgb)-1}  {a['action_name']}"
            if a["click_x"] is not None:
                label += f"  click=({a['click_x']},{a['click_y']})"
                # Map click (x, y) in 0..63 grid coords → pixels in the rendered frame.
                grid_w = raw_grid_width(captured)
                px = int((a["click_x"] + 0.5) * (W / grid_w))
                py = int((a["click_y"] + 0.5) * (H / grid_w))
                click_marker.center = (px, py)
                click_marker.set_alpha(0.85)
            else:
                click_marker.set_alpha(0.0)
            label += f"  state={captured['states'][i]}  level={captured['levels'][i]}"
            title.set_text(label)
        return [im, title, click_marker]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = anim.FuncAnimation(fig, update, frames=len(frames_rgb),
                            interval=int(1000 / fps), blit=False)
    writer = anim.FFMpegWriter(fps=fps, bitrate=2000)
    a.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)


def raw_grid_width(captured: dict) -> int:
    """Native grid width (always 64 for ARC-AGI-3)."""
    f0 = captured["frames_raw"][0]
    if f0.frame:
        return f0.frame[0].shape[1]
    return 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="v35", choices=["random", "v25", "v35"])
    parser.add_argument("--game", required=True, help="game id prefix (e.g. 'bt33', 'r11l')")
    parser.add_argument("--mode", default="OFFLINE", choices=["OFFLINE", "ONLINE"])
    parser.add_argument("--out", default=None, help="output mp4 path; default replays/<game>_<agent>.mp4")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=200)
    args = parser.parse_args()

    arc = Arcade(operation_mode=OperationMode[args.mode])
    game_id = find_game_id(arc, args.game)
    print(f"# replaying {game_id} with agent={args.agent}")

    out_path = Path(args.out) if args.out else Path(f"replays/{args.game}_{args.agent}.mp4")
    captured = run_and_capture(arc, game_id, args.agent, max_actions=args.max_actions)
    levels_cleared = captured["levels"][-1]
    print(f"# captured {captured['n_frames']} frames, levels={levels_cleared}/{len(captured['baselines'])}, "
          f"final state={captured['states'][-1]}")

    render_to_mp4(captured, out_path, fps=args.fps)
    print(f"# wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
