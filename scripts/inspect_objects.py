"""Sanity-check the object-list representation on real env observations.

Runs:
  - ARC-AGI-3 converter on bt11/bt33 (initial frame after reset)
  - MiniGrid converter on a synthetic 7x7 observation (since we may not have
    the minigrid package installed yet)

Prints decomposition summary, then round-trips through objects_to_grid for a
visual diff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc_agi import Arcade, OperationMode  # noqa: E402

from models.converters import arc_agi_3_to_frame, minigrid_to_frame  # noqa: E402
from models.object_list import objects_to_grid  # noqa: E402


def _summarize_frame(f, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"grid_shape={f.grid_shape}  env_marker={f.env_marker}  n_objects={len(f.objects)}")
    for i, o in enumerate(f.objects[:8]):
        print(
            f"  obj[{i}] color={o.color_id} rank={o.color_rank} size={o.size:3d} "
            f"bbox={o.bbox} centroid=({o.centroid_norm[0]:.2f},{o.centroid_norm[1]:.2f}) "
            f"aspect={o.aspect:.2f} singleton={o.is_singleton} edge={o.touches_edge} "
            f"neighbors={o.touches_others}"
        )
    if len(f.objects) > 8:
        print(f"  ... and {len(f.objects) - 8} more")


def _print_grid_side_by_side(orig: np.ndarray, recon: np.ndarray, max_dim: int = 16) -> None:
    H, W = orig.shape
    h, w = min(H, max_dim), min(W, max_dim)
    match = (orig[:h, :w] == recon[:h, :w]).all()
    print(f"  showing top-left {h}x{w} of original vs reconstructed (match={match}):")
    for y in range(h):
        line_o = " ".join(f"{int(orig[y, x]):2d}" for x in range(w))
        line_r = " ".join(f"{int(recon[y, x]):2d}" for x in range(w))
        print(f"    {line_o}  |  {line_r}")


def inspect_arc_agi_3() -> None:
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    for info in arc.available_environments:
        env = arc.make(info.game_id)
        if env is None:
            continue
        raw = env.observation_space
        frame = arc_agi_3_to_frame(raw)
        _summarize_frame(frame, f"ARC-AGI-3 / {info.game_id}")
        if frame.raw is not None:
            bg_color = int(np.bincount(frame.raw.flatten()).argmax())
            recon = objects_to_grid(frame, background_color=bg_color)
            _print_grid_side_by_side(frame.raw, recon)


def inspect_minigrid_synthetic() -> None:
    H, W = 7, 7
    obs = np.zeros((H, W, 3), dtype=np.int64)
    obs[..., 0] = 3   # floor everywhere
    obs[..., 1] = 5   # grey
    obs[0, :, 0] = 2; obs[-1, :, 0] = 2; obs[:, 0, 0] = 2; obs[:, -1, 0] = 2  # walls
    obs[0, :, 1] = 5; obs[-1, :, 1] = 5; obs[:, 0, 1] = 5; obs[:, -1, 1] = 5
    obs[3, 3, 0] = 10; obs[3, 3, 1] = 0  # agent (red)
    obs[1, 1, 0] = 5;  obs[1, 1, 1] = 4  # key (yellow)
    obs[5, 5, 0] = 8;  obs[5, 5, 1] = 1  # goal (green)

    frame = minigrid_to_frame(obs)
    _summarize_frame(frame, "MiniGrid (synthetic 7x7)")
    if frame.raw is not None:
        bg = int(np.bincount(frame.raw.flatten()).argmax())
        recon = objects_to_grid(frame, background_color=bg)
        _print_grid_side_by_side(frame.raw, recon)


def main() -> None:
    inspect_minigrid_synthetic()
    inspect_arc_agi_3()


if __name__ == "__main__":
    main()
