"""Tiny-batch sanity test: train predictor on 4 fixed trajectories for ~200 steps.

Verifies:
  1. Forward pass works.
  2. Loss is gated correctly (per-feature losses on padded slots don't dominate).
  3. The model can overfit a tiny batch — loss drops by ~1 nat or more on color/geom heads.

If any of these fail, real training is wasted. This script is the load-bearing
gate before scaling to the full corpus.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dataset import DatasetConfig, TrajectoryDataset, collate_fn  # noqa: E402
from models.loss import predictor_loss  # noqa: E402
from models.terminal_predictor import (  # noqa: E402
    PredictorConfig,
    TerminalPredictor,
    feature_mask_full,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"# device: {device}")

    # Load a tiny slice — only 4 trajectories, all from MiniGrid for shape stability.
    cfg = DatasetConfig(
        paths=[str(p) for p in sorted(Path("data").glob("minigrid*.parquet"))],
        prefix_strategy="uniform",
        seed=0,
    )
    ds = TrajectoryDataset(cfg)
    print(f"# loaded {len(ds)} trajectories")

    # Subset deterministically.
    indices = [0, 1, 2, 3]
    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=4, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, max_prefix_frames=8))

    pcfg = PredictorConfig(
        embed_dim=128,        # smaller for fast iteration
        n_heads=4,
        n_token_layers=1,
        n_frame_layers=2,
        n_terminal_slots=128,  # must match max_objects so target slot-i ↔ pred slot-i
    )
    model = TerminalPredictor(pcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"# model params: {n_params/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Build a single fixed batch we'll overfit.
    batch = next(iter(loader))
    batch_dev = {
        "prefix_tokens": batch["prefix_tokens"].to(device),
        "prefix_mask": batch["prefix_mask"].to(device),
        "target_tokens": batch["target_tokens"].to(device),
        "target_mask": batch["target_mask"].to(device),
        "loss_weights": batch["loss_weights"].to(device),
    }
    print(f"# batch shapes: prefix={tuple(batch_dev['prefix_tokens'].shape)} "
          f"target={tuple(batch_dev['target_tokens'].shape)}")
    print(f"# n valid target slots per sample: "
          f"{batch_dev['target_mask'].sum(-1).cpu().tolist()}")

    fmask = feature_mask_full(device)
    losses_history = []

    print("\n# training (overfit a single batch)")
    print(f"  {'step':>4s}  {'total':>8s}  {'color_id':>8s}  {'cx':>6s}  {'cy':>6s}  "
          f"{'aspect':>7s}  {'exists':>7s}  {'singletn':>9s}  {'edge':>7s}")
    t0 = time.time()
    for step in range(200):
        model.train()
        out = model(batch_dev["prefix_tokens"], batch_dev["prefix_mask"],
                     feature_mask=fmask)
        total, diag = predictor_loss(out,
                                       batch_dev["target_tokens"],
                                       batch_dev["target_mask"],
                                       loss_weights=batch_dev["loss_weights"])
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0 or step == 199:
            print(f"  {step:>4d}  {diag['total']:>8.3f}  {diag['loss_color_id']:>8.3f}  "
                  f"{diag['loss_cx']:>6.3f}  {diag['loss_cy']:>6.3f}  "
                  f"{diag['loss_aspect']:>7.3f}  {diag['loss_exists']:>7.3f}  "
                  f"{diag['loss_is_singleton']:>9.3f}  {diag['loss_touches_edge']:>7.3f}")
        losses_history.append(diag)
    elapsed = time.time() - t0

    initial = losses_history[0]
    final = losses_history[-1]
    print(f"\n# elapsed: {elapsed:.1f}s ({200/elapsed:.1f} steps/s)")
    print(f"\n# loss change (initial → final):")
    for k in ["total", "loss_color_id", "loss_color_rank", "loss_log_size",
              "loss_bbox", "loss_cx", "loss_cy", "loss_aspect",
              "loss_log_neighbors", "loss_is_singleton", "loss_touches_edge",
              "loss_exists"]:
        i, f = initial[k], final[k]
        delta = f - i
        flag = " ✓" if delta < -0.1 else ("" if delta < 0 else " ← did NOT decrease")
        print(f"    {k:22s}  {i:7.3f} -> {f:7.3f}  Δ={delta:+.3f}{flag}")

    # Sanity check: total loss must drop by at least 30% or this is a wiring bug.
    if final["total"] / initial["total"] > 0.7:
        print("\n# FAIL: total loss did not drop ≥30% in 200 steps. "
              "Probable wiring bug.")
        sys.exit(1)
    else:
        print("\n# PASS: total loss dropped, model can fit a tiny batch.")


if __name__ == "__main__":
    main()
