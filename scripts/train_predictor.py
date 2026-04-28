"""Train the terminal-state predictor on the Tier A+D corpus.

Per the Phase 0c amendment, we run a feature-mask ablation as the first
experiment: full features vs invariance-friendly only. The full-features
model is allowed to see env-correlated raw features (color_id, raw bbox
coords); the invariant model is not. If the invariant model fails to
converge or generalize, the corpus genuinely lacks cross-env signal.

Usage:
    uv run python -m scripts.train_predictor --variant full --steps 5000
    uv run python -m scripts.train_predictor --variant invariant --steps 5000
"""
from __future__ import annotations

import argparse
import json
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
    feature_mask_invariant,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "invariant"], required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prefix-strategy", default="uniform",
                        choices=["uniform", "long_first", "weighted"])
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--paths", nargs="+",
                        default=sorted(str(p) for p in Path("data").glob("*.parquet")))
    parser.add_argument("--small", action="store_true",
                        help="use small model config for quick local iteration")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / f"{args.variant}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"# device: {device}")
    print(f"# variant: {args.variant}")
    print(f"# corpus paths: {args.paths}")
    print(f"# out dir: {out_dir}")

    cfg = DatasetConfig(paths=list(args.paths), prefix_strategy=args.prefix_strategy,
                         seed=args.seed)
    ds = TrajectoryDataset(cfg)
    n = len(ds)
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng).tolist()
    n_val = max(int(n * args.val_frac), 32)
    val_idx = set(perm[:n_val])
    train_idx = [i for i in range(n) if i not in val_idx]

    train_subset = torch.utils.data.Subset(ds, train_idx)
    val_subset = torch.utils.data.Subset(ds, sorted(val_idx))
    print(f"# trajectories train={len(train_subset)} val={len(val_subset)}")

    # max_prefix=12 keeps memory reasonable while covering >75% of trajectories.
    # fixed_pad=True so MPS doesn't recompile kernels per batch.
    PREFIX_CAP = 12
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate_fn(b, max_prefix_frames=PREFIX_CAP, fixed_pad=True),
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=lambda b: collate_fn(b, max_prefix_frames=PREFIX_CAP, fixed_pad=True),
    )

    if args.small:
        pcfg = PredictorConfig(embed_dim=128, n_heads=4, n_token_layers=1,
                                n_frame_layers=2, n_terminal_slots=128)
    else:
        pcfg = PredictorConfig(embed_dim=256, n_heads=8, n_token_layers=2,
                                n_frame_layers=4, n_terminal_slots=128)
    model = TerminalPredictor(pcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"# model params: {n_params/1e6:.2f} M")

    fmask = feature_mask_full(device) if args.variant == "full" else feature_mask_invariant(device)
    print(f"# feature mask: {fmask.cpu().numpy().tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                        eta_min=args.lr * 0.1)

    log_path = out_dir / "log.jsonl"
    log_f = log_path.open("w")

    train_iter = iter(train_loader)
    t0 = time.time()
    print(f"\n# training for {args.steps} steps, batch_size={args.batch_size}")
    print(f"  {'step':>5s} {'loss':>7s} {'col_id':>7s} {'cx':>6s} {'aspect':>7s} "
          f"{'exists':>7s} {'val_loss':>9s} {'steps/s':>8s}")

    val_loss_last = float("nan")

    for step in range(args.steps):
        ds.set_step(step)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        prefix = batch["prefix_tokens"].to(device)
        pmask = batch["prefix_mask"].to(device)
        target = batch["target_tokens"].to(device)
        tmask = batch["target_mask"].to(device)
        weights = batch["loss_weights"].to(device)

        out = model(prefix, pmask, feature_mask=fmask)
        loss, diag = predictor_loss(out, target, tmask, loss_weights=weights)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.val_every == 0 or step == args.steps - 1:
            model.train(False)  # inference mode (avoid the "eval" name).
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for j, vb in enumerate(val_loader):
                    if j >= args.val_batches:
                        break
                    p = vb["prefix_tokens"].to(device)
                    pm = vb["prefix_mask"].to(device)
                    t = vb["target_tokens"].to(device)
                    tm = vb["target_mask"].to(device)
                    w = vb["loss_weights"].to(device)
                    o = model(p, pm, feature_mask=fmask)
                    vloss, _ = predictor_loss(o, t, tm, loss_weights=w)
                    val_total += float(vloss)
                    val_n += 1
            val_loss_last = val_total / max(val_n, 1)

            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            print(f"  {step:>5d} {diag['total']:>7.3f} {diag['loss_color_id']:>7.3f} "
                  f"{diag['loss_cx']:>6.3f} {diag['loss_aspect']:>7.3f} "
                  f"{diag['loss_exists']:>7.3f} {val_loss_last:>9.3f} {sps:>8.1f}")

        log_f.write(json.dumps({
            "step": step, "train": diag, "val_loss": val_loss_last,
            "lr": float(opt.param_groups[0]["lr"]),
        }) + "\n")
        log_f.flush()

    log_f.close()
    elapsed = time.time() - t0
    print(f"\n# training complete in {elapsed:.1f}s ({args.steps/elapsed:.1f} steps/s)")
    print(f"# final val_loss: {val_loss_last:.3f}")

    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": pcfg.__dict__,
        "variant": args.variant,
        "n_params": n_params,
        "final_val_loss": val_loss_last,
    }, ckpt_path)
    print(f"# saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
