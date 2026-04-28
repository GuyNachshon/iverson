"""Per-prediction diagnostics on held-out training data.

For each env_marker, sample N held-out trajectories. For each, decode the
predicted terminal and compute:
  - predicted vs actual object count
  - per-slot color accuracy (top-1)
  - centroid distance (predicted vs actual, when slots align by index)
  - exists head accuracy (precision/recall on the binary slot-occupied)

This tells us what the model is good at / bad at WITHIN the training
distribution. If centroid prediction is bad even within-distribution, no
amount of more training will fix transfer. If it's good within-distribution
but bad on ARC-AGI-3, the corpus distribution is the bottleneck.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dataset import DatasetConfig, TrajectoryDataset, collate_fn  # noqa: E402
from models.discretize import expected_value_decode  # noqa: E402
from models.terminal_predictor import (  # noqa: E402
    PredictorConfig,
    TerminalPredictor,
    feature_mask_full,
    feature_mask_invariant,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--paths", nargs="+",
                        default=sorted(str(p) for p in Path("data").glob("*.parquet")))
    parser.add_argument("--n-per-env", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    pcfg = PredictorConfig(**state["config"])
    model = TerminalPredictor(pcfg).to(device)
    model.load_state_dict(state["model_state"])
    model.train(False)
    variant = state.get("variant", "full")
    fmask = feature_mask_full(device) if variant == "full" else feature_mask_invariant(device)

    cfg = DatasetConfig(paths=list(args.paths), prefix_strategy="uniform",
                         seed=args.seed)
    ds = TrajectoryDataset(cfg)

    # Held-out 10% same as eval/train
    n = len(ds)
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=rng).tolist()
    n_val = max(int(n * 0.1), 32)
    val_idx = sorted(set(perm[:n_val]))

    by_marker = defaultdict(list)
    for i in val_idx:
        by_marker[ds._meta[i][0]].append(i)

    print(f"# variant: {variant}")
    print(f"\n{'env':12s}  {'pred_n':>7s}  {'true_n':>7s}  {'count_err':>9s}  "
          f"{'color_acc':>9s}  {'cent_err':>8s}  {'exists_p':>9s}  "
          f"{'exists_r':>9s}  {'n_samples':>9s}")

    for marker, idxs in sorted(by_marker.items()):
        sample_idxs = idxs[:args.n_per_env]
        # Compute per-trajectory metrics, then aggregate.
        true_counts = []
        pred_counts = []
        color_correct = []
        centroid_errors = []
        exists_correct = []
        exists_target = []
        for ix in sample_idxs:
            sample = ds[ix]
            prefix = sample["prefix_tokens"].unsqueeze(0).to(device)
            pmask = sample["prefix_mask"].unsqueeze(0).to(device)
            target = sample["target_tokens"].numpy()
            tmask = sample["target_mask"].numpy()
            with torch.no_grad():
                out = model(prefix, pmask, feature_mask=fmask)
            ep = torch.sigmoid(out["exists_logits"][0]).cpu().numpy()
            cid = out["color_id_logits"][0].argmax(-1).cpu().numpy()
            cx_dec = expected_value_decode(out["cx_logits"][0], 0.0, 1.0).cpu().numpy()
            cy_dec = expected_value_decode(out["cy_logits"][0], 0.0, 1.0).cpu().numpy()
            pred_active = ep > 0.5
            true_n = int(tmask.sum())
            pred_n = int(pred_active.sum())
            true_counts.append(true_n)
            pred_counts.append(pred_n)

            # Per-slot color accuracy: only on slots where target says active.
            true_active = tmask > 0.5
            true_color = target[:, 0].astype(int)
            color_match = (cid == true_color) & true_active
            color_correct.append(color_match.sum() / max(true_active.sum(), 1))

            # Centroid error on slots where both target and pred are active.
            both_active = pred_active & true_active
            if both_active.any():
                target_cx = target[:, 7]
                target_cy = target[:, 8]
                ce = np.sqrt(
                    (cx_dec[both_active] - target_cx[both_active]) ** 2
                    + (cy_dec[both_active] - target_cy[both_active]) ** 2
                ).mean()
                centroid_errors.append(float(ce))

            # Exists head: precision/recall against tmask
            tp = (pred_active & true_active).sum()
            fp = (pred_active & ~true_active).sum()
            fn = (~pred_active & true_active).sum()
            exists_correct.append((tp, fp, fn))
            exists_target.append(true_active.sum())

        total_tp = sum(t[0] for t in exists_correct)
        total_fp = sum(t[1] for t in exists_correct)
        total_fn = sum(t[2] for t in exists_correct)
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        avg_centroid = float(np.mean(centroid_errors)) if centroid_errors else float("nan")
        print(f"{marker:12s}  {np.mean(pred_counts):>7.1f}  {np.mean(true_counts):>7.1f}  "
              f"{abs(np.mean(pred_counts) - np.mean(true_counts)):>9.1f}  "
              f"{np.mean(color_correct):>9.2%}  {avg_centroid:>8.3f}  "
              f"{precision:>9.2%}  {recall:>9.2%}  {len(sample_idxs):>9d}")


if __name__ == "__main__":
    main()
