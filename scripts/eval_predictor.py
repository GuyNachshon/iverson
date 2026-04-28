"""Evaluate a trained terminal-state predictor checkpoint.

Two evaluations:
  1. Per-env held-out perplexity. For each env_marker in the corpus, average
     the predictor's loss on a held-out subset. Tells us whether the model
     generalizes within-env at all.
  2. Zero-shot ARC-AGI-3 transfer. For each ARC-AGI-3 game we have local
     env files for, run the wrapper, get a few prefix observations, and ask
     the predictor what the terminal looks like. We can't measure exact
     loss because we don't have ARC-AGI-3 terminals (the agent never won),
     but we can check: does the predicted "terminal state" make sense
     (sane object counts, sensible colors)?

Usage:
    uv run python -m scripts.eval_predictor --ckpt runs/full_NNN/model.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.dataset import DatasetConfig, TrajectoryDataset, collate_fn  # noqa: E402
from models.discretize import expected_value_decode  # noqa: E402
from models.loss import predictor_loss  # noqa: E402
from models.terminal_predictor import (  # noqa: E402
    PredictorConfig,
    TerminalPredictor,
    feature_mask_full,
    feature_mask_invariant,
)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[TerminalPredictor, str]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    pcfg = PredictorConfig(**state["config"])
    model = TerminalPredictor(pcfg).to(device)
    model.load_state_dict(state["model_state"])
    model.train(False)
    variant = state.get("variant", "full")
    return model, variant


def per_env_perplexity(model: TerminalPredictor, paths: list[str], variant: str,
                        device: torch.device, val_frac: float = 0.1,
                        max_batches_per_env: int = 50) -> dict:
    cfg = DatasetConfig(paths=paths, prefix_strategy="uniform", seed=0)
    ds = TrajectoryDataset(cfg)
    n = len(ds)
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=rng).tolist()
    n_val = max(int(n * val_frac), 32)
    val_idx = sorted(set(perm[:n_val]))

    by_marker = defaultdict(list)
    for i in val_idx:
        by_marker[ds._meta[i][0]].append(i)

    fmask = feature_mask_full(device) if variant == "full" else feature_mask_invariant(device)
    results = {}
    for marker, idxs in sorted(by_marker.items()):
        sub = Subset(ds, idxs)
        loader = DataLoader(sub, batch_size=16, shuffle=False, drop_last=False,
                            collate_fn=lambda b: collate_fn(b, max_prefix_frames=12, fixed_pad=True))
        total = 0.0
        n_batches = 0
        with torch.no_grad():
            for j, vb in enumerate(loader):
                if j >= max_batches_per_env:
                    break
                p = vb["prefix_tokens"].to(device)
                pm = vb["prefix_mask"].to(device)
                t = vb["target_tokens"].to(device)
                tm = vb["target_mask"].to(device)
                w = vb["loss_weights"].to(device)
                o = model(p, pm, feature_mask=fmask)
                vloss, _ = predictor_loss(o, t, tm, loss_weights=w)
                total += float(vloss.detach())
                n_batches += 1
        avg = total / max(n_batches, 1)
        results[marker] = {"avg_loss": avg, "n_trajs": len(idxs), "n_batches": n_batches}
        print(f"  {marker:12s}  n_trajs={len(idxs):4d}  n_batches={n_batches:3d}  "
              f"avg_loss={avg:.3f}")
    return results


def zero_shot_arc_agi_3(model: TerminalPredictor, variant: str,
                          device: torch.device, n_steps: int = 5,
                          mode: str = "OFFLINE", limit: int | None = None,
                          random_actions: bool = True) -> dict:
    """Take a few action steps in each ARC-AGI-3 env, ask the predictor to
    predict the terminal, and report aggregate stats across all envs.

    Returns a per-env dict for further analysis.
    """
    from collections import Counter

    print(f"\n## Zero-shot ARC-AGI-3 transfer  (mode={mode})")
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction
        from models.converters import arc_agi_3_to_frame
    except Exception as e:
        print(f"  (skipped: {e!r})")
        return {}

    arc = Arcade(operation_mode=OperationMode[mode])
    envs = arc.available_environments
    if not envs:
        print("  (no environments available)")
        return {}

    fmask = feature_mask_full(device) if variant == "full" else feature_mask_invariant(device)
    rng = np.random.default_rng(0)

    if limit is not None:
        envs = envs[:limit]

    summary: dict = {}
    counts = []
    pred_color_hist: Counter = Counter()
    last_color_hist: Counter = Counter()
    print(f"  {'game_id':50s}  {'prefix':>6s}  {'last':>4s}  {'pred':>4s}  "
          f"{'pred top colors':30s}  {'cx range':>16s}  {'cy range':>16s}")
    for info in envs:
        env = arc.make(info.game_id)
        if env is None:
            continue
        try:
            frames_list = [arc_agi_3_to_frame(env.observation_space)]
            for _ in range(n_steps):
                avail = env.observation_space.available_actions or [1]
                if random_actions:
                    action_id = int(rng.choice(avail))
                else:
                    action_id = avail[0]
                action = GameAction.from_id(action_id)
                data = {"game_id": info.game_id}
                if action.is_complex():
                    data.update({"x": int(rng.integers(0, 64)),
                                  "y": int(rng.integers(0, 64))})
                try:
                    env.step(action, data=data, reasoning={})
                    frames_list.append(arc_agi_3_to_frame(env.observation_space))
                except Exception:
                    break
        except Exception as e:
            print(f"  {info.game_id}: env interaction crashed {e!r}")
            continue

        prefix_tokens = np.zeros((len(frames_list), 128, 13), dtype=np.float32)
        prefix_mask = np.zeros((len(frames_list), 128), dtype=np.float32)
        for k, f in enumerate(frames_list):
            tok, m = f.to_array(max_objects=128)
            prefix_tokens[k] = tok
            prefix_mask[k] = m
        prefix_tokens_t = torch.from_numpy(prefix_tokens).unsqueeze(0).to(device)
        prefix_mask_t = torch.from_numpy(prefix_mask).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(prefix_tokens_t, prefix_mask_t, feature_mask=fmask)

        exists_prob = torch.sigmoid(out["exists_logits"][0]).cpu().numpy()
        n_predicted = int((exists_prob > 0.5).sum())
        color_id_pred = out["color_id_logits"][0].argmax(-1).cpu().numpy()
        cx_dec = expected_value_decode(out["cx_logits"][0], 0.0, 1.0).cpu().numpy()
        cy_dec = expected_value_decode(out["cy_logits"][0], 0.0, 1.0).cpu().numpy()

        last_n = int(prefix_mask[-1].sum())
        last_colors = prefix_tokens[-1, :last_n, 0].astype(int).tolist()
        active = exists_prob > 0.5
        active_idx = np.where(active)[0]
        pred_colors = [int(color_id_pred[i]) for i in active_idx]
        cx_active = cx_dec[active] if active.any() else np.array([])
        cy_active = cy_dec[active] if active.any() else np.array([])

        counts.append(n_predicted)
        for c in pred_colors:
            pred_color_hist[c] += 1
        for c in last_colors:
            last_color_hist[c] += 1

        top_pred = Counter(pred_colors).most_common(3)
        cx_range = (f"[{cx_active.min():.2f},{cx_active.max():.2f}]"
                     if active.any() else "—")
        cy_range = (f"[{cy_active.min():.2f},{cy_active.max():.2f}]"
                     if active.any() else "—")
        print(f"  {info.game_id:50s}  {len(frames_list):>6d}  {last_n:>4d}  "
              f"{n_predicted:>4d}  {str(top_pred):30s}  {cx_range:>16s}  {cy_range:>16s}")

        summary[info.game_id] = {
            "prefix_len": len(frames_list),
            "last_n_objs": last_n,
            "pred_count": n_predicted,
            "pred_top_colors": top_pred,
            "last_top_colors": Counter(last_colors).most_common(3),
        }

    print(f"\n  ## aggregate over {len(summary)} games:")
    if counts:
        print(f"    pred_count: mean={np.mean(counts):.1f}  median={int(np.median(counts))}  "
              f"zero={sum(1 for c in counts if c == 0)}  "
              f"non-zero={sum(1 for c in counts if c > 0)}")
        print(f"    top predicted colors (across all games): {pred_color_hist.most_common(5)}")
        print(f"    top last-frame colors (across all games): {last_color_hist.most_common(5)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--paths", nargs="+",
                        default=sorted(str(p) for p in Path("data").glob("*.parquet")))
    parser.add_argument("--skip-arc-agi-3", action="store_true")
    parser.add_argument("--arc-mode", default="OFFLINE",
                        choices=["OFFLINE", "ONLINE", "COMPETITION"])
    parser.add_argument("--arc-limit", type=int, default=None,
                        help="cap the number of ARC-AGI-3 games to evaluate")
    parser.add_argument("--arc-steps", type=int, default=5,
                        help="action steps per ARC-AGI-3 game before predicting")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                           ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"# device: {device}")

    model, variant = load_checkpoint(Path(args.ckpt), device)
    print(f"# loaded variant={variant}")

    print("\n## Per-env held-out perplexity")
    per_env_perplexity(model, args.paths, variant, device)

    if not args.skip_arc_agi_3:
        zero_shot_arc_agi_3(model, variant, device,
                              n_steps=args.arc_steps,
                              mode=args.arc_mode,
                              limit=args.arc_limit)


if __name__ == "__main__":
    main()
