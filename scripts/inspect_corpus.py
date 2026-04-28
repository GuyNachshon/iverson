"""Pre-Phase-1 corpus inspection.

Three checks:
  1. Per-env hand-inspect — print 10 sampled trajectories per env, showing the
     initial frame, a midpoint frame, and the terminal frame as raw grids and
     as object summaries. Catches Frame.raw-style bugs and obvious anomalies.
  2. Token distribution outliers — for each token feature (size, aspect, etc.)
     report mean/std/max across the corpus. Flags features with absurd ranges
     or near-constant values.
  3. Cluster-by-env — k-means (k=10) on terminal-frame token vectors (mean-pooled),
     report cluster x env_marker confusion matrix. If clusters are ~1:1 with envs,
     the cross-env diversity is nominal; if clusters mix envs, real abstraction
     signal exists.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.trajectory import read_trajectories, unpack_frames  # noqa: E402


def _env_id_from_run(run_id: str) -> str:
    return run_id.split("::", 1)[0]


def _print_grid_compact(grid: np.ndarray, max_dim: int = 12) -> None:
    H, W = grid.shape
    h, w = min(H, max_dim), min(W, max_dim)
    if H > max_dim or W > max_dim:
        print(f"    (showing top-left {h}x{w} of {H}x{w} grid)")
    for y in range(h):
        print("   ", " ".join(f"{int(grid[y, x]):3d}" for x in range(w)))


def hand_inspect(rows: list[dict], n_per_env: int = 2) -> None:
    print("\n## Hand-inspect (sample initial / mid / terminal per env)")
    by_env_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_env_id[_env_id_from_run(r["run_id"])].append(r)
    rng = np.random.default_rng(42)
    for env_id in sorted(by_env_id):
        sampled_idxs = rng.choice(len(by_env_id[env_id]),
                                   size=min(n_per_env, len(by_env_id[env_id])),
                                   replace=False)
        for idx in sampled_idxs:
            r = by_env_id[env_id][idx]
            tokens, mask = unpack_frames(r)
            n_frames = tokens.shape[0]
            n_objs_init = int(mask[0].sum())
            n_objs_mid = int(mask[n_frames // 2].sum()) if n_frames > 1 else 0
            n_objs_term = int(mask[-1].sum())
            print(f"\n  {env_id}  run={r['run_id']}  frames={n_frames}  "
                  f"actions={r['n_actions']}  success={r['success']}")
            print(f"    n_objects: init={n_objs_init}  mid={n_objs_mid}  term={n_objs_term}")
            # Centroid range across frames (sanity check: shouldn't be constant)
            init_cx = tokens[0, :n_objs_init, 7].mean() if n_objs_init else 0
            term_cx = tokens[-1, :n_objs_term, 7].mean() if n_objs_term else 0
            print(f"    centroid_x: init mean={init_cx:.3f}  term mean={term_cx:.3f}")
            # Color rank distribution at terminal
            colors_term = [int(tokens[-1, k, 0]) for k in range(n_objs_term)]
            color_count = Counter(colors_term)
            print(f"    terminal colors (top): {color_count.most_common(5)}")


def token_distribution(rows: list[dict]) -> None:
    """Per-feature mean/std/max across all (frame, object) tokens in the corpus."""
    print("\n## Token feature distributions")
    feature_names = [
        "color_id", "color_rank", "log_size",
        "x_min_n", "y_min_n", "x_max_n", "y_max_n",
        "cx_norm", "cy_norm", "aspect",
        "is_singleton", "touches_edge", "log_touches_others",
    ]
    all_vals = [[] for _ in range(13)]
    for r in rows:
        tokens, mask = unpack_frames(r)
        n_frames, max_obj, _ = tokens.shape
        for t in range(n_frames):
            n = int(mask[t].sum())
            for k in range(n):
                for f in range(13):
                    all_vals[f].append(float(tokens[t, k, f]))
    print(f"  total tokens: {len(all_vals[0])}")
    print(f"  {'feature':22s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}")
    for f, name in enumerate(feature_names):
        v = np.asarray(all_vals[f])
        print(f"  {name:22s}  {v.mean():8.3f}  {v.std():8.3f}  {v.min():8.3f}  {v.max():8.3f}")


def _terminal_token_vector(row: dict, drop_color_id: bool = False,
                            invariant_features_only: bool = False) -> tuple[np.ndarray, str]:
    """Return a fixed-size summary vector for the terminal frame + env_marker.

    If drop_color_id, exclude feature index 0 (raw color_id, which is
    near-perfectly env-correlated by construction).

    If invariant_features_only, use only features that are env-distribution-
    invariant: color_rank (per-env normalized, comparable across envs),
    geometric ratios (aspect, normalized centroid x/y), is_singleton,
    touches_edge, log_touches_others. Drops: color_id, log_size (env
    correlated via grid sizes), bbox coords (env-correlated via grid sizes).
    Also drops object-count from the summary by NOT taking a 'sum/count'
    aggregate.
    """
    tokens, mask = unpack_frames(row)
    if tokens.shape[0] == 0:
        return np.zeros(1, dtype=np.float32), row["env_marker"]
    final_t = tokens.shape[0] - 1
    n = int(mask[final_t].sum())
    if n == 0:
        return np.zeros(1, dtype=np.float32), row["env_marker"]
    real_tokens = tokens[final_t, :n]

    if invariant_features_only:
        # Indices: 1=color_rank, 7=cx, 8=cy, 9=aspect, 10=singleton, 11=edge, 12=log_neighbors
        keep_idx = [1, 7, 8, 9, 10, 11, 12]
        real_tokens = real_tokens[:, keep_idx]
    elif drop_color_id:
        real_tokens = real_tokens[:, 1:]

    feat = np.concatenate([
        real_tokens.mean(0),
        real_tokens.std(0),
        real_tokens.max(0),
    ]).astype(np.float32)
    return feat, row["env_marker"]


def _kmeans(X: np.ndarray, k: int, n_iter: int = 50,
             seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simple k-means. Returns (centroids, labels)."""
    rng = np.random.default_rng(seed)
    N, D = X.shape
    init_idx = rng.choice(N, size=k, replace=False)
    centroids = X[init_idx].copy()
    labels = np.zeros(N, dtype=np.int64)
    for _ in range(n_iter):
        # Distance to each centroid
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for c in range(k):
            members = X[labels == c]
            if len(members):
                centroids[c] = members.mean(0)
    return centroids, labels


def cluster_by_env(rows: list[dict], k: int = 10, drop_color_id: bool = False,
                    invariant_features_only: bool = False) -> None:
    if invariant_features_only:
        label = "invariant features only (color_rank, geometry; no count, no raw color)"
    elif drop_color_id:
        label = "geometric features only (color_id excluded)"
    else:
        label = "all 13 features"
    print(f"\n## k-means(k={k}) on terminal token summary vectors — {label}")
    vecs = []
    markers = []
    for r in rows:
        v, m = _terminal_token_vector(
            r,
            drop_color_id=drop_color_id,
            invariant_features_only=invariant_features_only,
        )
        vecs.append(v)
        markers.append(m)
    X = np.asarray(vecs, dtype=np.float32)
    # z-score normalize per feature
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    _, labels = _kmeans(Xn, k=k, seed=0)

    # Cluster x env confusion matrix
    distinct_markers = sorted(set(markers))
    n_markers = len(distinct_markers)
    marker_to_col = {m: i for i, m in enumerate(distinct_markers)}
    counts = np.zeros((k, n_markers), dtype=np.int64)
    for c, m in zip(labels, markers):
        counts[c, marker_to_col[m]] += 1
    print(f"  rows = clusters, cols = {distinct_markers}")
    header = "       " + "  ".join(f"{m[:8]:>8s}" for m in distinct_markers) + "    total"
    print(header)
    for c in range(k):
        row_total = counts[c].sum()
        if row_total == 0:
            continue
        cells = "  ".join(f"{counts[c, j]:>8d}" for j in range(n_markers))
        print(f"  c{c:>2d}  {cells}  {row_total:>8d}")
    # Diversity per cluster: entropy over env_markers
    print(f"\n  cluster purity (max env share within cluster):")
    for c in range(k):
        row_total = counts[c].sum()
        if row_total == 0:
            print(f"    c{c}: empty")
            continue
        purity = counts[c].max() / row_total
        dominant = distinct_markers[counts[c].argmax()]
        print(f"    c{c}: {row_total:5d} pts, purity={purity:.0%} (dominant={dominant})")
    # Aggregate metric
    total = counts.sum()
    weighted_purity = sum(counts[c].max() for c in range(k)) / total
    print(f"\n  weighted average purity: {weighted_purity:.0%}")
    print("  (purity 100% = clusters perfectly 1-1 with env_markers — bad sign;")
    print("   purity <40% = clusters mix envs heavily — strong abstraction signal)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="parquet files")
    parser.add_argument("--n-per-env", type=int, default=1,
                        help="trajectories to print per env in hand-inspect")
    parser.add_argument("--k", type=int, default=10, help="cluster count")
    parser.add_argument("--skip-inspect", action="store_true")
    parser.add_argument("--skip-distrib", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true")
    args = parser.parse_args()

    rows = []
    for p in args.paths:
        rows.extend(read_trajectories(Path(p)))
    print(f"# total trajectories loaded: {len(rows)}")

    if not args.skip_inspect:
        hand_inspect(rows, n_per_env=args.n_per_env)
    if not args.skip_distrib:
        token_distribution(rows)
    if not args.skip_cluster:
        cluster_by_env(rows, k=args.k, drop_color_id=False)
        cluster_by_env(rows, k=args.k, drop_color_id=True)
        cluster_by_env(rows, k=args.k, invariant_features_only=True)


if __name__ == "__main__":
    main()
