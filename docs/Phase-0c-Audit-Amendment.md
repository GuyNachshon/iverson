# Phase 0c Audit Amendment: pre-Phase-1 inspection

**Date:** 2026-04-28 (same session, post-review)
**Trigger:** Reviewer flagged a risk in the original audit's "diversity" claim — distinct terminal *positional signatures* might still be 1-1 with envs, in which case the corpus is nominally diverse but doesn't teach goal-shape abstraction. Action item: cluster terminals + inspect for cross-env mixing before committing to training.

## Findings

### 1. Hand-inspect (2 trajectories per env)

All sampled trajectories have correct structure:
- 15-puzzle: `centroid_moved=0.000` — symmetric tile swaps, mean position unchanged. Expected.
- Nonogram: `init_n=0` (empty board), `term_n=3-5` (filled cells). Expected.
- Sudoku: `init_n=40` (clues), `term_n=72` (full board minus background). Math checks out.
- Sokoban: `init_n=7-10`, `term_n=4-7`. Object count drops because boxes-on-target merge with targets. Expected.
- MiniGrid: small centroid drift (0.015–0.025). Expected (one moving agent, ~30 static objects).

**Frame.raw fix verified holding** — no trajectory shows `init == terminal`.

### 2. Token feature distribution

All 13 features have reasonable ranges:
- `color_id` 0–160 (cross-env vocabulary), `color_rank` 1–25 (per-env normalized).
- `log_size` 0.69–5.07 (singletons up to ~150 cells).
- `aspect` 0.10–11.00 (very tall/wide bounding boxes possible).
- `is_singleton` 90% on average (most objects are 1-cell sprites).
- `touches_edge` 14% on average (mostly interior objects).

No outliers, no near-constant features.

### 3. K-means cluster-by-env (THE LOAD-BEARING CHECK)

Three runs:

| features | weighted purity | interpretation |
|---|---|---|
| All 13 features | 91% | clusters mostly 1-1 with envs |
| Drop raw color_id | 90% | basically unchanged — color isn't the only env signal |
| Invariance-friendly only (color_rank, geometry, aspect, edge, neighbors) | 93% | even more env-distinguishable on "invariant" features |

**Per-env terminal fingerprints:**

| env | objs/term | size_mean | aspect | singletons | edge | neighbors |
|---|---|---|---|---|---|---|
| fifteen | 15.0 | 0.69 (log) | 1.00 | 100% | 73% | 1.39 |
| minigrid | 14.3 | 1.00 | 1.01 | 86% | 4% | 1.31 |
| nonogram | 5.2 | 1.19 | 1.11 | 48% | 83% | 1.08 |
| sokoban | 5.0 | 1.16 | 1.05 | 68% | 0% | 1.37 |
| sudoku | 72.0 | 0.69 | 1.00 | 100% | 40% | 1.51 |

Each env has a clearly distinct geometric signature. Sudoku's "72 singletons in a perfect grid" is unambiguous; Sokoban's "5 interior objects, none touching edges" is unambiguous; Fifteen's "15 singletons, 73% touching edges" is unambiguous.

## Honest interpretation

**The corpus has weak abstraction signal at the aggregate-statistics level.** A k-means classifier on summary vectors can identify the source env with ~93% purity. This means a poorly-designed predictor — one that lossy-compresses each frame to summary statistics before predicting the terminal — would learn `env_marker → terminal lookup` rather than cross-env structural abstraction.

**However, the test is a lower bound on the real diversity available to an attention-based model.** A token-level transformer with cross-attention can find specific structural patterns within the object lists (e.g., "player adjacent to a unique-rare-color object" applies across navigate-to-goal and reach-color-cell terminals) that summary statistics destroy. Whether this access is enough to overcome the env-correlation of aggregate features is an empirical question that only training can answer.

## What this changes about Phase 1

Three concrete adjustments:

1. **Mask out env-correlated absolute features at training time.** Specifically, randomly drop or noise-add to `color_id` and absolute bbox coordinates with probability 0.5 per batch. Keep `color_rank` (per-env normalized), centroids (normalized to [0,1]), aspect, and structural booleans. This forces the model to learn from invariance-friendly features rather than env shortcuts.

2. **Add a feature-ablation sanity check** as the very first training experiment. Train two models: full features vs invariance-friendly features only. If they validate equally, the model isn't really using env shortcuts (good). If full features wins big, the model is shortcutting (bad — apply mitigation #1 more aggressively).

3. **Lower expectations for cross-env transfer.** The corpus may produce a predictor that's good at within-env terminal prediction but transfers weakly to ARC-AGI-3. We have to test this empirically — and if transfer is weak, the corpus is the bottleneck and we add ARC-AGI-3-relevant envs.

## Phase 1 readiness verdict (revised)

**Conditional ship.** Train, but with three explicit checkpoints:

1. **First validation**: does invariance-feature-only training converge? If not, the corpus genuinely lacks cross-env signal and we expand it before scaling.
2. **Held-out perplexity per env**: does the model generalize within-env? Sanity check that training works at all.
3. **Zero-shot ARC-AGI-3 transfer**: does prediction on ARC-AGI-3 prefixes look qualitatively right? **This is the load-bearing question.** If yes, we proceed. If no, add envs that share terminal structure with ARC-AGI-3 (other 64×64 grid envs with multi-color symbolic encoding).

## What did NOT change

- Phase 0b (object-list representation) is sound.
- Trajectory format and serialization are sound.
- Per-env collectors are sound (100% bot success rate, no Frame.raw bugs after fix).
- 4,700 trajectories is enough to start training prototype-quality models. The risk is not "too small a corpus" but "wrong-shaped corpus." Adding more of the same probably won't help; the right addition is *types of envs* that share goal structure with each other and with ARC-AGI-3.

## Pause confirmed

Per the reviewer: pausing here. Phase 1 starts in a fresh session with these adjustments captured.
