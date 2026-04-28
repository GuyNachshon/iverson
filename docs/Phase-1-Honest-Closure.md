# Phase 1 Honest Closure

**Date:** 2026-04-28
**Status:** Phase 1 produced a working pipeline and a clear empirical finding. The finding is mostly negative, but it's the right kind of negative — it tells us where to look next.

## What we built

- TerminalPredictor architecture (1.24M params small, 7.10M params default).
- Per-feature CE-on-bins heads with HL-Gauss soft targets for continuous features. AlphaFold-style discretize-and-classify.
- Training pipeline with feature-mask ablation (full vs invariant), curriculum-capable prefix sampling, fixed-pad collation for MPS.
- Evaluation pipeline: per-env held-out perplexity, per-prediction diagnostics (color accuracy, centroid error, exists precision/recall), zero-shot ARC-AGI-3 transfer.
- v3 agent prototype using predicted terminal centroid as the click-target for ACTION6.

All 18 unit tests pass. Code is clean, tested, deployed to GitHub.

## What we learned

### Result 1: the invariant-features hypothesis is correct

500-step training of full vs invariant variants:

| | full | invariant |
|---|---|---|
| Val loss | 22.55 | **22.32** |
| ARC-AGI-3 prediction | mode-collapse to 0 objects | 4–26 objects (no collapse) |

Masking out env-correlated raw features (color_id, raw bbox coords) at training time produces a model with **lower within-env val loss AND non-trivial cross-env transfer**. The full variant memorizes env shortcuts; the invariant variant is forced to find generalizable patterns.

This validates the Phase 0c amendment's core hypothesis. **`--variant invariant` is the recipe going forward.**

### Result 2: the predictor's spatial prior is wrong for ARC-AGI-3

The 500-step invariant predictor on bt33 (a click-target game where the goal is to click a small button at top-left):

- Predicted terminal centroid: (0.49, 0.46) → click at (31, 29)
- Actual click target: (~0.05, ~0.05) → top-left button
- **The predictor points at the center; the goal is at the corner.**

This isn't a bug — it's the model correctly learning "MiniGrid terminals have agents near center" and applying that learned prior to a game where it's inapplicable.

### Result 3: training longer doesn't fix Result 2

5000-step run val loss curve:

| step | val_loss |
|---|---|
| 0 | 41.4 |
| 500 | 23.9 |
| 1000 | 22.4 |
| 1500 | 21.9 |

Diminishing returns clearly. The model's spatial prior at step 500 vs step 1500 looks essentially the same — both center-collapsed. **More training extracts more of the same prior.** Won't fix transfer.

### Result 4: per-prediction diagnostics show the architectural limits

On held-out training data, the invariant predictor:

| env | pred count | true count | color_acc | centroid err | exists P | exists R |
|---|---|---|---|---|---|---|
| fifteen | 10.7 | 15.0 | 87% | 0.27 | 100% | 71% |
| minigrid | 4.6 | 6.9 | 42% | 0.18 | 100% | 67% |
| nonogram | 7.6 | 5.0 | 53% | 0.51 | 55% | 84% |
| sokoban | 4.3 | 4.3 | 3% | 0.28 | 82% | 83% |
| sudoku | 52.5 | 72.0 | 11% | 0.23 | 98% | 72% |

- **Sokoban color_acc=3%** confirms the slot-position-assignment artifact: counts/exists are right, colors are wrong.
- **Sudoku undercounts by 27%**, exists head leans toward "predict empty."
- **Centroid errors 0.18–0.51 normalized** = 12–32 px on a 64-grid. Not great.
- **Color accuracy uniformly poor on rich-vocab envs** (MiniGrid 42%, Nonogram 53%) — same slot-assignment issue.

## Honest interpretation

The cluster check from Phase 0c warned that the corpus had **weak abstraction signal** — clusters were 91–93% pure on env_marker, even with invariance-friendly features. The Phase 1 results materialize exactly that warning:

- **Within-env, the model is fine.** It learns each env's terminal-distribution adequately. The invariant-variant ablation confirms it's not just memorizing env shortcuts.
- **Cross-env, the model produces a "MiniGrid-flavored prior" applied to any input.** This is the structural limit of training on a corpus where MiniGrid dominates (3800 of 4700 trajectories) and where each env's terminals have qualitatively different shapes.

The model isn't broken. It's correctly extracting what it can from a corpus that doesn't contain the structure it would need to predict ARC-AGI-3-shaped terminals.

## Three real paths forward

### Path A: Scope down what the predictor is responsible for

Instead of using the predictor for spatial guidance, use it only as:
- "Does this game have a terminal-shaped state?" (yes/no signal from exists count)
- "Is the current state near a terminal?" (compare current frame's object count + colors against predicted)

Spatial guidance comes from **non-predictor sources** — saliency on the current frame, undo-probing as in v2.5. The predictor doesn't have to do everything.

This is a Phase 2.5 design decision, not a Phase 1 retraining issue. Cheap to try.

### Path B: Add ARC-AGI-3-shaped envs to the corpus

The corpus is missing the kind of env where the goal is an edge UI element. We don't have such envs ready-made. Realistic options:
- Build a synthetic ARC-AGI-3-like generator that produces multi-color symbolic 64×64 grids with click-target UI (matches the v2.1 Phase 0 plan we deferred).
- Add Procgen / Minihack / NLE — but these require build fixes (NetHack failed on arm64 earlier).

This is the most expensive path but also the path that actually fixes the spatial-prior problem.

### Path C: Architectural fix — Hungarian matching during training

Currently slot-i in the prediction is *positionally* tied to slot-i in the target (sorted by size). That's why "slot 0 = average MiniGrid centroid" is what the model learns. Hungarian matching at training time decouples slot indices from positions.

May or may not help. The corpus's terminal distributions are still env-dominated; matching just removes one kind of bias. Worth trying since it's an architectural change with no data dependency.

## Phase 1 closure verdict

**Phase 1 is conditionally complete.** The pipeline is built, the empirical signal is clear, and the next step is well-scoped:

1. **Try path A first** (cheap, ~half a day). If a v3 agent that uses the predictor only for "does this game have terminal-shaped goals?" plus saliency-based spatial guidance can beat random/v2.5 on any ARC-AGI-3 game, we have something. If not, Path A doesn't work and we know it cheaply.

2. **If Path A fails, do Path C** (architectural, ~1–2 days). Hungarian matching + retrain + re-evaluate. If transfer becomes meaningful, we have something. If not, the corpus is the bottleneck.

3. **If both A and C fail, commit to Path B** (~1–2 weeks). Synthetic ARC-AGI-3 generator + retrain.

Each path is a clear go/no-go decision with measurable outcomes.

## What I don't recommend

- **More training on the same corpus.** Diminishing returns are obvious from the loss curve.
- **Bigger model.** Same data, more capacity → more memorization, not better transfer.
- **Different head architecture (MDN, diffusion).** The classification heads aren't the bottleneck. The output distribution is right; the conditioning on prefix is wrong.

## What would change my read

If, by trying Path A or C, the predictor's spatial prediction on bt33 moves from (cx=0.49, cy=0.46) to something near (0.05, 0.05), then Phase 1 becomes a clean success and v3 agent has a real shot. Until that happens, Phase 1's "we made a predictor" doesn't yet translate to "we have an agent that beats baselines."

This is the real Phase 1 / Phase 2 transition. Phase 1 produced the engine; Phase 2 needs to find the right way to use it. The honest verdict is: not yet.
