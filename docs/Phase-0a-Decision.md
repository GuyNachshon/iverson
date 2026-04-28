# Phase 0a Decision: Commit to v3-Terminal-Prediction

**Date:** 2026-04-28
**Decision:** Skip v2.5-ship, commit to revised v3 (terminal-state prediction).

## Evidence

Phase 0a built the minimal v2.5 agent (CNN+RSSM world model from `models/world_model.py` + `UndoReasoner` from `models/undo_reasoning.py` + entropy-driven action selection over the WM's imagined prior, 200-action budget per game, online training every 4 actions on a 512-transition buffer).

**Results:**

| Agent | Test envs (bt11/bt33) | Public games |
|---|---|---|
| Random | 0/5 levels (bt33) | **0/52 levels across 9 games** |
| v2.5 | 5/5 (bt11), 0/3 (bt33) | **0/30 levels across 5 games** |

The bt11 result (5/5 levels at human-baseline action count) is **not generalizable** — bt11 has only 2 available actions (ACTION3, ACTION4), and ACTION4 always immediately fails the level via `lose()`. Any agent that learns "don't pick the action that triggers GAME_OVER" wins bt11. v2.5 happens to learn this in its 12-probe phase. Real public games have 5–7 actions including the complex (x,y) click ACTION6, and there is no equivalent "obviously wrong" action to avoid.

The bt33 result (0/3) is the load-bearing signal. bt33 is click-only (`available_actions=[6]`); winning requires hitting a small "left" UI sprite (~2x2 cells in a 64x64 grid). Random clicking has near-zero hit probability. v2.5's entropy-driven action selector ranks all click positions roughly equally because the WM hasn't learned a click-target bias — and *can't* with a CNN encoder that has no object-centric inductive bias. **bt33 is a simplified version of what most public games require, and v2.5 is structurally incapable of solving it.**

The 5-public-game v2.5 run (ft09, r11l, sb26, tn36, cd82) confirmed: v2.5 ≈ random on real games. Same floor.

## Why v2.5 Fails Structurally

The WM learns transition dynamics, not affordances. It can predict "if I press ACTION3 the player moves left," but it has no representation in which "this region is a button" is expressible. Click-target inference requires an object-centric substrate that the pixel CNN does not produce. Adding more training steps does not fix this — it's a representation issue, not a fitting issue.

## Path Forward (Revised v3)

Per `docs/opinions/Interventional-World-Modeling.md` and the subsequent terminal-state-prediction correction:

1. **Phase 0b**: Shared object-list representation (connected components from grid → tokens; per-env converters for non-grid envs in the corpus). Note for later: revisit slot attention if connected components fails on visually noisy envs.
2. **Phase 0c**: Tier A+D corpus — completed-success trajectories from MiniGrid, Crafter, BabyAI, Sokoban, NetHack/MiniHack, plus scripted Sudoku/Nonogram/15-puzzle traces. ~10 envs, 50k–200k trajectories. Tier B (Procgen, NLE) and Tier C (Doom/Mario via VLM-perception) deferred until invariance signal is measured.
3. **Phase 1**: Terminal-state predictor — small transformer over object-list tokens with flow-matching or diffusion head, trained with InfoNCE-style contrastive auxiliary to prevent mode collapse. Train on Tier A+D, validate zero-shot on held-out envs and on ARC-AGI-3 prefixes.
4. **Phase 2**: Replace v2.5's entropy-driven action selector with: (a) terminal-state predictor produces distribution over predicted terminal grids, (b) shallow MCTS over WM, scoring leaves by distance-to-predicted-terminal-distribution, (c) strategy hyperparameters (exploration_horizon, probe_aggressiveness, reversibility_preference) derived from prediction properties (entropy/distance/sparsity), (d) ATT folded into MCTS as info-gain primitive.
5. **Tier 2 (parallel/lower-priority)**: Synthetic ARC-like generator → pretrain `world_model.py`. Useful but not on critical path.

## Risks

- **Out-of-distribution at test time**: held-out 110 ARC-AGI-3 games may have terminal-state structures not represented in the corpus. Mitigation: corpus diversity is the primary defense; we'll measure transfer quality on the 25 public games before committing to scale.
- **Object-list representation breaks on perceptually-noisy envs** (Doom RGB, Mario sprites). Mitigation: Tier A+D is grid/symbolic-only; we add visual envs only if the easier corpus doesn't deliver invariance.
- **Per-action latency budget**: MCTS over WM + terminal predictor at 130k actions in Kaggle's 6h means ~165ms/action. Tight but feasible if we keep MCTS shallow (depth ≤4, sims ≤50).

## Kept From v2.1 / v2.5

- World model (`models/world_model.py`) — same CNN+RSSM, but pretrained on Tier 2 synthetic data (when ready) and used for MCTS rollouts in Phase 2.
- Undo reasoner (`models/undo_reasoning.py`) — repurposed as MCTS info-gain primitive, not a separate mode-switched module.
- Bug fixes: `UNDO_KEY=7`, `num_key_actions=8`.
- Runner + scoring (`agents/base.py`) — RHAE per-level scoring with capping at 1.15× and 1-indexed level weights, exactly per methodology.

## Discarded From v2.1 / v2.5

- Saliency-based goal inference (replaced by terminal-state prediction).
- CEM planner (replaced by shallow MCTS over joint mechanics-and-goal uncertainty).
- Mode-switched undo (PROBE/VERIFY/EXPLOIT) — folded into MCTS as info-gain.
- Symbolic memory as designed in DECISION_LOOP.md — replaced by emergent post-hoc clustering of learned terminal-state representations (no hand-coded vocabulary).
