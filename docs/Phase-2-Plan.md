# Phase 2 Plan: Integrate Terminal-State Predictor into Agent

**Status:** Design sketch (Phase 1 still in progress). Will revise after Phase 1 results land.

## Recap of what we have

After Phase 1, the trained predictor produces, given a prefix of K observations:
- A distribution over the terminal frame's object list.
- Per-slot: existence (binary), color_rank, normalized centroid (cx, cy), aspect, structural booleans, log-size, log-neighbor-count.
- All as classifications over bins (HL-Gauss soft targets for continuous), so we can read out:
  - **Mode (argmax)**: most likely terminal token.
  - **Expectation (softmax over bin centers)**: smooth point estimate.
  - **Entropy**: per-feature uncertainty signal.

## The Phase 2 question

How does the agent *use* this prediction to choose actions?

The original v3 design (Interventional-World-Modeling.md) called for MCTS over the world model with the goal posterior as the value function. The terminal-state predictor IS our goal posterior. So:

```
Agent state → predicted terminal distribution → MCTS rollouts via WM → action
```

But the world model (v2.5) was the failure point of the whole previous baseline — it learned grid dynamics but couldn't infer "where to click" because it had no object-centric substrate. **The terminal-state predictor doesn't fix the WM**; it tells the agent *what to aim for*.

So Phase 2 is really three sub-questions:

1. How do we score a candidate action *without* a usable WM?
2. How do we compose "terminal prediction" with "current state" into an action choice?
3. Can we get away with a much simpler planner (no MCTS) by leaning hard on the predictor?

## Three candidate Phase 2 architectures

### A. Predictor + heuristic scoring (simplest, ship-first)

For each available action, simulate forward one step using the actual env (we already do this in v2.5's undo-probe loop). Score each successor state by **distance from the predicted terminal distribution** (in object-list feature space). Pick the action that minimizes distance.

```
choose_action(state):
    pred_terminal = predictor(prefix_of_recent_states)
    best_action, best_score = None, inf
    for action in available_actions:
        successor = env_step(action); env_undo()  # use ACTION7 to undo
        score = distance(successor, pred_terminal)
        if score < best_score:
            best_action, best_score = action, score
    return best_action
```

**Cost**: 2 actions per real step (act + undo) × |available_actions|. For ARC-AGI-3 with up to 7 actions, that's 14 probe-style actions per real action. Expensive but bounded.

**Pros**: simplest possible thing that uses the predictor. No MCTS, no learned world model. Gets us a working Phase 2 in a day.

**Cons**: only 1-step lookahead. Bad on games with long action sequences before reaching a goal-shaped state.

### B. Predictor + learned simple WM (medium effort)

Train a tiny world model (much smaller than v2.5's CNN+RSSM) on object-list transitions. (object_list, action) → next_object_list. Use this for shallow MCTS rollouts (depth ≤ 4).

**Pros**: lookahead beyond 1 step. Doesn't require pixel-level reconstruction.
**Cons**: another module to build and validate. Distribution shift between training-corpus dynamics and ARC-AGI-3 dynamics is a real risk.

### C. Predictor + intervention-driven exploration (closest to the v3 vision)

Use the predictor to maintain uncertainty over the goal. Frame each action as either "exploit toward predicted goal" or "probe to refine the prediction." When predictor entropy is high, probe; when low, exploit.

This is a more principled version of (A). Adds an explicit uncertainty signal.

**Pros**: the natural unification of v2.1's undo-probing with v3's goal acquisition.
**Cons**: needs careful tuning of exploit/probe trade-off. Harder to debug than (A).

## Recommendation

**Ship (A) first as the minimum viable Phase 2.** It's a few hundred lines. If it beats v2.5 on the public 25 games — even by 5% — we have proof the terminal predictor is doing useful work and we can iterate to (C). If it doesn't beat v2.5, the terminal predictor isn't ready and we need to fix it before any planner sophistication helps.

## Concrete implementation plan for path A

1. **`agents/distance.py`** — distance function between an actual frame and a predicted terminal distribution.
   - For each predicted slot with exists_prob > 0.5, find the closest actual object (by centroid + color_rank).
   - Sum per-pair feature distances.
   - Penalize "missing" predicted slots and "extra" actual slots.
   - This is a Hungarian-matching variant; can use scipy.optimize.linear_sum_assignment or a simple greedy approximation.

2. **`agents/iverson_v3.py`** — new agent subclass.
   - On reset: store the initial frame.
   - Each step: build prefix from frames so far, run predictor, get predicted terminal.
   - For each available action: probe via undo, compute distance(successor, predicted_terminal), pick the minimum.
   - Re-predict every K steps as the prefix grows (predictor sees more context).

3. **`scripts/run_v3.py`** — runner CLI similar to `run_baseline.py`.

4. **Evaluation**: run v3 against the 5 public ARC-AGI-3 games we used for v2.5 (ft09, r11l, sb26, tn36, cd82). v2.5 scored 0/30 levels. If v3 clears even 1 level, the predictor is doing real work.

## Risks for Phase 2

- **Distance function is hard to get right**. Hungarian matching is correct but slow; greedy is fast but may misalign slots. Need to ablate.
- **Predictor confidence is high on padded slots, low on real ones**. The exists head learned "most slots are empty" from training; on ARC-AGI-3 it may predict 13 objects when 30+ are real. Distance may need to be normalized to handle count mismatches gracefully.
- **Probe cost is real.** 14 probes per real action means a level with human baseline 20 actions becomes 280 actions for the agent. Squared RHAE = (20/280)² = 0.5%. We need to *reduce probe count* to be competitive — only probe if the predictor's preferred action isn't obvious.
- **Re-prediction frequency matters.** Re-running the predictor every step is expensive (~50ms × 200 steps = 10s). Re-running every 10 steps is cheap but might miss state changes. Tunable.

## Time-budget sanity check for Kaggle

130k actions in 6 hours = 165ms/action. Predictor forward = 50-100ms. Plus 14 env probes ≈ negligible (env is ~0.5ms). Plus distance computation ≈ 5ms. Total ~ 100-150ms/action. **Tight but feasible.**

If we cut probes to "only run if predictor is uncertain" (e.g. top-2 actions have similar scores), per-action cost drops to ~50ms. Comfortable.

## What gets cut from the original v3 vision

- **MCTS**: deferred to Phase 2.5 (after we know whether the simpler approach works).
- **Symbolic memory** (rule predicates): deferred indefinitely. The terminal-state predictor obviates the need for hand-coded rule-extraction; the model's emergent "this terminal looks like X" classification IS the rule.
- **WALL-E-style symbolic alignment**: not needed for Tier 1. Defer until after the simpler architecture is shipped.
