# Phase 0c Audit (Tier A+D corpus)

**Date:** 2026-04-28
**Verdict:** Ready for Phase 1 prototype training. Scale-up to RunPod deferred until model architecture is locked.

## Corpus summary

4,700 successful trajectories across 5 environment markers, 32 distinct envs:

| env_marker | trajectories | distinct envs | unique terminals (pos-sig) | diversity |
|---|---|---|---|---|
| minigrid (BabyAI) | 3,800 | 26 | 3,378 | 89% |
| sokoban | 300 | 3 | 299 | 99.7% |
| sudoku | 200 | 3 (cluestype) | 200 | 100% |
| nonogram | 200 | 3 (size) | 200 | 100% |
| fifteen | 200 | 4 (scramble) | 1 | 0% (canonical) |
| **TOTAL** | **4,700** | **39** | **4,078** | **87%** |

Disk size: ~3 MB across all parquet files. Trivially scalable.

## Distribution properties

**Trajectory length:** spans 4 to 62 actions across envs. Coverage of short (5–10), medium (15–30), long (40–60) regimes is balanced across the corpus.

**Objects per frame:** median 15 (minigrid) to 59 (sudoku). Max 72. The `max_objects=128` cap handles the full Sudoku terminal (81 cells minus 9 background-color cells = 72 non-background objects).

**Colors / vocabulary:** 27 distinct color_ids in MiniGrid (sprite type × color combinations), 16 in 15-puzzle, 10 in Sudoku, 5 in Sokoban, 3 in Nonogram. The cross-env color vocabulary is 50+ ids; envs have non-overlapping ranges so the model can learn env-conditional semantics.

**Terminal-state diversity (positional-signature metric):** 4,078 unique terminals out of 4,700 trajectories. The 622 duplicates are mostly:
- All 200 fifteen-puzzle terminals (canonical solved state).
- A small number of MiniGrid GoToObj/GoToLocal trajectories where the bot stops at the same goal cell from different starting points.

The positional-signature metric captures real diversity that the color-id-set metric undersells. Sokoban, Sudoku, and Nonogram look identical by color sets but have 100% unique terminals by position.

## What the corpus teaches the predictor

The terminal-state-prediction objective is: given a prefix of K frames + actions, predict the distribution over terminal-state tokens. The corpus provides:

1. **Cross-env terminal diversity.** No env's terminals dominate. The 5 markers have qualitatively different terminal structures (filled grids in puzzles, cluttered final positions in MiniGrid, target-aligned boxes in Sokoban). The model is forced to learn the *shape* of "this trajectory ends in a goal-achievement state" rather than memorizing one env's terminal manifold.

2. **Variable trajectory length.** 4–62 actions means the model sees both very-short prefixes (where prediction must rely on initial scene structure) and very-long prefixes (where the model can lean on action history). At training time we sample prefix lengths uniformly; the corpus supports this without bias.

3. **Variable object count.** From 5 (Sokoban terminal) to 72 (Sudoku terminal). Predictor architecture must handle variable-length input.

4. **Color/type diversity.** 50+ distinct color_ids across envs. Combined with normalized centroid features, the model has enough vocabulary to distinguish env types without env_marker as a hard label (we still pass env_marker for fast-path conditioning).

5. **One canonical-terminal env (fifteen).** All trajectories end at the same state. This trains the model to recognize "this env converges to a fixed point" — useful for ARC-AGI-3 levels that have similar fixed-target structures.

## Known biases (and why they're acceptable for now)

1. **Optimal scripted bots.** All trajectories are short-as-possible. Real human play is longer and noisier. The terminal-state predictor doesn't care (it learns terminals, not policies), but a *trajectory-length* feature would be biased. We don't use trajectory length as a predictor input.

2. **Per-puzzle fixed length.** Sudoku trajectories from 35-clue starts are *exactly* 47 frames every time. If we ever add a "predict trajectory length" objective, this would leak. We don't; current objective is terminal-only.

3. **Crafter and NetHack absent.** Crafter requires symbolic-state extraction work; NetHack/MiniHack don't build on arm64 macOS. Both deferred. Their absence reduces RGB-perception-needed coverage; we'll add them in a Phase 0c+ on RunPod when scaling up.

4. **15-puzzle has only 1 terminal state.** Adds zero diversity by terminal-state count. Kept because it provides a "this env has a canonical fixed point" example, which ARC-AGI-3 has analogs of.

5. **No NetHack means less raw mechanical variety in one env.** MiniGrid+BabyAI compensates with 26 distinct envs spanning navigation, pickup, door logic, key+lock, sequencing, color-conditioned tasks, and composite missions. Reasonable substitute for the diversity NetHack would have provided.

## Phase 1 readiness verdict

**Ship.** The corpus is sufficient for prototype training of the terminal-state predictor. Specific reasons:

- 4,700 successful trajectories is well above the ~1k floor for prototype-quality transformer training on a small input space.
- Terminal diversity (87% unique by pos-signature) means the predictor has a real distribution to learn, not a delta function.
- Trajectory-length and object-count distributions are wide enough to stress-test variable-length attention.
- Cross-env vocabulary is rich enough to validate that the model learns env-invariant terminal structure rather than env_marker → terminal lookups.

**What would change the verdict:**
- If the prototype trains to validation perplexity and *fails to transfer* to ARC-AGI-3 (held-out 25 public games), we'd scale to 100k–1M trajectories on RunPod.
- If the model overfits per-env (ignores cross-env structure), we'd add weight on the env_marker masking auxiliary or scale up corpus diversity (the deferred Crafter/NetHack/Tier B/C envs).

## Phase 1 plan (next session)

1. **`models/terminal_predictor.py`** — small transformer over object-list tokens. Config: 4-layer encoder, 256-dim, 8-head. Input: a prefix of K frames as `(K, max_objects, 13)` token embeddings + per-frame position embedding + per-token mask. Output: distribution over the *terminal* frame's tokens.

2. **Generative head**: start with a simple flow-matching head over per-token features (color_id treated as discrete + cross-entropy, geometric features as continuous + flow-matching). InfoNCE auxiliary across batch to prevent mode collapse to env-marginal terminal distributions.

3. **Training script**: PyTorch Lightning or plain PyTorch. Mixed precision. Train on local M-series CPU/MPS first to validate the loop, then move to RunPod GPU for serious training when batch size demands it.

4. **Validation**: held-out fraction of corpus + zero-shot evaluation on ARC-AGI-3 public game prefixes (we have 25 public games via API key). The held-out generalization metric is the load-bearing question — does cross-env diversity actually teach env-invariant terminal structure?

5. **Decision after Phase 1 prototype**:
   - If validation perplexity drops sharply and zero-shot ARC-AGI-3 prediction looks plausible → proceed to Phase 2 (integrate into agent).
   - If perplexity drops but ARC-AGI-3 prefixes look noise → scale up corpus first.
   - If perplexity doesn't drop → architecture is wrong, revisit.

## Risks for Phase 1

- **Mode collapse to env-marginal terminals.** If the model just learns "average terminal for this env_marker," it's useless on ARC-AGI-3 which has its own marker. Mitigation: InfoNCE-style contrastive auxiliary so the model must distinguish *this* trajectory's terminal from any other.
- **Object permutation invariance.** Object lists have no canonical order. The decomposition sorts by size, but two near-equal-size objects can swap positions. Mitigation: use a permutation-invariant Transformer (no per-position embedding within a frame; rely on the geometric features for position).
- **Variable terminal length.** Sudoku terminals have 72 objects, Sokoban 5. Predictor must handle this naturally; padding + mask is the standard solution.

## What does NOT need a decision before Phase 1

- Final model size. Start small (~5M params), scale if needed.
- Whether to pretrain WM (Tier 2) first. Independent track. Phase 1 doesn't depend on it.
- Kaggle packaging. Phase 5 concern. Phase 1 is locally trainable.
