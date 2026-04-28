# ARC-AGI-3 Agent v2: Architecture & Competition Strategy

## Competition: ARC Prize 2026 — ARC-AGI-3

**Prize Pool**: $850,000 ($150K progress + $700K bonus for 100% accuracy)  
**Deadline**: November 2, 2026  
**Current SOTA**: <1% (frontier AI)  
**Human baseline**: 100%

## The Problem

ARC-AGI-3 is fundamentally different from ARC-AGI-1/2:

| Feature | ARC-AGI-1/2 | ARC-AGI-3 |
|---------|-------------|-----------|
| Format | Static input→output grids | Interactive turn-based environments |
| Goal given? | Yes (implied by examples) | **No — must be inferred** |
| Rules given? | No, but examples show them | **No — must be discovered** |
| Metric | % tasks solved | **Action efficiency vs human baseline (RHAE)** |
| Levels | Single task | **≥6 levels per environment, building on each other** |

### The Four Capabilities Required (from the paper)
1. **Explore** an unknown environment
2. **Acquire goals** without being told them
3. **Build a world model** on the fly
4. **Learn continuously** (carry forward knowledge across levels)

### Technical Interface
- **Observation**: 64×64 grid, 16 colors
- **Actions**: 5 key actions + undo + cell-select (row, col) → up to 4,102 possible actions
- **Scoring**: RHAE = (human_actions / agent_actions)², capped at 1.15, weighted by level

## What Won the Preview Competition

| Approach | Score | Key Insight |
|----------|-------|-------------|
| CNN + RL (StochasticGoose, 1st) | 12.58% | Predict which actions cause frame changes |
| State graph search (Blind Squirrel, 2nd) | 6.71% | BFS/DFS over (state, action) graph |
| LRM + Python tools (Duke, academic) | Solved 3/3 public | Context management via code execution |
| Orchestrator + subagents (Symbolica, community) | Solved 3/3 public | Compressed textual summaries |

**Key finding**: Both Kaggle winners used **informed search**, not learned planning.

## v2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARC-AGI-3 Agent v2                               │
├─────────────────────────────────────────────────────────────────────────┤
│  Observation Grid (64x64, 16 colors)                                    │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ CNN Encoder       │  4-layer conv, 64→128→128→128 channels           │
│  │ (adaptive size)   │  Color embedding → conv → latent (256-dim)       │
│  └──────────────────┘                                                   │
│           ↓                                                             │
│  ┌──────────────────┐    ┌──────────────────┐                           │
│  │ RSSM Dynamics     │←──│ Action Effect     │                           │
│  │ (DreamerV3-style) │    │ Tracker           │                           │
│  │ GRU + categorical │    │ Learns: key 0 =   │                           │
│  │ latents (32×32)   │    │ "move right", etc  │                           │
│  └──────────────────┘    └──────────────────┘                           │
│           ↓                        ↓                                     │
│  ┌──────────────────┐    ┌──────────────────┐                           │
│  │ CEM Planner       │    │ Goal-Directed     │                           │
│  │ (when WM accurate │    │ Selector           │                           │
│  │  + goal known)    │    │ (navigate to       │                           │
│  └──────────────────┘    │  salient objects)  │                           │
│           ↓               └──────────────────┘                           │
│           ↓                        ↓                                     │
│  ┌──────────────────────────────────────────┐                           │
│  │           Action Selection                │                           │
│  │  Layered strategy:                        │                           │
│  │  1. Scan (first ~20 steps)                │                           │
│  │  2. Goal-directed (after mechanics known) │                           │
│  │  3. CEM planning (when WM accurate)       │                           │
│  │  4. Systematic exploration (fallback)     │                           │
│  └──────────────────────────────────────────┘                           │
│           ↓                                                             │
│  Action (key, position) → Environment                                   │
│           ↓                                                             │
│  ┌──────────────────┐                                                   │
│  │ Transition Buffer │  Records (obs, action, next_obs, reward, done)    │
│  │ + Online Training │  Updates world model every 4 steps                │
│  └──────────────────┘                                                   │
│                                                                         │
│  STATE PERSISTS ACROSS LEVELS (h_state, z_state, buffer, effect_tracker)│
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Online World Model (7.4M params)
- **CNN encoder/decoder** instead of ViT (10x faster for online learning)
- Learns from actual (obs, action, next_obs) transitions, not masked prediction
- DreamerV3-style categorical latents (32×32) with straight-through gradients
- **Reaches 99.9% prediction accuracy within ~100 transitions**
- Loss drops from 3.5 → 0.12 over the course of one environment

### 2. Systematic Exploration
- **Phase 1 (Scan)**: Test each of 6 key actions × each salient object position
  - Discovers which keys cause frame changes and in what contexts
- **Phase 2 (Targeted)**: Weight actions by observed effectiveness
  - Prefer keys with high change-rate, click on objects preferentially
- **Phase 3 (Frontier)**: BFS over unexplored (state, action) pairs

### 3. Action Effect Tracker
- Tracks displacement patterns: "key 0 moves player right by 1 cell"
- Infers action semantics: move_right, move_down, teleport, interact
- Enables goal-directed navigation: "go to position (10, 10)"

### 4. Goal Inference
- **Visual saliency**: Rare colored objects = potential goals
- **Terminal state tracking**: After level completion, learn what "winning" looks like
- **Subgoal cycling**: Navigate to each salient object in order

### 5. CEM Planning
- Cross-Entropy Method with world model imagination rollouts
- Only activates when world model is >70% accurate AND a goal is known
- Evaluates 64 candidate action sequences over 8-step horizon

## Verified Results (Mock Environment)

| Metric | Result |
|--------|--------|
| World model learning | 99.9% prediction accuracy after ~100 transitions |
| WM loss convergence | 3.5 → 0.12 in one environment |
| Action semantics | Correctly identifies move_right, move_down, teleport |
| State persistence | ✓ RSSM state carries across levels |
| Goal inference | Correctly identifies salient objects from visual scene |
| Speed | 2-3 seconds per 200-step level on T4 GPU |

## Competition Roadmap

### Phase 1: Foundation (Current) ✅
- [x] Online world model with CNN encoder + RSSM dynamics
- [x] Systematic exploration with action scanning
- [x] Action effect tracking and semantic inference
- [x] Visual goal inference from saliency
- [x] CEM planning with world model imagination
- [x] State persistence across levels

### Phase 2: Goal Acquisition (Next)
- [ ] **Subgoal cycling**: Try all actions at each goal position, not just navigate there
- [ ] **State-change as goal signal**: If an action at a goal position changes the grid differently than expected, that's progress
- [ ] **Terminal state feature extraction**: Learn what "winning frames" look like from successful level completions
- [ ] **Backtracking**: If stuck, try reversing to a previously successful state

### Phase 3: Advanced Planning
- [ ] **MCTS instead of CEM**: Monte Carlo Tree Search with world model for deeper planning
- [ ] **Symbolic rule extraction**: Convert learned world model into explicit rules (WALL-E 2.0 style)
- [ ] **Hypothesis testing**: Explicitly form and test hypotheses about game mechanics
- [ ] **Efficiency optimization**: Once a level is solvable, find the shortest action sequence

### Phase 4: Meta-Learning
- [ ] **Pre-training on procedural grid games**: Generate diverse environments for world model pre-training
- [ ] **Algorithm Distillation**: Train on learning histories to learn "how to learn" in-context
- [ ] **AdA-style meta-RL**: Memory-based learning that accumulates within-episode knowledge

### Phase 5: Competition Polish
- [ ] **Kaggle notebook integration**: Package agent for 6-hour Kaggle runtime
- [ ] **Time budgeting**: Allocate exploration/planning time per level based on remaining budget
- [ ] **Ensemble strategies**: Run multiple exploration strategies in parallel, pick best

## Literature Basis

This architecture draws from:

1. **DreamerV3** (Hafner et al., 2023) — RSSM world model with categorical latents
2. **StochasticGoose** (1st place preview) — CNN for frame-change prediction
3. **WALL-E 2.0** (2025) — Neurosymbolic world alignment, rule induction from failures
4. **AdA** (DeepMind, 2023) — Memory-based meta-RL for novel task adaptation
5. **Algorithm Distillation** (DeepMind, 2022) — In-context RL from learning histories
6. **BYOL-Explore** (2022) — Intrinsic exploration without reward
7. **Intelligent Go-Explore** (2024) — Foundation model-guided exploration
8. **TTT for ARC** (2024) — Test-time training with LoRA adapters

## Key Insight

The AutumnBench paper (Oct 2025) showed that **even o3 fails at systematic grid-world exploration**. The bottleneck is metacognitive: hypothesis → test → update belief. Current LLMs explore randomly instead of systematically.

This agent's design specifically addresses this by separating:
- **Exploration** (systematic, not random)
- **World model learning** (from actual transitions, not pre-training)
- **Planning** (only when the model is accurate)
- **Goal inference** (visual saliency + terminal state analysis)

The squared efficiency penalty in RHAE scoring means that being 10x less efficient than humans = 1% score. This makes every wasted action expensive. The agent must learn fast and act precisely.
