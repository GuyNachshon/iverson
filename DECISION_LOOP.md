# ARC-AGI-3 Agent v2.1: Integrated Decision Loop

## New Modules (Phase 2)

### 1. Undo-as-Reasoning (Action-Time-Training)
**Module**: `v2/models/undo_reasoning.py`

The Undo action is a **free experiment**, not a safety net.

```
Standard loop:         ATT loop:
  Act → Observe          Act → Observe → Undo → Re-observe
                         → Update world model
                         → Re-imagine all actions
                         → Pick the BEST one
                         → Execute for real
```

**Three modes**:
- **PROBE** (early game): Test each key action + undo. Cost: 2 actions per key × 6 keys = 12 actions. Gain: complete mechanics map + irreversibility detection.
- **VERIFY** (mid game): Only undo when world model prediction was wrong. Saves budget while correcting errors.
- **EXPLOIT** (late game): Never undo. Trust the model.

**Key insight**: Each probe gives us TWO transitions (action + undo) for the world model, doubling learning speed during exploration.

### 2. Symbolic Memory Buffer
**Module**: `v2/models/symbolic_memory.py`

Stores rules as explicit predicates:
```
[✓] R0: IF any_state() THEN key_0 → player_moves(right, 1)     (100% conf, 5✓/0✗)
[✓] R1: IF player_adjacent_to(3) THEN key_2 → object_removed(3) (100% conf, 3✓/0✗)
```

**Why this matters for cross-level generalization**:
- Level 1 teaches: \"key 0 = move right, key 1 = move down\"
- Level 3 teaches: \"key 2 = collect when adjacent\"
- Level 6 requires **composing** L1 + L3: \"navigate to object, then collect\"
- RSSM latent state from Level 1 degrades by Level 6. Symbolic rules don't.

**Rule Inducer**: Watches transitions and extracts patterns:
```python
inducer.observe_transition(grid_before, action_key=0, action_pos=55, grid_after)
# → Induces: IF action_on_object() THEN key_0 → player_moves(right, 1)
```

### 3. Boredom Detector
**Module**: `v2/models/symbolic_memory.py` (BoredomDetector class)

Detects three stuck patterns:
1. **Stagnation**: Same state for N consecutive actions
2. **Repetition**: Same action tried M times with no effect
3. **Oscillation**: A→B→A→B→A→B pattern

When bored, suggests diversification:
```python
{
  \"boredom_level\": 0.85,
  \"try_keys\": [1, 2, 3, 4, 5],  # Keys NOT tried recently
  \"avoid_positions\": [55],        # Positions that keep failing
  \"suggestion\": \"random_walk\"     # Break out of local loop
}
```

## Integrated Decision Loop (v2.1)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MAIN AGENT LOOP                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. OBSERVE grid                                                     │
│     │                                                                │
│  2. RECORD transition (if not first step)                            │
│     ├── Update transition buffer                                     │
│     ├── Feed action effect tracker                                   │
│     ├── Feed rule inducer → symbolic memory                          │
│     ├── Feed boredom detector                                        │
│     └── Check world model prediction accuracy                        │
│     │                                                                │
│  3. CHECK UNDO REASONER                                              │
│     ├── If awaiting undo result → process it, update WM              │
│     ├── If in PROBE mode → test current action with undo             │
│     └── If in VERIFY mode + prediction wrong → undo and re-learn     │
│     │                                                                │
│  4. CHECK BOREDOM                                                    │
│     ├── If bored → reset goal, try untested action                   │
│     └── If not bored → continue with current strategy                │
│     │                                                                │
│  5. SELECT ACTION (layered strategy)                                 │
│     │                                                                │
│     ├── Layer 0: Undo (if ATT says to probe/verify)                  │
│     │                                                                │
│     ├── Layer 1: Symbolic rule lookup                                │
│     │   \"Is there a CONFIRMED rule for my current goal?\"             │
│     │   e.g., \"I want to remove object(3). Rule R1 says:            │
│     │          press key 2 when adjacent to color 3\"                 │
│     │   IF YES → execute that rule's action                          │
│     │                                                                │
│     ├── Layer 2: Goal-directed navigation                            │
│     │   \"I know the mechanics. Navigate toward subgoal.\"             │
│     │   Uses action semantics (key 0 = right, key 1 = down, etc.)   │
│     │                                                                │
│     ├── Layer 3: CEM planning (world model imagination)              │
│     │   Only when: WM accuracy > 70% AND goal latent is known        │
│     │                                                                │
│     ├── Layer 4: Smart cell-select probing                           │
│     │   If cell-select budget remaining → probe next candidate       │
│     │                                                                │
│     └── Layer 5: Systematic exploration (fallback)                   │
│         Use targeted exploration with effective keys                  │
│     │                                                                │
│  6. EXECUTE action in environment                                    │
│     │                                                                │
│  7. ONLINE TRAIN world model (every N steps)                         │
│     │                                                                │
│  8. If level complete → persist state, advance level                 │
│     │                                                                │
│  └── LOOP                                                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Action Budget Allocation (for a typical level)

Assuming humans solve in ~15 actions → agent targets ≤30 actions for 25% RHAE floor.

| Phase | Actions | What Happens |
|-------|---------|-------------|
| Undo probes | 6-12 | Test each key + undo. Learn all mechanics. |
| Cell-select probes | 3-5 | Identify click purpose (teleport/toggle/etc.) |
| Goal setup | 0 | Visual analysis of salient objects (free) |
| Navigation | 5-10 | Move to each subgoal using learned mechanics |
| Interaction | 2-5 | Execute rules at each subgoal position |
| **Total** | **16-32** | Target: < 2× human baseline |

## Module Dependency Graph

```
           ┌─────────────────┐
           │  Environment     │
           └────────┬────────┘
                    │ grid, reward, done
                    ▼
    ┌───────────────────────────────┐
    │        Transition Buffer       │
    └───┬───┬───┬───┬───┬───┬──────┘
        │   │   │   │   │   │
        ▼   ▼   ▼   ▼   ▼   ▼
     ┌──┴─┐ ┌┴──┐ ┌┴──┐ ┌─┴──┐ ┌─┴──┐ ┌─┴──────┐
     │CNN │ │Ex-│ │Act│ │Rule│ │Bore│ │Undo    │
     │WM  │ │plo│ │Eff│ │Ind-│ │dom │ │Reason- │
     │RSSM│ │rer│ │Trk│ │ucer│ │Det │ │er(ATT) │
     └──┬─┘ └─┬─┘ └─┬─┘ └──┬┘ └─┬──┘ └──┬────┘
        │     │     │      │    │       │
        ▼     ▼     ▼      ▼    ▼       ▼
    ┌──────────────────────────────────────┐
    │        ACTION SELECTOR               │
    │  (6-layer priority: undo > symbolic  │
    │   > goal-directed > CEM > cell-sel   │
    │   > exploration)                     │
    └──────────────┬───────────────────────┘
                   │
                   ▼
              action (key, pos)
```

## What Persists Across Levels

| Component | Persists? | Why |
|-----------|-----------|-----|
| RSSM h_state, z_state | ✓ | Latent understanding of physics |
| Symbolic Memory rules | ✓ | Explicit mechanics knowledge |
| Action Effect Tracker | ✓ | \"key 0 = move right\" |
| Click Affordance Map | ✓ | Which cells are interactive |
| Undo knowledge | ✓ | Which keys are reversible |
| Transition Buffer | ✓ | Training data for world model |
| Boredom state | ✗ (reset) | Fresh patience per level |
| Goal/subgoals | ✗ (reset) | New objectives per level |
| Exploration phase | ✗ (reset to targeted) | Skip scan in later levels |

## Files

| File | Size | Purpose |
|------|------|---------|
| `v2/models/world_model.py` | 7.4M params | CNN encoder + RSSM dynamics + decoder |
| `v2/models/exploration.py` | — | Systematic multi-phase exploration |
| `v2/models/planning.py` | — | CEM planner with world model imagination |
| `v2/models/action_effects.py` | — | Action semantics learning |
| `v2/models/smart_cell_select.py` | — | Affordance map + budget + conditional mechanics |
| `v2/models/undo_reasoning.py` | — | ATT: undo-based hypothesis testing |
| `v2/models/symbolic_memory.py` | — | Rule buffer + inducer + boredom detector |

