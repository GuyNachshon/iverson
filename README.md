# iverson

ARC-AGI-3 agent. Targeting [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) (deadline Nov 2, 2026).

## Status

Phase 0a — minimal end-to-end loop running. Random baseline scores ~0 on the bundled test games (as expected; floor measurement). Next: wire the world model + undo reasoner into a v2.5 baseline agent, decide v2.5-ship vs v3-terminal-prediction.

## Layout

- `agents/` — agent base class, runner, baselines.
- `models/` — world model (CNN+RSSM), undo reasoner, and stubs for the rest of v2.1's modules.
- `environment_files/` — local ARC-AGI-3 game files (currently 2 test games; needs ARC API key for the full 25 public games).
- `scripts/` — CLI runners.
- `docs/` — competition info, design docs, architecture proposals.

## Run

```bash
uv sync
uv run python -m scripts.run_baseline --quiet
```

Requires `ARC_API_KEY` in `.env` for the 25 public games (free at [three.arcprize.org](https://three.arcprize.org/)).

## Reading

- `ARCHITECTURE.md` — v2.1 design (world model, undo-as-reasoning, symbolic memory).
- `DECISION_LOOP.md` — v2.1 integrated decision loop.
- `docs/opinions/Interventional-World-Modeling.md` — v3 critique and proposal (goal-acquisition as first-class peer to mechanics).
