"""Local Agent base + runner.

Decoupled from arcprize/ARC-AGI-3-Agents to keep our dev loop fast (no Swarm,
no scorecard plumbing). When we package for Kaggle Competition Mode we wrap
this in their `Agent` subclass.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import FrameData, FrameDataRaw, GameAction, GameState

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    game_id: str
    levels_completed: int
    win_levels: int
    actions_taken: int
    final_state: GameState
    seconds: float
    baseline_actions: list[int] = field(default_factory=list)
    actions_per_level: list[int] = field(default_factory=list)

    def per_level_score(self) -> list[float]:
        """RHAE per completed level: min(1.15, baseline/agent)^2.

        Uses actions_per_level when available (tracked by run_agent). Falls
        back to charging all actions to the last completed level if not.
        """
        scores: list[float] = []
        n_baselines = len(self.baseline_actions)
        for i in range(n_baselines):
            if i >= self.levels_completed:
                scores.append(0.0)
                continue
            baseline = self.baseline_actions[i]
            if i < len(self.actions_per_level):
                agent_actions = max(self.actions_per_level[i], 1)
            else:
                agent_actions = max(self.actions_taken, 1)
            ratio = min(1.15, baseline / agent_actions)
            scores.append(ratio * ratio)
        return scores

    def weighted_game_score(self) -> float:
        """Per-game score per methodology: sum(weight * level_score) / sum(weights for all levels).

        Weights are 1-indexed level numbers. Uncompleted levels score 0 but
        still contribute to the denominator (per methodology — partial
        completion caps your max score).
        """
        scores = self.per_level_score()
        if not scores:
            return 0.0
        total_weight = sum(i + 1 for i in range(len(scores)))
        weighted = sum((i + 1) * s for i, s in enumerate(scores))
        return weighted / total_weight if total_weight else 0.0


class Agent(ABC):
    """Minimal agent interface. Subclasses implement choose_action / is_done.

    Mirrors arcprize.agents.agent.Agent's required methods so wrapping is
    trivial when we package for Kaggle.
    """

    MAX_ACTIONS: int = 200  # higher than arcprize default (80) — we want headroom for probes

    def __init__(self, game_id: str, baseline_actions: Optional[list[int]] = None) -> None:
        self.game_id = game_id
        self.baseline_actions = baseline_actions or []
        self.frames: list[FrameData] = []
        self.action_counter = 0

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool: ...

    @abstractmethod
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction: ...

    # Lifecycle hooks — override as needed.
    def on_level_complete(self, level_idx: int) -> None:
        pass

    def on_game_start(self, initial_frame: FrameData) -> None:
        pass


def _to_frame_data(raw: FrameDataRaw) -> FrameData:
    return FrameData(
        game_id=raw.game_id,
        frame=[arr.tolist() for arr in raw.frame],
        state=raw.state,
        levels_completed=raw.levels_completed,
        win_levels=raw.win_levels,
        guid=raw.guid,
        full_reset=raw.full_reset,
        available_actions=raw.available_actions,
    )


def run_agent(agent: Agent, env: Any, max_actions: Optional[int] = None) -> GameResult:
    """Run one agent against one env, returning a GameResult."""
    cap = max_actions if max_actions is not None else agent.MAX_ACTIONS
    initial = _to_frame_data(env.observation_space)
    agent.frames.append(initial)
    agent.on_game_start(initial)

    start = time.time()
    last_levels = initial.levels_completed
    actions_per_level: list[int] = []
    actions_in_current_level = 0

    while not agent.is_done(agent.frames, agent.frames[-1]) and agent.action_counter < cap:
        action = agent.choose_action(agent.frames, agent.frames[-1])
        data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
        if "game_id" not in data:
            data["game_id"] = agent.game_id
        reasoning = data.pop("reasoning", {}) if isinstance(data, dict) else {}
        raw = env.step(action, data=data, reasoning=reasoning)
        frame = _to_frame_data(raw)
        agent.frames.append(frame)
        agent.action_counter += 1
        actions_in_current_level += 1
        if frame.levels_completed > last_levels:
            for _ in range(frame.levels_completed - last_levels):
                actions_per_level.append(actions_in_current_level)
                actions_in_current_level = 0
                agent.on_level_complete(len(actions_per_level) - 1)
            last_levels = frame.levels_completed

    elapsed = time.time() - start
    final = agent.frames[-1]
    return GameResult(
        game_id=agent.game_id,
        levels_completed=final.levels_completed,
        win_levels=final.win_levels,
        actions_taken=agent.action_counter,
        final_state=final.state,
        seconds=elapsed,
        baseline_actions=agent.baseline_actions,
        actions_per_level=actions_per_level,
    )


def make_arcade(mode: OperationMode = OperationMode.OFFLINE) -> Arcade:
    return Arcade(operation_mode=mode)
