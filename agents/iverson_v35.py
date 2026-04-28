"""Iverson v3.5 — state-graph search + online rule induction.

Per-step:
  1. Convert frame to object list.
  2. Update state graph + rule store from previous transition (if any).
  3. Score each available action by:
       base 1.0
       + α * rule_store.suggest(state, action)
       + β * saliency.score_action(state, action)
       + γ * state_graph.novelty(state, action)
       − reset_penalty when action == RESET
  4. Add an exploration jitter (small uniform random) to break ties.
  5. If best action is ACTION6, set click target via saliency.

Per-episode (level WIN or GAME_OVER): just clear per-level scratch state.
Rules and state graph persist across levels within a game.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from models.converters import arc_agi_3_to_frame
from models.object_list import Frame

from .base import Agent
from .rules import RuleStore
from .saliency import best_click_target as saliency_click_target
from .saliency import score_action as saliency_score_action
from .state_graph import StateGraph

logger = logging.getLogger(__name__)


# Action key conventions (per arcengine):
#   0 = RESET, 1-5 = simple, 6 = complex (x,y), 7 = undo
RESET_ID = 0
UNDO_ID = 7
COMPLEX_ID = 6


class IversonV35(Agent):
    MAX_ACTIONS = 200

    # Scoring weights.
    W_RULE = 1.0
    W_SALIENCY = 0.5
    W_NOVELTY = 0.8
    W_JITTER = 0.05  # small random tie-breaker
    RESET_PENALTY = 100.0  # only RESET when forced
    UNDO_PENALTY = 0.3     # mild bias against undo unless rule store rewards it

    def __init__(self, game_id: str, baseline_actions: Optional[list[int]] = None,
                 seed: int = 0) -> None:
        super().__init__(game_id, baseline_actions)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.state_graph = StateGraph()
        self.rule_store = RuleStore()

        # Pending transition for per-step update.
        self._prev_frame: Optional[Frame] = None
        self._prev_action: Optional[int] = None
        self._prev_levels: int = 0

        self._step_idx = 0
        self._used_click_targets: deque[tuple[int, int]] = deque(maxlen=8)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def on_level_complete(self, level_idx: int) -> None:
        # Per-level scratch reset; keep rule store + state graph (persist across levels).
        self._used_click_targets.clear()
        # Don't clear _prev_frame here — the next call processes the post-WIN observation
        # which is the new level's first frame, not the same level.

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData
                       ) -> GameAction:
        current_frame = arc_agi_3_to_frame(latest_frame)

        # Process the previous transition (state graph + rule store update).
        if self._prev_frame is not None and self._prev_action is not None:
            levels_increased = latest_frame.levels_completed > self._prev_levels
            game_over = latest_frame.state is GameState.GAME_OVER
            self.state_graph.record_transition(
                self._prev_frame, self._prev_action, current_frame,
                level_won=levels_increased, game_over=game_over,
            )
            self.rule_store.induce(
                action_key=self._prev_action,
                before=self._prev_frame, after=current_frame,
                levels_increased=levels_increased,
                step=self._step_idx,
            )
            self._prev_levels = latest_frame.levels_completed

        # Reset if game state demands it.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_frame = None
            self._prev_action = None
            return GameAction.RESET

        avail = list(latest_frame.available_actions or [])
        if not avail:
            return GameAction.ACTION1

        scores = self._score_actions(current_frame, avail)
        best_action_id = max(scores, key=scores.get)

        # Build the action.
        action = self._build_action(best_action_id, current_frame)

        # Bookkeeping for next step.
        self._prev_frame = current_frame
        self._prev_action = best_action_id
        self._step_idx += 1
        return action

    def _score_actions(self, frame: Frame, avail: list[int]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for action_id in avail:
            base = 1.0
            rule_score = self.rule_store.suggest(frame, action_id)
            sal_score = saliency_score_action(frame, action_id)
            novelty_score = self.state_graph.novelty(frame, action_id)
            jitter = self.np_rng.uniform(0, 1)

            score = (base
                     + self.W_RULE * rule_score
                     + self.W_SALIENCY * sal_score
                     + self.W_NOVELTY * novelty_score
                     + self.W_JITTER * jitter)

            if action_id == RESET_ID:
                score -= self.RESET_PENALTY
            if action_id == UNDO_ID:
                score -= self.UNDO_PENALTY

            scores[action_id] = float(score)
        return scores

    def _build_action(self, action_id: int, frame: Frame) -> GameAction:
        if action_id == RESET_ID:
            return GameAction.RESET
        action = GameAction.from_id(action_id)
        if action.is_complex():
            x, y = self._pick_click_target(frame)
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {"src": "v3.5", "click_x": int(x), "click_y": int(y)}
        else:
            action.reasoning = "v3.5"
        return action

    def _pick_click_target(self, frame: Frame) -> tuple[int, int]:
        """Pick a click target.

        Strategy: rank objects by saliency, pick the i-th one where i =
        number-of-times-this-state-already-clicked, modulo n_objects. This
        cycles through salient objects across repeated clicks at the same
        state without permanently excluding any.

        On first click at a state: pick #1 salient.
        On second click: pick #2 salient.
        Etc.

        Why: bt33 has two equally-edge-touching buttons; the agent needs to
        try both, but a hard exclude eats the right one too aggressively when
        the model needs many clicks at the *same* button to make progress.
        """
        from .saliency import score_objects
        if not frame.objects:
            return 32, 32
        # Count past clicks at this state by visit count.
        from .state_graph import hash_frame
        h = hash_frame(frame)
        node = self.state_graph.nodes.get(h)
        # COMPLEX_ID = 6
        n_clicked_here = node.actions_tried.get(6, None).times_tried if node and node.actions_tried.get(6) else 0
        scores = score_objects(frame)
        order = np.argsort(-scores)
        # Filter to clickable size range.
        from .saliency import SaliencyConfig
        cfg = SaliencyConfig()
        valid_order = [int(i) for i in order
                        if cfg.min_size_for_click <= frame.objects[int(i)].size <= cfg.max_size_for_click]
        if not valid_order:
            return 32, 32
        idx = valid_order[n_clicked_here % len(valid_order)]
        obj = frame.objects[idx]
        H, W = frame.grid_shape
        x = int(round(obj.centroid_norm[0] * (W - 1)))
        y = int(round(obj.centroid_norm[1] * (H - 1)))
        x = max(0, min(63, x))
        y = max(0, min(63, y))
        self._used_click_targets.append((x, y))
        return x, y
