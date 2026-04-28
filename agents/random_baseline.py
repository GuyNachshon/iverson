"""Random baseline. Honors `available_actions` so it doesn't waste actions
the env will reject. This is the floor we need v2.5 to beat by a wide margin.
"""
from __future__ import annotations

import random
from typing import Optional

from arcengine import FrameData, GameAction, GameState

from .base import Agent


class RandomBaseline(Agent):
    """Picks a uniformly random action from `available_actions` each step."""

    def __init__(self, game_id: str, baseline_actions: Optional[list[int]] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(game_id, baseline_actions)
        self.rng = random.Random(seed)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        avail = latest_frame.available_actions or []
        if not avail:
            # Degenerate: env reports no actions. Try ACTION1 as a defensive default.
            return GameAction.ACTION1

        action_id = self.rng.choice(avail)
        action = GameAction.from_id(action_id)
        if action.is_complex():
            action.set_data({"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)})
            action.reasoning = {"src": "random-baseline"}
        else:
            action.reasoning = "random-baseline"
        return action
