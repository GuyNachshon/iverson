"""Iverson v3 — terminal-state-prediction-driven agent.

Decision loop per step:
  1. RESET if game state requires it.
  2. Build a prefix from frames seen so far. Re-run the predictor every K steps
     (cheap if K is small enough; predictor is fast).
  3. For each available action, *probe* by acting and undoing (ACTION7) to see
     the successor state without committing.
  4. Score each successor by its distance to the predicted terminal.
  5. Pick the action with the lowest distance.

This deliberately uses NO learned world model; we use the actual env's undo
to roll out one step. That avoids the WM-as-imagined-rollouts failure mode
that broke v2.5. Cost: 2 actions per real action × |available_actions|
(probe + undo). For typical |actions|=5, that's 10 probe-actions per real
action.

Optimizations:
  - Skip probing if predictor confidence is low (don't waste actions).
  - Cache the predicted terminal — only re-run predictor every PRED_EVERY steps.
  - Fall back to random sampling among legal actions when predictor is undecided.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
from arcengine import FrameData, GameAction, GameState

from models.converters import arc_agi_3_to_frame
from models.object_list import Frame
from models.terminal_predictor import (
    PredictorConfig,
    TerminalPredictor,
    feature_mask_full,
    feature_mask_invariant,
)

from .base import Agent
from .distance import (
    TerminalPrediction,
    decode_predictor_output,
    distance_to_predicted_terminal,
)

logger = logging.getLogger(__name__)


# ARC-AGI-3 ACTION7 is the undo. (Set on UndoReasoner; matches arc-agi-3 docs.)
UNDO_ACTION_ID = 7


def _frame_to_token_array(frame: Frame, max_objects: int = 128) -> tuple[np.ndarray, np.ndarray]:
    return frame.to_array(max_objects=max_objects)


class IversonV3(Agent):
    """v3 baseline: terminal-state predictor + undo-based 1-step lookahead."""

    MAX_ACTIONS = 200
    PRED_EVERY = 4              # re-run predictor every N agent steps
    EXISTS_THRESHOLD = 0.5      # predictor exists-prob threshold
    PROBE_BUDGET_PER_LEVEL = 50  # cap probe-undo overhead per level
    MAX_PREFIX_FRAMES = 8       # truncate long prefixes for predictor

    def __init__(self, game_id: str, baseline_actions: Optional[list[int]] = None,
                 ckpt_path: str | None = None,
                 device: str = "cpu", seed: int = 0,
                 match_method: str = "hungarian") -> None:
        super().__init__(game_id, baseline_actions)
        if ckpt_path is None:
            raise ValueError("IversonV3 requires --ckpt path to a trained predictor")

        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # Load predictor.
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        pcfg = PredictorConfig(**state["config"])
        self.model = TerminalPredictor(pcfg).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.train(False)
        self.variant = state.get("variant", "full")
        if self.variant == "full":
            self.fmask = feature_mask_full(self.device)
        else:
            self.fmask = feature_mask_invariant(self.device)
        self.match_method = match_method

        # Per-level rolling state.
        self._frame_history: deque[Frame] = deque(maxlen=64)
        self._cached_pred: Optional[TerminalPrediction] = None
        self._steps_since_pred = 0
        self._probes_used = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def on_level_complete(self, level_idx: int) -> None:
        # Reset per-level state but keep the loaded predictor.
        self._frame_history.clear()
        self._cached_pred = None
        self._steps_since_pred = 0
        self._probes_used = 0

    def _make_prefix_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack the last MAX_PREFIX_FRAMES frames as a (1, K, max_objects, 13) tensor."""
        recent = list(self._frame_history)[-self.MAX_PREFIX_FRAMES:]
        if not recent:
            # Edge case: should never happen in choose_action since we add the current frame.
            empty = torch.zeros(1, 1, 128, 13, dtype=torch.float32, device=self.device)
            empty_mask = torch.zeros(1, 1, 128, dtype=torch.float32, device=self.device)
            return empty, empty_mask
        K = len(recent)
        tokens = np.zeros((K, 128, 13), dtype=np.float32)
        masks = np.zeros((K, 128), dtype=np.float32)
        for k, f in enumerate(recent):
            t, m = _frame_to_token_array(f, max_objects=128)
            tokens[k] = t
            masks[k] = m
        prefix_t = torch.from_numpy(tokens).unsqueeze(0).to(self.device)
        prefix_m = torch.from_numpy(masks).unsqueeze(0).to(self.device)
        return prefix_t, prefix_m

    def _refresh_prediction(self) -> TerminalPrediction:
        prefix_t, prefix_m = self._make_prefix_tensor()
        with torch.no_grad():
            out = self.model(prefix_t, prefix_m, feature_mask=self.fmask)
        pred = decode_predictor_output(out, exists_threshold=self.EXISTS_THRESHOLD)
        self._cached_pred = pred
        self._steps_since_pred = 0
        return pred

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData
                       ) -> GameAction:
        # Convert latest_frame to object-list and append.
        current_frame = arc_agi_3_to_frame(latest_frame)
        self._frame_history.append(current_frame)

        # Reset if needed.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        avail = list(latest_frame.available_actions or [])
        if not avail:
            return GameAction.ACTION1

        # Refresh prediction if cache is stale.
        if self._cached_pred is None or self._steps_since_pred >= self.PRED_EVERY:
            self._refresh_prediction()
        else:
            self._steps_since_pred += 1

        # If we've exhausted the probe budget for this level, fall back to
        # uniform random over legal actions (cheap sane baseline).
        if self._probes_used >= self.PROBE_BUDGET_PER_LEVEL:
            return self._random_action(avail)

        # Score each available action by 1-step probe (act + undo).
        # Note: ACTION6 (complex) needs (x,y) data; we sample a few candidates
        # per probe as a lightweight optimization.
        scores = self._score_actions(avail, current_frame)

        # If all scores are equal (predictor undecided), random.
        score_values = list(scores.values())
        if max(score_values) - min(score_values) < 1e-3:
            return self._random_action(avail)

        # Pick min-distance action.
        best_action_id = min(scores, key=scores.get)
        # Note: scores keys are encoded action descriptors below.
        return self._build_action(best_action_id)

    def _score_actions(self, avail: list[int], current_frame: Frame) -> dict:
        """Probe each available action via undo to estimate per-action score.

        For simple actions (1-5, 7), one probe each.
        For ACTION6 (complex), sample K positions and score each.

        Returns dict mapping action descriptor -> distance.
        """
        # IMPORTANT: probing requires we can call env.step + env.step(undo).
        # The Agent base class doesn't give us env access in choose_action.
        # We have to invert the architecture: for v3 we need a special run_agent.
        # For now, we score WITHOUT probing — pick the action whose
        # PREDICTED-NEXT-STATE looks closest to the predicted terminal,
        # using the cached prediction as a proxy.
        #
        # This is a degraded version of (A) from the Phase-2-Plan; without
        # access to undo, we can't see the true successor. Still useful as
        # a baseline.
        scores: dict = {}
        # Without probing, we score actions by their own intrinsic features
        # vs the predicted terminal. Simple actions get a constant score; the
        # complex action (ACTION6) gets scored by the click-target candidates.
        if self._cached_pred is None or self._cached_pred.n_active == 0:
            # No prediction; everything ties.
            for a in avail:
                scores[a] = 0.0
            return scores
        # If complex action available, score it by the predicted terminal's
        # most-confident object's centroid (the most likely click target).
        # For simple actions we score by current-frame distance to terminal
        # (no per-action discrimination — this is the limitation).
        base_d, _ = distance_to_predicted_terminal(
            self._cached_pred, current_frame, method=self.match_method
        )
        for a in avail:
            if a == 6:
                # Complex: prefer ACTION6 if predictor's most-confident object
                # is far from current state's objects (i.e., we need to place
                # something).
                scores[6] = base_d - 0.5  # bias toward ACTION6 when prediction is rich
            elif a == UNDO_ACTION_ID:
                scores[a] = base_d + 0.5  # prefer not to undo
            else:
                scores[a] = base_d  # baseline
        return scores

    def _random_action(self, avail: list[int]) -> GameAction:
        action_id = self.rng.choice([a for a in avail if a != 0])  # not RESET
        return self._build_action(action_id)

    def _build_action(self, action_id: int) -> GameAction:
        if action_id == 0:
            return GameAction.RESET
        action = GameAction.from_id(action_id)
        if action.is_complex():
            # Click target: use predicted terminal's most-confident object.
            x, y = self._best_click_target()
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {"src": "v3", "click_x": int(x), "click_y": int(y)}
        else:
            action.reasoning = "v3"
        return action

    def _best_click_target(self) -> tuple[int, int]:
        """Pick a (x, y) on a 64x64 grid where the predicted terminal has its
        most-confident object."""
        if self._cached_pred is None or self._cached_pred.n_active == 0:
            return self.rng.randint(0, 63), self.rng.randint(0, 63)
        # Pick the slot with highest exists_prob.
        idx = int(self._cached_pred.exists_prob.argmax())
        cx = self._cached_pred.cx[idx]
        cy = self._cached_pred.cy[idx]
        # Convert normalized [0,1] to grid [0,63].
        x = int(round(cx * 63))
        y = int(round(cy * 63))
        x = max(0, min(63, x))
        y = max(0, min(63, y))
        return x, y
