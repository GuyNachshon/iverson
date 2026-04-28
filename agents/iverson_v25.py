"""Iverson v2.5 — minimal world-model-driven agent.

Layered selection:
  1. RESET when needed.
  2. Undo-probe phase (UndoReasoner): for each available action key, take it
     and undo to learn reversibility + frame-change effect. Uses ACTION7=undo.
  3. Uncertainty-driven exploration: score each available action by predicted
     world-model loss (entropy of the prior over the next stochastic latent).
     Pick the highest-uncertainty legal action. This is the simplest non-trivial
     thing we can do with the WM and gives us a measurable signal.
  4. Random fallback among legal actions.

Trains the world model online every TRAIN_EVERY actions on a sliding window
of recent transitions.

Deliberately does NOT implement: symbolic memory, goal inference, CEM/MCTS,
slot attention, terminal-state prediction. Those are v3. v2.5 is the baseline
that tells us whether more sophisticated machinery is needed.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
from arcengine import FrameData, GameAction, GameState

from models.undo_reasoning import UndoReasoner
from models.world_model import OnlineWorldModel

from .base import Agent

logger = logging.getLogger(__name__)


def _frame_to_grid(frame: FrameData) -> np.ndarray:
    """Convert FrameData.frame (list of nested lists) into a (64,64) int array.

    Pads if smaller than 64x64; truncates if larger (shouldn't happen).
    """
    if not frame.frame:
        return np.zeros((64, 64), dtype=np.int64)
    arr = np.asarray(frame.frame[0], dtype=np.int64)
    H, W = arr.shape
    if H == 64 and W == 64:
        return arr
    out = np.zeros((64, 64), dtype=np.int64)
    out[:min(H, 64), :min(W, 64)] = arr[:64, :64]
    return out


def _entropy_of_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    return float(ent.mean().item())


class IversonV25(Agent):
    """v2.5 baseline: WM + undo-probes + uncertainty-driven action selection."""

    MAX_ACTIONS = 200
    PROBE_BUDGET = 12       # undo probes per game (one-shot at start; reused across levels)
    TRAIN_EVERY = 4         # train WM every N actions
    TRAIN_BATCH = 16        # transitions per training step
    LR = 3e-4

    def __init__(self, game_id: str, baseline_actions: Optional[list[int]] = None,
                 device: str = "cpu", seed: int = 0) -> None:
        super().__init__(game_id, baseline_actions)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.wm = OnlineWorldModel(num_key_actions=8, grid_size=64).to(self.device)
        self.opt = torch.optim.Adam(self.wm.parameters(), lr=self.LR)

        self.undo = UndoReasoner(num_key_actions=8, grid_size=64,
                                 probe_budget=self.PROBE_BUDGET)

        # transition buffer: dicts of {grid, action_key, action_pos, next_grid, reward, done}
        self.buffer: deque = deque(maxlen=512)
        # for recording (grid_before, action) so we can buffer the next observation
        self._pending_action: Optional[GameAction] = None
        self._pending_grid_before: Optional[np.ndarray] = None
        self._train_step_counter = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    # ---- Action selection ---------------------------------------------------

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Buffer the previous transition if we have one pending
        if self._pending_action is not None and self._pending_grid_before is not None:
            grid_after = _frame_to_grid(latest_frame)
            self._record_transition(self._pending_grid_before,
                                    self._pending_action, grid_after,
                                    done=latest_frame.state is GameState.WIN)
            # Online training pulse
            if self.action_counter > 0 and self.action_counter % self.TRAIN_EVERY == 0:
                self._train_step()

        # Reset if needed
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._pending_action = None
            self._pending_grid_before = None
            return GameAction.RESET

        avail = list(latest_frame.available_actions or [])
        if not avail:
            return GameAction.ACTION1

        grid_now = _frame_to_grid(latest_frame)

        # If we just probed and are awaiting the post-action observation,
        # the UndoReasoner will tell us to issue UNDO=ACTION7.
        if self.undo.awaiting_undo_result:
            self.undo.observe_action_result(grid_now)
            if self.undo.UNDO_KEY in avail:
                action = GameAction.from_id(self.undo.UNDO_KEY)
                action.reasoning = f"v25-probe-undo (key {self.undo.UNDO_KEY})"
                self._set_pending(action, grid_now)
                return action
            # Undo not available -- give up the probe, mark this key as
            # irreversible so we don't re-probe it.
            pending_key = self.undo.pending_action[0] if self.undo.pending_action else -1
            if pending_key >= 0:
                self.undo.undo_works[pending_key] = False
                self.undo.irreversible_keys.add(pending_key)
            self.undo.pending_action = None
            self.undo.awaiting_undo_result = False

        # If the previous frame was an undo result, finalize the probe
        if self.undo.pending_action is not None and not self.undo.awaiting_undo_result:
            self.undo.observe_undo_result(grid_now)

        # Probe phase: probe each unprobed available action
        if self.undo.mode == "probe":
            unprobed = [k for k in avail
                        if k not in self.undo.undo_works
                        and k != self.undo.UNDO_KEY
                        and k != GameAction.RESET.value]
            if unprobed and self.undo.probes_used < self.PROBE_BUDGET:
                key = unprobed[0]
                action = self._build_action(key)
                self.undo.begin_probe(grid_now, key,
                                      action.action_data.x * 64 + action.action_data.y
                                      if action.is_complex() else 0)
                self._set_pending(action, grid_now)
                return action

        # Past probe phase: uncertainty-driven action selection over available actions
        action = self._uncertainty_select(grid_now, avail)
        self._set_pending(action, grid_now)
        return action

    def on_level_complete(self, level_idx: int) -> None:
        self.undo.reset_for_new_level()

    # ---- Helpers ------------------------------------------------------------

    def _build_action(self, key_id: int) -> GameAction:
        action = GameAction.from_id(key_id)
        if action.is_complex():
            x = self.rng.randint(0, 63)
            y = self.rng.randint(0, 63)
            action.set_data({"x": x, "y": y})
            action.reasoning = {"src": "v25", "action": action.value}
        else:
            action.reasoning = "v25"
        return action

    def _set_pending(self, action: GameAction, grid_before: np.ndarray) -> None:
        self._pending_action = action
        self._pending_grid_before = grid_before

    def _record_transition(self, grid_before: np.ndarray, action: GameAction,
                           grid_after: np.ndarray, done: bool) -> None:
        action_key = action.value
        if action.is_complex():
            x = action.action_data.x
            y = action.action_data.y
            action_pos = int(x) * 64 + int(y)
        else:
            action_pos = 0
        self.buffer.append({
            "grid": torch.from_numpy(grid_before).long(),
            "next_grid": torch.from_numpy(grid_after).long(),
            "action_key": action_key,
            "action_pos": action_pos,
            "reward": 0.0,
            "done": done,
        })

    def _train_step(self) -> None:
        if len(self.buffer) < self.TRAIN_BATCH:
            return
        batch = list(self.rng.sample(list(self.buffer), self.TRAIN_BATCH))
        self.wm.train()
        losses = self.wm.compute_loss(batch)
        self.opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt.step()
        self._train_step_counter += 1
        if self._train_step_counter % 25 == 0:
            logger.debug(
                f"[{self.game_id}] wm step {self._train_step_counter} "
                f"total={losses['total'].item():.3f} recon={losses['recon'].item():.3f}"
            )

    def _uncertainty_select(self, grid: np.ndarray, avail: list[int]) -> GameAction:
        """Pick the available action whose imagined-prior has highest entropy.

        Uses the world model in imagine mode for each candidate action. High
        prior entropy means "world model is uncertain about what this action does".
        Per the design, that's where to spend the next action.
        """
        legal = [k for k in avail if k != GameAction.RESET.value]
        if not legal:
            return GameAction.RESET

        # Until the WM has trained a bit, fall back to uniform random.
        if self._train_step_counter < 5:
            return self._build_action(self.rng.choice(legal))

        self.wm.train(False)  # switch to inference mode without using .eval() name
        device = self.device
        grid_t = torch.from_numpy(grid).long().unsqueeze(0).to(device)
        obs_latent = self.wm.encoder(grid_t)
        h, z = self.wm.dynamics.init_state(1, device)

        best_key = legal[0]
        best_entropy = -1.0
        with torch.no_grad():
            for key in legal:
                action = GameAction.from_id(key)
                if action.is_complex():
                    entropies = []
                    for _ in range(4):
                        x = self.rng.randint(0, 63)
                        y = self.rng.randint(0, 63)
                        pos = torch.tensor([[x / 64.0, y / 64.0]], device=device)
                        key_t = torch.tensor([key], device=device)
                        action_emb = self.wm.dynamics.embed_action(key_t, pos)
                        _, _, prior_logits = self.wm.dynamics.imagine(action_emb, h, z)
                        entropies.append(_entropy_of_logits(prior_logits))
                    ent = float(np.mean(entropies))
                else:
                    pos = torch.zeros(1, 2, device=device)
                    key_t = torch.tensor([key], device=device)
                    action_emb = self.wm.dynamics.embed_action(key_t, pos)
                    _, _, prior_logits = self.wm.dynamics.imagine(action_emb, h, z)
                    ent = _entropy_of_logits(prior_logits)
                if ent > best_entropy:
                    best_entropy = ent
                    best_key = key

        return self._build_action(best_key)
