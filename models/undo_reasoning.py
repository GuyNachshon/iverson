"""
Undo-as-Reasoning (Action-Time-Training) for ARC-AGI-3.

Uses Undo as a systematic hypothesis testing primitive:
  1. PROBE mode: test each action type with undo to learn reversibility + effects
  2. VERIFY mode: undo when world model prediction was wrong, re-learn, re-plan
  3. EXPLOIT mode: trust the model, never undo (budget preservation)

Generates testable hypotheses: "Key 0 moves object right" (confirmed 5/5).
Detects irreversible actions to prevent catastrophic mistakes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class HypothesisStatus(Enum):
    UNTESTED = "untested"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    PARTIALLY_CONFIRMED = "partially"


@dataclass
class Hypothesis:
    description: str
    action_key: int
    context: Dict
    predicted_effect: str
    status: HypothesisStatus = HypothesisStatus.UNTESTED
    test_count: int = 0
    confirm_count: int = 0
    refute_count: int = 0
    confidence: float = 0.0

    def update(self, confirmed: bool):
        self.test_count += 1
        if confirmed: self.confirm_count += 1
        else: self.refute_count += 1
        self.confidence = self.confirm_count / self.test_count
        if self.confidence > 0.8 and self.test_count >= 3:
            self.status = HypothesisStatus.CONFIRMED
        elif self.confidence < 0.2 and self.test_count >= 3:
            self.status = HypothesisStatus.REFUTED
        else:
            self.status = HypothesisStatus.PARTIALLY_CONFIRMED


class UndoReasoner:
    # ARC-AGI-3 ACTION7 is the undo. (ACTION6 is the complex (x,y) action.)
    UNDO_KEY = 7

    def __init__(self, num_key_actions=7, grid_size=64, probe_budget=12, verify_threshold=0.8):
        self.num_key_actions = num_key_actions
        self.grid_size = grid_size
        self.probe_budget = probe_budget
        self.verify_threshold = verify_threshold
        self.mode = "probe"
        self.probes_used = 0
        self.total_undos = 0
        self.hypotheses: List[Hypothesis] = []
        self.undo_works: Dict[int, bool] = {}
        self.irreversible_keys: set = set()
        self.pre_action_state = None
        self.post_action_state = None
        self.pending_action = None
        self.awaiting_undo_result = False

    def should_probe(self, action_key, grid):
        if self.mode == "exploit": return False
        if self.probes_used >= self.probe_budget:
            self.mode = "verify"
            return False
        return self.mode == "probe" and action_key not in self.undo_works

    def begin_probe(self, grid_before, action_key, action_pos):
        self.pre_action_state = grid_before.copy()
        self.pending_action = (action_key, action_pos)
        self.awaiting_undo_result = False

    def observe_action_result(self, grid_after):
        if self.pre_action_state is None or self.pending_action is None: return None
        self.post_action_state = grid_after.copy()
        action_key = self.pending_action[0]
        if self.mode == "probe" and action_key not in self.undo_works:
            self.awaiting_undo_result = True
            self.probes_used += 1
            return (self.UNDO_KEY, 0)
        return None

    def observe_undo_result(self, grid_after_undo):
        if not self.awaiting_undo_result: return {}
        self.awaiting_undo_result = False
        self.total_undos += 1
        action_key = self.pending_action[0]
        restored = np.array_equal(self.pre_action_state, grid_after_undo)
        self.undo_works[action_key] = restored
        if not restored: self.irreversible_keys.add(action_key)
        result = {"restored": restored, "action_key": action_key}
        self.pre_action_state = None
        self.post_action_state = None
        self.pending_action = None
        if self.probes_used >= self.probe_budget: self.mode = "verify"
        return result

    def reset_for_new_level(self):
        self.pre_action_state = None
        self.post_action_state = None
        self.pending_action = None
        self.awaiting_undo_result = False
        if self.mode == "probe" and self.undo_works: self.mode = "verify"

    def reset_for_new_environment(self):
        self.mode = "probe"
        self.probes_used = 0
        self.total_undos = 0
        self.hypotheses.clear()
        self.undo_works.clear()
        self.irreversible_keys.clear()
        self.pre_action_state = None
        self.post_action_state = None
        self.pending_action = None
        self.awaiting_undo_result = False
