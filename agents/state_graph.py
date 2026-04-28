"""StateGraph: per-game memory of (state, action, next_state) edges.

Used for novelty signaling and revisit detection. Resets per game
(between games we know nothing); persists across levels within a game
(Competition Mode allows level resets).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models.object_list import Frame


@dataclass
class ActionStats:
    times_tried: int = 0
    delta_sum: float = 0.0          # cumulative |frame change|
    leads_to_win: int = 0           # times this action immediately preceded a level WIN
    leads_to_game_over: int = 0     # times it triggered GAME_OVER
    last_used_step: int = -1

    def avg_delta(self) -> float:
        return self.delta_sum / max(self.times_tried, 1)


@dataclass
class StateNode:
    state_hash: int
    visit_count: int = 0
    actions_tried: dict[int, ActionStats] = field(default_factory=dict)

    def stats_for(self, action_id: int) -> ActionStats:
        if action_id not in self.actions_tried:
            self.actions_tried[action_id] = ActionStats()
        return self.actions_tried[action_id]


def hash_frame(frame: Frame, max_objects: int = 32, position_bits: int = 6) -> int:
    """Collapse a frame to a small canonical signature.

    `max_objects`: only the first N objects (by size, descending sort already in Frame).
    `position_bits`: centroids quantized to 2^position_bits buckets per axis.
                     position_bits=6 → 64 buckets per axis (matches grid resolution).

    The hash is `hash(tuple_of_signature_tuples)`. Two frames map to the same
    state iff their first-N objects have matching colors + bucketed positions.
    """
    sig: list[tuple] = []
    bucket = 1 << position_bits
    for obj in frame.objects[:max_objects]:
        cx_b = int(obj.centroid_norm[0] * bucket)
        cy_b = int(obj.centroid_norm[1] * bucket)
        sig.append((obj.color_id, cx_b, cy_b, obj.size, obj.is_singleton, obj.touches_edge))
    return hash(tuple(sig))


def frame_diff_magnitude(before: Frame, after: Frame) -> float:
    """A coarse 'how much did the frame change' scalar.

    We compare the multiset of (color_id, bucketed centroid) across frames:
    each removed or added object counts 1.0; each shifted-but-same-color
    object counts as the L2 of its centroid drift.
    """
    if not before.objects and not after.objects:
        return 0.0
    sig_before = {(o.color_id, round(o.centroid_norm[0], 2), round(o.centroid_norm[1], 2)): o
                  for o in before.objects}
    sig_after = {(o.color_id, round(o.centroid_norm[0], 2), round(o.centroid_norm[1], 2)): o
                 for o in after.objects}
    keys_before = set(sig_before)
    keys_after = set(sig_after)
    added = len(keys_after - keys_before)
    removed = len(keys_before - keys_after)
    # Cheap: don't bother matching shifted objects across positions; the
    # add+remove counting is approximately right.
    return float(added + removed)


class StateGraph:
    """Per-game graph of states and actions tried at each."""

    def __init__(self) -> None:
        self.nodes: dict[int, StateNode] = {}
        self.edges: dict[tuple[int, int], int] = {}  # (state_hash, action_id) → next_state_hash
        self._step_counter: int = 0

    def get_or_create(self, frame: Frame) -> StateNode:
        h = hash_frame(frame)
        node = self.nodes.get(h)
        if node is None:
            node = StateNode(state_hash=h)
            self.nodes[h] = node
        node.visit_count += 1
        return node

    def record_transition(self, before: Frame, action_id: int, after: Frame,
                           reward: float = 0.0, level_won: bool = False,
                           game_over: bool = False) -> None:
        before_node = self.get_or_create(before)
        after_hash = hash_frame(after)
        stats = before_node.stats_for(action_id)
        stats.times_tried += 1
        stats.delta_sum += frame_diff_magnitude(before, after)
        stats.last_used_step = self._step_counter
        if level_won:
            stats.leads_to_win += 1
        if game_over:
            stats.leads_to_game_over += 1
        self.edges[(before_node.state_hash, action_id)] = after_hash
        self._step_counter += 1

    def novelty(self, frame: Frame, action_id: int) -> float:
        """0..1: how unfamiliar is taking action_id from this state.

        1.0 = never tried at this state.
        0.0 = tried K times with consistent outcome.
        Scaled in between.
        """
        h = hash_frame(frame)
        node = self.nodes.get(h)
        if node is None:
            return 1.0
        stats = node.actions_tried.get(action_id)
        if stats is None or stats.times_tried == 0:
            return 1.0
        # Decay novelty with sqrt(times_tried).
        return 1.0 / (1.0 + (stats.times_tried) ** 0.5)

    def reset(self) -> None:
        """Reset between games. Within a game (across levels), we keep state."""
        self.nodes.clear()
        self.edges.clear()
        self._step_counter = 0

    def __len__(self) -> int:
        return len(self.nodes)
