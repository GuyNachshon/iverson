"""Symbolic rules. Each rule is a (precondition, action, expected_effect)
predicate. Rules are induced from observed transitions and queried at
decision time.

Three simple effect families used for v3.5:
  - GLOBAL_DELTA(min_cells): action causes ≥N cells to change globally.
  - COLOR_REMOVED(color_id): action removes objects of the given color.
  - LEVEL_PROGRESS: action immediately preceded levels_completed increment.

Each rule has confirm/refute counts updated as we observe matching
transitions. Rules are intentionally generic (one-template-fits-many) to
keep the rule store small and queries fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from models.object_list import Frame


# ---------------------------------------------------------------------------
# Effect representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Effect:
    kind: str                       # "global_delta" | "color_removed" | "level_progress" | "no_op"
    color_id: int = -1              # for color_removed
    min_cells: int = 0              # for global_delta

    @classmethod
    def global_delta(cls, min_cells: int) -> "Effect":
        return cls(kind="global_delta", min_cells=min_cells)

    @classmethod
    def color_removed(cls, color_id: int) -> "Effect":
        return cls(kind="color_removed", color_id=color_id)

    @classmethod
    def level_progress(cls) -> "Effect":
        return cls(kind="level_progress")

    @classmethod
    def no_op(cls) -> "Effect":
        return cls(kind="no_op")


@dataclass(frozen=True)
class Precondition:
    """Currently a stub — v3.5 rules treat preconditions as 'always true'.

    v3.7 may add precondition templates (player_adjacent_to_color,
    n_objects_in_frame > N, etc.).
    """
    kind: str = "any"

    def evaluate(self, frame: Frame) -> bool:
        return True


@dataclass
class Rule:
    action_key: int
    effect: Effect
    precondition: Precondition = field(default_factory=Precondition)
    confirm_count: int = 0
    refute_count: int = 0
    last_observed_step: int = -1

    @property
    def confidence(self) -> float:
        n = self.confirm_count + self.refute_count
        if n == 0:
            return 0.0
        return self.confirm_count / n

    @property
    def support(self) -> int:
        """Total observations supporting this rule's denominator."""
        return self.confirm_count + self.refute_count

    def matches(self, action_key: int, frame: Frame) -> bool:
        return self.action_key == action_key and self.precondition.evaluate(frame)

    def __repr__(self) -> str:
        return (f"Rule(action={self.action_key}, effect={self.effect.kind}"
                + (f"[{self.effect.color_id}]" if self.effect.kind == "color_removed" else "")
                + f", conf={self.confidence:.2f}, support={self.support})")


# ---------------------------------------------------------------------------
# Effect verification on observed transitions
# ---------------------------------------------------------------------------

def _frame_color_counts(frame: Frame) -> dict[int, int]:
    out: dict[int, int] = {}
    for obj in frame.objects:
        out[obj.color_id] = out.get(obj.color_id, 0) + obj.size
    return out


def _diff_n_cells(before: Frame, after: Frame) -> int:
    """Coarse cell-level diff; counts color cell-count changes."""
    cb = _frame_color_counts(before)
    ca = _frame_color_counts(after)
    keys = set(cb) | set(ca)
    return sum(abs(cb.get(k, 0) - ca.get(k, 0)) for k in keys)


def _color_was_removed(before: Frame, after: Frame, color_id: int) -> bool:
    cb = _frame_color_counts(before).get(color_id, 0)
    ca = _frame_color_counts(after).get(color_id, 0)
    return ca < cb


def effect_observed(effect: Effect, before: Frame, after: Frame,
                     levels_increased: bool) -> bool:
    """Did `before -> after` (with a flag for level-progress) match `effect`?"""
    if effect.kind == "no_op":
        return _diff_n_cells(before, after) == 0
    if effect.kind == "global_delta":
        return _diff_n_cells(before, after) >= effect.min_cells
    if effect.kind == "color_removed":
        return _color_was_removed(before, after, effect.color_id)
    if effect.kind == "level_progress":
        return levels_increased
    return False


# ---------------------------------------------------------------------------
# RuleStore: induce rules from transitions, score actions
# ---------------------------------------------------------------------------

class RuleStore:
    rules: list[Rule]

    def __init__(self) -> None:
        self.rules = []

    def reset(self) -> None:
        self.rules.clear()

    def __len__(self) -> int:
        return len(self.rules)

    def _find(self, action_key: int, effect: Effect) -> Optional[Rule]:
        for r in self.rules:
            if (r.action_key == action_key
                and r.effect.kind == effect.kind
                and r.effect.color_id == effect.color_id
                and r.effect.min_cells == effect.min_cells):
                return r
        return None

    def add_or_update(self, action_key: int, effect: Effect, observed: bool,
                       step: int = -1) -> Rule:
        r = self._find(action_key, effect)
        if r is None:
            r = Rule(action_key=action_key, effect=effect)
            self.rules.append(r)
        if observed:
            r.confirm_count += 1
        else:
            r.refute_count += 1
        r.last_observed_step = max(r.last_observed_step, step)
        return r

    def induce(self, action_key: int, before: Frame, after: Frame,
                levels_increased: bool, step: int = -1) -> list[Rule]:
        """Generate candidate rules from this transition; record observations.

        We propose rules from the same set of templates each time:
          - global_delta(min_cells = 1, 4, 16) — coarse buckets of "did something."
          - color_removed for each color present in `before` but reduced in `after`.
          - level_progress (only if levels_increased was true).
          - no_op (if no diff).

        Each candidate is then verified — observed=True if effect matches the
        actual transition, False otherwise. This builds confidence over time.
        """
        rules_touched: list[Rule] = []
        delta = _diff_n_cells(before, after)

        # global_delta family
        for min_cells in (1, 4, 16):
            ok = delta >= min_cells
            r = self.add_or_update(action_key, Effect.global_delta(min_cells), ok, step)
            rules_touched.append(r)

        # color_removed: only consider colors that actually decreased.
        cb = _frame_color_counts(before)
        ca = _frame_color_counts(after)
        for color_id in cb:
            ok = ca.get(color_id, 0) < cb[color_id]
            if ok:
                r = self.add_or_update(action_key, Effect.color_removed(color_id), True, step)
                rules_touched.append(r)
            # NOTE: we don't propose a color_removed rule and refute it for every
            # color in every transition — that would explode the rule store.

        # level_progress
        r = self.add_or_update(action_key, Effect.level_progress(), levels_increased, step)
        rules_touched.append(r)

        # no_op
        r = self.add_or_update(action_key, Effect.no_op(), delta == 0, step)
        rules_touched.append(r)

        return rules_touched

    # ---- Action scoring ----

    def suggest(self, frame: Frame, action_key: int) -> float:
        """Score an action by the rules that fire at this state.

        Higher = the rule store thinks this action is useful. Combines:
          - level_progress confidence (huge, if rules say this action wins levels)
          - global_delta(>=4) confidence (medium, "this action does meaningful stuff")
          - global_delta(>=16) confidence (large delta = often interaction with goal)
          - subtract no_op confidence (penalize "this action does nothing")
        """
        score = 0.0
        for r in self.rules:
            if r.action_key != action_key:
                continue
            if r.support < 2:
                continue  # don't trust 1-shot rules
            c = r.confidence
            if r.effect.kind == "level_progress":
                score += 5.0 * c
            elif r.effect.kind == "global_delta":
                if r.effect.min_cells == 1:
                    score += 0.5 * c
                elif r.effect.min_cells == 4:
                    score += 1.0 * c
                elif r.effect.min_cells == 16:
                    score += 2.0 * c
            elif r.effect.kind == "color_removed":
                # Only relevant if the color is present in the current frame.
                if any(o.color_id == r.effect.color_id for o in frame.objects):
                    score += 1.5 * c
            elif r.effect.kind == "no_op":
                score -= 2.0 * c
        return score

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for r in self.rules:
            kinds[r.effect.kind] = kinds.get(r.effect.kind, 0) + 1
        return {
            "n_rules": len(self.rules),
            "by_kind": kinds,
            "high_confidence_rules": sum(1 for r in self.rules
                                          if r.confidence > 0.8 and r.support >= 3),
        }
