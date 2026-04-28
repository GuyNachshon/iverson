"""SkillBank: persistent (within a game) record of action subsequences
that achieved level wins.

When a level is completed, we look back over the trajectory and identify
the contiguous slice of actions that produced the level transition. That
slice is stored as a "skill" — a sequence of (action_id, click_coords)
that the agent can replay-attempt at future levels.

Skills are persisted across levels within a game (Competition Mode allows
level resets, not game resets). Across games, the bank is reset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillStep:
    action_id: int
    click_x: int = -1   # only meaningful if action is complex (ACTION6)
    click_y: int = -1


@dataclass
class Skill:
    """A contiguous action subsequence that produced a level win."""
    steps: list[SkillStep]
    level_idx: int                 # which level it solved (0-indexed)
    n_uses: int = 0
    n_successes: int = 0           # times re-applying it worked

    @property
    def confidence(self) -> float:
        if self.n_uses == 0:
            return 1.0  # untested but learned-from-success
        return self.n_successes / self.n_uses


@dataclass
class TransitionLog:
    """One step in a trajectory."""
    action_id: int
    click_x: int = -1
    click_y: int = -1
    levels_completed_after: int = 0
    state_hash: int = 0


class SkillBank:
    """Holds skills, supports promote-from-trajectory and lookup-by-state."""

    def __init__(self) -> None:
        self.skills: list[Skill] = []
        self._trajectory: list[TransitionLog] = []
        self._levels_at_start: int = 0

    def reset(self) -> None:
        """Reset between games."""
        self.skills.clear()
        self._trajectory.clear()
        self._levels_at_start = 0

    def reset_trajectory(self) -> None:
        """Reset the per-level trajectory log without clearing skills."""
        self._trajectory.clear()

    def log_step(self, action_id: int, click_x: int, click_y: int,
                 levels_completed_after: int, state_hash: int = 0) -> None:
        self._trajectory.append(TransitionLog(
            action_id=action_id, click_x=click_x, click_y=click_y,
            levels_completed_after=levels_completed_after, state_hash=state_hash,
        ))

    def promote_on_level_win(self, level_idx: int, lookback: int = 8) -> Optional[Skill]:
        """Look back at the last `lookback` steps and promote them as a skill.

        Returns the new skill if promoted, None if nothing to promote.
        """
        if not self._trajectory:
            return None
        # Find the step where levels_completed actually increased.
        # The win is at trajectory[-1] (the action that triggered it). We
        # promote the last `lookback` actions including the trigger.
        slice_start = max(0, len(self._trajectory) - lookback)
        slice_end = len(self._trajectory)
        steps = [
            SkillStep(action_id=t.action_id, click_x=t.click_x, click_y=t.click_y)
            for t in self._trajectory[slice_start:slice_end]
        ]
        skill = Skill(steps=steps, level_idx=level_idx)
        self.skills.append(skill)
        return skill

    def candidate_actions_at_step(self, position_in_skill: int = 0
                                   ) -> list[tuple[int, int, int, float]]:
        """Return (action_id, click_x, click_y, weight) tuples from skills
        whose `position_in_skill`-th step we'd replay.

        Weight = skill confidence * recency-decay (newer skills weighted more).
        """
        out: list[tuple[int, int, int, float]] = []
        for i, skill in enumerate(self.skills):
            if position_in_skill < len(skill.steps):
                step = skill.steps[position_in_skill]
                # Recency: more recent skills get higher weight.
                recency = 0.5 + 0.5 * ((i + 1) / max(len(self.skills), 1))
                weight = skill.confidence * recency
                out.append((step.action_id, step.click_x, step.click_y, weight))
        return out

    def stats(self) -> dict:
        return {
            "n_skills": len(self.skills),
            "skill_lengths": [len(s.steps) for s in self.skills],
            "skill_levels": [s.level_idx for s in self.skills],
        }
