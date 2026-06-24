"""TrajectoryEvidence — compresses executor outputs into compact notebook records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrajectoryEvidence:
    """单条轨迹证据，由 Executor 产生，可转换为 NotebookEntry。"""

    subgoal: str
    task_mode: str
    progress: str
    salient: list[str] = field(default_factory=list)
    outcome: str = ""  # arrived_near_target | target_not_reached | detection_failed | object_found | panorama_complete
    gd_quality: str = "ok"  # ok | bbox_too_large | score_too_low | no_detection
    key_frames: list[str] = field(default_factory=list)
    room_id: int = -1
    objects_nearby: list[str] = field(default_factory=list)

    def to_notebook_entry(self, step: int):
        """根据 outcome 和 task_mode 将证据转换为合适的 NotebookEntry。"""
        from src.agent_notebook import NotebookEntry

        negation = "not" in self.progress.lower()
        confidence = 0.7 if self.gd_quality == "ok" else 0.4

        if self.outcome == "detection_failed":
            entry_type = "hypothesis_rejected"
            content = (
                f"GD detection failed for '{self.subgoal}': {self.gd_quality}. "
                f"Objects nearby: {', '.join(self.objects_nearby)}."
            )
            negation = True
        elif self.outcome == "object_found" or (
            self.task_mode == "navigate_to_object" and self.outcome == "target_not_reached"
        ):
            entry_type = "object_observed"
            content = (
                f"Object observed: {self.subgoal}. "
                f"Salient: {', '.join(self.salient)}. "
                f"Room {self.room_id}."
            )
            negation = self.gd_quality == "score_too_low"
        elif self.task_mode == "explore_seed":
            entry_type = "seed_visited"
            content = (
                f"Seed visited: {self.subgoal}. "
                f"Arrived at Room {self.room_id}. "
                f"Objects: {', '.join(self.objects_nearby)}."
            )
            negation = "not" in self.progress.lower()
        elif self.task_mode == "explore_frontier":
            entry_type = "frontier_visited"
            content = (
                f"Frontier visited: {self.subgoal}. "
                f"Arrived at Room {self.room_id}. "
                f"Outcome: {self.outcome}."
            )
            negation = "not" in self.progress.lower()
        else:
            entry_type = "room_explored"
            content = (
                f"Room {self.room_id} explored. "
                f"Objects: {', '.join(self.objects_nearby)}. "
                f"Progress: {self.progress}."
            )
            negation = "not" in self.progress.lower()

        return NotebookEntry(
            step=step,
            entry_type=entry_type,
            content=content,
            negation=negation,
            confidence=confidence,
            key_frame_id=self.key_frames[0] if self.key_frames else None,
        )
