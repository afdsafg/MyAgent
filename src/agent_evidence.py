"""TrajectoryEvidence data class — compresses executor outputs into compact records.

TODO: This stub is a temporary stand-in until the wt-notebook branch's
agent_evidence.py is merged.  Replace with the canonical implementation when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrajectoryEvidence:
    """Compressed record of one executor tool invocation."""
    subgoal: str
    task_mode: str
    progress: str
    salient: List[str]
    outcome: str
    gd_quality: str = "ok"
    key_frames: List[str] = field(default_factory=list)
    room_id: int = -1
    objects_nearby: List[str] = field(default_factory=list)

    def to_notebook_entry(self, step: int) -> "NotebookEntry":
        from src.agent_notebook import NotebookEntry

        if self.outcome == "detection_failed":
            entry_type = "hypothesis_rejected"
            content = (
                f"GD detection failed for '{self.subgoal}': {self.gd_quality}. "
                f"Objects nearby: {self.objects_nearby}."
            )
            negation = True
        elif self.outcome == "object_found":
            entry_type = "object_observed"
            content = (
                f"Object observed: {self.subgoal}. "
                f"Salient: {', '.join(self.salient)}. "
                f"Room {self.room_id}."
            )
            negation = False
        elif self.task_mode == "explore_seed":
            entry_type = "seed_visited"
            content = (
                f"Seed visited: {self.subgoal}. "
                f"Arrived at room {self.room_id}. "
                f"Objects: {self.objects_nearby}."
            )
            negation = "NOT" in self.progress
        elif self.task_mode == "explore_frontier":
            entry_type = "frontier_visited"
            content = (
                f"Frontier visited: {self.subgoal}. "
                f"Arrived at room {self.room_id}. "
                f"Outcome: {self.outcome}."
            )
            negation = "NOT" in self.progress
        else:
            entry_type = "room_explored"
            content = (
                f"Room {self.room_id} explored. "
                f"Objects: {self.objects_nearby}. "
                f"Progress: {self.progress}."
            )
            negation = "NOT" in self.progress

        return NotebookEntry(
            step=step,
            entry_type=entry_type,
            content=content,
            negation=negation,
            confidence=0.7,
            key_frame_id=self.key_frames[0] if self.key_frames else None,
        )
