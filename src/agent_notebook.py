"""EvidenceNotebook — persistent cross-stage memory for the two-tier planner.

TODO: This stub is a temporary stand-in until the wt-notebook branch's
agent_notebook.py is merged.  Replace with the canonical implementation when available.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NotebookEntry:
    step: int
    timestamp: str = ""
    entry_type: str = ""
    content: str = ""
    negation: bool = False
    confidence: float = 0.0
    key_frame_id: Optional[str] = None


class EvidenceNotebook:
    """Persistent evidence store for the Planner-Executor loop."""

    def __init__(self):
        self.entries: list[NotebookEntry] = []
        self._exhausted_ids: dict[str, int] = defaultdict(int)
        self._last_outcomes: dict[str, list[str]] = defaultdict(list)

    def add_entry(
        self,
        step: int,
        entry_type: str,
        content: str,
        negation: bool = False,
        confidence: float = 0.0,
        key_frame_id: Optional[str] = None,
    ) -> NotebookEntry:
        entry = NotebookEntry(
            step=step,
            entry_type=entry_type,
            content=content,
            negation=negation,
            confidence=confidence,
            key_frame_id=key_frame_id,
        )
        self.entries.append(entry)

        if entry_type == "seed_visited":
            seed_id = self._extract_id(content, "Seed_")
            self._last_outcomes[seed_id].append(content)
            self._exhausted_ids[seed_id] += 1
        elif entry_type == "frontier_visited":
            fid = self._extract_id(content, "Frontier_")
            self._last_outcomes[fid].append(content)
            self._exhausted_ids[fid] += 1
        return entry

    def is_exhausted(self, entity_id: str) -> bool:
        return self._exhausted_ids.get(entity_id, 0) >= 3

    def get_injection_text(self, max_entries: int = 10) -> str:
        recent = self.entries[-max_entries:]
        lines = []
        for e in recent:
            marker = "NOT" if e.negation else ""
            line = f"- [Step {e.step}] {e.content}"
            lines.append(line)
        return "## History\nYou have explored the following:\n" + "\n".join(lines)

    def get_visited_seeds(self) -> set[str]:
        return {
            self._extract_id(e.content, "Seed_")
            for e in self.entries
            if e.entry_type == "seed_visited"
        }

    def get_visited_frontiers(self) -> set[str]:
        return {
            self._extract_id(e.content, "Frontier_")
            for e in self.entries
            if e.entry_type == "frontier_visited"
        }

    def update_from_evidence(self, evidence, step: int):
        from src.agent_evidence import TrajectoryEvidence

        if isinstance(evidence, TrajectoryEvidence):
            entry = evidence.to_notebook_entry(step)
            self.entries.append(entry)
        else:
            # Fallback: append as raw dict if available
            if hasattr(evidence, "__dict__"):
                self.entries.append(NotebookEntry(step=step, content=str(evidence.__dict__)))

    def _extract_id(self, content: str, prefix: str) -> str:
        match = re.search(f"{prefix}(\\d+)", content)
        return f"{prefix}{match.group(1)}" if match else ""
