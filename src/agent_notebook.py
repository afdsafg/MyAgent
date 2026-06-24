"""Evidence Notebook — persistent cross-stage memory for two-tier agent."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NotebookEntry:
    """单个证据条目。"""
    step: int
    timestamp: str = ""
    entry_type: str = ""  # room_explored | object_observed | hypothesis_rejected | seed_visited | frontier_visited
    content: str = ""
    negation: bool = False
    confidence: float = 0.0
    key_frame_id: Optional[str] = None


class EvidenceNotebook:
    """Agent 证据笔记本：记录探索历史、检测循环、注入 Planner prompt。"""

    def __init__(self):
        self.entries: list[NotebookEntry] = []
        # entity_id (lowercase) -> visit count
        self._exhausted_ids: dict[str, int] = defaultdict(int)
        # entity_id (lowercase) -> list of outcome content strings
        self._last_outcomes: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Mutating API
    # ------------------------------------------------------------------

    def add_entry(
        self,
        step: int,
        entry_type: str,
        content: str,
        negation: bool = False,
        confidence: float = 0.0,
        key_frame_id: Optional[str] = None,
    ) -> NotebookEntry:
        """添加一条证据记录并更新循环检测状态。"""
        entry = NotebookEntry(
            step=step,
            entry_type=entry_type,
            content=content,
            negation=negation,
            confidence=confidence,
            key_frame_id=key_frame_id,
        )
        self.entries.append(entry)

        # 追踪 seed / frontier 访问次数以检测循环
        if entry_type == "seed_visited":
            entity_id = self._extract_id(content, "Seed_")
            if entity_id:
                self._last_outcomes[entity_id].append(content)
                self._exhausted_ids[entity_id] += 1
        elif entry_type == "frontier_visited":
            entity_id = self._extract_id(content, "Frontier_")
            if entity_id:
                self._last_outcomes[entity_id].append(content)
                self._exhausted_ids[entity_id] += 1

        return entry

    def update_from_evidence(self, evidence, step: int):
        """将 TrajectoryEvidence 转换为 NotebookEntry 并添加（复用追踪逻辑）。"""
        entry = evidence.to_notebook_entry(step)
        # 复用 add_entry 的追踪逻辑，不要直接 append
        self.add_entry(
            step=entry.step,
            entry_type=entry.entry_type,
            content=entry.content,
            negation=entry.negation,
            confidence=entry.confidence,
            key_frame_id=entry.key_frame_id,
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def is_exhausted(self, entity_id: str) -> bool:
        """检测同一实体是否已访问 3 次以上。"""
        return self._exhausted_ids.get(entity_id.lower(), 0) >= 3

    def get_injection_text(self, max_entries: int = 10) -> str:
        """生成注入 Planner prompt 的历史文本。"""
        recent = self.entries[-max_entries:]
        lines: list[str] = []
        for e in recent:
            marker = "NOT " if e.negation else ""
            line = f"- [Step {e.step}] {marker}{e.content}"
            lines.append(line)
        return "## History\nYou have explored the following:\n" + "\n".join(lines)

    def get_visited_seeds(self) -> set[str]:
        """返回已访问过的 seed ID 集合。"""
        return {
            self._extract_id(e.content, "Seed_")
            for e in self.entries
            if e.entry_type == "seed_visited"
        }

    def get_visited_frontiers(self) -> set[str]:
        """返回已访问过的 frontier ID 集合。"""
        return {
            self._extract_id(e.content, "Frontier_")
            for e in self.entries
            if e.entry_type == "frontier_visited"
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_id(content: str, prefix: str) -> str:
        """从内容中提取实体 ID，例如从 "Seed_3 visited..." 提取 "seed_3"。"""
        match = re.search(rf"{re.escape(prefix)}(\d+)", content, re.IGNORECASE)
        if match:
            return f"{prefix.lower()}{match.group(1)}"
        return ""
