"""Tests for EvidenceNotebook and TrajectoryEvidence (Task 1)."""

from __future__ import annotations

import pytest

from src.agent_notebook import EvidenceNotebook, NotebookEntry
from src.agent_evidence import TrajectoryEvidence


# ── NotebookEntry ──────────────────────────────────────────────────────

def test_notebook_entry_defaults():
    entry = NotebookEntry(step=1)
    assert entry.step == 1
    assert entry.timestamp == ""
    assert entry.entry_type == ""
    assert entry.content == ""
    assert entry.negation is False
    assert entry.confidence == 0.0
    assert entry.key_frame_id is None


def test_notebook_entry_with_values():
    entry = NotebookEntry(
        step=15,
        entry_type="room_explored",
        content="Bedroom explored, oven NOT found.",
        negation=True,
        confidence=0.9,
        key_frame_id="snap_step15_view0",
    )
    assert entry.step == 15
    assert entry.entry_type == "room_explored"
    assert entry.negation is True
    assert entry.confidence == 0.9
    assert entry.key_frame_id == "snap_step15_view0"


# ── EvidenceNotebook ───────────────────────────────────────────────────

class TestEvidenceNotebook:
    def setup_method(self):
        self.nb = EvidenceNotebook()

    def test_add_room_explored_entry(self):
        entry = self.nb.add_entry(
            step=15,
            entry_type="room_explored",
            content="Bedroom (room_5) explored. Objects: [bed, chair, door]. Target oven NOT found.",
            negation=True,
            confidence=0.9,
            key_frame_id="snap_step15_view0",
        )
        assert entry.step == 15
        assert entry.negation is True
        assert len(self.nb.entries) == 1

    def test_add_multiple_entry_types(self):
        self.nb.add_entry(step=1, entry_type="room_explored", content="Hallway explored.")
        self.nb.add_entry(step=5, entry_type="object_observed", content="Table found.")
        self.nb.add_entry(step=10, entry_type="hypothesis_rejected", content="Kitchen not here.")
        self.nb.add_entry(step=15, entry_type="seed_visited", content="Seed_3 visited.")
        self.nb.add_entry(step=20, entry_type="frontier_visited", content="Frontier_0 explored.")
        assert len(self.nb.entries) == 5
        assert all(isinstance(e, NotebookEntry) for e in self.nb.entries)

    def test_loop_detection_seed(self):
        """3 visits to the same seed → is_exhausted returns True."""
        for i in range(3):
            self.nb.add_entry(
                step=8 + i,
                entry_type="seed_visited",
                content=f"Seed_3 visited → dining area (not kitchen)",
                negation=False,
                confidence=0.8,
            )
        assert self.nb.is_exhausted("seed_3") is True

    def test_loop_detection_seed_not_exhausted_after_2(self):
        """2 visits to the same seed → is_exhausted returns False."""
        for i in range(2):
            self.nb.add_entry(
                step=8 + i,
                entry_type="seed_visited",
                content="Seed_3 visited → dining area",
                negation=False,
                confidence=0.8,
            )
        assert self.nb.is_exhausted("seed_3") is False

    def test_loop_detection_frontier(self):
        for i in range(3):
            self.nb.add_entry(
                step=8 + i,
                entry_type="frontier_visited",
                content="Frontier_1 visited → dead end",
                negation=False,
                confidence=0.7,
            )
        assert self.nb.is_exhausted("frontier_1") is True

    def test_loop_detection_different_entities(self):
        """Visiting seed_3 twice and seed_5 once should not exhaust either."""
        self.nb.add_entry(step=1, entry_type="seed_visited", content="Seed_3 visited.")
        self.nb.add_entry(step=2, entry_type="seed_visited", content="Seed_3 visited.")
        self.nb.add_entry(step=3, entry_type="seed_visited", content="Seed_5 visited.")
        assert self.nb.is_exhausted("seed_3") is False
        assert self.nb.is_exhausted("seed_5") is False

    def test_is_exhausted_case_insensitive(self):
        for i in range(3):
            self.nb.add_entry(
                step=8 + i,
                entry_type="seed_visited",
                content="SEED_3 visited",
                negation=False,
                confidence=0.8,
            )
        # Both uppercase and lowercase should work
        assert self.nb.is_exhausted("SEED_3") is True
        assert self.nb.is_exhausted("seed_3") is True
        assert self.nb.is_exhausted("Seed_3") is True

    def test_get_visited_seeds(self):
        self.nb.add_entry(step=1, entry_type="seed_visited", content="Seed_1 visited.")
        self.nb.add_entry(step=2, entry_type="seed_visited", content="Seed_3 visited.")
        self.nb.add_entry(step=3, entry_type="frontier_visited", content="Frontier_0 visited.")
        visited = self.nb.get_visited_seeds()
        assert "seed_1" in visited
        assert "seed_3" in visited
        assert "frontier_0" not in visited

    def test_get_visited_frontiers(self):
        self.nb.add_entry(step=1, entry_type="frontier_visited", content="Frontier_0 explored.")
        self.nb.add_entry(step=2, entry_type="frontier_visited", content="Frontier_2 explored.")
        self.nb.add_entry(step=3, entry_type="seed_visited", content="Seed_1 visited.")
        visited = self.nb.get_visited_frontiers()
        assert "frontier_0" in visited
        assert "frontier_2" in visited
        assert "seed_1" not in visited

    def test_injection_text_format(self):
        self.nb.add_entry(
            step=5,
            entry_type="room_explored",
            content="Bedroom: bed, chair. Oven NOT found.",
            negation=True,
            confidence=0.9,
        )
        self.nb.add_entry(
            step=12,
            entry_type="seed_visited",
            content="Seed_3 -> dining area, not kitchen.",
            negation=False,
            confidence=0.7,
        )
        text = self.nb.get_injection_text(max_entries=5)
        assert "## History" in text
        assert "You have explored the following" in text
        assert "Bedroom: bed, chair. Oven NOT found." in text
        assert "Seed_3 -> dining area, not kitchen." in text
        assert "Step 5" in text
        assert "Step 12" in text

    def test_injection_text_max_entries(self):
        for i in range(15):
            self.nb.add_entry(
                step=i,
                entry_type="room_explored",
                content=f"Room {i} explored.",
            )
        text = self.nb.get_injection_text(max_entries=5)
        # Should only include the last 5 entries
        assert "Room 14" in text
        assert "Room 10" in text
        assert "Room 0" not in text

    def test_update_from_evidence(self):
        """update_from_evidence should convert TrajectoryEvidence and add entry."""
        ev = TrajectoryEvidence(
            subgoal="Navigate to oven via view_2",
            task_mode="navigate_to_object",
            progress="Moved 3 steps toward kitchen",
            salient=["cabinet", "oven-like(score=0.44)"],
            outcome="target_not_reached",
            gd_quality="score_too_low",
            key_frames=["snap_step12_view2"],
            room_id=5,
            objects_nearby=["bed", "chair"],
        )
        self.nb.update_from_evidence(ev, step=12)
        assert len(self.nb.entries) == 1
        entry = self.nb.entries[0]
        assert entry.step == 12
        assert entry.entry_type == "object_observed"
        assert "oven-like" in entry.content
        assert "Room 5" in entry.content


# ── TrajectoryEvidence ─────────────────────────────────────────────────

class TestTrajectoryEvidence:
    def test_defaults(self):
        ev = TrajectoryEvidence(
            subgoal="Go to oven",
            task_mode="navigate_to_object",
            progress="Moving",
        )
        assert ev.outcome == ""
        assert ev.gd_quality == "ok"
        assert ev.key_frames == []
        assert ev.room_id == -1
        assert ev.objects_nearby == []

    def test_detection_failed_to_entry(self):
        ev = TrajectoryEvidence(
            subgoal="Navigate to oven",
            task_mode="navigate_to_object",
            progress="Detection failed",
            salient=["cabinet"],
            outcome="detection_failed",
            gd_quality="score_too_low",
            key_frames=["snap_1"],
            room_id=5,
            objects_nearby=["bed"],
        )
        entry = ev.to_notebook_entry(step=10)
        assert entry.entry_type == "hypothesis_rejected"
        assert entry.negation is True
        assert "score_too_low" in entry.content
        assert entry.key_frame_id == "snap_1"

    def test_object_found_to_entry(self):
        ev = TrajectoryEvidence(
            subgoal="Find oven",
            task_mode="navigate_to_object",
            progress="Target in view",
            salient=["oven"],
            outcome="object_found",
            gd_quality="ok",
            key_frames=["snap_2"],
            room_id=3,
            objects_nearby=["stove", "counter"],
        )
        entry = ev.to_notebook_entry(step=20)
        assert entry.entry_type == "object_observed"
        assert entry.negation is False
        assert "oven" in entry.content
        assert "Room 3" in entry.content

    def test_explore_seed_to_entry(self):
        ev = TrajectoryEvidence(
            subgoal="Explore seed 3",
            task_mode="explore_seed",
            progress="Arrived at dining area, not kitchen",
            salient=["dining table", "chair"],
            outcome="arrived_near_target",
            gd_quality="ok",
            room_id=7,
            objects_nearby=["dining table"],
        )
        entry = ev.to_notebook_entry(step=30)
        assert entry.entry_type == "seed_visited"
        assert "seed visited" in entry.content.lower() or "Seed visited" in entry.content
        assert "Room 7" in entry.content

    def test_explore_frontier_to_entry(self):
        ev = TrajectoryEvidence(
            subgoal="Explore frontier 0",
            task_mode="explore_frontier",
            progress="Dead end found",
            salient=["wall"],
            outcome="target_not_reached",
            gd_quality="ok",
            room_id=2,
            objects_nearby=["wall"],
        )
        entry = ev.to_notebook_entry(step=40)
        assert entry.entry_type == "frontier_visited"
        assert "Frontier_0" not in entry.content  # subgoal is "Explore frontier 0"
        assert "Room 2" in entry.content

    def test_unknown_task_mode_defaults_to_room_explored(self):
        ev = TrajectoryEvidence(
            subgoal="Do something",
            task_mode="unknown_mode",
            progress="Nothing special happened",
            outcome="unknown",
            room_id=1,
            objects_nearby=["box"],
        )
        entry = ev.to_notebook_entry(step=50)
        assert entry.entry_type == "room_explored"
        assert "Room 1" in entry.content

    def test_no_key_frames_none_id(self):
        ev = TrajectoryEvidence(
            subgoal="Test",
            task_mode="navigate_to_object",
            progress="Moving",
            outcome="detection_failed",
            gd_quality="no_detection",
        )
        entry = ev.to_notebook_entry(step=0)
        assert entry.key_frame_id is None

    def test_trajectory_evidence_to_entry(self):
        """Full integration test matching the plan's example."""
        ev = TrajectoryEvidence(
            subgoal="Navigate to oven via view_2",
            task_mode="navigate_to_object",
            progress="Moved 3 steps toward kitchen",
            salient=["cabinet", "oven-like(score=0.44)"],
            outcome="target_not_reached",
            gd_quality="score_too_low",
            key_frames=["snap_step12_view2"],
            room_id=5,
            objects_nearby=["bed", "chair"],
        )
        entry = ev.to_notebook_entry(step=12)
        assert entry.entry_type == "object_observed"
        assert "oven-like" in entry.content
        assert entry.key_frame_id == "snap_step12_view2"
