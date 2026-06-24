"""Executor: wraps 6 structured tools over the existing agent_tools layer.

Each tool returns a TrajectoryEvidence instance that compresses the low-level
outcome for the upper-tier Planner / EvidenceNotebook.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.agent_evidence import TrajectoryEvidence

logger = logging.getLogger(__name__)


class Executor:
    """Dispatches PlannerAction to the appropriate low-level tool."""

    def __init__(
        self,
        scene,
        tsdf_planner,
        memory_store,
        cfg,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
    ):
        self.scene = scene
        self.tsdf = tsdf_planner
        self.memory = memory_store
        self.cfg = cfg
        self.models = {
            "detection": detection_model,
            "sam": sam_predictor,
            "clip": clip_model,
            "clip_preprocess": clip_preprocess,
            "clip_tokenizer": clip_tokenizer,
        }
        self._pts = None
        self._angle = None
        self._step_counter = 0

    # ── state ─────────────────────────────────────────────────────────

    def set_state(self, pts, angle, step_counter: int):
        self._pts = pts
        self._angle = angle
        self._step_counter = step_counter

    # ── helpers ───────────────────────────────────────────────────────

    def _m(self) -> dict:
        return self.models

    def _collect_nearby(self, pts) -> list:
        if not hasattr(self.scene, "objects"):
            return []
        return [
            obj["class_name"]
            for obj in self.scene.objects.values()
            if hasattr(obj, "bbox") and obj.get("bbox") is not None
            and __import__("numpy").linalg.norm(
                obj["bbox"].center[[0, 2]] - pts[[0, 2]]
            ) < self.cfg.scene_graph.obj_include_dist + 0.5
        ]

    # ── 6 tools ───────────────────────────────────────────────────────

    def explore_panorama(self, config: Optional[dict] = None) -> TrajectoryEvidence:
        from src.agent_tools import observe_panorama

        pts, angle, _mosaic_b64, text = observe_panorama(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            self._step_counter,
            self.memory,
            self.scene.cam_intrinsic,
            self.cfg,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
        )
        self._pts, self._angle = pts, angle

        room_id = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else -1
        )
        return TrajectoryEvidence(
            subgoal="Explore panorama for re-orientation",
            task_mode="explore_panorama",
            progress=text,
            salient=[text],
            outcome="panorama_complete",
            room_id=room_id,
            objects_nearby=self._collect_nearby(pts),
        )

    def navigate_to_object(
        self, object_name: str, view_idx: Optional[int] = None
    ) -> TrajectoryEvidence:
        from src.agent_tools import navigate_to_object

        pts, angle, success, status, _img = navigate_to_object(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            object_name,
            self.memory,
            self.scene.cam_intrinsic,
            self.cfg,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
            self._step_counter,
        )
        self._pts, self._angle = pts, angle

        gd_quality = (
            "ok"
            if success
            else ("detection_failed" if "GD" in status else "target_not_reached")
        )
        room_id = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else -1
        )

        return TrajectoryEvidence(
            subgoal=f"Navigate to {object_name} via view {view_idx}",
            task_mode="navigate_to_object",
            progress=f"Navigation status: {status}",
            salient=[object_name, status],
            outcome="arrived_near_target" if success else "target_not_reached",
            gd_quality=gd_quality,
            room_id=room_id,
            objects_nearby=self._collect_nearby(pts),
        )

    def explore_seed(self, seed_id: str) -> TrajectoryEvidence:
        from src.agent_tools import navigate_to_seed

        try:
            room_id = int(seed_id)
        except (ValueError, TypeError):
            room_id = 0

        pts, angle, success, status, _img = navigate_to_seed(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            room_id,
            self.cfg,
            self.memory,
            self.scene.cam_intrinsic,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
            self._step_counter,
        )
        self._pts, self._angle = pts, angle

        arrived_room = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else room_id
        )

        return TrajectoryEvidence(
            subgoal=f"Navigate to seed {seed_id}",
            task_mode="explore_seed",
            progress=f"Arrived at seed {seed_id}, room {arrived_room}",
            salient=[f"seed_{seed_id}", f"room_{arrived_room}"],
            outcome="arrived_near_target" if success else "target_not_reached",
            room_id=arrived_room,
            objects_nearby=self._collect_nearby(pts),
        )

    def explore_frontier(self, frontier_id: str) -> TrajectoryEvidence:
        from src.agent_tools import navigate_to_frontier

        try:
            fid = int(frontier_id)
        except (ValueError, TypeError):
            fid = 0

        pts, angle, success, status, _img = navigate_to_frontier(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            fid,
            self.cfg,
            self.memory,
            self.scene.cam_intrinsic,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
            self._step_counter,
        )
        self._pts, self._angle = pts, angle

        arrived_room = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else -1
        )

        return TrajectoryEvidence(
            subgoal=f"Navigate to frontier {frontier_id}",
            task_mode="explore_frontier",
            progress=f"Arrived at frontier {frontier_id}, room {arrived_room}",
            salient=[f"frontier_{frontier_id}", f"room_{arrived_room}"],
            outcome="arrived_near_target" if success else "target_not_reached",
            room_id=arrived_room,
            objects_nearby=self._collect_nearby(pts),
        )

    def inspect_object(self, object_name: str) -> TrajectoryEvidence:
        from src.agent_tools import silent_perception_step

        pts, angle = silent_perception_step(
            self.scene,
            self.tsdf,
            self._pts,
            self._angle,
            self._step_counter,
            self.memory,
            self.scene.cam_intrinsic,
            self.cfg,
            self.models["detection"],
            self.models["sam"],
            self.models["clip"],
            self.models["clip_preprocess"],
            self.models["clip_tokenizer"],
        )
        self._pts, self._angle = pts, angle

        room_id = (
            self.tsdf.get_room_id_at(self.tsdf.habitat2voxel(pts)[:2])
            if hasattr(self.tsdf, "get_room_id_at")
            else -1
        )

        return TrajectoryEvidence(
            subgoal=f"Inspect {object_name} at current position",
            task_mode="inspect_object",
            progress=f"Close inspection of {object_name}",
            salient=[object_name],
            outcome="inspection_complete",
            room_id=room_id,
            objects_nearby=self._collect_nearby(pts),
        )

    # ── dispatch ──────────────────────────────────────────────────────

    def execute_action(self, action) -> TrajectoryEvidence:
        if action.action_type == "explore_panorama":
            return self.explore_panorama()
        elif action.action_type == "navigate_to_object":
            return self.navigate_to_object(action.object_name, action.view_idx)
        elif action.action_type == "explore_seed":
            return self.explore_seed(action.seed_id)
        elif action.action_type == "explore_frontier":
            return self.explore_frontier(action.frontier_id)
        elif action.action_type == "inspect_object":
            return self.inspect_object(action.object_name)
        elif action.action_type == "submit_answer":
            return TrajectoryEvidence(
                subgoal="Submit answer",
                task_mode="submit_answer",
                progress=f"Answer: {action.answer}",
                salient=[action.answer or ""],
                outcome="answer_submitted",
                room_id=-1,
                objects_nearby=[],
            )
        else:
            logger.warning("Unknown action_type: %s", action.action_type)
            return TrajectoryEvidence(
                subgoal="Unknown action",
                task_mode="unknown",
                progress="Unknown action type",
                outcome="error",
                salient=[],
                gd_quality="no_detection",
            )
