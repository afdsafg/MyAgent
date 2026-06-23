# Visualization Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a `RunLogger` module that saves every robot step, frontier, and memory snapshot to `results/<timestamp>/<question_id>/` with per-stage subdirectories, covering all 6 stages of the agent workflow.

**Architecture:** New lightweight `src/run_logger.py` module (does not touch existing `logger_aeqa.py`). Insert logging hooks at 8 key points in `agent_workflow.py`, `scene_aeqa.py`, `agent_tools.py`, and `seed_views.py`. Port the `render_topdown` function from the debug script for per-step topdown visualization. Record full VLM trace with `reason` field to `trace.jsonl`.

**Tech Stack:** Python 3.9, matplotlib, numpy, PIL, OmegaConf

**Worktree:** `.worktrees/add-visualization` (create from `fix-navigation` after Plan 1 merges)

**Design doc:** `docs/plans/2026-06-23-visualization-logging-design.md`

**Prerequisite:** Plan 1 (navigation-fix) must be complete and merged.

---

## Phase A: RunLogger Core

### Task 1: Create `src/run_logger.py` with `__init__` and `init_episode`

**Files:**
- Create: `.worktrees/add-visualization/src/run_logger.py`
- Test: `.worktrees/add-visualization/tests/test_run_logger.py`

**Step 1: Write failing test**

```python
# tests/test_run_logger.py
import os
import tempfile
from src.run_logger import RunLogger


def test_init_creates_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        assert os.path.isdir(os.path.join(tmp, logger.run_timestamp))


def test_init_episode_creates_subdirs():
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        logger.init_episode("q-001", "what is on the oven?", "towel")
        ep_dir = os.path.join(tmp, logger.run_timestamp, "q-001")
        assert os.path.isdir(ep_dir)
        for sub in ["panorama", "stage2_decision", "stage2_5a_seed_selection",
                    "stage3_object_selection", "stage4_navigation",
                    "stage5_decision", "stage6_frontier_selection",
                    "seed_views", "snapshot"]:
            assert os.path.isdir(os.path.join(ep_dir, sub)), f"missing {sub}"


def test_trace_jsonl_exists_after_init():
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        logger.init_episode("q-001", "question?", "answer")
        trace_path = os.path.join(
            tmp, logger.run_timestamp, "q-001", "trace.jsonl")
        assert os.path.isfile(trace_path)
```

**Step 2: Run test to verify it fails**

Run: `cd .worktrees/add-visualization && python -m pytest tests/test_run_logger.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/run_logger.py
"""RunLogger: lightweight per-stage visualization and trace logging.

Writes to results/<timestamp>/<question_id>/ with subdirectories per stage.
Does NOT modify existing logger_aeqa.py (kept for run_aeqa_evaluation.py).
"""
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


class RunLogger:
    """Per-episode structured logger writing to results/<ts>/<qid>/."""

    SUBDIRS = [
        "panorama",
        "stage2_decision",
        "stage2_5a_seed_selection",
        "stage3_object_selection",
        "stage4_navigation",
        "stage5_decision",
        "stage6_frontier_selection",
        "seed_views",
        "snapshot",
    ]

    def __init__(self, output_root: str = "results", enabled: bool = True,
                 save_nav_topdown: bool = True, save_nav_views: bool = False,
                 save_seed_history: bool = False, dpi: int = 110):
        self.output_root = output_root
        self.enabled = enabled
        self.save_nav_topdown = save_nav_topdown
        self.save_nav_views = save_nav_views
        self.save_seed_history = save_seed_history
        self.dpi = dpi
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_root, self.run_timestamp)
        self.episode_dir: Optional[str] = None
        self._step_counter = 0

        if self.enabled:
            os.makedirs(self.run_dir, exist_ok=True)

    def init_episode(self, question_id: str, question: str, answer: str = ""):
        """Create episode directory tree and initialize trace.jsonl."""
        if not self.enabled:
            return
        self.episode_dir = os.path.join(self.run_dir, question_id)
        os.makedirs(self.episode_dir, exist_ok=True)
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(self.episode_dir, sub), exist_ok=True)

        # Initialize trace.jsonl with episode metadata
        self.log_trace("episode_start", {
            "question_id": question_id,
            "question": question,
            "answer": answer,
        })

    def log_trace(self, event_type: str, data: Dict[str, Any],
                  reason: Optional[str] = None):
        """Append a JSON event to trace.jsonl."""
        if not self.enabled or self.episode_dir is None:
            return
        trace_path = os.path.join(self.episode_dir, "trace.jsonl")
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            **data,
        }
        if reason is not None:
            event["reason"] = reason
        with open(trace_path, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _episode_subdir(self, name: str) -> str:
        """Get path to episode subdirectory."""
        return os.path.join(self.episode_dir, name)

    def log_summary(self, result: Dict[str, Any]):
        """Write final summary.json."""
        if not self.enabled or self.episode_dir is None:
            return
        summary_path = os.path.join(self.episode_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        self.log_trace("episode_end", result)
```

**Step 4: Run tests**

Run: `cd .worktrees/add-visualization && python -m pytest tests/test_run_logger.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
cd .worktrees/add-visualization
git add src/run_logger.py tests/test_run_logger.py
git commit -m "feat: add RunLogger core with episode directory structure

- results/<timestamp>/<question_id>/ output layout
- 9 subdirectories per stage (panorama, stage2-6, seed_views, snapshot)
- trace.jsonl with JSON-per-line event log
- Configurable: enabled, save_nav_topdown, save_nav_views, save_seed_history"
```

---

### Task 2: Port `render_topdown` from debug script

**Files:**
- Modify: `.worktrees/add-visualization/src/run_logger.py` (append method)

**Step 1: Port the render_topdown function**

Reference: `MSGNav-main/tools/debug_render/debug_iterative_spiral_navigate.py:305-500`

Append to `RunLogger` class:

```python
    def render_topdown(self, tsdf_planner, pts, angle, nav_trace,
                       target_voxel_xy, spiral_results_history,
                       output_path, phrase="", score=0.0, iteration=0,
                       pcd_voxel_list=None):
        """Render topdown map with rooms, trajectory, spiral history.

        Ported from debug_iterative_spiral_navigate.py:render_topdown.
        """
        if not self.enabled:
            return
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from scipy import ndimage
        import cv2

        h, w = tsdf_planner._tsdf_vol_cpu.shape[:2]
        fig, ax = plt.subplots(figsize=(8, 8 * h / w))
        ft_map = np.full((h, w, 3), 255, dtype=np.uint8)

        # Room segmentation (for rendering)
        room_height = 1.8
        high_voxel = int(room_height / tsdf_planner._voxel_size) + tsdf_planner.min_height_voxel
        envelope = (tsdf_planner._tsdf_vol_cpu[:, :, high_voxel] > 0) & \
                   (tsdf_planner._tsdf_vol_cpu[:, :, 0] < 0)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        envelope = (cv2.morphologyEx(
            (envelope.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel3, iterations=1
        ) > 0) & envelope

        # Room overlay
        tab20 = plt.cm.tab20
        if hasattr(tsdf_planner, 'room_map') and tsdf_planner.room_map is not None:
            room_map = tsdf_planner.room_map
            for rid in np.unique(room_map):
                if rid == 0:
                    continue
                mask = room_map == rid
                color = np.array(tab20((rid - 1) % 20))[:3] * 255
                ft_map[mask] = color.astype(np.uint8)
                center = ndimage.center_of_mass(mask)
                if center[0] > 0:
                    ax.text(center[1], center[0], f"R{rid}", fontsize=11,
                            fontweight="bold", color="black", ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        # Explored area (green tint)
        navigable = envelope
        explored = (tsdf_planner.unexplored == 0) & navigable
        ft_map[explored] = (ft_map[explored].astype(float) * 0.7 +
                            np.array([180, 230, 180]) * 0.3).astype(np.uint8)

        ax.imshow(ft_map, origin="upper")

        # Navigation trace (red polyline)
        if nav_trace:
            trace_vy = [s["voxel_xy"][0] for s in nav_trace]
            trace_vx = [s["voxel_xy"][1] for s in nav_trace]
            ax.plot(trace_vx, trace_vy, "r-", linewidth=2, alpha=0.8)
            ax.scatter(trace_vx, trace_vy, c="red", s=20, zorder=5)

        # 3D point cloud (red scatter)
        if pcd_voxel_list:
            ax.scatter([v[1] for v in pcd_voxel_list],
                       [v[0] for v in pcd_voxel_list],
                       c="red", s=8, alpha=0.6, edgecolors="darkred",
                       linewidths=0.3, zorder=10)

        # Spiral search history (multi-color markers)
        colors = ["orange", "purple", "brown", "pink", "gray"]
        for i, sr in enumerate(spiral_results_history):
            c = colors[i % len(colors)]
            vy, vx = sr["voxel_xy"]
            ax.scatter(vx, vy, c=c, s=100, marker="*",
                       edgecolors="black", linewidths=1, zorder=8)
            ax.text(vx, vy, f"i{i+1}", fontsize=8, ha="center",
                    va="bottom", color=c, fontweight="bold")

        # Target voxel (red cross)
        if target_voxel_xy:
            tv_y, tv_x = target_voxel_xy
            ax.plot([tv_x, tv_x], [tv_y - 3, tv_y + 3], "r-", linewidth=2)
            ax.plot([tv_x - 3, tv_x + 3], [tv_y, tv_y], "r-", linewidth=2)
            ax.text(tv_x, tv_y - 5, phrase or "target",
                    color="red", fontsize=10, fontweight="bold")

        # Agent (blue circle + heading tick)
        agent_voxel = tsdf_planner.habitat2voxel(pts)
        ay, ax_ = agent_voxel[0], agent_voxel[1]
        ax.scatter(ax_, ay, c="blue", s=80, zorder=9, edgecolors="darkblue")
        tick_len = 5
        ax.plot([ax_, ax_ + tick_len * np.cos(angle)],
                [ay, ay + tick_len * np.sin(angle)], "b-", linewidth=2)

        ax.set_title(f"iter {iteration} | {phrase} (score={score:.2f})", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
```

**Step 2: Verify compile**

Run: `cd .worktrees/add-visualization && python -c "from src.run_logger import RunLogger; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/run_logger.py
git commit -m "feat: port render_topdown from debug script to RunLogger

- Room segmentation fill + labels
- Navigation trace (red polyline + scatter)
- 3D point cloud (red scatter)
- Spiral search history (multi-color star markers)
- Target voxel (red cross + label)
- Agent (blue circle + heading tick)"
```

---

### Task 3: Implement `log_panorama`

**Files:**
- Modify: `.worktrees/add-visualization/src/run_logger.py`

**Step 1: Add method to RunLogger**

```python
    def log_panorama(self, panorama_views, mosaic_img, question):
        """Stage 1: save 8 views + mosaic + meta.json."""
        if not self.enabled or self.episode_dir is None:
            return
        import matplotlib.pyplot as plt

        pano_dir = self._episode_subdir("panorama")

        # Save individual views
        for v in panorama_views:
            fname = f"view{v['view_idx']}_{v['direction']}.png"
            plt.imsave(os.path.join(pano_dir, fname), v["rgb"])

        # Save mosaic
        if mosaic_img is not None:
            mosaic_path = os.path.join(pano_dir, "mosaic.png")
            if isinstance(mosaic_img, str):  # base64
                import base64
                with open(mosaic_path, "wb") as f:
                    f.write(base64.b64decode(mosaic_img))
            else:  # numpy array
                plt.imsave(mosaic_path, mosaic_img)

        # Save meta
        meta = {
            "question": question,
            "view_count": len(panorama_views),
            "views": [
                {"view_idx": v["view_idx"], "direction": v["direction"],
                 "angle": float(v["angle"])}
                for v in panorama_views
            ],
        }
        with open(os.path.join(pano_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.log_trace("panorama", {
            "stage": 1, "view_count": len(panorama_views),
            "pts": panorama_views[0]["cam_pose"][:3, 3].tolist()
                   if panorama_views else None,
        })
```

**Step 2: Verify compile**

Run: `cd .worktrees/add-visualization && python -c "from src.run_logger import RunLogger; r=RunLogger(enabled=False); print(hasattr(r,'log_panorama'))"`
Expected: `True`

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/run_logger.py
git commit -m "feat: add log_panorama for Stage 1 8-view panorama logging"
```

---

### Task 4: Implement `log_vlm_decision` (with reason)

**Files:**
- Modify: `.worktrees/add-visualization/src/run_logger.py`

**Step 1: Add method**

```python
    def log_vlm_decision(self, stage, input_image, response_text,
                         parsed, reason, latency_ms=None):
        """Stage 2/2.5a/3/5/6: save VLM input image + response JSON + reason."""
        if not self.enabled or self.episode_dir is None:
            return
        import base64
        import matplotlib.pyplot as plt

        stage_map = {
            2: "stage2_decision",
            "2.5a": "stage2_5a_seed_selection",
            3: "stage3_object_selection",
            5: "stage5_decision",
            6: "stage6_frontier_selection",
        }
        subdir = stage_map.get(stage, f"stage{stage}")
        stage_dir = self._episode_subdir(subdir)
        os.makedirs(stage_dir, exist_ok=True)

        # Save input image
        if input_image is not None:
            img_path = os.path.join(stage_dir, "input.png")
            if isinstance(input_image, str):  # base64
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(input_image))
            elif isinstance(input_image, np.ndarray):
                plt.imsave(img_path, input_image)

        # Save response
        resp_data = {
            "stage": stage,
            "reason": reason,
            "raw_response": response_text,
            "parsed": parsed,
            "latency_ms": latency_ms,
        }
        with open(os.path.join(stage_dir, "response.json"), "w") as f:
            json.dump(resp_data, f, indent=2, ensure_ascii=False, default=str)

        # Trace
        self.log_trace("vlm_call", {
            "stage": stage,
            "input": os.path.basename(img_path) if input_image is not None else None,
            "response": parsed,
            "latency_ms": latency_ms,
        }, reason=reason)
```

**Step 2: Verify compile**

Run: `cd .worktrees/add-visualization && python -c "from src.run_logger import RunLogger; r=RunLogger(enabled=False); print(hasattr(r,'log_vlm_decision'))"`
Expected: `True`

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/run_logger.py
git commit -m "feat: add log_vlm_decision with reason field and trace logging"
```

---

### Task 5: Implement `log_gd_detection` and `log_backprojection`

**Files:**
- Modify: `.worktrees/add-visualization/src/run_logger.py`

**Step 1: Add methods**

```python
    def log_gd_detection(self, rgb, bbox, mask, phrase, score):
        """Stage 4: GD bbox + SAM mask visualization."""
        if not self.enabled or self.episode_dir is None:
            return
        import matplotlib.pyplot as plt

        nav_dir = self._episode_subdir("stage4_navigation")
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(rgb)
        x1, y1, x2, y2 = bbox
        from matplotlib.patches import Rectangle
        axs[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    fill=False, edgecolor="red", linewidth=2))
        axs[0].set_title(f"GD: {phrase} (score={score:.3f})")
        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("SAM mask")
        axs[2].imshow(rgb)
        axs[2].imshow(mask, alpha=0.4, cmap="Reds")
        axs[2].set_title("Overlay")
        for a in axs:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(nav_dir, "gd_detection.png"),
                    dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        self.log_trace("gd_detect", {
            "phrase": phrase, "score": float(score),
            "bbox": [float(x) for x in bbox],
        })

    def log_backprojection(self, tsdf_planner, target_normal, target_voxel,
                           pcd_voxels=None):
        """Stage 4: 3D back-projection point cloud + target voxel on topdown."""
        if not self.enabled or self.episode_dir is None:
            return
        import matplotlib.pyplot as plt

        nav_dir = self._episode_subdir("stage4_navigation")
        h, w = tsdf_planner._tsdf_vol_cpu.shape[:2]
        fig, ax = plt.subplots(figsize=(8, 8 * h / w))

        # Render occupancy base
        if hasattr(tsdf_planner, 'island') and tsdf_planner.island is not None:
            base = np.full((h, w, 3), 240, dtype=np.uint8)
            base[tsdf_planner.island > 0] = [200, 200, 200]
            ax.imshow(base, origin="upper")
        else:
            ax.set_facecolor("white")

        # 3D point cloud voxels
        if pcd_voxels:
            ax.scatter([v[1] for v in pcd_voxels],
                       [v[0] for v in pcd_voxels],
                       c="red", s=6, alpha=0.6, zorder=10)

        # Target voxel (cross)
        tv_y, tv_x = int(target_voxel[0]), int(target_voxel[1])
        ax.plot([tv_x, tv_x], [tv_y - 3, tv_y + 3], "r-", linewidth=2)
        ax.plot([tv_x - 3, tv_x + 3], [tv_y, tv_y], "r-", linewidth=2)
        ax.text(tv_x, tv_y - 5, f"target {target_voxel.tolist()}",
                color="red", fontsize=9)

        ax.set_title(f"Back-projection: {len(pcd_voxels or [])} points → "
                     f"voxel {target_voxel.tolist()}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(nav_dir, "backprojection.png"),
                    dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        self.log_trace("backproject", {
            "target_normal": [float(x) for x in target_normal],
            "target_voxel": [int(x) for x in target_voxel],
            "pcd_points": len(pcd_voxels) if pcd_voxels else 0,
        })
```

**Step 2: Verify compile**

Run: `cd .worktrees/add-visualization && python -c "from src.run_logger import RunLogger; r=RunLogger(enabled=False); print(hasattr(r,'log_gd_detection'), hasattr(r,'log_backprojection'))"`
Expected: `True True`

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/run_logger.py
git commit -m "feat: add log_gd_detection and log_backprojection visualizations"
```

---

### Task 6: Implement navigation step logging

**Files:**
- Modify: `.worktrees/add-visualization/src/run_logger.py`

**Step 1: Add methods**

```python
    def log_spiral_search(self, iteration, target_voxel_xy, spiral_result,
                          tsdf_planner):
        """Stage 4: spiral search result topdown."""
        if not self.enabled or self.episode_dir is None:
            return
        iter_dir = os.path.join(self._episode_subdir("stage4_navigation"),
                                f"iter{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        # Render simple topdown with target + spiral result
        import matplotlib.pyplot as plt
        h, w = tsdf_planner._tsdf_vol_cpu.shape[:2]
        fig, ax = plt.subplots(figsize=(8, 8 * h / w))
        if tsdf_planner.island is not None:
            base = np.full((h, w, 3), 240, dtype=np.uint8)
            base[tsdf_planner.island > 0] = [200, 200, 200]
            ax.imshow(base, origin="upper")

        # Target
        ty, tx = target_voxel_xy
        ax.scatter(tx, ty, c="red", s=100, marker="x", linewidths=2, zorder=10)

        # Spiral result
        if spiral_result:
            sy, sx = spiral_result["voxel_xy"]
            ax.scatter(sx, sy, c="orange", s=120, marker="*",
                       edgecolors="black", linewidths=1, zorder=9)
            ax.text(sx, sy, f"dist={spiral_result['spiral_dist']}",
                    fontsize=9, color="orange", fontweight="bold")

        ax.set_title(f"iter {iteration} spiral search (dist={spiral_result['spiral_dist'] if spiral_result else 'N/A'})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(iter_dir, "spiral_search.png"),
                    dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        self.log_trace("spiral_search", {
            "iter": iteration,
            "target_voxel": list(target_voxel_xy),
            "spiral_dist": spiral_result["spiral_dist"] if spiral_result else None,
            "voxel": list(spiral_result["voxel_xy"]) if spiral_result else None,
        })

    def log_nav_step(self, iteration, step, pts, angle, tsdf_planner,
                     nav_trace, target_voxel_xy, spiral_history, fig=None):
        """Stage 4: per-step topdown with trajectory."""
        if not self.enabled or self.episode_dir is None:
            return
        if not self.save_nav_topdown:
            return

        iter_dir = os.path.join(self._episode_subdir("stage4_navigation"),
                                f"iter{iteration}")
        nav_walk_dir = os.path.join(iter_dir, "nav_walk")
        os.makedirs(nav_walk_dir, exist_ok=True)

        # Use provided fig (from agent_step) or render our own
        if fig is not None:
            try:
                fig.savefig(os.path.join(nav_walk_dir, f"step{step:02d}_nav.png"),
                            dpi=self.dpi, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                logging.warning(f"log_nav_step fig save failed: {e}")

        # Also render our own topdown with full history
        out_path = os.path.join(nav_walk_dir, f"step{step:02d}_topdown.png")
        self.render_topdown(tsdf_planner, pts, angle, nav_trace,
                           target_voxel_xy, spiral_history, out_path,
                           iteration=iteration)

        self.log_trace("nav_step", {
            "iter": iteration, "step": step,
            "pts": [float(x) for x in pts],
            "voxel": tsdf_planner.habitat2voxel(pts)[:2].tolist(),
        })

    def log_iter_summary(self, iteration, tsdf_planner, nav_trace,
                         spiral_history, target_voxel_xy):
        """Stage 4: iteration summary topdown."""
        if not self.enabled or self.episode_dir is None:
            return
        iter_dir = os.path.join(self._episode_subdir("stage4_navigation"),
                                f"iter{iteration}")
        out_path = os.path.join(iter_dir, "topdown_iter_summary.png")

        if nav_trace:
            pts = np.array(nav_trace[-1]["pts"])
            angle = nav_trace[-1]["angle_rad"]
        else:
            return

        self.render_topdown(tsdf_planner, pts, angle, nav_trace,
                           target_voxel_xy, spiral_history, out_path,
                           iteration=iteration)

        arrived = nav_trace[-1].get("target_arrived", False) if nav_trace else False
        self.log_trace("iter_summary", {
            "iter": iteration, "arrived": arrived,
            "nav_steps": len(nav_trace),
        })

    def log_final_topdown(self, tsdf_planner, nav_trace, spiral_history,
                          target_voxel_xy):
        """Stage 4: final topdown with all iteration history."""
        if not self.enabled or self.episode_dir is None:
            return
        nav_dir = self._episode_subdir("stage4_navigation")
        out_path = os.path.join(nav_dir, "final_topdown.png")

        if not nav_trace:
            return
        pts = np.array(nav_trace[-1]["pts"])
        angle = nav_trace[-1]["angle_rad"]

        self.render_topdown(tsdf_planner, pts, angle, nav_trace,
                           target_voxel_xy, spiral_history, out_path,
                           iteration=len(spiral_history))
```

**Step 2: Verify compile**

Run: `cd .worktrees/add-visualization && python -c "from src.run_logger import RunLogger; r=RunLogger(enabled=False); print(all(hasattr(r, m) for m in ['log_spiral_search','log_nav_step','log_iter_summary','log_final_topdown']))"`
Expected: `True`

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/run_logger.py
git commit -m "feat: add spiral search, nav step, iter summary, final topdown logging"
```

---

### Task 7: Implement seed view and snapshot logging

**Files:**
- Modify: `.worktrees/add-visualization/src/run_logger.py`

**Step 1: Add methods**

```python
    def log_seed_view_update(self, seed_id, image, position, reason,
                             angle=None):
        """Seed view image: save current + optionally history."""
        if not self.enabled or self.episode_dir is None:
            return
        import matplotlib.pyplot as plt

        seed_dir = self._episode_subdir("seed_views")
        current_path = os.path.join(seed_dir, f"seed{seed_id}_current.png")
        plt.imsave(current_path, image)

        # Optional history (debug)
        if self.save_seed_history:
            hist_dir = os.path.join(seed_dir, f"seed{seed_id}_history")
            os.makedirs(hist_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            plt.imsave(os.path.join(hist_dir, f"{ts}.png"), image)

        self.log_trace("seed_view_update", {
            "seed_id": seed_id,
            "reason": reason,
            "position": [float(x) for x in position],
        })

    def log_snapshot(self, snapshot_id, image, room_id, objects_in_view,
                     position):
        """silent_perception snapshot: save RGB to snapshot/ dir."""
        if not self.enabled or self.episode_dir is None:
            return
        import matplotlib.pyplot as plt

        snap_dir = self._episode_subdir("snapshot")
        plt.imsave(os.path.join(snap_dir, f"{snapshot_id}.png"), image)

        self.log_trace("snapshot", {
            "snapshot_id": snapshot_id,
            "room_id": room_id,
            "objects_in_view": objects_in_view,
            "position": [float(x) for x in position],
        })
```

**Step 2: Verify compile**

Run: `cd .worktrees/add-visualization && python -c "from src.run_logger import RunLogger; r=RunLogger(enabled=False); print(hasattr(r,'log_seed_view_update'), hasattr(r,'log_snapshot'))"`
Expected: `True True`

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/run_logger.py
git commit -m "feat: add seed view and snapshot logging to RunLogger"
```

---

## Phase B: Hook Insertions

### Task 8: Insert hooks in `agent_workflow.py`

**Files:**
- Modify: `.worktrees/add-visualization/src/agent_workflow.py`

**Step 1: Add RunLogger instance to `run_episode`**

In `run_episode` function, after setup:
```python
    # Initialize RunLogger
    from src.run_logger import RunLogger
    run_logger = RunLogger(
        output_root=getattr(cfg, 'visualization_output_root', 'results'),
        enabled=getattr(cfg, 'save_visualization', True),
        save_nav_topdown=getattr(cfg.visualization, 'save_nav_topdown', True)
                        if hasattr(cfg, 'visualization') else True,
    )
    run_logger.init_episode(question_id, question, answer="")
```

**Step 2: Insert `log_panorama` after Stage 1**

After `observe_panorama` call:
```python
    run_logger.log_panorama(panorama_views, mosaic_b64, question)
```

**Step 3: Insert `log_vlm_decision` after each `call_vlm`**

Wrap all VLM calls:
```python
    import time as _time
    _t0 = _time.time()
    vlm_response = call_vlm(messages, image_b64=last_img)
    _latency = int((_time.time() - _t0) * 1000)
    vlm_parsed = _parse_vlm_response(vlm_response)
    run_logger.log_vlm_decision(
        stage=current_stage, input_image=last_img,
        response_text=vlm_response, parsed=vlm_parsed,
        reason=vlm_parsed.get("reason", ""),
        latency_ms=_latency)
```

**Step 4: Commit**

```bash
cd .worktrees/add-visualization
git add src/agent_workflow.py
git commit -m "feat: insert RunLogger hooks in agent_workflow.py

- log_panorama after Stage 1 observe_panorama
- log_vlm_decision after every call_vlm (with latency + reason)
- RunLogger initialized from cfg.visualization settings"
```

---

### Task 9: Insert hooks in `scene_aeqa.py` (grounded_navigate_to_object)

**Files:**
- Modify: `.worktrees/add-visualization/src/scene_aeqa.py`

**Step 1: Add `run_logger` parameter to `grounded_navigate_to_object`**

Add `run_logger=None` parameter. At each key point:

```python
    # After GD detection + SAM:
    if run_logger is not None:
        run_logger.log_gd_detection(rgb, bbox, mask, phrase, score)

    # After back-projection:
    if run_logger is not None:
        pcd_voxel_list = [(int(v[0]), int(v[1]))
                          for v in pcd_np_clipped]
        # convert pcd_np_clipped to voxels
        pcd_voxel_list = []
        for p in pcd_np_clipped:
            v = tsdf_planner.normal2voxel(p)[:2]
            pcd_voxel_list.append((int(v[0]), int(v[1])))
        run_logger.log_backprojection(
            tsdf_planner, target_normal, target_voxel, pcd_voxel_list)

    # After each spiral search:
    if run_logger is not None:
        run_logger.log_spiral_search(
            iteration, target_voxel_xy, spiral_result, tsdf_planner)

    # After each nav_step:
    if run_logger is not None:
        run_logger.log_nav_step(
            iteration, nav_step, cur_pts, cur_angle, tsdf_planner,
            nav_trace, target_voxel_xy, spiral_results_history)

    # After each iteration:
    if run_logger is not None:
        run_logger.log_iter_summary(
            iteration, tsdf_planner, nav_trace, spiral_results_history,
            target_voxel_xy)

    # After all iterations:
    if run_logger is not None:
        run_logger.log_final_topdown(
            tsdf_planner, nav_trace, spiral_results_history, target_voxel_xy)
```

**Step 2: Thread `run_logger` through `navigate_to_object` in `agent_tools.py`**

Add `run_logger=None` param to `navigate_to_object`, pass to `gd_nav`.

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/scene_aeqa.py src/agent_tools.py
git commit -m "feat: insert RunLogger hooks in grounded_navigate_to_object

- log_gd_detection after GD+SAM
- log_backprojection after 3D back-projection
- log_spiral_search after each spiral search
- log_nav_step after each agent_step
- log_iter_summary after each iteration
- log_final_topdown after all iterations
- Thread run_logger through navigate_to_object"
```

---

### Task 10: Insert hooks in `agent_tools.py` (silent_perception_step)

**Files:**
- Modify: `.worktrees/add-visualization/src/agent_tools.py`

**Step 1: Add `run_logger` parameter to `silent_perception_step`**

After `memory_store.add_snapshot(...)`:
```python
    if run_logger is not None:
        run_logger.log_snapshot(
            snapshot_id=f"step{step_id}_view{i}",
            image=view_rgb,
            room_id=room_id,
            objects_in_view=objs_in_view,
            position_3d=pts.tolist())
```

**Step 2: Thread `run_logger` through all callers**

Update `_navigate_to_target_with_agent_step`, `navigate_to_object`, `navigate_to_seed`, `navigate_to_frontier` to accept and pass `run_logger`.

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/agent_tools.py
git commit -m "feat: insert RunLogger snapshot hook in silent_perception_step

- log_snapshot after each memory_store.add_snapshot
- Thread run_logger through navigation helper functions"
```

---

### Task 11: Insert hooks in `seed_views.py` (SeedViewManager)

**Files:**
- Modify: `.worktrees/add-visualization/src/seed_views.py`

**Step 1: Add `run_logger` to SeedViewManager**

```python
class SeedViewManager:
    def __init__(self, run_logger=None):
        self.seeds = {}
        self.run_logger = run_logger
```

**Step 2: Log on register and update**

In `register_seed`:
```python
    if self.run_logger is not None:
        self.run_logger.log_seed_view_update(
            seed_id, obs["color_sensor"][..., :3], agent_pts,
            reason="registered", angle=angle_to_seed)
```

In `update_after_step` (after successful update):
```python
    if self.run_logger is not None:
        self.run_logger.log_seed_view_update(
            seed_id, seed["image"], cur_pts,
            reason=f"dist_decreased ({last_dist:.2f}→{cur_dist:.2f})",
            angle=angle_to_seed)
```

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add src/seed_views.py
git commit -m "feat: insert RunLogger hooks in SeedViewManager

- log_seed_view_update on register and on lazy update
- Includes reason for update (registered vs dist_decreased)"
```

---

## Phase C: Config & Smoke Test

### Task 12: Add visualization config fields

**Files:**
- Modify: `.worktrees/add-visualization/cfg/` (config files)

**Step 1: Add visualization section to config**

In the HM-GE config yaml:
```yaml
visualization:
  enabled: true
  output_root: "results"
  save_nav_topdown: true
  save_nav_views: false
  save_seed_history: false
  dpi: 110
```

**Step 2: Verify config loads**

Run: `cd .worktrees/add-visualization && python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('cfg/<config>.yaml'); print(cfg.visualization)"`
Expected: prints the visualization section

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add cfg/
git commit -m "feat: add visualization config section to HM-GE config"
```

---

### Task 13: Smoke test for RunLogger

**Files:**
- Create: `.worktrees/add-visualization/tests/test_run_logger_smoke.py`

**Step 1: Write smoke test**

```python
# tests/test_run_logger_smoke.py
"""Smoke test: verify RunLogger can be instantiated and all methods exist."""
import os
import tempfile
import numpy as np
from src.run_logger import RunLogger


def test_all_logging_methods_exist():
    """Verify all expected methods are defined."""
    logger = RunLogger(enabled=False)
    methods = [
        "init_episode", "log_trace", "log_summary",
        "render_topdown", "log_panorama", "log_vlm_decision",
        "log_gd_detection", "log_backprojection",
        "log_spiral_search", "log_nav_step", "log_iter_summary",
        "log_final_topdown", "log_seed_view_update", "log_snapshot",
    ]
    for m in methods:
        assert hasattr(logger, m), f"missing method: {m}"


def test_trace_writes_jsonl():
    """Verify trace.jsonl accumulates events."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        logger.init_episode("q-test", "question?", "answer")
        logger.log_trace("test_event", {"key": "value"}, reason="test reason")
        logger.log_trace("another", {"num": 42})

        trace_path = os.path.join(tmp, logger.run_timestamp, "q-test", "trace.jsonl")
        with open(trace_path) as f:
            lines = f.readlines()
        assert len(lines) >= 3  # episode_start + 2 events
        import json
        for line in lines:
            evt = json.loads(line)
            assert "ts" in evt
            assert "event" in evt


def test_disabled_logger_no_files():
    """When disabled, no files should be created."""
    logger = RunLogger(enabled=False)
    logger.init_episode("q-test", "q", "a")
    logger.log_trace("test", {})
    # No exception, no files
```

**Step 2: Run smoke test**

Run: `cd .worktrees/add-visualization && python -m pytest tests/test_run_logger_smoke.py -v`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
cd .worktrees/add-visualization
git add tests/test_run_logger_smoke.py
git commit -m "test: add RunLogger smoke test for method existence and trace.jsonl"
```

---

### Task 14: Manual integration test on server

**This task runs manually on the server after merging Plan 1 + Plan 2.**

**Step 1: Merge both plans**

```bash
cd /root/MyAgent
git merge fix-navigation
git merge add-visualization
```

**Step 2: Run a single AEQA question**

```bash
conda run -n 3dmem python run_hmge_evaluation.py \
  --cfg_file cfg/eval_aeqa_hmge_41.yaml \
  --question_id 00c2be2a-1377-4fae-a889-30936b7890c3
```

**Step 3: Verify output structure**

```bash
ls results/<timestamp>/00c2be2a-1377-4fae-a889-30936b7890c3/
# Should see: trace.jsonl, panorama/, stage2_decision/, stage4_navigation/,
#              seed_views/, snapshot/, summary.json
```

**Step 4: Verify trace.jsonl content**

```bash
cat results/<timestamp>/<qid>/trace.jsonl | python -m json.tool
# Each line should be valid JSON with 'ts', 'event', and 'reason' (for vlm_call)
```

**Step 5: Verify topdown images**

```bash
ls results/<timestamp>/<qid>/stage4_navigation/iter1/nav_walk/
# Should see: step01_topdown.png, step02_topdown.png, ...
```

---

## Verification Checklist

- [ ] `RunLogger` core tests pass (3 tests in test_run_logger.py)
- [ ] `RunLogger` smoke tests pass (3 tests in test_run_logger_smoke.py)
- [ ] All 14 logging methods exist on RunLogger
- [ ] `trace.jsonl` accumulates valid JSON events with `reason` field
- [ ] Disabled logger creates no files
- [ ] Server integration: `results/<ts>/<qid>/` directory tree created
- [ ] Server integration: `trace.jsonl` contains VLM calls with reasons
- [ ] Server integration: `stage4_navigation/iter{N}/nav_walk/step{NN}_topdown.png` exists
- [ ] Server integration: `panorama/mosaic.png` exists with 8 views + compass
- [ ] Server integration: `seed_views/seed{N}_current.png` exists

---

## Key References

- Design doc: `docs/plans/2026-06-23-visualization-logging-design.md`
- Debug script (render_topdown source): `MSGNav-main/tools/debug_render/debug_iterative_spiral_navigate.py:305-500`
- Original production visualization (reference): `/media/afdsafg/系统/Users/afdsafg/Downloads/exp_eval_results/exp_eval_aeqa_41_hmge_region_frontier_pred_yolov8m_samb_640/00c2be2a-1377-4fae-a889-30936b7890c3/`
