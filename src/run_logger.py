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

        ax.set_title(f"Back-projection: {len(pcd_voxels or [])} points -> "
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
