"""Seed view manager: lazy rendering of seed direction images.

Each seed has ONE representative image. Agent cannot see through walls,
cannot teleport to seed. Image updates only when:
1. Agent moves closer to seed (Euclidean distance decreased)
2. No tall obstacle (>=1.2m) blocks the ray agent->seed
"""
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class SeedViewManager:
    """Manages seed direction images with lazy updates."""

    def __init__(self, run_logger=None):
        self.seeds: Dict[int, dict] = {}
        self.run_logger = run_logger
        # Each entry: {"image": np.ndarray, "view_image_pos": np.ndarray,
        #              "view_image_angle": float, "seed_position": np.ndarray}

    def register_seed(self, seed_id: int, position: np.ndarray,
                      scene, tsdf_planner, agent_pts: np.ndarray):
        """Register a new seed and render its initial view image.

        Renders the view from agent's current position toward the seed.
        Habitat-sim forward is -z, so yaw = atan2(-dx, -dz).
        (Per debug_render_episode.py:907-908)
        """
        angle_to_seed = math.atan2(
            -(position[0] - agent_pts[0]),
            -(position[2] - agent_pts[2]))
        obs, _ = scene.get_observation(agent_pts, angle_to_seed)
        self.seeds[seed_id] = {
            "image": obs["color_sensor"][..., :3],
            "view_image_pos": agent_pts.copy(),
            "view_image_angle": float(angle_to_seed),
            "seed_position": position.copy(),
        }
        logging.info(f"  SeedViewManager: registered seed {seed_id} at {position.tolist()}")

        # Log seed view update
        if self.run_logger is not None:
            self.run_logger.log_seed_view_update(
                seed_id, obs["color_sensor"][..., :3], agent_pts,
                reason="registered", angle=angle_to_seed)

    def update_after_step(self, active_seed_ids: List[int],
                          cur_pts: np.ndarray, tsdf_planner, scene,
                          min_blocking_height: float = 1.2):
        """Check all seeds and update images if conditions met.

        Conditions for update:
        1. Ray agent->seed not blocked by tall obstacles
        2. Euclidean distance (xz-plane) decreased by >0.1m (anti-jitter)
        """
        from src.geom import check_ray_blocked

        for seed_id in active_seed_ids:
            if seed_id not in self.seeds:
                continue
            seed = self.seeds[seed_id]
            seed_pos = seed["seed_position"]

            # Condition 1: no tall obstacle blocking
            ray_blocked = check_ray_blocked(
                tsdf_planner, cur_pts, seed_pos,
                min_blocking_height=min_blocking_height)
            if ray_blocked:
                continue

            # Condition 2: distance decreased
            cur_dist = np.linalg.norm(cur_pts[[0, 2]] - seed_pos[[0, 2]])
            last_dist = np.linalg.norm(
                seed["view_image_pos"][[0, 2]] - seed_pos[[0, 2]])
            if cur_dist >= last_dist - 0.1:  # 0.1m anti-jitter
                continue

            # Update
            angle_to_seed = math.atan2(
                -(seed_pos[0] - cur_pts[0]),
                -(seed_pos[2] - cur_pts[2]))
            obs, _ = scene.get_observation(cur_pts, angle_to_seed)
            seed["image"] = obs["color_sensor"][..., :3]
            seed["view_image_pos"] = cur_pts.copy()
            seed["view_image_angle"] = float(angle_to_seed)
            logging.info(f"  SeedViewManager: updated seed {seed_id} "
                        f"(dist {last_dist:.2f}->{cur_dist:.2f})")

            # Log seed view update
            if self.run_logger is not None:
                self.run_logger.log_seed_view_update(
                    seed_id, obs["color_sensor"][..., :3], cur_pts,
                    reason=f"dist_decreased ({last_dist:.2f}->{cur_dist:.2f})",
                    angle=angle_to_seed)

    def get_mosaic(self, question: str, seed_ids: Optional[List[int]] = None,
                   max_seeds: int = 8) -> Optional[np.ndarray]:
        """Build a mosaic of seed images with seed_id labels.

        If seed_ids is provided, only those seeds are shown.
        Otherwise shows all registered seeds.

        Returns RGB numpy array, or None if no seeds.
        """
        if not self.seeds:
            return None

        if seed_ids is not None:
            filtered = [(sid, self.seeds[sid]) for sid in seed_ids if sid in self.seeds]
        else:
            filtered = list(self.seeds.items())
        seeds = filtered[:max_seeds]
        n = len(seeds)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for i, (seed_id, seed) in enumerate(seeds):
            r, c = i // cols, i % cols
            axes[r, c].imshow(seed["image"])
            axes[r, c].set_title(f"Seed {seed_id}", fontsize=14, fontweight='bold')
            axes[r, c].axis('off')

        # Hide unused subplots
        for i in range(n, rows * cols):
            r, c = i // cols, i % cols
            axes[r, c].axis('off')

        fig.suptitle(f"Seed views (question: {question[:60]}...)", fontsize=12)
        fig.tight_layout()
        # Rasterize to numpy (use renderer.buffer_rgba for backend compatibility)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        raw = renderer.buffer_rgba()
        mosaic = np.asarray(raw)[:, :, :3]  # drop alpha
        plt.close(fig)
        return mosaic

    def get_unexplored_seed_ids(self, explored_seed_ids: set) -> List[int]:
        """Return seed IDs not in the explored set."""
        return [sid for sid in self.seeds if sid not in explored_seed_ids]
