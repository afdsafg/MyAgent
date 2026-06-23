# Navigation Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the broken Habitat robot navigation by rewriting `grounded_navigate_to_object` with the verified pipeline from `debug_iterative_spiral_navigate.py`, and refactor the agent workflow into a 6-stage state machine where VLM selects the view direction before GD detection.

**Architecture:** Port the 4 helper functions (`_spiral_search_navigable_point`, `_check_voxel_navigable`, `_refresh_planner_grids`, `bresenham_2d`) from the debug script. Rewrite `grounded_navigate_to_object` with corrected 3D back-projection (`pose_habitat_to_tsdf(cam_pose)` + `camera_convention="z_forward"`). Refactor `agent_workflow.py` into 6 stages with VLM JSON schema requiring `reason` field. Add `SeedViewManager` for lazy seed-view updates with height-aware occlusion.

**Tech Stack:** Python 3.9, Habitat-Sim, GroundingDINO (Swin-T), SAM (sam_l.pt), open_clip (ViT-B-32), OmegaConf, numpy, matplotlib

**Worktree:** `.worktrees/fix-navigation` (existing branch, commit `6ab91c9`)

**Design doc:** `docs/plans/2026-06-23-navigation-fix-design.md`

**Reference (verified working):** `MSGNav-main/tools/debug_render/debug_iterative_spiral_navigate.py` (990 lines)

---

## Phase A: Pure Helpers (unit-testable, no Habitat)

### Task 1: Add `bresenham_2d` to `src/geom.py`

**Files:**
- Modify: `.worktrees/fix-navigation/src/geom.py` (append at end)
- Test: `.worktrees/fix-navigation/tests/test_geom.py`

**Step 1: Create test directory and write failing test**

```python
# tests/test_geom.py
import numpy as np
from src.geom import bresenham_2d


def test_bresenham_horizontal():
    """Horizontal ray along x-axis."""
    pts = bresenham_2d((5, 0), (5, 10))
    assert pts[0] == (5, 0)
    assert pts[-1] == (5, 10)
    assert len(pts) == 11


def test_bresenham_vertical():
    """Vertical ray along y-axis."""
    pts = bresenham_2d((0, 3), (8, 3))
    assert pts[0] == (0, 3)
    assert pts[-1] == (8, 3)
    assert len(pts) == 9


def test_bresenham_diagonal():
    """45-degree diagonal."""
    pts = bresenham_2d((0, 0), (5, 5))
    assert (0, 0) in pts
    assert (5, 5) in pts
    # Diagonal should hit every (i, i)
    for i in range(6):
        assert (i, i) in pts


def test_bresenham_single_point():
    """Start == end."""
    pts = bresenham_2d((3, 3), (3, 3))
    assert pts == [(3, 3)]


def test_bresenham_negative_direction():
    """Ray going in negative y direction."""
    pts = bresenham_2d((10, 5), (2, 5))
    assert pts[0] == (10, 5)
    assert pts[-1] == (2, 5)
    assert len(pts) == 9
```

**Step 2: Run test to verify it fails**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_geom.py -v`
Expected: FAIL with `ImportError: cannot import name 'bresenham_2d' from 'src.geom'`

**Step 3: Write minimal implementation**

Append to `src/geom.py`:

```python
def bresenham_2d(start, end):
    """Bresenham line algorithm on 2D grid.

    Args:
        start: (y, x) tuple or array
        end: (y, x) tuple or array
    Returns:
        List of (y, x) tuples from start to end inclusive.
    """
    y0, x0 = int(start[0]), int(start[1])
    y1, x1 = int(end[0]), int(end[1])
    points = []
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    while True:
        points.append((y0, x0))
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points
```

**Step 4: Run test to verify it passes**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_geom.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
cd .worktrees/fix-navigation
git add src/geom.py tests/test_geom.py
git commit -m "feat: add bresenham_2d helper for ray casting on voxel grid"
```

---

### Task 2: Add `check_ray_blocked` (height-aware occlusion) to `src/geom.py`

**Files:**
- Modify: `.worktrees/fix-navigation/src/geom.py` (append)
- Test: `.worktrees/fix-navigation/tests/test_geom.py`

**Step 1: Write failing test**

Append to `tests/test_geom.py`:

```python
class _FakePlanner:
    """Minimal stand-in for TSDFPlanner exposing only what check_ray_blocked needs."""
    def __init__(self, tsdf_vol, voxel_size=0.05, min_height_voxel=0):
        self._tsdf_vol_cpu = tsdf_vol  # shape (H, W, Z)
        self._voxel_size = voxel_size
        self.min_height_voxel = min_height_voxel

    def habitat2voxel(self, pos):
        # Simplified: assume pos already in voxel coords (test only)
        return np.array(pos, dtype=int)


def test_check_ray_blocked_clear_path():
    """No obstacles above 1.2m → not blocked."""
    # 10x10x40 volume, all zeros (free space)
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    blocked = check_ray_blocked(planner, [5, 0, 5], [5, 0, 1],
                                 min_blocking_height=1.2)
    assert blocked is False


def test_check_ray_blocked_wall():
    """Wall (TSDF<0) above 1.2m → blocked."""
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    # Place a wall at voxel (5, 3) from z=12 to z=30 (1.2m to 3.0m)
    vol[5, 3, 12:30] = -0.5  # occupied
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    blocked = check_ray_blocked(planner, [5, 0, 5], [5, 0, 1],
                                 min_blocking_height=1.2)
    assert blocked is True


def test_check_ray_blocked_low_obstacle():
    """Table (0.75m) below 1.2m threshold → not blocked."""
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    # Table at z=0..7 (0 to 0.7m), should NOT block
    vol[5, 3, 0:8] = -0.5
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    blocked = check_ray_blocked(planner, [5, 0, 5], [5, 0, 1],
                                 min_blocking_height=1.2)
    assert blocked is False


def test_check_ray_blocked_endpoints_skipped():
    """Endpoints (agent and target voxels) should be skipped."""
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    # Obstacle AT agent voxel — should be skipped
    vol[5, 0, 12:30] = -0.5
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    blocked = check_ray_blocked(planner, [5, 0, 5], [5, 0, 1],
                                 min_blocking_height=1.2)
    assert blocked is False  # endpoint skipped
```

**Step 2: Run test to verify it fails**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_geom.py::test_check_ray_blocked_clear_path -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Append to `src/geom.py`:

```python
def check_ray_blocked(tsdf_planner, agent_pos, target_pos, min_blocking_height=1.2):
    """Check if ray from agent to target is blocked by tall obstacles.

    Only obstacles taller than min_blocking_height are considered blocking.
    Tables (0.75m), chairs (0.5m) don't block; walls (2.4m+), cabinets (1.8m) do.

    Args:
        tsdf_planner: TSDFPlanner with _tsdf_vol_cpu, _voxel_size, min_height_voxel, habitat2voxel
        agent_pos: 3D habitat position [x, y, z]
        target_pos: 3D habitat position [x, y, z]
        min_blocking_height: height threshold in meters (default 1.2m)
    Returns:
        True if blocked, False if clear.
    """
    agent_voxel = tsdf_planner.habitat2voxel(agent_pos)
    target_voxel = tsdf_planner.habitat2voxel(target_pos)

    ray_voxels = bresenham_2d(agent_voxel[:2], target_voxel[:2])

    voxel_size = tsdf_planner._voxel_size
    floor_z = tsdf_planner.min_height_voxel
    min_block_z = floor_z + int(min_blocking_height / voxel_size)
    max_z = tsdf_planner._tsdf_vol_cpu.shape[2]
    H, W = tsdf_planner._tsdf_vol_cpu.shape[:2]

    for vy, vx in ray_voxels[1:-1]:  # skip endpoints (agent and target)
        if not (0 <= vy < H and 0 <= vx < W):
            return True  # out of bounds = blocked
        column = tsdf_planner._tsdf_vol_cpu[vy, vx, min_block_z:max_z]
        if (column < -0.1).any():  # TSDF < 0 = behind surface = occupied
            return True
    return False
```

**Step 4: Run tests**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_geom.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
cd .worktrees/fix-navigation
git add src/geom.py tests/test_geom.py
git commit -m "feat: add height-aware check_ray_blocked for seed view occlusion"
```

---

## Phase B: Navigation Core (grounded_navigate_to_object rewrite)

### Task 3: Rewrite `grounded_navigate_to_object` Phase A (GD detect + SAM + back-projection)

**Files:**
- Modify: `.worktrees/fix-navigation/src/scene_aeqa.py:1021-1380` (replace entire function)

**Step 1: Read current implementation to confirm what's being replaced**

Run: `cd .worktrees/fix-navigation && sed -n '1021,1160p' src/scene_aeqa.py`
Confirm: current function starts at line 1021, uses `trans_pose=cam_pose` (raw Habitat, line 1105) without `camera_convention`.

**Step 2: Rewrite the function signature and Phase A (back-projection)**

Replace the entire `grounded_navigate_to_object` function (lines 1021-1380) with:

```python
def grounded_navigate_to_object(
    scene, tsdf_planner, pts, angle,
    view_idx, view_angle, view_cam_pose,  # VLM-selected view
    object_desc,
    max_steps=20, gd_dir=None,
    max_consecutive_failures=5,
    max_iterations=5, converge_dist_voxels=5,
    max_nav_steps_per_iter=15,
    memory_store=None, cam_intr_ext=None, cfg_ext=None,
    detection_model=None, sam_predictor=None,
    clip_model=None, clip_preprocess=None, clip_tokenizer=None,
    cnt_step_base=0, step_budget=None,
    gd_model=None,
):
    """GD 导航链：VLM 选定视角 → GD 检测 → 3D 反投影 → 迭代螺旋搜索导航。

    视角由 VLM 在 Stage 2 选定（view_idx + view_angle + view_cam_pose）。
    代码不做视角扫描，只用 VLM 选定的那张图做 GD 检测。

    Key fixes from HM-GE stage-3 notes:
    - trans_pose = pose_habitat_to_tsdf(cam_pose)  (NOT raw Habitat cam_pose)
    - camera_convention = "z_forward"  (matches TSDF pose)
    - Result pcd is already in normal coords; no pos_habitat_to_normal conversion
    - Z-clip: normal[2] > 3.0m → pin to floor
    - Target = pcd mean (x,y), pin normal[2] to floor

    Returns: (new_pts, new_angle, success_bool, status_text, images_list)
    """
    from PIL import Image
    from src.habitat import pos_habitat_to_normal, pos_normal_to_habitat, pose_habitat_to_tsdf
    from src.conceptgraph.slam.utils import (
        detections_to_obj_pcd_and_bbox,
        init_process_pcd,
        get_bounding_box,
    )
    import open3d as o3d

    images = []
    cam_intr = cam_intr_ext if cam_intr_ext is not None else scene.cam_intrinsic
    cfg_cg = scene.cfg_cg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    floor_height = float(pts[1])
    cfg = cfg_ext if cfg_ext is not None else scene.cfg

    # ── Phase A: GD detect + SAM + 3D back-project → target_normal ──
    # Use VLM-selected view (view_angle / view_cam_pose)
    obs, cam_pose_habitat = scene.get_observation(pts, view_angle)
    rgb = obs["color_sensor"]
    depth = obs["depth_sensor"]

    bbox, phrase, score = _gd_detect(rgb, object_desc, gd_model)
    if bbox is None:
        return pts, angle, False, f"GD no detection for '{object_desc}'", images

    logging.info(f"  GD: '{phrase}' score={score:.3f}")
    images.append(("gd_detection", rgb.copy()))

    # SAM mask
    try:
        sam_out = scene.sam_predictor.predict(
            rgb, bboxes=[bbox.tolist()], verbose=False)
        mask = sam_out[0].masks.data.cpu().numpy()[0].astype(bool)
    except Exception as e:
        logging.warning(f"  GD: SAM failed: {e}, using bbox as mask")
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        x1, y1, x2, y2 = bbox.astype(int)
        mask[y1:y2, x1:x2] = True

    # 3D back-projection — STRICTLY follow debug_iterative_spiral_navigate.py
    # CRITICAL FIX: use TSDF pose + z_forward convention (not raw Habitat pose)
    cam_pose_tsdf = pose_habitat_to_tsdf(cam_pose_habitat)
    try:
        obj_list = detections_to_obj_pcd_and_bbox(
            depth_array=depth,
            masks=mask[None, :, :].astype(np.float32),
            cam_K=cam_intr,
            image_rgb=rgb,
            trans_pose=cam_pose_tsdf,            # TSDF pose (FIXED)
            camera_convention="z_forward",        # matches TSDF pose (FIXED)
            min_points_threshold=5,
            spatial_sim_type=cfg_cg.spatial_sim_type,
            obj_pcd_max_points=cfg_cg.obj_pcd_max_points,
            downsample_voxel_size=cfg_cg.get('downsample_voxel_size',
                cfg_cg.get('downsample_voxcel_size', 0.02))
                if isinstance(cfg_cg, dict)
                else getattr(cfg_cg, 'downsample_voxel_size',
                    getattr(cfg_cg, 'downsample_voxcel_size', 0.02)),
            device=device,
        )
    except Exception as e:
        logging.warning(f"  GD: back-project failed: {e}")
        return pts, angle, False, f"Back-projection failed: {e}", images

    if not obj_list or obj_list[0] is None:
        return pts, angle, False, "Back-projection returned None", images

    obj = obj_list[0]
    pcd = obj["pcd"]
    if len(pcd.points) == 0:
        return pts, angle, False, "Empty point cloud", images

    # pcd_np is already in normal coords (because we used TSDF pose + z_forward)
    pcd_np = np.asarray(pcd.points)
    logging.info(f"  GD: back-projected {pcd_np.shape[0]} points")

    # Z-clip: pin points above 3.0m to floor height (normal[2] is height)
    z_clip = cfg.tsdf_grid_size * 30  # 3.0m
    pcd_np_clipped = pcd_np.copy()
    over_z = pcd_np_clipped[:, 2] > z_clip
    if over_z.any():
        logging.info(f"  GD: z-clipping {int(over_z.sum())}/{len(pcd_np)} points")
        pcd_np_clipped[over_z, 2] = floor_height

    # Target = pcd mean (x,y more accurate than OBB center which z-clip pulls)
    target_normal = pcd_np_clipped.mean(axis=0)
    target_normal[2] = floor_height  # pin height to floor

    target_voxel = tsdf_planner.normal2voxel(target_normal)
    target_voxel_xy = (int(target_voxel[0]), int(target_voxel[1]))
    logging.info(
        f"  GD: target normal={target_normal.tolist()} "
        f"voxel={target_voxel.tolist()}")

    # ── Phase B placeholder — implemented in Task 4 ──
    # (return early for now; Task 4 will add the iterative spiral loop)
    return pts, angle, True, f"GD detected '{phrase}' at voxel {target_voxel.tolist()}", images
```

**Step 3: Verify imports compile**

Run: `cd .worktrees/fix-navigation && python -c "from src.scene_aeqa import grounded_navigate_to_object; print('OK')"`
Expected: `OK` (no ImportError)

**Step 4: Commit**

```bash
cd .worktrees/fix-navigation
git add src/scene_aeqa.py
git commit -m "fix: rewrite grounded_navigate_to_object Phase A with correct back-projection

- Use pose_habitat_to_tsdf(cam_pose) instead of raw Habitat cam_pose
- Explicitly pass camera_convention='z_forward' to match TSDF pose
- Result pcd is already in normal coords (remove pos_habitat_to_normal)
- Z-clip normal[2]>3.0m to floor, use pcd mean, pin normal[2] to floor

Phase B (iterative spiral loop) added in next commit."
```

---

### Task 4: Implement Phase B (iterative spiral search navigation loop)

**Files:**
- Modify: `.worktrees/fix-navigation/src/scene_aeqa.py` (replace the Phase B placeholder from Task 3)

**Step 1: Replace the Phase B placeholder**

In `grounded_navigate_to_object`, find the placeholder:
```python
    # ── Phase B placeholder — implemented in Task 4 ──
    return pts, angle, True, f"GD detected '{phrase}' at voxel {target_voxel.tolist()}", images
```

Replace with:

```python
    # ── Phase B: Iterative spiral search + per-step navigation ──
    # Port from debug_iterative_spiral_navigate.py lines 661-906
    # Per HM-GE stage-3 notes (迭代螺旋搜索多段导航.md):
    #   1. Refresh grids, check if target on island
    #   2. Spiral search nearest navigable point to target
    #   3. set_next_navigation_point(target_type="image", ...)
    #   4. Per-step: agent_step → silent_perception → refresh → update_frontier
    #   5. On arrival: if converged → done; if not converged → re-spiral (continue)
    from src.agent_tools import silent_perception_step as gd_silent_step

    cur_pts, cur_angle = pts.copy(), angle
    converged = False
    arrived_any = False
    map_h, map_w = tsdf_planner._tsdf_vol_cpu.shape[:2]
    max_spiral_radius = max(map_h, map_w)

    for iteration in range(1, max_iterations + 1):
        if step_budget is not None and gd_silent_step._step_counter >= step_budget:
            logging.info(f"  GD iter {iteration}: step budget exhausted, stopping")
            break

        logging.info(f"  GD iter {iteration}/{max_iterations}, pts={cur_pts.tolist()}")

        # 1. Refresh grids from current TSDF state
        tsdf_planner.refresh_planner_grids(cur_pts)
        island_sum = int(tsdf_planner.island.sum()) if tsdf_planner.island is not None else 0
        logging.info(f"  GD iter {iteration}: island={island_sum} voxels")

        # 2. Check if target is already navigable (on island)
        spiral_result = None
        tv_y, tv_x = target_voxel_xy
        if (0 <= tv_y < map_h and 0 <= tv_x < map_w
                and tsdf_planner.island[tv_y, tv_x]):
            logging.info(f"  GD iter {iteration}: target on island, converging")
            normal_pos = tsdf_planner.voxel2normal(np.array([tv_y, tv_x]))
            normal_3d = np.array([normal_pos[0], normal_pos[1], floor_height])
            hab_pos = pos_normal_to_habitat(normal_3d)
            snapped = scene.pathfinder.snap_point(hab_pos[:3])
            if snapped is not None and not np.isnan(snapped).any():
                spiral_result = {
                    "habitat_pos": snapped,
                    "voxel_xy": target_voxel_xy,
                    "search_steps": 0,
                    "spiral_dist": 0,
                }
                converged = True

        # 3. Spiral search (if not already on island)
        if spiral_result is None:
            spiral_result = tsdf_planner.spiral_search_navigable_point(
                pathfinder=scene.pathfinder,
                target_voxel_xy=target_voxel_xy,
                agent_habitat=cur_pts,
                max_radius_voxels=max_spiral_radius,
                floor_height=floor_height,
            )
            if spiral_result is None:
                logging.warning(f"  GD iter {iteration}: spiral found nothing")
                break

        # 4. Convergence check
        spiral_dist = spiral_result["spiral_dist"]
        logging.info(
            f"  GD iter {iteration}: spiral dist={spiral_dist} "
            f"voxel={spiral_result['voxel_xy']}")
        if spiral_dist <= converge_dist_voxels:
            converged = True

        # 5. Set navigation point — use production code's target_type="image"
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None
        pathfinder_target = np.array(spiral_result["habitat_pos"])
        set_ok = tsdf_planner.set_next_navigation_point(
            target_type="image",
            choice=pathfinder_target,
            pts=cur_pts.tolist(),
            objects=None, obs_points=None,
            cfg=cfg.planner, pathfinder=scene.pathfinder,
            random_position=False, observe_snapshot=False,
        )
        if not set_ok:
            logging.error(f"  GD iter {iteration}: set_next_navigation_point failed")
            break

        # 6. Per-step navigation loop
        arrived = False
        for nav_step in range(1, max_nav_steps_per_iter + 1):
            if step_budget is not None and gd_silent_step._step_counter >= step_budget:
                break

            result = tsdf_planner.agent_step(
                pts=cur_pts, angle=cur_angle,
                objects=scene.objects, snapshots=scene.snapshots,
                pathfinder=scene.pathfinder, cfg=cfg.planner,
                save_visualization=False,
            )
            if result[0] is None:
                logging.warning(f"  GD iter {iteration} step {nav_step}: agent_step failed")
                break

            cur_pts, cur_angle, _, _, _, target_arrived = result

            # Per-step silent perception (scene graph + TSDF + snapshot)
            gd_silent_step(
                scene, tsdf_planner, cur_pts, cur_angle,
                cnt_step_base + iteration * max_nav_steps_per_iter + nav_step,
                memory_store, cam_intr, cfg,
                detection_model, sam_predictor,
                clip_model, clip_preprocess, clip_tokenizer,
            )

            # Refresh grids (map grows with new observations)
            tsdf_planner.refresh_planner_grids(cur_pts)

            # Update frontier map (includes room segmentation)
            try:
                tsdf_planner.update_frontier_map(
                    cur_pts, cfg.planner, scene,
                    cnt_step_base + iteration * max_nav_steps_per_iter + nav_step,
                    save_frontier_image=False,
                )
            except Exception as e:
                logging.warning(f"  GD iter {iteration} step {nav_step}: update_frontier_map failed: {e}")

            logging.info(
                f"  GD iter {iteration} step {nav_step}: "
                f"voxel={tsdf_planner.habitat2voxel(cur_pts)[:2].tolist()} "
                f"arrived={target_arrived}")

            if target_arrived:
                arrived = True
                break

        arrived_any = arrived or arrived_any

        # Clear navigation state for next iteration
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None

        # 7. Iteration termination logic
        if converged and arrived:
            logging.info(f"  GD: converged and arrived at iteration {iteration}")
            break
        if arrived and not converged:
            logging.info(f"  GD iter {iteration}: arrived but not converged, re-spiraling")
            continue
        logging.info(f"  GD iter {iteration}: didn't arrive, continuing from current pos")

    status = f"GD nav: {'converged' if converged else 'not converged'}, arrived={arrived_any}"
    return cur_pts, cur_angle, arrived_any, status, images
```

**Step 2: Verify imports compile**

Run: `cd .worktrees/fix-navigation && python -c "from src.scene_aeqa import grounded_navigate_to_object; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
cd .worktrees/fix-navigation
git add src/scene_aeqa.py
git commit -m "feat: add iterative spiral search navigation loop to grounded_navigate_to_object

Port from debug_iterative_spiral_navigate.py lines 661-906:
- Refresh grids → check target on island → spiral search
- set_next_navigation_point(target_type='image', ...) — uses production code
- Per-step: agent_step → silent_perception → refresh_grids → update_frontier
- Arrival + not converged → continue (re-spiral with expanded map)
- Step budget awareness"
```

---

### Task 5: Update `navigate_to_object` signature in `agent_tools.py`

**Files:**
- Modify: `.worktrees/fix-navigation/src/agent_tools.py:266-308`

**Step 1: Update the function signature and body**

Replace `navigate_to_object` (lines 266-308) with:

```python
def navigate_to_object(
    scene, tsdf_planner, pts, angle,
    view_idx, view_angle, view_cam_pose, object_desc,
    memory_store, cam_intr, cfg, detection_model, sam_predictor,
    clip_model, clip_preprocess, clip_tokenizer, cnt_step,
    max_steps=20, step_budget=None, gd_model=None,
) -> Tuple[np.ndarray, np.ndarray, bool, str, Optional[str]]:
    """GD 导航到指定物体。返回 (pts, angle, success, status, img_b64)。

    视角由 VLM 选定（view_idx + view_angle + view_cam_pose）。
    GD 检测使用该视角，不做方向扫描。
    step_budget 用于限制导航步数，避免超出总步数配额。
    """
    from src.scene_aeqa import grounded_navigate_to_object as gd_nav
    from src.agent_image_utils import numpy_to_base64

    max_nav = 15
    max_iter = 5
    if step_budget is not None:
        max_nav = min(max_nav, max(1, step_budget))
        max_iter = min(max_iter, max(1, step_budget // 3))

    new_pts, new_angle, success, status, _images = gd_nav(
        scene, tsdf_planner, pts, angle,
        view_idx=view_idx, view_angle=view_angle, view_cam_pose=view_cam_pose,
        object_desc=object_desc,
        max_consecutive_failures=5,
        max_iterations=max_iter, converge_dist_voxels=5,
        max_nav_steps_per_iter=max_nav,
        memory_store=memory_store, cam_intr_ext=cam_intr, cfg_ext=cfg,
        detection_model=detection_model, sam_predictor=sam_predictor,
        clip_model=clip_model, clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        cnt_step_base=cnt_step, step_budget=step_budget,
        gd_model=gd_model,
    )

    # GD 导航内部每子步已做 silent_perception + refresh + update_frontier
    # 这里只返回当前视角图像给 VLM
    obs, _ = scene.get_observation(new_pts, new_angle)
    img_b64 = numpy_to_base64(obs["color_sensor"][..., :3])

    return new_pts, new_angle, success, status, img_b64
```

**Step 2: Verify imports compile**

Run: `cd .worktrees/fix-navigation && python -c "from src.agent_tools import navigate_to_object; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
cd .worktrees/fix-navigation
git add src/agent_tools.py
git commit -m "refactor: update navigate_to_object signature to accept VLM-selected view

- Add view_idx, view_angle, view_cam_pose, gd_model params
- GD detection now uses VLM-selected view (no direction scanning)
- Pass through to grounded_navigate_to_object"
```

---

## Phase C: Panorama & Seed Views

### Task 6: Modify `observe_panorama` to 8 views with directional labels

**Files:**
- Modify: `.worktrees/fix-navigation/src/agent_tools.py:165-264` (observe_panorama)

**Step 1: Rewrite `observe_panorama`**

Replace `observe_panorama` (lines 165-264) with:

```python
def observe_panorama(
    scene, tsdf_planner, pts, angle, cnt_step,
    memory_store, cam_intr, cfg, detection_model,
    sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
) -> Tuple[np.ndarray, np.ndarray, str, str, list]:
    """8 视角全景观测，返回 (pts, angle, mosaic_b64, text, panorama_views)。

    8 视角：前/右前/右/右后/后/左后/左/左前（相对 agent 朝向，顺时针每 45°）
    拼图布局：3×3 网格，中心是方位指南针
    """
    from src.agent_image_utils import make_mosaic, numpy_to_base64
    import matplotlib.pyplot as plt

    DIRECTIONS = ["前", "右前", "右", "右后", "后", "左后", "左", "左前"]
    # 顺时针每 45°，view_idx 0 = agent 当前朝向 = "前"
    angles = [angle + i * 2 * np.pi / 8 for i in range(8)]

    panorama_views = []
    views_rgb = []
    for i, ang in enumerate(angles):
        obs, cam_pose = scene.get_observation(pts, ang)
        rgb = obs["color_sensor"][..., :3]
        views_rgb.append(rgb)
        panorama_views.append({
            "view_idx": i,
            "direction": DIRECTIONS[i],
            "angle": float(ang),
            "cam_pose": cam_pose,
            "rgb": rgb,
        })

    # 静默执行感知（3视角 + TSDF + 场景图更新；snapshot 由下方 8 视角存档）
    silent_perception_step(
        scene, tsdf_planner, pts, angle, cnt_step, memory_store,
        cam_intr, cfg, detection_model, sam_predictor,
        clip_model, clip_preprocess, clip_tokenizer,
        skip_snapshots=True,
    )

    # 保存全景 8 张视角到 MemoryStore
    room_id = tsdf_planner.get_room_id_at(
        tsdf_planner.habitat2voxel(pts)[:2])
    step_id = silent_perception_step._step_counter
    for ang_idx, view_rgb in enumerate(views_rgb):
        objs_in_view = [
            scene.objects[oid]["class_name"]
            for oid in scene.objects
            if np.linalg.norm(
                scene.objects[oid]["bbox"].center[[0, 2]] - pts[[0, 2]]
            ) < cfg.scene_graph.obj_include_dist + 0.5
        ]
        memory_store.add_snapshot(
            snapshot_id=f"pano_step{step_id}_view{ang_idx}",
            image=view_rgb,
            room_id=room_id,
            objects_in_view=objs_in_view,
            position_3d=pts.tolist(),
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
        )

    # 构建 3×3 拼图（中心是方位指南针）
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    ax_ord = np.array([[7, 0, 1], [6, -1, 2], [5, 4, 3]])
    for row in range(3):
        for col in range(3):
            idx = ax_ord[row, col]
            ax = axes[row, col]
            if idx == -1:
                # 中心格：方位指南针
                ax.axis('off')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                cx, cy = 0.5, 0.5
                al = 0.25
                ax.annotate('', xy=(cx, cy+al), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
                ax.annotate('', xy=(cx, cy-al), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
                ax.annotate('', xy=(cx-al, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
                ax.annotate('', xy=(cx+al, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
                ax.text(cx, cy+al+0.05, '前', ha='center', fontsize=12, fontweight='bold')
                ax.text(cx, cy-al-0.05, '后', ha='center', fontsize=12, fontweight='bold')
                ax.text(cx-al-0.05, cy, '左', va='center', fontsize=12, fontweight='bold')
                ax.text(cx+al+0.05, cy, '右', va='center', fontsize=12, fontweight='bold')
            else:
                ax.imshow(views_rgb[idx])
                ax.set_title(DIRECTIONS[idx], fontsize=11, fontweight='bold')
                ax.axis('off')

    fig.tight_layout()
    # Rasterize to numpy
    fig.canvas.draw()
    mosaic = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    mosaic = mosaic.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    mosaic_b64 = numpy_to_base64(mosaic)
    text = f"Panorama: 8 views (前/右前/右/右后/后/左后/左/左前) at step {cnt_step}"
    return pts, angle, mosaic_b64, text, panorama_views
```

**Step 2: Verify imports compile**

Run: `cd .worktrees/fix-navigation && python -c "from src.agent_tools import observe_panorama; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
cd .worktrees/fix-navigation
git add src/agent_tools.py
git commit -m "feat: 8-view panorama with directional labels and cam_pose storage

- 8 views (前/右前/右/右后/后/左后/左/左前) clockwise from agent heading
- 3x3 mosaic with compass rose in center (Spatial-X style)
- Store cam_pose per view for GD back-projection
- Return panorama_views list for downstream Stage 2 view selection"
```

---

### Task 7: Create `SeedViewManager` in `src/seed_views.py`

**Files:**
- Create: `.worktrees/fix-navigation/src/seed_views.py`
- Test: `.worktrees/fix-navigation/tests/test_seed_views.py`

**Step 1: Write failing test**

```python
# tests/test_seed_views.py
import numpy as np
from src.seed_views import SeedViewManager


class _FakeScene:
    def get_observation(self, pts, angle):
        return {"color_sensor": np.full((100, 100, 3), int(np.degrees(angle)) % 256, dtype=np.uint8)}, None


class _FakePlanner:
    _voxel_size = 0.1
    min_height_voxel = 0
    _tsdf_vol_cpu = np.zeros((20, 20, 40), dtype=np.float32)

    def habitat2voxel(self, pos):
        return np.array(pos, dtype=int)


def test_register_seed_renders_image():
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    mgr.register_seed(1, np.array([5.0, 0.0, 5.0]), scene, planner,
                      np.array([0.0, 0.0, 0.0]))
    assert 1 in mgr.seeds
    assert mgr.seeds[1]["image"] is not None
    assert mgr.seeds[1]["image"].shape == (100, 100, 3)
    assert np.array_equal(mgr.seeds[1]["view_image_pos"], [0.0, 0.0, 0.0])


def test_update_after_step_no_change_when_dist_increases():
    """Agent moves away from seed → no update."""
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    mgr.register_seed(1, np.array([5.0, 0.0, 0.0]), scene, planner,
                      np.array([0.0, 0.0, 0.0]))  # dist=5
    original_image = mgr.seeds[1]["image"].copy()
    # Agent moves away (dist=8)
    mgr.update_after_step([1], np.array([8.0, 0.0, 0.0]), planner, scene)
    assert np.array_equal(mgr.seeds[1]["image"], original_image)


def test_update_after_step_updates_when_dist_decreases():
    """Agent moves closer to seed → update."""
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    mgr.register_seed(1, np.array([5.0, 0.0, 0.0]), scene, planner,
                      np.array([0.0, 0.0, 0.0]))  # dist=5
    original_image = mgr.seeds[1]["image"].copy()
    # Agent moves closer (dist=2)
    mgr.update_after_step([1], np.array([3.0, 0.0, 0.0]), planner, scene)
    assert not np.array_equal(mgr.seeds[1]["image"], original_image)


def test_get_mosaic_returns_image():
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    for sid in [1, 2, 3]:
        mgr.register_seed(sid, np.array([5.0, 0.0, float(sid)]),
                          scene, planner, np.array([0.0, 0.0, 0.0]))
    mosaic = mgr.get_mosaic("test question")
    assert mosaic is not None
    assert mosaic.ndim == 3  # H, W, 3
```

**Step 2: Run test to verify it fails**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_seed_views.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.seed_views'`

**Step 3: Write implementation**

```python
# src/seed_views.py
"""Seed view manager: lazy rendering of seed direction images.

Each seed has ONE representative image. Agent cannot see through walls,
cannot teleport to seed. Image updates only when:
1. Agent moves closer to seed (Euclidean distance decreased)
2. No tall obstacle (>=1.2m) blocks the ray agent→seed
"""
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


class SeedViewManager:
    """Manages seed direction images with lazy updates."""

    def __init__(self):
        self.seeds: Dict[int, dict] = {}
        # Each entry: {"image": np.ndarray, "view_image_pos": np.ndarray,
        #              "view_image_angle": float, "seed_position": np.ndarray}

    def register_seed(self, seed_id: int, position: np.ndarray,
                      scene, tsdf_planner, agent_pts: np.ndarray):
        """Register a new seed and render its initial view image.

        Renders the view from agent's current position toward the seed.
        """
        angle_to_seed = math.atan2(
            position[0] - agent_pts[0],
            position[2] - agent_pts[2])
        obs, _ = scene.get_observation(agent_pts, angle_to_seed)
        self.seeds[seed_id] = {
            "image": obs["color_sensor"][..., :3],
            "view_image_pos": agent_pts.copy(),
            "view_image_angle": float(angle_to_seed),
            "seed_position": position.copy(),
        }
        logging.info(f"  SeedViewManager: registered seed {seed_id} at {position.tolist()}")

    def update_after_step(self, active_seed_ids: List[int],
                          cur_pts: np.ndarray, tsdf_planner, scene,
                          min_blocking_height: float = 1.2):
        """Check all seeds and update images if conditions met.

        Conditions for update:
        1. Ray agent→seed not blocked by tall obstacles
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
                seed_pos[0] - cur_pts[0],
                seed_pos[2] - cur_pts[2])
            obs, _ = scene.get_observation(cur_pts, angle_to_seed)
            seed["image"] = obs["color_sensor"][..., :3]
            seed["view_image_pos"] = cur_pts.copy()
            seed["view_image_angle"] = float(angle_to_seed)
            logging.info(f"  SeedViewManager: updated seed {seed_id} "
                        f"(dist {last_dist:.2f}→{cur_dist:.2f})")

    def get_mosaic(self, question: str, max_seeds: int = 8) -> Optional[np.ndarray]:
        """Build a mosaic of all seed images with seed_id labels.

        Returns RGB numpy array, or None if no seeds.
        """
        if not self.seeds:
            return None

        seeds = list(self.seeds.items())[:max_seeds]
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
        fig.canvas.draw()
        mosaic = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        mosaic = mosaic.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return mosaic

    def get_unexplored_seed_ids(self, explored_seed_ids: set) -> List[int]:
        """Return seed IDs not in the explored set."""
        return [sid for sid in self.seeds if sid not in explored_seed_ids]
```

**Step 4: Run tests**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_seed_views.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd .worktrees/fix-navigation
git add src/seed_views.py tests/test_seed_views.py
git commit -m "feat: add SeedViewManager for lazy seed direction image updates

- register_seed: render initial view toward seed at registration time
- update_after_step: re-render only if agent closer and no tall obstacle
- Height-aware occlusion check (1.2m threshold) — tables don't block
- 0.1m anti-jitter threshold
- get_mosaic: build labeled mosaic for VLM seed selection
- get_unexplored_seed_ids: filter explored seeds"
```

---

## Phase D: VLM Workflow Refactor

### Task 8: Update `_parse_vlm_response` to enforce `reason` field

**Files:**
- Modify: `.worktrees/fix-navigation/src/agent_workflow.py:684+`
- Test: `.worktrees/fix-navigation/tests/test_vlm_parse.py`

**Step 1: Write failing test**

```python
# tests/test_vlm_parse.py
import json
from src.agent_workflow import _parse_vlm_response


def test_parse_with_reason():
    """Valid response with reason field."""
    resp = json.dumps({
        "reason": "I see an oven in view 3",
        "action": "navigate_to_object",
        "view_idx": 3,
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "navigate_to_object"
    assert parsed["reason"] == "I see an oven in view 3"
    assert parsed["view_idx"] == 3


def test_parse_missing_reason():
    """Response without reason → flagged as missing_reason."""
    resp = json.dumps({"action": "explore_other_room"})
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "missing_reason"


def test_parse_submit_answer():
    resp = json.dumps({
        "reason": "I can see the towel on the oven handle",
        "action": "submit_answer",
        "answer": "yes",
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "submit_answer"
    assert parsed["answer"] == "yes"


def test_parse_explore_seed():
    resp = json.dumps({
        "reason": "seed 2 is toward the kitchen",
        "action": "explore_seed",
        "seed_id": 2,
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "explore_seed"
    assert parsed["seed_id"] == 2


def test_parse_explore_frontier():
    resp = json.dumps({
        "reason": "all seeds are bedrooms, fallback to frontier",
        "action": "explore_frontier",
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "explore_frontier"


def test_parse_frontier_selection():
    resp = json.dumps({
        "reason": "frontier 0 leads to unexplored hallway",
        "frontier_id": 0,
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "explore_frontier"
    assert parsed["frontier_id"] == 0


def test_parse_object_selection():
    resp = json.dumps({
        "reason": "stainless steel appliance with square door",
        "object": "oven",
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "object_selected"
    assert parsed["object"] == "oven"


def test_parse_invalid_json():
    parsed = _parse_vlm_response("not json at all")
    assert parsed["tool"] == "parse_error"
```

**Step 2: Run test to verify it fails**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_vlm_parse.py -v`
Expected: FAIL (existing `_parse_vlm_response` doesn't enforce reason)

**Step 3: Read current `_parse_vlm_response`**

Run: `cd .worktrees/fix-navigation && sed -n '684,730p' src/agent_workflow.py`

**Step 4: Rewrite `_parse_vlm_response`**

Replace the existing `_parse_vlm_response` function with:

```python
def _parse_vlm_response(response: str) -> dict:
    """Parse VLM JSON response. Enforce mandatory 'reason' field.

    Returns dict with at least:
        - tool: str (action name, or 'parse_error'/'missing_reason')
        - reason: str (may be empty if missing)
        - raw: str (original response, only on error)
    """
    import json as _json

    # Try to extract JSON from response (VLM may add prose around it)
    text = response.strip()
    # Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return {"tool": "parse_error", "reason": "", "raw": response}

    try:
        parsed = _json.loads(text[start:end + 1])
    except _json.JSONDecodeError:
        return {"tool": "parse_error", "reason": "", "raw": response}

    # Enforce reason field
    reason = parsed.get("reason", "").strip()
    if not reason:
        return {"tool": "missing_reason", "reason": "", "raw": response}

    parsed["reason"] = reason

    # Determine tool from action or frontier_id presence
    if "frontier_id" in parsed:
        parsed["tool"] = "explore_frontier"
    elif "object" in parsed and "action" not in parsed:
        parsed["tool"] = "object_selected"
    else:
        parsed["tool"] = parsed.get("action", "")

    return parsed
```

**Step 5: Run tests**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_vlm_parse.py -v`
Expected: PASS (8 tests)

**Step 6: Commit**

```bash
cd .worktrees/fix-navigation
git add src/agent_workflow.py tests/test_vlm_parse.py
git commit -m "feat: enforce mandatory reason field in VLM response parsing

- _parse_vlm_response now flags missing_reason when reason absent
- Handle all action types: navigate_to_object, explore_seed, explore_frontier,
  submit_answer, object_selected
- Robust JSON extraction (VLM may add prose around JSON)
- Returns parse_error for invalid JSON"
```

---

### Task 9: Rewrite Stage prompts with JSON schema (reason required)

**Files:**
- Modify: `.worktrees/fix-navigation/src/agent_workflow.py:154-227` (prompt templates)

**Step 1: Replace all STAGE*_PROMPT templates**

Replace lines 154-227 with:

```python
# ── VLM Output Schema (shared across stages) ──
SCHEMA_REQUIREMENT = """
你必须输出以下 JSON 格式（不要输出其他内容）：
{
  "reason": "<一句话解释为什么做这个选择，必须包含你观察到的具体视觉线索>",
  ...action-specific fields...
}

reason 字段要求：
- 必须包含你从图片中观察到的具体视觉线索（如"view3 中看到不锈钢家电"）
- 必须解释该选择如何帮助回答问题
- 不允许输出"我决定..."等空泛表述，必须有具体依据
"""

STAGE1_PROMPT = """Stage 1: Initial Exploration

You are at the starting position. Call observe_panorama to look around.
Based on the panorama, describe what you see and which direction is most promising.

Question: "{question}"
"""

STAGE2_PROMPT = """Stage 2: Main Direction Decision

Look at the 8-view panorama above. The views are labeled:
  view0=前 view1=右前 view2=右 view3=右后 view4=后 view5=左后 view6=左 view7=左前

For the question: "{question}"

Decide:
- If you see a relevant object in one of the views → navigate_to_object with view_idx
- If no relevant object visible in any view → explore_other_room

{SCHEMA_REQUIREMENT}

Actions:
1. navigate_to_object: {{"reason": "...", "action": "navigate_to_object", "view_idx": <0-7>}}
2. explore_other_room: {{"reason": "...", "action": "explore_other_room"}}
"""

STAGE2_5A_PROMPT = """Stage 2.5a: Seed Selection

You decided to explore other rooms. Here are the available unexplored seeds
(each image shows the view from your current position toward that seed):

{seed_info}

For the question: "{question}"

Decide:
- If a seed seems relevant → explore_seed with seed_id
- If all seeds seem irrelevant → explore_frontier (fallback)

{SCHEMA_REQUIREMENT}

Actions:
1. explore_seed: {{"reason": "...", "action": "explore_seed", "seed_id": <id>}}
2. explore_frontier: {{"reason": "...", "action": "explore_frontier"}}
"""

STAGE3_PROMPT = """Stage 3: Object Selection

You selected view_idx {view_idx}. Here is the large image of that view.

For the question: "{question}"

You MUST output ONE concrete physical object name visible in this image that
will serve as your navigation anchor. The object must be:
- A concrete noun phrase a detector can find (e.g. "oven", "the red door", "towel")
- NOT a room name, direction, or abstract concept

{SCHEMA_REQUIREMENT}

Output: {{"reason": "...", "object": "<object_name>"}}
"""

STAGE5_PROMPT = """Stage 5: Re-decision After Arrival

You've arrived near the target. Here are the 3 frontal views from your
current position (left 60°, front, right 60°):
  view0=left view1=front view2=right

For the question: "{question}"

Decide:
- If you can answer the question now → submit_answer
- If you see a new relevant object in one of the 3 views → navigate_to_object
- If you need to explore other rooms → explore_other_room

{SCHEMA_REQUIREMENT}

Actions:
1. navigate_to_object: {{"reason": "...", "action": "navigate_to_object", "view_idx": <0-2>}}
2. explore_other_room: {{"reason": "...", "action": "explore_other_room"}}
3. submit_answer: {{"reason": "...", "action": "submit_answer", "answer": "<your_answer>"}}
"""

STAGE6_PROMPT = """Stage 6: Frontier Selection

You decided to explore frontiers. Here are all available frontiers:

{frontier_info}

For the question: "{question}"

Select the most promising frontier to explore next.

{SCHEMA_REQUIREMENT}

Output: {{"reason": "...", "frontier_id": <id>}}
"""
```

**Step 2: Verify imports compile**

Run: `cd .worktrees/fix-navigation && python -c "from src.agent_workflow import STAGE2_PROMPT; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
cd .worktrees/fix-navigation
git add src/agent_workflow.py
git commit -m "feat: rewrite stage prompts with JSON schema and mandatory reason field

- All stages require JSON output with 'reason' field
- Stage 2: binary decision (navigate_to_object vs explore_other_room)
- Stage 2.5a: seed selection or frontier fallback
- Stage 3: object name selection from large view image
- Stage 5: re-decision with 3 frontal views (navigate/explore/submit)
- Stage 6: frontier_id selection
- SCHEMA_REQUIREMENT shared constant enforces reason quality"
```

---

### Task 10-14: Stage state machine implementations

**Note:** Tasks 10-14 implement the 6-stage state machine in `agent_workflow.py:run_episode`. Due to tight coupling, these are implemented together in one commit.

**Files:**
- Modify: `.worktrees/fix-navigation/src/agent_workflow.py:232+` (run_episode)

**Step 1: Rewrite `run_episode` with 6-stage state machine**

This is a large rewrite. Replace the existing `run_episode` function (from line 232 onwards) with the new state machine. The full implementation follows the design in `docs/plans/2026-06-23-navigation-fix-design.md` section 2.1.

Key structure:
```python
def run_episode(...):
    # ... setup (same as before) ...

    # Stage 1: Panorama
    pts, angle, mosaic_b64, _, panorama_views = observe_panorama(...)
    # Store panorama_views for Stage 2 view_idx lookup

    # Stage 2-6 loop
    current_stage = 2
    while current_stage != "done":
        if current_stage == 2:
            # VLM call 1: main direction decision
            # Input: 8-view mosaic
            # Output: navigate_to_object + view_idx, OR explore_other_room
            ...
        elif current_stage == "2.5a":
            # VLM call 2: seed selection
            # Input: seed mosaic from SeedViewManager
            # Output: explore_seed + seed_id, OR explore_frontier
            ...
        elif current_stage == 3:
            # VLM call 3: object selection
            # Input: large image of selected view_idx
            # Output: object name
            # Then: GD navigation (Stage 4 inline)
            ...
        elif current_stage == 5:
            # VLM call 4: re-decision after arrival
            # Input: 3 frontal views
            # Output: navigate_to_object, explore_other_room, or submit_answer
            ...
        elif current_stage == 6:
            # VLM call 5: frontier selection
            # Input: frontier mosaic
            # Output: frontier_id
            # Then: navigate_to_frontier, back to Stage 5
            ...

    return result
```

**Step 2: Implement each stage**

Refer to the design doc `docs/plans/2026-06-23-navigation-fix-design.md` section 2.1 for the full state machine logic. Key transitions:
- Stage 2 → Stage 3 (navigate_to_object) or Stage 2.5a (explore_other_room)
- Stage 2.5a → navigate_to_seed then Stage 5, or Stage 6 (explore_frontier)
- Stage 3 → Stage 4 (GD nav inline) → Stage 5
- Stage 5 → Stage 3 (navigate), Stage 2.5a (explore), or done (submit_answer)
- Stage 6 → navigate_to_frontier → Stage 5

**Step 3: Verify imports compile**

Run: `cd .worktrees/fix-navigation && python -c "from src.agent_workflow import run_episode; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
cd .worktrees/fix-navigation
git add src/agent_workflow.py
git commit -m "feat: implement 6-stage state machine in run_episode

Stage 1: 8-view panorama (no VLM)
Stage 2: main direction decision (VLM call 1) → navigate_to_object or explore_other_room
Stage 2.5a: seed selection (VLM call 2) → explore_seed or explore_frontier
Stage 3: object selection (VLM call 3) → object name
Stage 4: GD navigation (code, no VLM) → iterative spiral search
Stage 5: re-decision after arrival (VLM call 4) → navigate/explore/submit
Stage 6: frontier selection (VLM call 5) → frontier_id

SeedViewManager integrated for lazy seed view updates.
All VLM calls enforce reason field in JSON output."
```

---

### Task 15: Integrate SeedViewManager into navigation flow

**Files:**
- Modify: `.worktrees/fix-navigation/src/agent_tools.py` (navigate_to_seed, navigate_to_frontier)
- Modify: `.worktrees/fix-navigation/src/tsdf_planner.py` (seed generation hook)

**Step 1: Add SeedViewManager to navigate_to_seed and navigate_to_frontier**

After each `agent_step` in `_navigate_to_target_with_agent_step`, call `seed_view_manager.update_after_step(...)`.

**Step 2: Hook into seed generation in tsdf_planner.py**

Find `_commit_room_regions` or wherever new seeds are created, and call `seed_view_manager.register_seed(...)`.

**Step 3: Pass SeedViewManager through run_episode**

Add `seed_view_manager` parameter to navigation tool functions, or use a module-level singleton.

**Step 4: Commit**

```bash
cd .worktrees/fix-navigation
git add src/agent_tools.py src/tsdf_planner.py
git commit -m "feat: integrate SeedViewManager into navigation flow

- Register seeds when generated in tsdf_planner._commit_room_regions
- Update seed views after each agent_step in _navigate_to_target_with_agent_step
- Pass SeedViewManager instance through run_episode"
```

---

### Task 16: End-to-end smoke test

**Files:**
- Create: `.worktrees/fix-navigation/tests/test_e2e_smoke.py`

**Step 1: Write smoke test**

```python
# tests/test_e2e_smoke.py
"""End-to-end smoke test: verify imports and basic structure.
Does NOT run Habitat (too slow for CI). Run full E2E manually on server.
"""
import pytest


def test_imports():
    """Verify all modules import cleanly."""
    from src.scene_aeqa import grounded_navigate_to_object
    from src.agent_tools import (
        navigate_to_object, observe_panorama,
        navigate_to_seed, navigate_to_frontier,
        silent_perception_step,
    )
    from src.agent_workflow import (
        run_episode, _parse_vlm_response,
        STAGE2_PROMPT, STAGE2_5A_PROMPT, STAGE3_PROMPT,
        STAGE5_PROMPT, STAGE6_PROMPT,
    )
    from src.seed_views import SeedViewManager
    from src.geom import bresenham_2d, check_ray_blocked
    print("All imports OK")


def test_navigate_to_object_signature():
    """Verify navigate_to_object accepts new params."""
    import inspect
    from src.agent_tools import navigate_to_object
    sig = inspect.signature(navigate_to_object)
    params = list(sig.parameters.keys())
    assert "view_idx" in params
    assert "view_angle" in params
    assert "view_cam_pose" in params
    assert "gd_model" in params


def test_grounded_navigate_signature():
    """Verify grounded_navigate_to_object accepts new params."""
    import inspect
    from src.scene_aeqa import grounded_navigate_to_object
    sig = inspect.signature(grounded_navigate_to_object)
    params = list(sig.parameters.keys())
    assert "view_idx" in params
    assert "view_angle" in params
    assert "view_cam_pose" in params
    assert "gd_model" in params


def test_observe_panorama_returns_views():
    """Verify observe_panorama returns panorama_views list."""
    import inspect
    from src.agent_tools import observe_panorama
    # Check return annotation mentions panorama_views or is a tuple of 5
    sig = inspect.signature(observe_panorama)
    ret = sig.return_annotation
    # Just verify it's annotated (don't enforce exact type)
    assert ret is not inspect.Parameter.empty
```

**Step 2: Run smoke test**

Run: `cd .worktrees/fix-navigation && python -m pytest tests/test_e2e_smoke.py -v`
Expected: PASS (4 tests)

**Step 3: Commit**

```bash
cd .worktrees/fix-navigation
git add tests/test_e2e_smoke.py
git commit -m "test: add E2E smoke test for import and signature verification"
```

---

### Task 17: Manual integration test on server

**This task is run manually on the server (root@8.147.163.63), not in CI.**

**Step 1: Push to remote**

```bash
cd .worktrees/fix-navigation
git push origin fix-navigation
```

**Step 2: On server, pull and run**

```bash
sshpass -p '<password>' ssh root@8.147.163.63 -p 59961
cd /root/MyAgent
git fetch origin
git checkout fix-navigation
git pull origin fix-navigation
```

**Step 3: Run the test script**

Use the existing `/tmp/test_nav.py` (from previous session) or create a new one that:
1. Loads scene `00824-Dd4bFSTQ8gi` with question `00c2be2a-1377-4fae-a889-30936b7890c3`
2. Renders 8-view panorama
3. Calls `grounded_navigate_to_object` with `view_idx=3, view_angle=<panorama view 3 angle>, view_cam_pose=<...>`
4. Verifies: GD back-projection produces DIFFERENT voxels for "oven" vs "door" (not both [156,0,2])
5. Verifies: agent moves >2m (not stuck at 0.5m)

**Step 4: Verify success criteria**

- GD back-projection: oven and door project to **different** voxels
- Navigation: agent executes >5 steps, displacement >2m
- Iterative spiral: search distance decreases (e.g., 37→4)
- Map growth: island/unoccupied voxels increase per iteration
- At least one `converged and arrived` iteration

---

## Verification Checklist

- [ ] `bresenham_2d` unit tests pass (5 tests)
- [ ] `check_ray_blocked` unit tests pass (4 tests)
- [ ] `SeedViewManager` unit tests pass (4 tests)
- [ ] `_parse_vlm_response` unit tests pass (8 tests)
- [ ] E2E smoke test passes (4 tests)
- [ ] Server integration: GD back-projection produces different voxels for different objects
- [ ] Server integration: agent moves >2m (not stuck at 0.5m)
- [ ] Server integration: iterative spiral search converges (distance decreases)

---

## Key References

- Design doc: `docs/plans/2026-06-23-navigation-fix-design.md`
- Verified working script: `MSGNav-main/tools/debug_render/debug_iterative_spiral_navigate.py`
- Spatial-X 8-view reference: `/home/afdsafg/spatialNav/Spatial-X/spatialx/mp3d_extensions/discrete_env.py:486-558`
- Obsidian notes:
  - `HM-GE科研开发日志/MSGNav-调试笔记-20260614-迭代螺旋搜索多段导航.md`
  - `HM-GE科研开发日志/MSGNav-调试笔记-20260614-GD-SAM-3D-2D-导航-完整管线技术总结.md`
