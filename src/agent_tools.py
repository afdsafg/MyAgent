"""HM-GE Agent 工具集。7 个 VLM 可调用的工具函数。

每 step 静默执行：3 视角观测 + YOLO/SAM/CLIP/3D + TSDF + 房间分割 + Snapshot 存档。
"""

import logging
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── 每 step 静默感知 ────────────────────────────────────────────────────

def silent_perception_step(
    scene, tsdf_planner, pts, angle, cnt_step, memory_store,
    cam_intr, cfg, detection_model, sam_predictor,
    clip_model, clip_preprocess, clip_tokenizer,
    skip_snapshots=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """每 step 静默执行：3 视角观测 + 全管线更新 + Snapshot 存档。

    按 plan §3，每个 step 都必须存档 snapshot（不按位移阈值过滤）。
    Returns: (new_pts, new_angle) — 不变，移动由调用方完成。
    """
    # step 计数器：每次调用都递增，保证 snapshot_id 唯一
    if not hasattr(silent_perception_step, '_step_counter'):
        silent_perception_step._step_counter = -1
        silent_perception_step._last_pos = None
    silent_perception_step._step_counter += 1
    silent_perception_step._last_pos = pts.tolist() if hasattr(pts, 'tolist') else list(pts)
    step_id = silent_perception_step._step_counter

    angles = [angle - np.pi / 3, angle, angle + np.pi / 3]
    all_added_obj_ids = []
    rgb_views = []

    for view_idx, ang in enumerate(angles):
        obs, cam_pose = scene.get_observation(pts, ang)
        rgb = obs["color_sensor"]
        depth = obs["depth_sensor"]
        obs_name = f"step{step_id}_view{view_idx}"

        annotated_rgb, added_obj_ids, _ = scene.update_scene_graph(
            image_rgb=rgb[..., :3], depth=depth,
            intrinsics=cam_intr, cam_pos=cam_pose,
            pts=pts, pts_voxel=tsdf_planner.habitat2voxel(pts),
            img_path=obs_name, frame_idx=step_id * 3 + view_idx,
            target_obj_mask=None,
        )
        all_added_obj_ids += added_obj_ids
        rgb_views.append(rgb[..., :3])

        from src.habitat import pose_habitat_to_tsdf
        tsdf_planner.integrate(
            color_im=rgb, depth_im=depth, cam_intr=cam_intr,
            cam_pose=pose_habitat_to_tsdf(cam_pose),
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * cfg.img_height),
            margin_w=int(cfg.margin_w_ratio * cfg.img_width),
            explored_depth=cfg.explored_depth,
        )

        scene.periodic_cleanup_objects(
            frame_idx=step_id * 3 + view_idx, pts=pts)

    # Snapshot 聚类更新（场景图层）
    scene.update_snapshots(
        obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection)

    # 每 step 存档到 MemoryStore（plan §3 要求每 step 都存档）
    if not skip_snapshots:
        room_id = tsdf_planner.get_room_id_at(
            tsdf_planner.habitat2voxel(pts)[:2])
        for i, view_rgb in enumerate(rgb_views):
            objs_in_view = [
                scene.objects[oid]["class_name"]
                for oid in scene.objects
                if np.linalg.norm(
                    scene.objects[oid]["bbox"].center[[0, 2]] - pts[[0, 2]]
                ) < cfg.scene_graph.obj_include_dist + 0.5
            ]
            memory_store.add_snapshot(
                snapshot_id=f"step{step_id}_view{i}",
                image=view_rgb,
                room_id=room_id,
                objects_in_view=objs_in_view,
                position_3d=pts.tolist(),
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
            )

    return pts, angle


# ── 导航辅助：循环 agent_step 直到抵达或超限 ────────────────────────────

def _navigate_to_target_with_agent_step(
    scene, tsdf_planner, pts, angle, choice, cfg,
    memory_store, cam_intr, detection_model, sam_predictor,
    clip_model, clip_preprocess, clip_tokenizer, cnt_step,
    max_substeps=25, step_budget=None,
    seed_view_manager=None, active_seed_ids=None,
) -> Tuple[np.ndarray, np.ndarray, bool, str, int]:
    """循环调用 set_next_navigation_point + agent_step 直到抵达目标。

    决策节奏：VLM 调用此函数后不参与每子步决策，只在抵达后由调用方唤醒 VLM。
    每个子步后执行 silent_perception_step（plan §3：每 step 存档）。
    step_budget: 剩余底层步数配额，超出后停止导航。
    seed_view_manager: optional SeedViewManager for lazy seed view updates.
    Returns: (final_pts, final_angle, arrived, status, substeps_taken)
    """
    # 确保上一次的导航状态被清空，避免 set_next 拒绝
    tsdf_planner.max_point = None
    tsdf_planner.target_point = None

    success = tsdf_planner.set_next_navigation_point(
        choice=choice, pts=pts, objects=scene.objects,
        cfg=cfg.planner, pathfinder=scene.pathfinder,
    )
    if not success:
        return pts, angle, False, "Failed to set navigation target", 0

    substeps = 0
    arrived = False
    cur_pts, cur_angle = pts, angle

    for substeps in range(1, max_substeps + 1):
        # 检查底层步数配额
        if step_budget is not None and silent_perception_step._step_counter >= step_budget:
            break

        result = tsdf_planner.agent_step(
            pts=cur_pts, angle=cur_angle, objects=scene.objects,
            snapshots=scene.snapshots, pathfinder=scene.pathfinder,
            cfg=cfg.planner, save_visualization=False,
        )
        if result[0] is None:
            tsdf_planner.max_point = None
            tsdf_planner.target_point = None
            return cur_pts, cur_angle, False, "agent_step failed", substeps - 1

        cur_pts, cur_angle, _, _, _, target_arrived = result

        # 每个子步都做静默感知并存档
        silent_perception_step(
            scene, tsdf_planner, cur_pts, cur_angle, cnt_step + substeps,
            memory_store, cam_intr, cfg, detection_model, sam_predictor,
            clip_model, clip_preprocess, clip_tokenizer,
        )

        # Update seed views if conditions met (lazy update)
        if seed_view_manager is not None and active_seed_ids:
            seed_view_manager.update_after_step(
                active_seed_ids, cur_pts, tsdf_planner, scene)

        if target_arrived:
            arrived = True
            break

    # 无论是否抵达都清空，确保下一次 set_next 可用
    tsdf_planner.max_point = None
    tsdf_planner.target_point = None

    status = "Arrived at target" if arrived else f"Stopped after {substeps} substeps"
    return cur_pts, cur_angle, arrived, status, substeps


# ── 7 个 VLM 工具 ───────────────────────────────────────────────────────

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

    DIRECTIONS = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
    # Clockwise every 45°, view_idx 0 = agent's current heading = "front"
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

    # 触发房间分割 + frontier 更新（关键：8 视角全景后必须调用，
    # 否则 room_regions 为空，SeedViewManager 注册不到任何 seed）
    # 先刷新 grid（update_frontier_map 内部会做，但如果 frontier 为空
    # 它会提前 return False，跳过 update_room_map，所以这里先手动刷新）
    try:
        from src.habitat import pos_habitat_to_normal
        import scipy.ndimage as _ndimage
        pts_normal = pos_habitat_to_normal(pts)
        island, unoccupied = tsdf_planner.get_island_around_pts(
            pts_normal, height=tsdf_planner.occupancy_height)
        tsdf_planner.unoccupied = unoccupied
        tsdf_planner.island = island
        tsdf_planner.unexplored = (np.sum(tsdf_planner._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in tsdf_planner.init_points:
            tsdf_planner.unexplored[point[0], point[1]] = 0
        tsdf_planner.occupied = np.logical_not(tsdf_planner.unoccupied).astype(int)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        tsdf_planner.unexplored_neighbors = _ndimage.convolve(
            tsdf_planner.unexplored, kernel, mode="constant", cval=0.0)
        tsdf_planner.occupied_map_camera = np.logical_not(
            tsdf_planner.get_island_around_pts(pts_normal, height=tsdf_planner.vision_height)[0])
    except Exception as e:
        logging.warning(f"observe_panorama: grid refresh failed: {e}")

    # 直接调用 update_room_map（不依赖 update_frontier_map，因为后者
    # 在 frontier 为空时会提前 return，跳过 room segmentation）
    try:
        tsdf_planner.update_room_map(cfg=cfg.planner, pts=pts_normal)
    except Exception as e:
        logging.warning(f"observe_panorama: update_room_map failed: {e}")

    # 也尝试 update_frontier_map（可能发现 frontier）
    try:
        tsdf_planner.update_frontier_map(
            pts, cfg.planner, scene, cnt_step,
            save_frontier_image=False)
    except Exception as e:
        logging.warning(f"observe_panorama: update_frontier_map failed: {e}")

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
                ax.text(cx, cy+al+0.05, 'Front', ha='center', fontsize=12, fontweight='bold')
                ax.text(cx, cy-al-0.05, 'Back', ha='center', fontsize=12, fontweight='bold')
                ax.text(cx-al-0.05, cy, 'Left', va='center', fontsize=12, fontweight='bold')
                ax.text(cx+al+0.05, cy, 'Right', va='center', fontsize=12, fontweight='bold')
            else:
                ax.imshow(views_rgb[idx])
                ax.set_title(DIRECTIONS[idx], fontsize=11, fontweight='bold')
                ax.axis('off')

    fig.tight_layout()
    # Rasterize to numpy (use renderer.buffer_rgba for backend compatibility)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    raw = renderer.buffer_rgba()
    mosaic = np.asarray(raw)[:, :, :3]  # drop alpha
    plt.close(fig)

    mosaic_b64 = numpy_to_base64(mosaic)
    text = f"Panorama: 8 views (front/front-right/right/back-right/back/back-left/left/front-left) at step {cnt_step}"
    return pts, angle, mosaic_b64, text, panorama_views


def view_direction(
    scene, tsdf_planner, pts, angle, direction_desc,
    memory_store, cam_intr, cfg, detection_model, sam_predictor,
    clip_model, clip_preprocess, clip_tokenizer, cnt_step,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """朝向指定方向观察，返回 (pts, angle, img_b64)。

    转向后执行一次 silent_perception_step，存档新视角的 snapshot。
    """
    from src.agent_image_utils import numpy_to_base64

    shifts = {
        "left": -np.pi / 3,
        "right": np.pi / 3,
        "forward": 0.0,
        "backward": np.pi,
    }

    shift = shifts.get(direction_desc.lower(), 0.0)
    new_angle = angle + shift

    obs, _ = scene.get_observation(pts, new_angle)
    rgb = obs["color_sensor"][..., :3]
    img_b64 = numpy_to_base64(rgb)

    # 静默感知并存档（plan §3：每 step 存档）
    silent_perception_step(
        scene, tsdf_planner, pts, new_angle, cnt_step, memory_store,
        cam_intr, cfg, detection_model, sam_predictor,
        clip_model, clip_preprocess, clip_tokenizer,
    )

    return pts, new_angle, img_b64


def navigate_to_object(
    scene, tsdf_planner, pts, angle,
    view_idx, view_angle, view_cam_pose, object_desc,
    memory_store, cam_intr, cfg, detection_model, sam_predictor,
    clip_model, clip_preprocess, clip_tokenizer, cnt_step,
    max_steps=20, step_budget=None,
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
    )

    # GD 导航内部每子步已做 silent_perception + refresh + update_frontier
    # 这里只返回当前视角图像给 VLM
    obs, _ = scene.get_observation(new_pts, new_angle)
    img_b64 = numpy_to_base64(obs["color_sensor"][..., :3])

    return new_pts, new_angle, success, status, img_b64


def navigate_to_seed(
    scene, tsdf_planner, pts, angle, room_id, cfg,
    memory_store, cam_intr, detection_model, sam_predictor,
    clip_model, clip_preprocess, clip_tokenizer, cnt_step,
    max_substeps=25, step_budget=None,
    seed_view_manager=None, active_seed_ids=None,
) -> Tuple[np.ndarray, np.ndarray, bool, str, Optional[str]]:
    """导航到指定房间种子点。返回 (pts, angle, success, status, img_b64)。

    循环 agent_step 直到抵达或超限，每个子步后做 silent_perception_step。
    VLM 在调用此函数后不参与每子步决策，只在抵达后由调用方唤醒。
    """
    from src.agent_image_utils import numpy_to_base64
    from src.tsdf_planner import Frontier

    room = None
    for r in tsdf_planner.room_regions:
        if r.room_id == room_id:
            room = r
            break
    if room is None:
        return pts, angle, False, f"Room {room_id} not found", None

    # 构造临时 Frontier 作为导航目标
    cur_voxel = tsdf_planner.habitat2voxel(pts)[:2]
    direction = room.center.astype(np.float64) - cur_voxel
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 1e-6:
        direction = direction / direction_norm
    else:
        direction = np.array([0.0, 0.0])
    temp_frontier = Frontier(
        position=room.center.astype(np.float64),
        orientation=direction,
        region=room.region,
        frontier_id=-room_id,
    )

    cur_pts, cur_angle = pts, angle
    final_pts, final_angle, arrived, status, substeps = _navigate_to_target_with_agent_step(
        scene, tsdf_planner, cur_pts, cur_angle, temp_frontier, cfg,
        memory_store, cam_intr, detection_model, sam_predictor,
        clip_model, clip_preprocess, clip_tokenizer, cnt_step, max_substeps,
        step_budget=step_budget,
        seed_view_manager=seed_view_manager,
        active_seed_ids=active_seed_ids or [],
    )

    # 抵达后更新 frontier / 房间分割
    if np.linalg.norm(final_pts - cur_pts) > 1e-3:
        tsdf_planner.update_frontier_map(
            final_pts, cfg.planner, scene, cnt_step, save_frontier_image=False)

    obs, _ = scene.get_observation(final_pts, final_angle)
    img_b64 = numpy_to_base64(obs["color_sensor"][..., :3])

    full_status = f"Room {room_id}: {status} ({substeps} substeps)"
    return final_pts, final_angle, arrived, full_status, img_b64


def navigate_to_frontier(
    scene, tsdf_planner, pts, angle, frontier_id, cfg,
    memory_store, cam_intr, detection_model, sam_predictor,
    clip_model, clip_preprocess, clip_tokenizer, cnt_step,
    max_substeps=25, step_budget=None,
    seed_view_manager=None, active_seed_ids=None,
) -> Tuple[np.ndarray, np.ndarray, bool, str, Optional[str]]:
    """导航到指定 frontier。返回 (pts, angle, success, status, img_b64)。

    循环 agent_step 直到抵达或超限，每个子步后做 silent_perception_step。
    VLM 在调用此函数后不参与每子步决策，只在抵达后由调用方唤醒。
    """
    from src.agent_image_utils import numpy_to_base64

    frontier = None
    for ft in tsdf_planner.frontiers:
        if ft.frontier_id == frontier_id:
            frontier = ft
            break
    if frontier is None:
        return pts, angle, False, f"Frontier {frontier_id} not found", None

    cur_pts, cur_angle = pts, angle
    final_pts, final_angle, arrived, status, substeps = _navigate_to_target_with_agent_step(
        scene, tsdf_planner, cur_pts, cur_angle, frontier, cfg,
        memory_store, cam_intr, detection_model, sam_predictor,
        clip_model, clip_preprocess, clip_tokenizer, cnt_step, max_substeps,
        step_budget=step_budget,
        seed_view_manager=seed_view_manager,
        active_seed_ids=active_seed_ids or [],
    )

    if np.linalg.norm(final_pts - cur_pts) > 1e-3:
        tsdf_planner.update_frontier_map(
            final_pts, cfg.planner, scene, cnt_step, save_frontier_image=False)

    obs, _ = scene.get_observation(final_pts, final_angle)
    img_b64 = numpy_to_base64(obs["color_sensor"][..., :3])

    full_status = f"Frontier {frontier_id}: {status} ({substeps} substeps)"
    return final_pts, final_angle, arrived, full_status, img_b64


def query_memory(
    memory_store, text_query, top_k=8,
) -> Optional[str]:
    """查询记忆存储，返回拼接图的 base64。"""
    from src.agent_image_utils import numpy_to_base64

    mosaic = memory_store.make_query_mosaic(text_query, top_k)
    if mosaic is None:
        return None

    return numpy_to_base64(mosaic)


def submit_answer(
    answer_text,
) -> str:
    """提交最终答案。"""
    return f"ANSWER: {answer_text}"
