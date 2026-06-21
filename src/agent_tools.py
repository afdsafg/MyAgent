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

    Returns: (new_pts, new_angle) — 通常不变，只在 GD 导航步中变化。
    如果 skip_snapshots=True，只做 TSDF/场景图更新，不存档。
    """
    # 检查是否与上次存档位置相同，避免重复快照
    if not hasattr(silent_perception_step, '_last_pos'):
        silent_perception_step._last_pos = None
        silent_perception_step._step_counter = 0
    pos_changed = (
        silent_perception_step._last_pos is None
        or np.linalg.norm(np.array(pts) - np.array(silent_perception_step._last_pos)) > 0.5
    )
    if pos_changed:
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
        obs_name = f"step{cnt_step}_view{view_idx}"

        annotated_rgb, added_obj_ids, _ = scene.update_scene_graph(
            image_rgb=rgb[..., :3], depth=depth,
            intrinsics=cam_intr, cam_pos=cam_pose,
            pts=pts, pts_voxel=tsdf_planner.habitat2voxel(pts),
            img_path=obs_name, frame_idx=cnt_step * 3 + view_idx,
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
            frame_idx=cnt_step * 3 + view_idx, pts=pts)

    # Snapshot 存档 — 仅在位置变化时保存
    scene.update_snapshots(
        obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection)

    if pos_changed and not skip_snapshots:
        room_id = tsdf_planner.get_room_id_at(
            tsdf_planner.habitat2voxel(pts)[:2])
        for i, view_rgb in enumerate(rgb_views):
            # 收集当前视图中所有 object（不仅仅是新增的）
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


# ── 7 个 VLM 工具 ───────────────────────────────────────────────────────

def observe_panorama(
    scene, tsdf_planner, pts, angle, cnt_step,
    memory_store, cam_intr, cfg, detection_model,
    sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """7 视角全景观测，返回 (pts, angle, mosaic_b64, text)。"""
    from src.agent_image_utils import make_mosaic, numpy_to_base64

    angles = np.linspace(-np.pi, np.pi, 8)[:7]
    views = []
    for ang in angles:
        obs, _ = scene.get_observation(pts, ang)
        views.append(obs["color_sensor"][..., :3])

    # 静默执行感知（3视角 + TSDF + 场景图更新，不存档3视角图）
    silent_perception_step(
        scene, tsdf_planner, pts, angle, cnt_step, memory_store,
        cam_intr, cfg, detection_model, sam_predictor,
        clip_model, clip_preprocess, clip_tokenizer,
        skip_snapshots=True,
    )

    # 保存全景 7 张视角到 MemoryStore（仅位置变化 + 有检测到物体）
    room_id = tsdf_planner.get_room_id_at(
        tsdf_planner.habitat2voxel(pts)[:2])
    for ang_idx, view_rgb in enumerate(views):
            objs_in_view = [
                scene.objects[oid]["class_name"]
                for oid in scene.objects
                if np.linalg.norm(
                    scene.objects[oid]["bbox"].center[[0, 2]] - pts[[0, 2]]
                ) < cfg.scene_graph.obj_include_dist + 0.5
            ]
            memory_store.add_snapshot(
                snapshot_id=f"step{silent_perception_step._step_counter}_pano_view{ang_idx}",
                image=view_rgb,
                room_id=room_id,
                objects_in_view=objs_in_view,
                position_3d=pts.tolist(),
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
            )

    # 更新房间分割
    tsdf_planner.update_frontier_map(
        pts, cfg.planner, scene, cnt_step, save_frontier_image=False)

    mosaic = make_mosaic(views, target_h=200)
    mosaic_b64 = numpy_to_base64(mosaic)

    room_info = ""
    if hasattr(tsdf_planner, "room_regions") and tsdf_planner.room_regions:
        room_info = f"\nRooms detected: {len(tsdf_planner.room_regions)}"

    text = (
        f"Panorama observed. {len(tsdf_planner.frontiers)} frontiers available."
        f"{room_info}"
    )
    return pts, angle, mosaic_b64, text


def view_direction(
    scene, pts, angle, direction_desc
) -> Tuple[np.ndarray, np.ndarray, str]:
    """朝向指定方向观察，返回 (pts, angle, img_b64)。"""
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

    return pts, new_angle, img_b64


def navigate_to_object(
    scene, tsdf_planner, pts, angle, object_desc, max_steps=20,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """GD 导航到指定物体。"""
    from src.scene_aeqa import grounded_navigate_to_object as gd_nav

    new_pts, new_angle, success, status, images = gd_nav(
        scene, tsdf_planner, pts, angle, object_desc, max_steps=max_steps,
    )
    return new_pts, new_angle, success, status


def navigate_to_seed(
    scene, tsdf_planner, pts, angle, room_id, cfg,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """使用原 3D-Mem 导航链走向房间中心（set_next_navigation_point + agent_step）。"""
    from src.habitat import pos_habitat_to_normal

    room = None
    for r in tsdf_planner.room_regions:
        if r.room_id == room_id:
            room = r
            break
    if room is None:
        return pts, angle, False, f"Room {room_id} not found"

    # 构造一个临时 Frontier 用作导航目标
    from src.tsdf_planner import Frontier
    temp_frontier = Frontier(
        position=room.center.astype(np.float64),
        orientation=np.array([0.0, 0.0]),
        region=room.region,
        frontier_id=-room_id,
    )

    # 使用原 3D-Mem 逻辑设置导航目标
    success = tsdf_planner.set_next_navigation_point(
        choice=temp_frontier, pts=pts, objects=scene.objects,
        cfg=cfg.planner, pathfinder=scene.pathfinder,
    )
    if not success:
        return pts, angle, False, f"Failed to set nav target for room {room_id}"

    # 执行一步 agent_step
    result = tsdf_planner.agent_step(
        pts=pts, angle=angle, objects=scene.objects,
        snapshots=scene.snapshots, pathfinder=scene.pathfinder,
        cfg=cfg.planner, save_visualization=False,
    )
    if result[0] is None:
        tsdf_planner.max_point = None
        tsdf_planner.target_point = None
        return pts, angle, False, f"agent_step failed for room {room_id}"

    new_pts, new_angle, _, _, _, _ = result
    return new_pts, new_angle, True, f"Stepping toward room {room_id}"


def navigate_to_frontier(
    scene, tsdf_planner, pts, angle, frontier_id,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """使用 pathfinder 导航到指定 frontier 区域。"""
    from src.habitat import pos_normal_to_habitat

    frontier = None
    for ft in tsdf_planner.frontiers:
        if ft.frontier_id == frontier_id:
            frontier = ft
            break

    if frontier is None:
        return pts, angle, False, f"Frontier {frontier_id} not found"

    # Convert frontier voxel position to habitat
    voxel_pos = frontier.position.astype(np.float64)
    pos_normal = voxel_pos * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
    pos_normal = np.append(pos_normal, pts[2])
    target_habitat = pos_normal_to_habitat(pos_normal)

    # Find navigable point near the frontier
    nav = scene.pathfinder.get_random_navigable_point_near(
        circle_center=target_habitat, radius=2.0, max_tries=20)
    if np.isnan(nav).any():
        return pts, angle, False, f"No navigable point near frontier {frontier_id}"

    # Step toward the target
    direction = nav - pts
    direction[1] = 0
    dist = np.linalg.norm(direction)
    if dist < 0.1:
        return pts, angle, True, f"Already at frontier {frontier_id}"

    step_size = min(1.0, dist)
    new_pts = pts + (direction / dist) * step_size
    new_angle = np.arctan2(direction[0], -direction[2]) - np.pi / 2

    return new_pts, new_angle, True, f"Moving toward frontier {frontier_id} ({dist:.1f}m away)"


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
