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
) -> Tuple[np.ndarray, np.ndarray]:
    """每 step 静默执行：3 视角观测 + 全管线更新 + Snapshot 存档。

    Returns: (new_pts, new_angle) — 通常不变，只在 GD 导航步中变化。
    """
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

    # Snapshot 存档
    scene.update_snapshots(
        obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection)

    # 存档到 MemoryStore
    room_id = tsdf_planner.get_room_id_at(
        tsdf_planner.habitat2voxel(pts)[:2])
    for i, view_rgb in enumerate(rgb_views):
        objs_in_view = [scene.objects[oid]["class_name"]
                        for oid in all_added_obj_ids
                        if oid in scene.objects]
        memory_store.add_snapshot(
            snapshot_id=f"step{cnt_step}_view{i}",
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

    # 静默执行感知
    silent_perception_step(
        scene, tsdf_planner, pts, angle, cnt_step, memory_store,
        cam_intr, cfg, detection_model, sam_predictor,
        clip_model, clip_preprocess, clip_tokenizer,
    )

    # 更新房间分割
    tsdf_planner.update_frontier_map(
        pts, cfg, scene, cnt_step, save_frontier_image=False)

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
    tsdf_planner, pts, angle, room_id, seed_idx=0,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """导航到指定房间的 seed 点。"""
    from src.geom import get_nearest_true_point

    room = None
    for r in tsdf_planner.room_regions:
        if r.room_id == room_id:
            room = r
            break

    if room is None:
        return pts, angle, False, f"Room {room_id} not found"

    center = room.center.astype(np.float64)
    if seed_idx > 0:
        coords = np.argwhere(room.region)
        if len(coords) > seed_idx:
            idx = min(seed_idx, len(coords) - 1)
            center = coords[idx].astype(np.float64)

    nav = get_nearest_true_point(
        center.astype(int), tsdf_planner.unoccupied)
    if nav is None:
        return pts, angle, False, f"No navigable point near room {room_id} center"

    direction = nav - tsdf_planner.habitat2voxel(pts)[:2]
    direction_habitat = np.array([
        direction[1] * tsdf_planner._voxel_size,
        0,
        -direction[0] * tsdf_planner._voxel_size,
    ])
    direction_norm = np.linalg.norm(direction_habitat)

    if direction_norm > 0:
        new_angle = np.arctan2(
            direction_habitat[0], -direction_habitat[2]) - np.pi / 2
    else:
        new_angle = angle

    return pts, new_angle, True, f"Facing room {room_id}"


def navigate_to_frontier(
    tsdf_planner, pts, angle, frontier_id,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """导航到指定 frontier。"""
    frontier = None
    for ft in tsdf_planner.frontiers:
        if ft.frontier_id == frontier_id:
            frontier = ft
            break

    if frontier is None:
        return pts, angle, False, f"Frontier {frontier_id} not found"

    direction = np.array(frontier.position, dtype=float)
    direction -= tsdf_planner.habitat2voxel(pts)[:2]
    direction_norm_val = np.linalg.norm(direction)

    if direction_norm_val > 0:
        new_angle = np.arctan2(direction[1], direction[0]) - np.pi / 2
    else:
        new_angle = angle

    return pts, new_angle, True, f"Facing frontier {frontier_id}"


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
