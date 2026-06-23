"""HM-GE Agent Workflow 控制器。

6 阶段主循环、VLM API 调用、阶段切换逻辑。
"""

import json
import logging
import os
import sys
import time

# Ensure project root is on sys.path (needed when running from within src/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import requests
import numpy as np
from typing import List, Optional, Tuple, Dict

from src.agent_context import ContextManager
from src.agent_memory import MemoryStore
from src.agent_tools import (
    silent_perception_step,
    observe_panorama,
    view_direction,
    navigate_to_object,
    navigate_to_seed,
    navigate_to_frontier,
    query_memory,
    submit_answer,
)
from src.agent_image_utils import numpy_to_base64, make_mosaic
from src.seed_views import SeedViewManager

logger = logging.getLogger(__name__)


# ── VLM API ─────────────────────────────────────────────────────────────

def call_vlm(
    messages: List[dict],
    image_b64: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "mimo-v2.5",
) -> str:
    """调用 mimo-v2.5 API。"""

    if api_key is None:
        from src.const import OPENAI_API_KEY as _key
        api_key = _key
    if base_url is None:
        from src.const import OPENAI_BASE_URL as _url
        base_url = _url

    # Deep copy messages to avoid mutation
    api_messages = []
    for msg in messages:
        api_messages.append(dict(msg))

    # Build the last message with optional image
    last_msg = api_messages[-1]
    # Handle numpy array input (convert to base64)
    if image_b64 is not None:
        if isinstance(image_b64, np.ndarray):
            from src.agent_image_utils import numpy_to_base64
            image_b64 = numpy_to_base64(image_b64)
        if image_b64:  # now it's a string (or empty)
            content_list = [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": last_msg["content"]},
            ]
            api_messages[-1] = {"role": last_msg["role"], "content": content_list}

    payload = {
        "model": model_name,
        "messages": api_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Use xray proxy if available
    proxies = None
    proxy_http = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    proxy_https = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if proxy_http or proxy_https:
        proxies = {}
        if proxy_http:
            proxies["http"] = proxy_http
        if proxy_https:
            proxies["https"] = proxy_https

    try:
        resp = requests.post(
            base_url, json=payload, headers=headers,
            timeout=180, proxies=proxies)
    except requests.exceptions.Timeout:
        logger.error("VLM API timeout")
        return ""
    except requests.exceptions.ConnectionError as e:
        logger.error(f"VLM API connection error: {e}")
        return ""
    except Exception as e:
        logger.error(f"VLM API request failed: {e}")
        return ""

    if resp.status_code != 200:
        logger.error(f"VLM API error: {resp.status_code} {resp.text[:500]}")
        return ""

    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content")
    if content is None:
        content = message.get("reasoning_content", "")
    return content


# ── System Prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an embodied navigation agent searching for the answer to a question in a 3D indoor environment.

You operate in a 6-stage workflow. In each stage you have specific goals and tools available.

Available tools:
- observe_panorama: Take an 8-view panorama. Returns a mosaic image showing all directions and room/frontier information.
- view_direction <direction>: Look toward "left", "right", "forward", or "backward". Returns the RGB image from that direction.
- navigate_to_object <object_description>: Use GroundingDINO to detect the described object and navigate toward it. Returns success/failure and status. The <object_description> MUST be a concrete noun phrase that GroundingDINO can detect, e.g. "oven", "the red door", "towel hanging on oven handle", "coffee table". Do NOT use room names, directions, or abstract concepts — only physical objects.
- navigate_to_seed <room_id>: Navigate toward the center of the specified room (e.g. "1").
- navigate_to_frontier <frontier_id>: Navigate toward the specified unexplored frontier (e.g. "0").
- query_memory <text_query>: Search past observations for relevant images. Returns a mosaic of matching snapshots (max 2 queries per episode).
- submit_answer <answer_text>: Submit your final answer to the question.

Always respond in this JSON format:
{
    "reasoning": "<your reasoning about what you observe and what to do next>",
    "tool": "<tool_name>",
    "arguments": "<arguments for the tool, if any>",
    "answer": "<your answer, only when using submit_answer>",
    "next_stage": <integer 1-6, only set when you want to transition stages; omit otherwise>
}

Stage transition guide:
- Stage 1 -> 2 after panorama.
- Stage 2 -> 3 if target likely in current room; -> 4 if not.
- Stage 3 -> 3 to keep navigating; -> 6 if target found; -> 4 to switch room; -> 5 if current room has no value.
- Stage 4 -> 1 to enter a chosen room/frontier; -> 5 if all regions/frontiers explored.
- Stage 5 -> 6 after memory query or to give up.
- Stage 6: call submit_answer.
"""

# ── VLM Output Schema (shared across stages) ──
# Note: braces are escaped ({{ }}) so .format() doesn't treat them as placeholders
SCHEMA_REQUIREMENT = """
You MUST output the following JSON format (output nothing else):
{{
  "reason": "<one sentence explaining your choice, must include specific visual clues you observed>",
  ...action-specific fields...
}}

reason field requirements:
- Must include specific visual clues you observed from the image (e.g. "I see a stainless steel appliance in view3")
- Must explain how this choice helps answer the question
- Vague statements like "I decided to..." are NOT allowed; you must provide concrete evidence
"""

STAGE1_PROMPT = """Stage 1: Initial Exploration

You are at the starting position. Call observe_panorama to look around.
Based on the panorama, describe what you see and which direction is most promising.

Question: "{question}"
"""

STAGE2_PROMPT = """Stage 2: Main Direction Decision

Look at the 8-view panorama above. The views are labeled:
  view0=front view1=front-right view2=right view3=back-right view4=back view5=back-left view6=left view7=front-left

For the question: "{question}"

Decide:
- If you see a relevant object in one of the views -> navigate_to_object with view_idx
- If no relevant object visible in any view -> explore_other_room
""" + SCHEMA_REQUIREMENT + """

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
- If a seed seems relevant -> explore_seed with seed_id
- If all seeds seem irrelevant -> explore_frontier (fallback)
""" + SCHEMA_REQUIREMENT + """

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
""" + SCHEMA_REQUIREMENT + """

Output: {{"reason": "...", "object": "<object_name>"}}
"""

STAGE5_PROMPT = """Stage 5: Re-decision After Arrival

You've arrived near the target. Here are the 3 frontal views from your
current position (left 60°, front, right 60°):
  view0=left view1=front view2=right

For the question: "{question}"

Decide:
- If you can answer the question now -> submit_answer
- If you see a new relevant object in one of the 3 views -> navigate_to_object
- If you need to explore other rooms -> explore_other_room
""" + SCHEMA_REQUIREMENT + """

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
""" + SCHEMA_REQUIREMENT + """

Output: {{"reason": "...", "frontier_id": <id>}}
"""


# ── Main Workflow ───────────────────────────────────────────────────────

def run_episode(
    scene_id: str,
    question: str,
    question_id: str,
    cfg,
    detection_model,
    sam_predictor,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    output_dir: str = "/root/MyAgent/results/hmge",
    max_total_steps: int = 50,
    start_pts: Optional[np.ndarray] = None,
    start_angle: float = 0.0,
) -> Dict:
    """Run a single HM-GE workflow episode.

    Returns: dict with keys:
        scene_id, question_id, question, answer, success, steps_taken,
        stages_completed, error
    """
    import habitat_sim
    from src.scene_aeqa import Scene
    from src.habitat import pos_normal_to_habitat
    from src.tsdf_planner import TSDFPlanner

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== Episode {question_id}: {scene_id} ===")
    logger.info(f"Question: {question}")

    result = {
        "scene_id": scene_id,
        "question_id": question_id,
        "question": question,
        "answer": "",
        "success": False,
        "steps_taken": 0,
        "stages_completed": 0,
        "error": "",
    }

    # Initialize scene, planner, memory, context
    scene = None
    tsdf_planner = None
    try:
        # 每 episode 重置步数计数器
        from src.agent_tools import silent_perception_step
        silent_perception_step._last_pos = None
        silent_perception_step._step_counter = -1

        # Load concept graph config if not provided
        import yaml
        from omegaconf import OmegaConf, DictConfig

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif hasattr(cfg, "concept_graph_config_path"):
            pass  # OmegaConf object
        else:
            from easydict import EasyDict
            cfg = EasyDict(cfg)

        # Load separate concept graph config
        graph_cfg_path = getattr(cfg, "concept_graph_config_path", None)
        if graph_cfg_path and os.path.exists(graph_cfg_path):
            graph_cfg = OmegaConf.load(graph_cfg_path)
            OmegaConf.resolve(graph_cfg)
        else:
            graph_cfg = getattr(cfg, "scene_graph", {})

        # Load scene
        scene = Scene(
            scene_id=scene_id, cfg=cfg, graph_cfg=graph_cfg,
            detection_model=detection_model, sam_predictor=sam_predictor,
            clip_model=clip_model, clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
        )

        # Determine starting position — prefer AEQA-provided position
        if start_pts is not None and not np.isnan(start_pts).any():
            pts = start_pts.copy()
            angle = start_angle
        else:
            start_pts_random = scene.pathfinder.get_random_navigable_point()
            if np.isnan(start_pts_random).any():
                start_pts_random = np.array([0.0, 1.5, 0.0])
            pts = start_pts_random.copy()
            angle = 0.0

        # Initialize TSDF planner — match original 3D-Mem approach
        from src.geom import get_scene_bnds
        vol_bnds, _ = get_scene_bnds(scene.pathfinder, floor_height=pts[1])
        tsdf_planner = TSDFPlanner(
            vol_bnds=vol_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height=pts[1],
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=cfg.init_clearance * 2,
            save_visualization=False,
        )

        # Initial observation (angle already set above from start_angle / random fallback)
        obs, cam_pose = scene.get_observation(pts, angle)

        cam_intr = scene.cam_intrinsic
        memory_store = MemoryStore(
            output_dir=os.path.join(output_dir, f"memory_{question_id}"))
        context = ContextManager()

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        result["error"] = str(e)
        return result

    total_steps = 0
    answer = ""
    stages_completed = 0

    try:
        # Override room_segmentation config for broader room discovery
        # (default area_source="explored" + observed_ratio_threshold=0.30
        # only finds rooms the agent has physically visited; we need all
        # navigable rooms visible from the panorama, per Obsidian notes
        # MSGNav-调试渲染笔记-20260612-room-seg.md section "改进 v2")
        # Note: _cfg_get reads attrs directly from cfg.planner, not from
        # a room_segmentation sub-config, so we set attrs on cfg.planner
        from omegaconf import OmegaConf as _OC
        if not hasattr(cfg.planner, "area_source"):
            cfg.planner.area_source = "navigable"
        else:
            cfg.planner.area_source = "navigable"
        if not hasattr(cfg.planner, "observed_ratio_threshold"):
            cfg.planner.observed_ratio_threshold = 0.0
        else:
            cfg.planner.observed_ratio_threshold = 0.0
        if not hasattr(cfg.planner, "max_unobserved_room_hops"):
            cfg.planner.max_unobserved_room_hops = 99
        else:
            cfg.planner.max_unobserved_room_hops = 99

        # ═══ STAGE 1: 8-View Panorama ═══
        logger.info("--- Stage 1: Initial Panorama ---")
        pts, angle, mosaic_b64, pano_text, panorama_views = observe_panorama(
            scene, tsdf_planner, pts, angle, total_steps,
            memory_store, cam_intr, cfg, detection_model,
            sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
        )
        total_steps += 1

        # SeedViewManager: register seeds from current frontier/room map
        seed_view_manager = SeedViewManager()
        if hasattr(tsdf_planner, "room_regions") and tsdf_planner.room_regions:
            logger.info(f"Stage 1: found {len(tsdf_planner.room_regions)} rooms")
            _register_new_seeds(seed_view_manager, tsdf_planner, scene, pts)
            logger.info(f"Stage 1: registered {len(seed_view_manager.seeds)} seeds")
        else:
            logger.info("Stage 1: no room_regions found (room segmentation may have failed)")

        # ═══ STAGE 2-6: 6-Stage State Machine ═══
        current_stage = 2
        consecutive_missing_reason = 0
        max_vlm_calls = max_total_steps // 2

        def _low_level_steps():
            return silent_perception_step._step_counter

        vlm_call_count = 0

        while current_stage != "done" and _low_level_steps() < max_total_steps:
            if current_stage == 2:
                # ── Stage 2: Main Direction Decision (VLM call 1) ──
                logger.info("--- Stage 2: Main Direction Decision ---")
                stage_prompt = STAGE2_PROMPT.format(question=question)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": pano_text + "\n" + stage_prompt},
                ]
                vlm_response = call_vlm(messages, image_b64=mosaic_b64)
                vlm_call_count += 1
                vlm_parsed = _parse_vlm_response(vlm_response)

                if vlm_parsed.get("tool") == "missing_reason":
                    consecutive_missing_reason += 1
                    if consecutive_missing_reason >= 2:
                        logger.info("Stage 2: 2x missing_reason, fallback to explore_other_room")
                        current_stage = "2.5a"
                        consecutive_missing_reason = 0
                        continue
                    continue

                consecutive_missing_reason = 0
                logger.info(f"Stage 2 VLM: {vlm_parsed}")

                if vlm_parsed.get("tool") == "navigate_to_object":
                    view_idx = vlm_parsed.get("view_idx", 0)
                    if view_idx is None:
                        view_idx = 0
                    view_idx = int(view_idx)
                    view_idx = max(0, min(view_idx, len(panorama_views) - 1))
                    view_info = panorama_views[view_idx]
                    # Store view info for Stage 3
                    pending_view = {
                        "view_idx": view_idx,
                        "angle": view_info["angle"],
                        "cam_pose": view_info["cam_pose"],
                        "rgb": view_info["rgb"],
                    }
                    current_stage = 3
                else:
                    current_stage = "2.5a"

            elif current_stage == "2.5a":
                # ── Stage 2.5a: Seed Selection (VLM call 2) ──
                logger.info("--- Stage 2.5a: Seed Selection ---")
                explored_seed_ids = set()
                seed_ids = seed_view_manager.get_unexplored_seed_ids(explored_seed_ids)

                if not seed_ids:
                    logger.info("Stage 2.5a: no seeds available, fallback to frontier")
                    current_stage = 6
                    continue

                seed_mosaic = seed_view_manager.get_mosaic(question)
                seed_info = f"Available seeds: {seed_ids}"
                stage_prompt = STAGE2_5A_PROMPT.format(
                    seed_info=seed_info, question=question)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stage_prompt},
                ]
                vlm_response = call_vlm(messages, image_b64=seed_mosaic)
                vlm_call_count += 1
                vlm_parsed = _parse_vlm_response(vlm_response)

                if vlm_parsed.get("tool") == "missing_reason":
                    consecutive_missing_reason += 1
                    if consecutive_missing_reason >= 2:
                        logger.info("Stage 2.5a: 2x missing_reason, fallback to frontier")
                        current_stage = 6
                        consecutive_missing_reason = 0
                        continue
                    continue

                consecutive_missing_reason = 0
                logger.info(f"Stage 2.5a VLM: {vlm_parsed}")

                if vlm_parsed.get("tool") == "explore_seed":
                    seed_id = vlm_parsed.get("seed_id", seed_ids[0])
                    if seed_id is None:
                        seed_id = seed_ids[0]
                    seed_id = int(seed_id)
                    step_budget = max_total_steps - _low_level_steps()
                    pts, angle, success, status, obs_image = navigate_to_seed(
                        scene, tsdf_planner, pts, angle, seed_id, cfg,
                        memory_store, cam_intr, detection_model, sam_predictor,
                        clip_model, clip_preprocess, clip_tokenizer, total_steps,
                        step_budget=step_budget,
                        seed_view_manager=seed_view_manager,
                        active_seed_ids=[sid for sid in seed_view_manager.seeds],
                    )
                    total_steps += 1
                    # Register any new seeds discovered after navigation
                    _register_new_seeds(seed_view_manager, tsdf_planner, scene, pts)
                    current_stage = 5
                else:
                    current_stage = 6

            elif current_stage == 3:
                # ── Stage 3: Object Selection (VLM call 3) ──
                logger.info("--- Stage 3: Object Selection ---")
                rgb = pending_view["rgb"]
                stage_prompt = STAGE3_PROMPT.format(
                    view_idx=pending_view["view_idx"], question=question)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stage_prompt},
                ]
                vlm_response = call_vlm(messages, image_b64=numpy_to_base64(rgb))
                vlm_call_count += 1
                vlm_parsed = _parse_vlm_response(vlm_response)

                if vlm_parsed.get("tool") == "missing_reason":
                    consecutive_missing_reason += 1
                    if consecutive_missing_reason >= 2:
                        logger.info("Stage 3: 2x missing_reason, fallback to explore_other_room")
                        current_stage = "2.5a"
                        consecutive_missing_reason = 0
                        continue
                    continue

                consecutive_missing_reason = 0
                object_desc = vlm_parsed.get("object", "")

                if not _is_valid_object_desc(object_desc):
                    logger.warning(f"Stage 3: invalid object '{object_desc}', retrying")
                    continue

                logger.info(f"Stage 3: object='{object_desc}'")

                # ── Stage 4: GD Navigation (code, no VLM) ──
                step_budget = max_total_steps - _low_level_steps()
                pts, angle, success, status, _ = navigate_to_object(
                    scene, tsdf_planner, pts, angle,
                    view_idx=pending_view["view_idx"],
                    view_angle=pending_view["angle"],
                    view_cam_pose=pending_view["cam_pose"],
                    object_desc=object_desc,
                    memory_store=memory_store, cam_intr=cam_intr, cfg=cfg,
                    detection_model=detection_model, sam_predictor=sam_predictor,
                    clip_model=clip_model, clip_preprocess=clip_preprocess,
                    clip_tokenizer=clip_tokenizer, cnt_step=total_steps,
                    step_budget=step_budget,
                )
                total_steps += 1
                current_stage = 5

            elif current_stage == 5:
                # ── Stage 5: Re-decision After Arrival (VLM call 4) ──
                logger.info("--- Stage 5: Re-decision After Arrival ---")
                # Render 3 frontal views (left 60°, front, right 60°) at current position
                obs_angles = [angle - np.pi / 3, angle, angle + np.pi / 3]
                frontal_views = []
                frontal_rgb = []
                for i, ang in enumerate(obs_angles):
                    obs, cam_pose = scene.get_observation(pts, ang)
                    rgb = obs["color_sensor"][..., :3]
                    frontal_rgb.append(rgb)
                    frontal_views.append({
                        "view_idx": i,
                        "angle": float(ang),
                        "cam_pose": cam_pose,
                        "rgb": rgb,
                    })

                # Build 3-view mosaic
                mosaic_3 = make_mosaic(frontal_rgb, target_h=300)

                stage_prompt = STAGE5_PROMPT.format(question=question)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stage_prompt},
                ]
                vlm_response = call_vlm(messages, image_b64=mosaic_3)
                vlm_call_count += 1
                vlm_parsed = _parse_vlm_response(vlm_response)

                if vlm_parsed.get("tool") == "missing_reason":
                    consecutive_missing_reason += 1
                    if consecutive_missing_reason >= 2:
                        logger.info("Stage 5: 2x missing_reason, fallback to submit_answer")
                        answer = vlm_parsed.get("answer", "unanswerable")
                        result["answer"] = answer
                        result["success"] = True
                        result["steps_taken"] = _low_level_steps()
                        result["stages_completed"] = 5
                        current_stage = "done"
                        break
                    continue

                consecutive_missing_reason = 0
                logger.info(f"Stage 5 VLM: {vlm_parsed}")

                tool = vlm_parsed.get("tool", "")
                if tool == "submit_answer":
                    answer = vlm_parsed.get("answer", "unanswerable")
                    result["answer"] = answer
                    result["success"] = True
                    result["steps_taken"] = _low_level_steps()
                    result["stages_completed"] = 5
                    current_stage = "done"
                elif tool == "navigate_to_object":
                    view_idx = vlm_parsed.get("view_idx", 1)
                    if view_idx is None:
                        view_idx = 1
                    view_idx = int(view_idx)
                    view_idx = max(0, min(view_idx, len(frontal_views) - 1))
                    view_info = frontal_views[view_idx]
                    pending_view = {
                        "view_idx": view_idx,
                        "angle": view_info["angle"],
                        "cam_pose": view_info["cam_pose"],
                        "rgb": view_info["rgb"],
                    }
                    current_stage = 3
                elif tool == "explore_other_room":
                    current_stage = "2.5a"
                else:
                    # Unknown action -> fallback to explore_other_room
                    current_stage = "2.5a"

            elif current_stage == 6:
                # ── Stage 6: Frontier Selection (VLM call 5) ──
                logger.info("--- Stage 6: Frontier Selection ---")
                frontier_info = _format_frontiers_info(tsdf_planner)
                stage_prompt = STAGE6_PROMPT.format(
                    frontier_info=frontier_info, question=question)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": stage_prompt},
                ]
                # Use frontier visualization if available
                frontier_img = None
                if hasattr(tsdf_planner, '_last_frontier_image') and tsdf_planner._last_frontier_image is not None:
                    frontier_img = numpy_to_base64(tsdf_planner._last_frontier_image)

                vlm_response = call_vlm(messages, image_b64=frontier_img)
                vlm_call_count += 1
                vlm_parsed = _parse_vlm_response(vlm_response)

                if vlm_parsed.get("tool") == "missing_reason":
                    consecutive_missing_reason += 1
                    if consecutive_missing_reason >= 2:
                        logger.info("Stage 6: 2x missing_reason, fallback to frontier 0")
                        vlm_parsed = {"tool": "explore_frontier", "frontier_id": 0, "reason": "fallback"}
                        consecutive_missing_reason = 0
                    else:
                        continue

                frontier_id = vlm_parsed.get("frontier_id", 0)
                if frontier_id is None:
                    frontier_id = 0
                frontier_id = int(frontier_id)
                step_budget = max_total_steps - _low_level_steps()
                pts, angle, success, status, obs_image = navigate_to_frontier(
                    scene, tsdf_planner, pts, angle, frontier_id, cfg,
                    memory_store, cam_intr, detection_model, sam_predictor,
                    clip_model, clip_preprocess, clip_tokenizer, total_steps,
                    step_budget=step_budget,
                    seed_view_manager=seed_view_manager,
                    active_seed_ids=[sid for sid in seed_view_manager.seeds],
                )
                total_steps += 1
                # Register any new seeds discovered after navigation
                _register_new_seeds(seed_view_manager, tsdf_planner, scene, pts)
                current_stage = 5

        # ═══ Final Answer (Stage 6 fallback) ═══
        if not answer and current_stage != "done":
            logger.info("--- Stage 6: Submit Answer ---")
            stage_prompt = STAGE6_PROMPT.format(question=question)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": stage_prompt},
            ]
            vlm_response = call_vlm(messages)
            vlm_parsed = _parse_vlm_response(vlm_response)
            answer = vlm_parsed.get("answer", vlm_parsed.get("arguments", "unanswerable"))
            logger.info(f"Final answer: {answer}")

            result["answer"] = answer
            result["success"] = "unanswerable" not in answer.lower()
            result["steps_taken"] = _low_level_steps()
            result["stages_completed"] = 6

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)

    finally:
        # Cleanup
        if scene is not None:
            try:
                scene.__del__()
            except:
                pass

    return result


def _register_new_seeds(seed_view_manager, tsdf_planner, scene, agent_pts):
    """Scan room_regions and register any new seeds not yet in SeedViewManager.

    A seed is any room OTHER than the one the agent is currently in.
    room_state can be 'observed', 'hypothesis', or 'unknown' — all are
    valid seeds (we want to navigate to other rooms to explore them).
    """
    from src.habitat import pos_normal_to_habitat
    if not hasattr(tsdf_planner, "room_regions") or not tsdf_planner.room_regions:
        return
    existing_ids = set(seed_view_manager.seeds.keys())
    # Find which room the agent is currently in
    agent_voxel = tsdf_planner.habitat2voxel(agent_pts)[:2]
    agent_room_id = tsdf_planner.get_room_id_at(agent_voxel)
    logger.info(f"_register_new_seeds: agent_room_id={agent_room_id}, "
                f"existing_ids={existing_ids}, "
                f"room_ids={[r.room_id for r in tsdf_planner.room_regions]}")
    for room in tsdf_planner.room_regions:
        # Skip the room the agent is already in
        if room.room_id == agent_room_id:
            continue
        if room.room_id not in existing_ids:
            try:
                # room.center is 2D voxel [vy, vx], convert to 3D habitat
                # Per debug_render_episode.py:878-885, use _vol_bnds + 0.5 offset
                # (voxel center, not corner) and pin height to eye level 1.5m
                vy, vx = int(room.center[0]), int(room.center[1])
                voxel_size = tsdf_planner._voxel_size
                world_y = tsdf_planner._vol_bnds[0, 0] + (vy + 0.5) * voxel_size
                world_x = tsdf_planner._vol_bnds[1, 0] + (vx + 0.5) * voxel_size
                seed_normal = np.asarray([world_y, world_x, 1.5], dtype=float)
                center_habitat = pos_normal_to_habitat(seed_normal)
                seed_view_manager.register_seed(
                    room.room_id, center_habitat,
                    scene, tsdf_planner, agent_pts)
            except Exception as e:
                logger.warning(f"_register_new_seeds: failed to register "
                              f"seed {room.room_id}: {e}")


# ── Helpers ──────────────────────────────────────────────────────────────

# Invalid arguments for navigate_to_object — these are not object descriptions
_NAV_OBJ_INVALID = {
    "", "forward", "backward", "left", "right", "up", "down",
    "explore", "navigate", "search", "look", "go", "move",
    "room", "room 0", "room 1", "room 2", "room 3", "room 4",
    "frontier", "frontier 0", "frontier 1", "frontier 2",
    "yes", "no", "true", "false", "none", "null",
    "the kitchen", "the bathroom", "the bedroom", "the living room",
    "kitchen", "bathroom", "bedroom", "living room",
}

def _is_valid_object_desc(desc: str) -> bool:
    """Check if a string is a valid concrete object description for GroundingDINO.

    Rejects empty strings, directions, room names, and other non-object terms.
    """
    if not desc or not isinstance(desc, str):
        return False
    desc_clean = desc.strip().lower()
    if desc_clean in _NAV_OBJ_INVALID:
        return False
    if len(desc_clean) < 2:
        return False
    # Reject pure numbers (room/frontier IDs)
    try:
        int(desc_clean)
        return False
    except ValueError:
        pass
    return True


def _build_messages(context: ContextManager) -> List[dict]:
    """Build the message list for VLM from context manager state."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add stage transition summaries from previous stages
    for transition in context.transitions:
        if transition.from_stage != context.current_stage:
            summary_text = (
                f"[Stage {transition.from_stage}→{transition.to_stage} summary]\n"
                f"{transition.summary}"
            )
            messages.append({"role": "assistant", "content": summary_text})

    # Add current stage messages
    messages.extend(context.stage_messages)

    return messages


def _format_rooms_info(tsdf_planner) -> str:
    """Format room information for VLM prompt."""
    if not hasattr(tsdf_planner, "room_regions") or not tsdf_planner.room_regions:
        return "No room segmentation available."

    lines = []
    for room in tsdf_planner.room_regions:
        lines.append(
            f"  Room {room.room_id}: area={room.area}, "
            f"state={room.room_state}, "
            f"observed={room.observed_ratio:.1%}, "
            f"frontiers={room.frontier_ids}"
        )
    return "Rooms:\n" + "\n".join(lines) if lines else "No rooms."


def _format_frontiers_info(tsdf_planner) -> str:
    """Format frontier information for VLM prompt."""
    if not tsdf_planner.frontiers:
        return "No frontiers available."

    lines = []
    for ft in tsdf_planner.frontiers:
        room_str = f"room={ft.room_id}" if hasattr(ft, "room_id") and ft.room_id >= 0 else ""
        lines.append(f"  Frontier {ft.frontier_id}: {room_str}")
    return "Frontiers:\n" + "\n".join(lines) if lines else "No frontiers."


def _parse_vlm_response(response: str) -> dict:
    """Parse VLM JSON response. Enforce mandatory 'reason' field.

    Returns dict with at least:
        - tool: str (action name, or 'parse_error'/'missing_reason')
        - reason: str (may be empty if missing)
        - raw: str (original response, only on error)
    """
    import json as _json

    # Try to extract JSON from response (VLM may add prose around it)
    text = response.strip() if response else ""
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


# ── Direct Run (for testing) ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--cfg", type=str,
                       default="cfg/eval_aeqa.yaml")
    parser.add_argument("--output", type=str,
                       default="/root/MyAgent/results/hmge")
    args = parser.parse_args()

    # Load config
    import yaml
    from omegaconf import OmegaConf
    from src.utils import get_pts_angle_aeqa

    with open(args.cfg, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    OmegaConf.resolve(cfg)

    # Look up AEQA start position for this scene+question
    start_pts = None
    start_angle = 0.0
    try:
        questions_list = json.load(open(cfg.questions_list_path, "r"))
        for qd in questions_list:
            if qd["episode_history"] == args.scene and qd["question"] == args.question:
                start_pts, start_angle = get_pts_angle_aeqa(
                    qd["position"], qd["rotation"])
                logging.info(f"AEQA start position: {start_pts}, angle: {start_angle}")
                break
    except Exception as e:
        logging.warning(f"Could not find AEQA start position: {e}")

    # Load models (same as run_aeqa_evaluation.py)
    from ultralytics import SAM, YOLOWorld
    import open_clip

    detection_model = YOLOWorld(cfg.yolo_model_name)
    sam_predictor = SAM(cfg.sam_model_name)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    result = run_episode(
        scene_id=args.scene,
        question=args.question,
        question_id="test",
        cfg=cfg,
        detection_model=detection_model,
        sam_predictor=sam_predictor,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        output_dir=args.output,
        start_pts=start_pts,
        start_angle=start_angle,
    )

    print(json.dumps(result, indent=2))
