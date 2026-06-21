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

logger = logging.getLogger(__name__)


# ── VLM API ─────────────────────────────────────────────────────────────

def call_vlm(
    messages: List[dict],
    image_b64: Optional[str] = None,
    max_tokens: int = 1024,
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
    if image_b64:
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
- observe_panorama: Take a 7-view panorama. Returns a mosaic image showing all directions and room/frontier information.
- view_direction <direction>: Look toward "left", "right", "forward", or "backward". Returns the RGB image from that direction.
- navigate_to_object <object_description>: Use GroundingDINO to detect the described object and navigate toward it. Returns success/failure and status.
- navigate_to_seed <room_id>: Face toward the center of the specified room.
- navigate_to_frontier <frontier_id>: Face toward the specified unexplored frontier.
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

STAGE1_PROMPT = """Stage 1: Initial Exploration

You are at the starting position. First, call observe_panorama to look around and understand your surroundings. Based on the panorama, identify:
1. What rooms you can see
2. What objects are visible
3. What unexplored areas (frontiers) exist

Describe what you see and which direction you think is most promising for finding the answer to: "{question}"
"""

STAGE2_PROMPT = """Stage 2: Direction Judgment

Based on the panorama you just observed, look at the objects and rooms visible. 

For the question: "{question}"

Do you see the target objects or relevant clues in the current view?
- If YES: Navigate in that direction with navigate_to_object or view_direction.
- If NO: Choose the most promising unexplored frontier or room to explore.

Available rooms: {rooms_info}
Available frontiers: {frontiers_info}
"""

STAGE3_PROMPT = """Stage 3: Targeted Navigation

You decided to search for: "{target}". 

Try using navigate_to_object with a description that matches what you're looking for. 
If GD navigation fails, try viewing different directions or navigating to promising frontiers/rooms.

Available rooms: {rooms_info}
Available frontiers: {frontiers_info}
"""

STAGE4_PROMPT = """Stage 4: Final Exploration

You've explored several areas but haven't found the answer yet. 

Question: "{question}"

Choose which room or frontier to explore next. Consider:
- Which rooms have you not fully explored?
- Which frontiers seem most promising?

Available rooms: {rooms_info}
Available frontiers: {frontiers_info}
"""

STAGE5_PROMPT = """Stage 5: Memory Fallback

You've explored extensively but still need to answer: "{question}"

You can use query_memory to search past observations. You have a limited number of queries remaining.
The query result will be shown to you as an image on the next turn.

If you think you have enough information, use submit_answer to provide your answer.
If you truly cannot find the answer, submit "unanswerable".
When ready to answer, set next_stage to 6.
"""

STAGE6_PROMPT = """Stage 6: Submit Answer

Based on all observations and reasoning, submit your final answer.

Question: "{question}"

Respond with submit_answer and your answer.
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
        # ═══════ STAGE 1: Initial Panorama ═══════
        logger.info("--- Stage 1: Initial Panorama ---")
        context.start_stage(1)

        pts, angle, mosaic_b64, pano_text = observe_panorama(
            scene, tsdf_planner, pts, angle, total_steps,
            memory_store, cam_intr, cfg, detection_model,
            sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
        )
        total_steps += 1

        # Build rooms/frontiers info string
        rooms_info = _format_rooms_info(tsdf_planner)
        frontiers_info = _format_frontiers_info(tsdf_planner)

        s1_msg = STAGE1_PROMPT.format(question=question)
        context.add_message("system", SYSTEM_PROMPT)
        context.add_message("user", pano_text + "\n" + s1_msg)
        context.add_image(mosaic_b64)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pano_text + "\n" + s1_msg},
        ]
        vlm_response = call_vlm(messages, image_b64=mosaic_b64)
        vlm_parsed = _parse_vlm_response(vlm_response)
        context.add_message("assistant", vlm_response)
        stages_completed = 1
        logger.info(f"Stage 1 VLM: {vlm_parsed}")

        # Determine next stage from VLM (default 2)
        current_stage = int(vlm_parsed.get("next_stage", 2) or 2)
        context.transition(current_stage, vlm_parsed.get("reasoning", "Stage 1 done"))

        # ═══════ STAGE 2-5: Exploration Loop (non-linear) ═══════
        # 决策节奏：VLM 只在导航抵达目标后才被唤醒做下一次决策（类似 choose_every_step=false）。
        # 导航工具内部循环 agent_step 直到抵达，每子步做 silent_perception。
        # max_total_steps 限制的是底层步数（silent_perception 次数），不是 VLM 决策次数。
        stage_visit_count = {}
        max_stage_visits = {2: 3, 3: 6, 4: 4, 5: 2}
        max_stage_decisions = 8  # 每阶段最多 VLM 决策次数

        def _low_level_steps():
            return silent_perception_step._step_counter

        while current_stage in (2, 3, 4, 5) and _low_level_steps() < max_total_steps:
            stage_visit_count[current_stage] = stage_visit_count.get(current_stage, 0) + 1
            if stage_visit_count[current_stage] > max_stage_visits.get(current_stage, 3):
                logger.info(f"Stage {current_stage} visited too many times, forcing advance")
                current_stage = min(current_stage + 1, 5)
                continue

            logger.info(f"--- Stage {current_stage} (visit {stage_visit_count[current_stage]}) ---")
            context.start_stage(current_stage)

            # Build stage prompt
            if current_stage == 2:
                stage_prompt = STAGE2_PROMPT.format(
                    question=question, rooms_info=rooms_info,
                    frontiers_info=frontiers_info)
            elif current_stage == 3:
                target = vlm_parsed.get("target", vlm_parsed.get("arguments", "relevant objects"))
                stage_prompt = STAGE3_PROMPT.format(
                    target=target, rooms_info=rooms_info,
                    frontiers_info=frontiers_info)
            elif current_stage == 4:
                stage_prompt = STAGE4_PROMPT.format(
                    question=question, rooms_info=rooms_info,
                    frontiers_info=frontiers_info)
            else:  # stage 5
                stage_prompt = STAGE5_PROMPT.format(question=question)

            context.add_message("user", stage_prompt)

            stage_decisions = 0
            next_stage = None

            while stage_decisions < max_stage_decisions and _low_level_steps() < max_total_steps:
                messages = _build_messages(context)
                last_img = context.stage_images[-1] if context.stage_images else None
                vlm_response = call_vlm(messages, image_b64=last_img)
                context.add_message("assistant", vlm_response)

                stage_decisions += 1
                total_steps += 1  # VLM-turn counter (for naming/logging only)

                vlm_parsed = _parse_vlm_response(vlm_response)
                tool = vlm_parsed.get("tool", "")
                args = vlm_parsed.get("arguments", "")
                reasoning = vlm_parsed.get("reasoning", "")

                logger.info(f"Stage {current_stage} decision {stage_decisions} (low-level steps={_low_level_steps()}): tool={tool}, args={args}")

                # Check for answer submission
                if tool == "submit_answer":
                    answer = vlm_parsed.get("answer", args)
                    logger.info(f"Answer submitted in stage {current_stage}: {answer}")
                    result["answer"] = answer
                    result["success"] = True
                    result["steps_taken"] = _low_level_steps()
                    result["stages_completed"] = current_stage
                    return result

                # Step budget for navigation tools — prevents overshooting max_total_steps
                step_budget = max_total_steps - _low_level_steps()

                # Execute tool — navigation tools run to arrival internally, VLM wakes after
                obs_text = ""
                obs_image = None

                if tool == "observe_panorama":
                    pts, angle, obs_image, obs_text = observe_panorama(
                        scene, tsdf_planner, pts, angle, total_steps,
                        memory_store, cam_intr, cfg, detection_model,
                        sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
                    )
                    rooms_info = _format_rooms_info(tsdf_planner)
                    frontiers_info = _format_frontiers_info(tsdf_planner)
                    obs_text += f"\n{rooms_info}\n{frontiers_info}"

                elif tool == "view_direction":
                    pts, angle, obs_image = view_direction(
                        scene, tsdf_planner, pts, angle, args,
                        memory_store, cam_intr, cfg, detection_model,
                        sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
                        total_steps,
                    )
                    obs_text = f"Viewed direction: {args}"

                elif tool == "navigate_to_object":
                    pts, angle, success, status, obs_image = navigate_to_object(
                        scene, tsdf_planner, pts, angle, args,
                        memory_store, cam_intr, cfg, detection_model,
                        sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
                        total_steps, step_budget=step_budget,
                    )
                    obs_text = f"navigate_to_object: success={success}, status={status}"
                    rooms_info = _format_rooms_info(tsdf_planner)
                    frontiers_info = _format_frontiers_info(tsdf_planner)
                    obs_text += f"\n{rooms_info}\n{frontiers_info}"

                elif tool == "navigate_to_seed":
                    try:
                        room_id = int(args)
                    except (ValueError, TypeError):
                        room_id = 0
                    pts, angle, success, status, obs_image = navigate_to_seed(
                        scene, tsdf_planner, pts, angle, room_id, cfg,
                        memory_store, cam_intr, detection_model, sam_predictor,
                        clip_model, clip_preprocess, clip_tokenizer, total_steps,
                        step_budget=step_budget,
                    )
                    obs_text = f"navigate_to_seed: success={success}, status={status}"
                    rooms_info = _format_rooms_info(tsdf_planner)
                    frontiers_info = _format_frontiers_info(tsdf_planner)
                    obs_text += f"\n{rooms_info}\n{frontiers_info}"

                elif tool == "navigate_to_frontier":
                    try:
                        frontier_id = int(args)
                    except (ValueError, TypeError):
                        frontier_id = 0
                    pts, angle, success, status, obs_image = navigate_to_frontier(
                        scene, tsdf_planner, pts, angle, frontier_id, cfg,
                        memory_store, cam_intr, detection_model, sam_predictor,
                        clip_model, clip_preprocess, clip_tokenizer, total_steps,
                        step_budget=step_budget,
                    )
                    obs_text = f"navigate_to_frontier: success={success}, status={status}"
                    rooms_info = _format_rooms_info(tsdf_planner)
                    frontiers_info = _format_frontiers_info(tsdf_planner)
                    obs_text += f"\n{rooms_info}\n{frontiers_info}"

                elif tool == "query_memory":
                    obs_image = query_memory(memory_store, args)
                    obs_text = f"Memory query for: {args}" if obs_image else "Memory query returned no results or quota exhausted"

                else:
                    obs_text = f"Tool '{tool}' not recognized."

                # Provide observation back to VLM (VLM wakes up here after navigation arrival)
                context.add_message("user", f"Observation: {obs_text}")
                if obs_image:
                    context.add_image(obs_image)

                # Check explicit next_stage transition from VLM
                ns = vlm_parsed.get("next_stage")
                if ns is not None:
                    try:
                        next_stage = int(ns)
                    except (ValueError, TypeError):
                        next_stage = None
                    if next_stage in (1, 2, 3, 4, 5, 6) and next_stage != current_stage:
                        break

            # Stage transition summary
            summary = vlm_parsed.get("reasoning", f"Stage {current_stage} completed")
            target_stage = next_stage if next_stage is not None else min(current_stage + 1, 6)
            context.transition(target_stage, summary)
            stages_completed = current_stage
            current_stage = target_stage

        # ═══════ STAGE 6: Final Answer ═══════
        if not answer:
            logger.info("--- Stage 6: Submit Answer ---")
            context.start_stage(6)

            s6_prompt = STAGE6_PROMPT.format(question=question)
            context.add_message("user", s6_prompt)

            messages = _build_messages(context)
            vlm_response = call_vlm(messages)
            vlm_parsed = _parse_vlm_response(vlm_response)

            answer = vlm_parsed.get("answer", vlm_parsed.get("arguments", "unanswerable"))
            logger.info(f"Final answer: {answer}")

            result["answer"] = answer
            result["success"] = "unanswerable" not in answer.lower()
            result["steps_taken"] = silent_perception_step._step_counter
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


# ── Helpers ──────────────────────────────────────────────────────────────

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
    """Parse VLM JSON response."""
    if response is None:
        response = ""
    try:
        # Try to find JSON block
        if "```" in response:
            # Extract content between first ``` and last ```
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:]
                try:
                    return json.loads(part.strip())
                except json.JSONDecodeError:
                    continue

        # Try direct JSON parse
        return json.loads(response.strip())
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse VLM response as JSON: {e}")
        return {
            "reasoning": response[:200],
            "tool": "unknown",
            "arguments": "",
            "answer": "",
        }


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
