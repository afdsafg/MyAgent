# Two-Tier Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor HM-GE from free-format VLM loop to Planner-Executor two-tier architecture with Evidence Notebook, matching Qwen-RobotNav's design philosophy.

**Architecture:** Upper Planner (Qwen3.6-Plus via cloud API) makes structured decisions through 4-component prompts, controlling Lower Executor (existing GD+spiral) via 6 structured tools. Evidence Notebook provides persistent cross-stage memory. TrajectoryEvidence compresses executor outputs into compact records.

**Tech Stack:** Python, Habitat simulator, GroundingDINO, SAM, CLIP, Qwen3.6-Plus (Alibaba Cloud API), OpenAI-compatible API client

---

## Worktree Strategy

**Phase 1-3 developed in parallel worktrees, then merged into refactor-two-tier:**

| Worktree | Branch | Tasks | Merged Into |
|---|---|---|---|
| wt-notebook | feat/notebook-evidence | Task 1-2 (Notebook + Evidence) | refactor-two-tier |
| wt-gd-filter | feat/gd-quality-filter | Task 3 (GD filtering) | refactor-two-tier |
| wt-panorama | feat/panorama-improve | Task 4 (Panorama 8-view) | refactor-two-tier |
| wt-planner | feat/planner-executor | Task 5-6 (Planner + Workflow) | refactor-two-tier |
| refactor-two-tier | refactor/two-tier | Merge all + integration | main (after validation) |

---

## Task 1: Evidence Notebook

**Branch:** feat/notebook-evidence
**Files:**
- Create: `src/agent_notebook.py`
- Create: `src/agent_evidence.py`
- Test: `tests/test_notebook.py`

**Step 1: Write the failing test**

```python
# tests/test_notebook.py
import pytest
from src.agent_notebook import EvidenceNotebook, NotebookEntry
from src.agent_evidence import TrajectoryEvidence

def test_add_room_explored_entry():
    nb = EvidenceNotebook()
    entry = nb.add_entry(
        step=15, entry_type="room_explored",
        content="Bedroom (room_5) explored. Objects: [bed, chair, door]. Target oven NOT found.",
        negation=True, confidence=0.9, key_frame_id="snap_step15_view0"
    )
    assert entry.step == 15
    assert entry.negation is True
    assert len(nb.entries) == 1

def test_loop_detection():
    nb = EvidenceNotebook()
    for i in range(3):
        nb.add_entry(step=8+i, entry_type="seed_visited",
                     content=f"Seed_3 visited → dining area (not kitchen)", 
                     negation=False, confidence=0.8)
    # After 3 same (seed_3, "dining area"), should be marked
    assert nb.is_exhausted("seed_3") is True

def test_injection_text():
    nb = EvidenceNotebook()
    nb.add_entry(step=5, entry_type="room_explored",
                 content="Bedroom: bed, chair. Oven NOT found.", negation=True, confidence=0.9)
    nb.add_entry(step=12, entry_type="seed_visited",
                 content="Seed_3 → dining area, not kitchen.", negation=False, confidence=0.7)
    text = nb.get_injection_text(max_entries=5)
    assert "Bedroom" in text
    assert "NOT found" in text

def test_trajectory_evidence_to_entry():
    ev = TrajectoryEvidence(
        subgoal="Navigate to oven via view_2",
        task_mode="navigate_to_object",
        progress="Moved 3 steps toward kitchen",
        salient=["cabinet", "oven-like(score=0.44)"],
        outcome="target_not_reached",
        gd_quality="score_too_low",
        key_frames=["snap_step12_view2"],
        room_id=5,
        objects_nearby=["bed", "chair"]
    )
    entry = ev.to_notebook_entry(step=12)
    assert entry.entry_type == "object_observed"
    assert "oven-like" in entry.content
```

**Step 2: Run test to verify it fails**

```bash
cd /home/afdsafg/下载/new/3D-Mem && python -m pytest tests/test_notebook.py -v
```
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
# src/agent_notebook.py
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

@dataclass
class NotebookEntry:
    step: int
    timestamp: str = ""
    entry_type: str = ""  # room_explored | object_observed | hypothesis_rejected | seed_visited | frontier_visited
    content: str = ""
    negation: bool = False
    confidence: float = 0.0
    key_frame_id: Optional[str] = None

class EvidenceNotebook:
    def __init__(self):
        self.entries: list[NotebookEntry] = []
        self._exhausted_ids: dict[str, int] = defaultdict(int)  # id -> count of same outcome
        self._last_outcomes: dict[str, list[str]] = defaultdict(list)  # id -> list of outcomes
    
    def add_entry(self, step: int, entry_type: str, content: str,
                  negation: bool = False, confidence: float = 0.0,
                  key_frame_id: Optional[str] = None) -> NotebookEntry:
        entry = NotebookEntry(
            step=step, entry_type=entry_type, content=content,
            negation=negation, confidence=confidence, key_frame_id=key_frame_id
        )
        self.entries.append(entry)
        # Track outcomes for loop detection
        if entry_type == "seed_visited":
            seed_id = self._extract_id(content, "Seed_")
            self._last_outcomes[seed_id].append(content)
            self._exhausted_ids[seed_id] += 1
        elif entry_type == "frontier_visited":
            fid = self._extract_id(content, "Frontier_")
            self._last_outcomes[fid].append(content)
            self._exhausted_ids[fid] += 1
        return entry
    
    def is_exhausted(self, entity_id: str) -> bool:
        """Check if entity has been visited 3+ times with same outcome."""
        return self._exhausted_ids.get(entity_id, 0) >= 3
    
    def get_injection_text(self, max_entries: int = 10) -> str:
        """Generate text for Planner prompt injection."""
        recent = self.entries[-max_entries:]
        lines = []
        for e in recent:
            marker = "NOT" if e.negation else ""
            line = f"- [Step {e.step}] {e.content}"
            lines.append(line)
        return "## History\nYou have explored the following:\n" + "\n".join(lines)
    
    def get_visited_seeds(self) -> set[str]:
        """Return set of visited seed IDs."""
        return {self._extract_id(e.content, "Seed_") 
                for e in self.entries if e.entry_type == "seed_visited"}
    
    def get_visited_frontiers(self) -> set[str]:
        """Return set of visited frontier IDs."""
        return {self._extract_id(e.content, "Frontier_") 
                for e in self.entries if e.entry_type == "frontier_visited"}
    
    def _extract_id(self, content: str, prefix: str) -> str:
        """Extract entity ID from content like 'Seed_3 visited...'"""
        import re
        match = re.search(f'{prefix}(\\d+)', content)
        return f"{prefix}{match.group(1)}" if match else ""
    
    def update_from_evidence(self, evidence: 'TrajectoryEvidence', step: int):
        """Convert TrajectoryEvidence to NotebookEntry and add."""
        entry = evidence.to_notebook_entry(step)
        self.entries.append(entry)
```

```python
# src/agent_evidence.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrajectoryEvidence:
    subgoal: str
    task_mode: str
    progress: str
    salient: list[str]
    outcome: str  # arrived_near_target | target_not_reached | detection_failed | object_found
    gd_quality: str = "ok"  # ok | bbox_too_large | score_too_low | no_detection
    key_frames: list[str] = []
    room_id: int = -1
    objects_nearby: list = []
    
    def to_notebook_entry(self, step: int):
        """Convert to NotebookEntry based on outcome."""
        from src.agent_notebook import NotebookEntry
        
        if self.outcome == "detection_failed":
            entry_type = "hypothesis_rejected"
            content = f"GD detection failed for '{self.subgoal}': {self.gd_quality}. Objects nearby: {self.objects_nearby}."
            negation = True
        elif self.outcome == "object_found":
            entry_type = "object_observed"
            content = f"Object observed: {self.subgoal}. Salient: {', '.join(self.salient)}. Room {self.room_id}."
            negation = False
        elif self.task_mode == "explore_seed":
            entry_type = "seed_visited"
            content = f"Seed visited: {self.subgoal}. Arrived at room {self.room_id}. Objects: {self.objects_nearby}."
            negation = "NOT" in self.progress
        elif self.task_mode == "explore_frontier":
            entry_type = "frontier_visited"
            content = f"Frontier visited: {self.subgoal}. Arrived at room {self.room_id}. Outcome: {self.outcome}."
            negation = "NOT" in self.progress
        else:
            entry_type = "room_explored"
            content = f"Room {self.room_id} explored. Objects: {self.objects_nearby}. Progress: {self.progress}."
            negation = "NOT" in self.progress
        
        return NotebookEntry(
            step=step, entry_type=entry_type, content=content,
            negation=negation, confidence=0.7,
            key_frame_id=self.key_frames[0] if self.key_frames else None
        )
```

**Step 4: Run test to verify it passes**

```bash
cd /home/afdsafg/下载/new/3D-Mem && python -m pytest tests/test_notebook.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agent_notebook.py src/agent_evidence.py tests/test_notebook.py
git commit -m "feat: add EvidenceNotebook and TrajectoryEvidence data classes"
```

---

## Task 2: GD Quality Filtering + Navigation Parameter Adjustments

**Branch:** feat/gd-quality-filter
**Files:**
- Modify: `src/scene_aeqa.py:985-1018` (_gd_detect) and `src/scene_aeqa.py:1021-1332` (grounded_navigate_to_object)
- Modify: `src/agent_tools.py:266-307` (navigate_to_object)
- Test: `tests/test_gd_filter.py`

**Step 1: Write the failing test**

```python
# tests/test_gd_filter.py
import numpy as np
from src.scene_aeqa import gd_quality_filter

def test_bbox_too_large():
    # bbox covers 60% of 1280x1280 image
    bbox = np.array([0, 0, 768, 768])
    result, reason = gd_quality_filter(bbox, score=0.5, image_shape=(1280, 1280))
    assert result is None
    assert reason == "bbox_too_large"

def test_score_too_low():
    # small bbox but low score
    bbox = np.array([100, 100, 200, 200])
    result, reason = gd_quality_filter(bbox, score=0.25, image_shape=(1280, 1280))
    assert result is None
    assert reason == "score_too_low"

def test_good_detection():
    bbox = np.array([100, 100, 300, 300])
    result, reason = gd_quality_filter(bbox, score=0.55, image_shape=(1280, 1280))
    assert result is not None
    assert reason == "ok"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_gd_filter.py -v
```
Expected: FAIL (function not found)

**Step 3: Add gd_quality_filter function**

In `src/scene_aeqa.py`, add before `_gd_detect` (around line 985):

```python
def gd_quality_filter(bbox, score, image_shape, max_bbox_ratio=0.30, min_score=0.35):
    """Filter GD detections by bbox area ratio and confidence score.
    
    Args:
        bbox: [x1, y1, x2, y2] detection box
        score: detection confidence
        image_shape: (height, width) of the image
        max_bbox_ratio: maximum bbox area / image area ratio (default 0.30)
        min_score: minimum confidence score (default 0.35)
    
    Returns:
        (bbox, "ok") if quality passes, (None, reason) if rejected
    """
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    image_area = image_shape[0] * image_shape[1]
    ratio = bbox_area / image_area
    
    if ratio > max_bbox_ratio:
        return None, "bbox_too_large"
    if score < min_score:
        return None, "score_too_low"
    return bbox, "ok"
```

**Step 4: Integrate filter into grounded_navigate_to_object**

In `src/scene_aeqa.py`, inside `grounded_navigate_to_object`, after `_gd_detect` call (around line 1078):

```python
# After getting best_bbox, best_score from _gd_detect:
filtered_bbox, filter_reason = gd_quality_filter(
    best_bbox, best_score, (rgb.shape[0], rgb.shape[1])
)
if filtered_bbox is None:
    # Log rejection and skip this detection attempt
    print(f"[GD Filter] Rejected detection: {filter_reason} (score={best_score:.2f}, bbox={best_bbox})")
    continue  # try next detect_attempt with different angle
# Use filtered_bbox instead of best_bbox for SAM segmentation
best_bbox = filtered_bbox
```

Also modify the converge_dist_voxels default:
- Change default parameter from 5 to 12 at line ~1029
- Add arrival verification after navigation completion

**Step 5: Add arrival verification**

In `src/agent_tools.py`, inside `navigate_to_object` (around line 290), after `grounded_navigate_to_object` returns:

```python
# After GD navigation completes:
if success:
    # Verify arrival: check Euclidean distance to target
    from src.geom import pos_habitat_to_normal
    agent_normal = pos_habitat_to_normal(np.array([pts[0], pts[1], 0]))
    # If we have target_voxel from GD, check distance
    # Otherwise trust the spiral convergence result
    # For now, add a flag in the return value
```

**Step 6: Run test to verify it passes**

```bash
python -m pytest tests/test_gd_filter.py -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add src/scene_aeqa.py src/agent_tools.py tests/test_gd_filter.py
git commit -m "feat: add GD quality filter (bbox>30% reject, score<0.35 reject) + converge_dist=12"
```

---

## Task 3: Panorama Improvements (8-view + labels + resolution)

**Branch:** feat/panorama-improve
**Files:**
- Modify: `src/agent_tools.py:165-228` (observe_panorama)
- Modify: `src/scene_aeqa.py` (get_observation if needed)
- Test: `tests/test_panorama.py`

**Step 1: Write the failing test**

```python
# tests/test_panorama.py
import numpy as np
from src.agent_tools import observe_panorama_config

def test_panorama_8_views():
    config = observe_panorama_config()
    assert config["view_count"] == 8
    assert config["resolution"] == 400
    assert len(config["view_labels"]) == 8
    assert "Front" in config["view_labels"]

def test_view_angles_8():
    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    assert len(angles) == 8
    # Should cover full 360 degrees
    assert angles[0] == -np.pi
    assert abs(angles[-1] - (np.pi - np.pi/4)) < 0.01
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_panorama.py -v
```
Expected: FAIL

**Step 3: Modify observe_panorama in agent_tools.py**

Key changes to `observe_panorama` (line 165-228):

1. Change from 7 views to 8 views:
```python
# OLD: angles = np.linspace(-math.pi, math.pi, 7, endpoint=False)
# NEW:
n_views = 8
angles = np.linspace(-math.pi, math.pi, n_views, endpoint=False)
```

2. Add view labels:
```python
VIEW_LABELS = ["Front", "Front-Right", "Right", "Back-Right", 
               "Back", "Back-Left", "Left", "Front-Left"]
```

3. Increase resolution:
```python
# OLD: target_h = 200
# NEW:
target_h = 400
```

4. Add cam_pose storage:
```python
cam_poses = []  # store each view's cam_pose for GD use
for i, (view_angle, label) in enumerate(zip(angles, VIEW_LABELS)):
    obs, cam_pose = scene.get_observation(pts, view_angle)
    cam_poses.append(cam_pose)
```

5. Add YOLO object annotations per view:
```python
# After each view's scene graph update, collect detected objects
view_objects = [obj for obj in scene.objects.values() if obj.detected_in_current_frame]
objects_per_view.append(view_objects)
```

6. Add text annotations on mosaic:
```python
# Add "Front View: [bed, door]" labels next to each view in mosaic
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_panorama.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agent_tools.py src/scene_aeqa.py tests/test_panorama.py
git commit -m "feat: panorama 8-view + labels + resolution 400 + cam_pose storage"
```

---

## Task 4: Planner Module (Qwen3.6-Plus API + 4-component prompt)

**Branch:** feat/planner-executor
**Files:**
- Create: `src/agent_planner.py`
- Create: `src/agent_executor.py`
- Modify: `src/const.py` (add Qwen3.6-Plus API config)
- Test: `tests/test_planner.py`

**Step 1: Write the failing test**

```python
# tests/test_planner.py
import pytest
from src.agent_planner import Planner, PlannerAction

def test_parse_planner_response():
    planner = Planner(api_key="test", base_url="test")
    response = '{"reason": "oven likely in kitchen", "action": "navigate_to_object", "object_name": "oven", "confidence": 0.7}'
    action = planner.parse_response(response)
    assert action.action_type == "navigate_to_object"
    assert action.object_name == "oven"
    assert action.confidence == 0.7

def test_build_prompt_components():
    planner = Planner(api_key="test", base_url="test")
    history = "## History\n- [Step 5] Bedroom: no oven found"
    scene = "## Scene Analysis\n- View 0: [cabinet, door]"
    progress = "## Progress\nTarget oven not found. Kitchen to explore."
    actions = "## Actions\n1. navigate_to_object\n2. explore_seed"
    prompt = planner.build_prompt(question="What color is the towel?", 
                                   history=history, scene=scene,
                                   progress=progress, actions=actions)
    assert "History" in prompt
    assert "Scene Analysis" in prompt
    assert "Progress" in prompt
    assert "Actions" in prompt

def test_planner_action_dataclass():
    action = PlannerAction(action_type="explore_seed", seed_id="3", confidence=0.6, reason="seed 3 near kitchen")
    assert action.action_type == "explore_seed"
    assert action.seed_id == "3"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_planner.py -v
```
Expected: FAIL

**Step 3: Write agent_planner.py**

```python
# src/agent_planner.py
"""Upper Planner: Qwen3.6-Plus API + 4-component structured prompt."""
from dataclasses import dataclass
from typing import Optional
import json
import openai

@dataclass
class PlannerAction:
    action_type: str  # explore_panorama | navigate_to_object | explore_seed | explore_frontier | inspect_object | submit_answer
    reason: str = ""
    confidence: float = 0.0
    object_name: Optional[str] = None
    seed_id: Optional[str] = None
    frontier_id: Optional[str] = None
    view_idx: Optional[int] = None
    answer: Optional[str] = None

class Planner:
    def __init__(self, api_key: str, base_url: str, model_name: str = "qwen3.6-plus"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_rounds = 10
    
    def decide(self, question: str, history: str, scene: str, 
               progress: str, actions: str, image_b64: Optional[str] = None) -> PlannerAction:
        """Call Qwen3.6-Plus with 4-component prompt, parse action."""
        prompt = self.build_prompt(question, history, scene, progress, actions)
        messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
        
        if image_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        response_text = self._call_api(messages)
        return self.parse_response(response_text)
    
    def build_prompt(self, question, history, scene, progress, actions):
        return f"""Answer this embodied question by navigating to find the answer.

Question: {question}

{history}

{scene}

{progress}

{actions}

Output your decision as JSON: {"reason": "...", "action": "...", "confidence": 0.0-1.0, [optional: "object_name"/"seed_id"/"frontier_id"/"view_idx"/"answer"]}"""
    
    def parse_response(self, response: str) -> PlannerAction:
        """Parse VLM JSON response into PlannerAction."""
        # Try to extract JSON from response
        try:
            # Find JSON block in response
            json_str = response
            if "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            data = json.loads(json_str)
            return PlannerAction(
                action_type=data.get("action", "explore_panorama"),
                reason=data.get("reason", ""),
                confidence=data.get("confidence", 0.5),
                object_name=data.get("object_name"),
                seed_id=data.get("seed_id"),
                frontier_id=data.get("frontier_id"),
                view_idx=data.get("view_idx"),
                answer=data.get("answer")
            )
        except json.JSONDecodeError:
            return PlannerAction(action_type="explore_panorama", reason="Failed to parse VLM response", confidence=0.0)
    
    def _call_api(self, messages):
        """Call Qwen3.6-Plus via OpenAI-compatible API."""
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content

PLANNER_SYSTEM_PROMPT = """You are a high-level navigation planner for an embodied agent. 
Your job is to decide what the agent should do next to find the answer to a question.
You receive structured information about what has been explored, what is currently visible, and how much progress has been made.
You must output a JSON decision with a reason, action type, and confidence score.
Do NOT repeat actions that have already been tried with the same outcome.
Be strategic: use the History and Progress information to avoid redundant exploration."""
```

**Step 4: Write agent_executor.py**

```python
# src/agent_executor.py
"""Executor interface: wraps 6 structured tools."""
from src.agent_evidence import TrajectoryEvidence

class Executor:
    def __init__(self, scene, tsdf_planner, memory_store, cfg, 
                 detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer):
        self.scene = scene
        self.tsdf = tsdf_planner
        self.memory = memory_store
        self.cfg = cfg
        self.models = {
            "detection": detection_model,
            "sam": sam_predictor,
            "clip": clip_model,
            "clip_preprocess": clip_preprocess,
            "clip_tokenizer": clip_tokenizer
        }
        self._pts = None
        self._angle = None
        self._step_counter = 0
    
    def set_state(self, pts, angle, step_counter):
        self._pts = pts
        self._angle = angle
        self._step_counter = step_counter
    
    def explore_panorama(self, config: dict = None) -> TrajectoryEvidence:
        """Take 8-view panorama and return evidence."""
        from src.agent_tools import observe_panorama
        if config is None:
            config = {"view_count": 8, "resolution": 400}
        
        pts, angle, rooms_info, frontiers_info = observe_panorama(
            self.scene, self.tsdf, self._pts, self._angle, 
            self._step_counter, self.memory, self.scene.cam_intr,
            self.cfg, self.models["detection"], self.models["sam"],
            self.models["clip"], self.models["clip_preprocess"], 
            self.models["clip_tokenizer"]
        )
        self._pts, self._angle = pts, angle
        
        # Collect nearby objects from scene graph
        objects_nearby = list(self.scene.objects.keys()) if hasattr(self.scene, 'objects') else []
        room_id = self.tsdf.get_room_id_at(pts) if hasattr(self.tsdf, 'get_room_id_at') else -1
        
        return TrajectoryEvidence(
            subgoal="Explore panorama for re-orientation",
            task_mode="explore_panorama",
            progress=f"8-view panorama taken at room {room_id}",
            salient=rooms_info.split(", ") if rooms_info else [],
            outcome="panorama_complete",
            room_id=room_id,
            objects_nearby=objects_nearby
        )
    
    def navigate_to_object(self, object_name: str, view_idx: int = None) -> TrajectoryEvidence:
        """Navigate to object using GD detection."""
        from src.agent_tools import navigate_to_object
        pts, angle, success, status, image = navigate_to_object(
            self.scene, self.tsdf, self._pts, self._angle, object_name,
            self.memory, self.scene.cam_intr, self.cfg,
            self.models["detection"], self.models["sam"],
            self.models["clip"], self.models["clip_preprocess"],
            self.models["clip_tokenizer"], self._step_counter
        )
        self._pts, self._angle = pts, angle
        
        gd_quality = "ok" if success else ("detection_failed" if "GD" in status else "target_not_reached")
        room_id = self.tsdf.get_room_id_at(pts) if hasattr(self.tsdf, 'get_room_id_at') else -1
        objects_nearby = list(self.scene.objects.keys()) if hasattr(self.scene, 'objects') else []
        
        return TrajectoryEvidence(
            subgoal=f"Navigate to {object_name} via view {view_idx}",
            task_mode="navigate_to_object",
            progress=f"Navigation status: {status}",
            salient=[object_name, status],
            outcome="arrived_near_target" if success else "target_not_reached",
            gd_quality=gd_quality,
            room_id=room_id,
            objects_nearby=objects_nearby
        )
    
    def explore_seed(self, seed_id: str) -> TrajectoryEvidence:
        """Navigate to seed viewpoint."""
        from src.agent_tools import navigate_to_seed
        pts, angle, success, status, image = navigate_to_seed(
            self.scene, self.tsdf, self._pts, self._angle, seed_id,
            self.cfg, self.memory, self.scene.cam_intr,
            self.models["detection"], self.models["sam"],
            self.models["clip"], self.models["clip_preprocess"],
            self.models["clip_tokenizer"], self._step_counter
        )
        self._pts, self._angle = pts, angle
        
        room_id = self.tsdf.get_room_id_at(pts) if hasattr(self.tsdf, 'get_room_id_at') else -1
        objects_nearby = list(self.scene.objects.keys()) if hasattr(self.scene, 'objects') else []
        
        return TrajectoryEvidence(
            subgoal=f"Navigate to seed {seed_id}",
            task_mode="explore_seed",
            progress=f"Arrived at seed {seed_id}, room {room_id}",
            salient=[f"seed_{seed_id}", f"room_{room_id}"],
            outcome="arrived_near_target" if success else "target_not_reached",
            room_id=room_id,
            objects_nearby=objects_nearby
        )
    
    def explore_frontier(self, frontier_id: str) -> TrajectoryEvidence:
        """Navigate to frontier."""
        from src.agent_tools import navigate_to_frontier
        pts, angle, success, status, image = navigate_to_frontier(
            self.scene, self.tsdf, self._pts, self._angle, frontier_id,
            self.cfg, self.memory, self.scene.cam_intr,
            self.models["detection"], self.models["sam"],
            self.models["clip"], self.models["clip_preprocess"],
            self.models["clip_tokenizer"], self._step_counter
        )
        self._pts, self._angle = pts, angle
        
        room_id = self.tsdf.get_room_id_at(pts) if hasattr(self.tsdf, 'get_room_id_at') else -1
        objects_nearby = list(self.scene.objects.keys()) if hasattr(self.scene, 'objects') else []
        
        return TrajectoryEvidence(
            subgoal=f"Navigate to frontier {frontier_id}",
            task_mode="explore_frontier",
            progress=f"Arrived at frontier {frontier_id}, room {room_id}",
            salient=[f"frontier_{frontier_id}", f"room_{room_id}"],
            outcome="arrived_near_target" if success else "target_not_reached",
            room_id=room_id,
            objects_nearby=objects_nearby
        )
    
    def inspect_object(self, object_name: str) -> TrajectoryEvidence:
        """Stay in place, multi-view close inspection."""
        from src.agent_tools import silent_perception_step
        # Take 3 close views at current position
        pts, angle = silent_perception_step(
            self.scene, self.tsdf, self._pts, self._angle,
            self._step_counter, self.memory, self.scene.cam_intr, self.cfg,
            self.models["detection"], self.models["sam"],
            self.models["clip"], self.models["clip_preprocess"],
            self.models["clip_tokenizer"]
        )
        self._pts, self._angle = pts, angle
        
        room_id = self.tsdf.get_room_id_at(pts) if hasattr(self.tsdf, 'get_room_id_at') else -1
        objects_nearby = list(self.scene.objects.keys()) if hasattr(self.scene, 'objects') else []
        
        return TrajectoryEvidence(
            subgoal=f"Inspect {object_name} at current position",
            task_mode="inspect_object",
            progress=f"Close inspection of {object_name}",
            salient=[object_name],
            outcome="inspection_complete",
            room_id=room_id,
            objects_nearby=objects_nearby
        )
    
    def execute_action(self, action) -> TrajectoryEvidence:
        """Dispatch action to appropriate executor tool."""
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
        else:
            return TrajectoryEvidence(
                subgoal="Unknown action", task_mode="unknown",
                progress="Unknown action type", outcome="error",
                salient=[], gd_quality="no_detection"
            )
```

**Step 5: Add Qwen3.6-Plus API config to const.py**

```python
# Add to src/const.py:
QWEN_PLANNER_API_KEY = "<to-be-configured>"  # Qwen3.6-Plus API key
QWEN_PLANNER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # Alibaba Cloud DashScope API
QWEN_PLANNER_MODEL = "qwen-plus-latest"  # or "qwen3.6-plus" when available
```

**Step 6: Run test to verify it passes**

```bash
python -m pytest tests/test_planner.py -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add src/agent_planner.py src/agent_executor.py src/const.py tests/test_planner.py
git commit -m "feat: add Planner (Qwen3.6-Plus) and Executor (6-tool interface) modules"
```

---

## Task 5: Workflow Rewrite (Planner-Executor Loop)

**Branch:** feat/planner-executor (same as Task 4)
**Files:**
- Rewrite: `src/agent_workflow.py`
- Modify: `src/agent_context.py` (preserve notebook on transition)

**Step 1: Rewrite run_episode in agent_workflow.py**

The new `run_episode` replaces the free VLM loop with a structured Planner-Executor loop:

```python
def run_episode_two_tier(scene_id, question, question_id, cfg, 
                          detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
                          output_dir="/root/MyAgent/results/hmge",
                          max_planner_rounds=10, max_total_steps=50):
    """Two-tier Planner-Executor episode loop."""
    
    # Initialize components
    scene = Scene(scene_id, cfg, ...)
    tsdf_planner = TSDFPlanner(...)
    memory_store = MemoryStore(output_dir)
    notebook = EvidenceNotebook()
    planner = Planner(api_key=QWEN_PLANNER_API_KEY, base_url=QWEN_PLANNER_BASE_URL)
    executor = Executor(scene, tsdf_planner, memory_store, cfg, ...)
    
    # Step 1: Initial panorama
    pts, angle = scene.init_agent()
    executor.set_state(pts, angle, 0)
    evidence = executor.explore_panorama()
    notebook.update_from_evidence(evidence, step=0)
    
    # Planner-Executor loop
    for round_num in range(max_planner_rounds):
        # Build 4-component prompt
        history = notebook.get_injection_text()
        scene_analysis = build_scene_analysis(scene, tsdf_planner)
        progress = build_progress(question, notebook, round_num, max_planner_rounds)
        actions = build_available_actions(notebook, tsdf_planner, scene)
        
        # Optionally include panorama image
        image_b64 = evidence.key_frames[0] if evidence.key_frames else None
        
        # Planner decision
        action = planner.decide(question, history, scene_analysis, progress, actions, image_b64)
        
        # Check if submit_answer
        if action.action_type == "submit_answer":
            return {"answer": action.answer, "steps": executor._step_counter}
        
        # Executor execution
        evidence = executor.execute_action(action)
        notebook.update_from_evidence(evidence, step=executor._step_counter)
        
        # Loop detection
        if notebook.is_exhausted(action.seed_id or action.frontier_id or ""):
            # Force strategy switch
            continue
        
        # Step budget check
        if executor._step_counter >= max_total_steps:
            break
    
    # Fallback: submit best guess based on notebook
    fallback_action = planner.decide(
        question, notebook.get_injection_text(), 
        "## Scene Analysis\nNo current observation (budget exhausted)",
        "## Progress\nBudget exhausted. Must submit answer now.",
        "## Actions\n1. submit_answer(your best guess)",
        image_b64=None
    )
    return {"answer": fallback_action.answer or "unknown", "steps": executor._step_counter}
```

**Step 2: Modify ContextManager to preserve notebook**

In `src/agent_context.py`, modify `start_stage` method (line 29-33):

```python
# OLD: clears everything
# NEW: preserve notebook reference
def start_stage(self, stage_num: int, notebook: EvidenceNotebook = None):
    self.current_stage = stage_num
    self.stage_messages = []
    self.stage_images = []
    self.notebook = notebook  # Notebook persists across stages
```

**Step 3: Commit**

```bash
git add src/agent_workflow.py src/agent_context.py
git commit -m "feat: rewrite run_episode as Planner-Executor two-tier loop + preserve notebook"
```

---

## Task 6: Integration Test + Merge

**Branch:** refactor/two-tier
**Files:**
- All merged files from previous worktrees

**Step 1: Create refactor-two-tier worktree**

```bash
cd /home/afdsafg/下载/new/3D-Mem
git worktree add .worktrees/refactor-two-tier -b refactor/two-tier
```

**Step 2: Merge feature branches**

```bash
cd .worktrees/refactor-two-tier
git merge feat/notebook-evidence
git merge feat/gd-quality-filter
git merge feat/panorama-improve
git merge feat/planner-executor
```

**Step 3: Run end-to-end integration test**

```bash
python -m pytest tests/test_notebook.py tests/test_gd_filter.py tests/test_panorama.py tests/test_planner.py -v
```

**Step 4: Run the actual agent on test scene**

```bash
# Run on same scene as original trace for comparison
python src/scene_aeqa.py --scene_id=<same_scene> --question="What color is the towel on the oven?" --output_dir=<new_results_dir>
```

**Step 5: Compare results with original trace**

- Compare: steps taken, time elapsed, whether answer found, loop count
- Expected: <10 Planner rounds, no death loops, answer submitted

**Step 6: Commit and report**

```bash
git add -A && git commit -m "feat: complete two-tier refactor integration"
```

---

## Summary: Task Dependencies & Parallel Execution

```
Task 1 (Notebook+Evidence) ──┐
Task 2 (GD Filter)          ──┤── parallel worktrees ──┐
Task 3 (Panorama 8-view)    ──┤                        ├── merge into refactor/two-tier
                              │                        │
Task 4 (Planner+Executor)   ──┤── same worktree       ──┤
Task 5 (Workflow rewrite)   ──┘── same worktree       ──┤
                                                     │
Task 6 (Integration+Merge) ──────────────────────────┘
```

Tasks 1, 2, 3 can run in **parallel** (no dependencies).
Task 4 depends on Task 1 (imports TrajectoryEvidence).
Task 5 depends on Tasks 1-4.
Task 6 depends on all previous tasks.
