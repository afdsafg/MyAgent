# Two-Tier Architecture Refactor Design

> Inspired by Qwen-RobotNav (2606.18112v2): Planner-Executor 二层架构 + Evidence Notebook + 结构化推理链

**Date**: 2026-06-24
**Status**: Design Approved
**Worktree**: refactor-two-tier (to be created from main)

---

## 1. Problem Statement

Current HM-GE agent uses a free-format VLM decision loop where the VLM (mimo-v2.5) freely chooses tools at each stage. This causes:

- **P0-1**: Design doc defines 6-stage state machine but code uses free loop — architecture-code mismatch
- **P0-2**: VLM death loop (17min, seed_3 chosen 7x) — no visited-tracking, no termination mechanism
- **P0-3**: GD navigation extremely inefficient (bbox covers 50%+ image, 1-step "arrival") — no quality filtering
- **P0-4**: Each VLM call is independent, zero history context — no information accumulation

Qwen-RobotNav solves similar problems with: structured two-tier architecture, evidence notebook, parameterized interface, structured reasoning chain.

---

## 2. Architecture: Two-Tier Planner-Executor

### Upper Planner (Qwen3.6-Plus via cloud API)

- Responsible for: high-level reasoning, task decomposition, memory management, dynamic strategy switching
- Does NOT directly operate on raw observation streams
- Controls executor through **structured tool interface**: `nav_call(sub_goal, task_mode, config)`
- Maintains Evidence Notebook as persistent cross-stage memory
- Each decision injects: notebook entries + current stage perception info
- Max 10 decision rounds per episode; fallback submits best guess

### Lower Executor (existing GD + spiral search + silent_perception)

- Remains unchanged internally, but called through structured interface by Planner
- After execution, returns **TrajectoryEvidence** (compact JSON-like record, not raw observation stream)
- Includes: subgoal, task_mode, progress, salient observations, outcome, GD quality, key_frames, room_id, nearby objects

### Core Interface

```python
def nav_call(sub_goal: str, task_mode: str, config: dict) -> TrajectoryEvidence:
    """
    task_mode: 'explore_panorama' | 'navigate_to_object' | 'explore_seed' | 'explore_frontier' | 'inspect_object'
    sub_goal: natural language description
    config: {view_count, resolution, max_steps, view_idx, object_name, seed_id, frontier_id}
    """
```

---

## 3. Evidence Notebook

### Data Model

```python
class NotebookEntry:
    step: int
    timestamp: str
    entry_type: str  # 'room_explored' | 'object_observed' | 'hypothesis_rejected' | 'seed_visited' | 'frontier_visited'
    content: str     # natural language description
    negation: bool   # whether this is negative information (e.g., "oven NOT found here")
    confidence: float  # 0-1
    key_frame_id: str | None  # retrievable visual evidence index
```

### 5 Entry Types

1. **room_explored**: `[step 15] Bedroom (room_5) explored. Objects: [bed, chair, door]. Target oven NOT found here.`
2. **object_observed**: `[step 12] Oven-like object at view_2, score=0.44, bbox 50%+ (low quality). Key frame: snap_step12_view2.`
3. **hypothesis_rejected**: `[step 20] Hypothesis "seed_3 → kitchen" REJECTED. Seed_3 actually in dining area (room_7).`
4. **seed_visited**: `[step 8] Seed_3 visited → dining area (room_7), not kitchen. Kitchen visible through doorway.`
5. **frontier_visited**: `[step 22] Frontier_14 visited → still bedroom area. No kitchen access from this frontier.`

### Key Mechanisms

- **Negation explicitly recorded**: every exploration records "target NOT found"
- **Belief revision**: new entries can revise old ones; update history remains auditable
- **Planner injection**: recent 5-10 entries injected into Planner prompt as "History" component
- **Loop detection**: if 3x same (seed_id, outcome) → mark as "not worth retrying"
- **Survives context compression**: notebook entries are NOT cleared between stages

---

## 4. Planner Prompt: 4-Component Structured Reasoning Chain

Inspired by Qwen-RobotNav's Structured Multi-Perspective Reasoning (History / Scene Analysis / Instruction Progress / Action Reasoning).

### Component 1: History

```
## History
You have explored the following areas so far:
- [Step 15] Bedroom (room_5): bed, chair, door. Oven NOT found.
- [Step 20] Dining area (room_7): table, cabinet. Kitchen entrance visible.
- [Step 22] Seed_3 visited 2x → always dining area, NOT kitchen.
Target: oven — still not located. Kitchen remains most likely room.
```

### Component 2: Scene Analysis

```
## Current Scene Analysis (from object detector)
- View 0 (Front): [cabinet, doorway, white_wall]
- View 1 (Left-60°): [bed_frame, nightstand]
- View 2 (Right-60°): [hallway, picture_frame]
Nearby objects from scene graph: [bed, chair, cabinet] in room_5
```

### Component 3: Progress

```
## Task Progress
Question: "What color is the towel on the oven?"
Status: Target (oven) NOT yet found.
Explored rooms: bedroom, dining area — 2 of ~5 rooms.
Remaining candidates: kitchen (high probability), bathroom, hallway.
```

### Component 4: Action

```
## Available Actions
1. explore_panorama: Take new 8-view panorama for re-orientation
2. navigate_to_object(object_name): Use GD to navigate toward specific object
3. explore_seed(seed_id): Navigate to seed viewpoint (available: seed_1[kitchen?], seed_5[bathroom])
4. explore_frontier(frontier_id): Navigate to unexplored frontier
5. submit_answer(answer): If you've seen the answer in previous observations
6. inspect_object(object_name): Stay and carefully examine a previously glimpsed object

Output: {"reason": "...", "action": "...", "confidence": 0.0-1.0}
```

Key improvements:
- Planner sees **complete history** not "starting from zero"
- **Scene graph detection results** injected into Scene Analysis
- **Negation and progress** help Planner judge when to submit_answer
- **New actions**: inspect_object and explore_panorama solve Stage 5's 3-narrow-view limitation
- **Visited seeds/frontiers removed** from action options

---

## 5. Executor Interface & GD Quality Filtering

### 6 Structured Tools

```python
# Tool 1: Panorama re-orientation
explore_panorama(config: PanoramaConfig) -> TrajectoryEvidence
  PanoramaConfig: {view_count: 8, resolution: 400, show_labels: true}

# Tool 2: Navigate to specific object
navigate_to_object(object_name: str, view_idx: int | None) -> TrajectoryEvidence

# Tool 3: Navigate to seed point
explore_seed(seed_id: str) -> TrajectoryEvidence  # only unvisited seeds

# Tool 4: Navigate to frontier
explore_frontier(frontier_id: str) -> TrajectoryEvidence  # only unvisited frontiers

# Tool 5: Inspect object in place
inspect_object(object_name: str) -> TrajectoryEvidence  # no movement, multi-view close inspection

# Tool 6: Submit answer
submit_answer(answer: str) -> FinalResult
```

### GD Quality Filtering (NEW)

```python
def gd_quality_filter(detection):
    bbox_area_ratio = detection.bbox_area / image_area
    if bbox_area_ratio > 0.30:  # bbox covers >30% of image → reject
        return None, "bbox_too_large"
    if detection.score < 0.35:  # confidence too low → reject
        return None, "score_too_low"
    return detection, "ok"
```

### Navigation Parameter Adjustments

| Parameter | Current | New | Reason |
|---|---|---|---|
| converge_dist_voxels | 5 | 12 | Prevent premature "arrival" (~0.6m threshold) |
| Arrival verification | None | Euclidean distance < 1.5m | GD claims "arrived" but may be far from target |
| GD bbox filter | None | >30% reject | Prevent full-image bbox |
| GD score filter | None | <0.35 reject | Prevent low-quality detection |

### TrajectoryEvidence Return Format

```python
class TrajectoryEvidence:
    subgoal: str           # "Navigate to oven via view_2"
    task_mode: str         # "navigate_to_object"
    progress: str          # "Moved 3 steps toward kitchen area"
    salient: list[str]     # ["cabinet", "doorway_to_kitchen", "oven-like(score=0.44, low_quality)"]
    outcome: str           # "arrived_near_target" | "target_not_reached" | "detection_failed"
    gd_quality: str        # "ok" | "bbox_too_large" | "score_too_low"
    key_frames: list[str]  # ["snap_step23_view0", "snap_step23_view2"]
    room_id: str           # current room
    objects_nearby: list   # YOLO detected nearby objects
```

---

## 6. Episode Flow

```
Episode Start:
  1. Executor: silent_perception init + render 8-view panorama
  2. Executor returns -> TrajectoryEvidence(progress="initial position", salient=[objects])

  Loop (max 10 Planner decision rounds):
    3. Planner: inject notebook + last evidence + 4-component prompt -> choose action

    4a. action=explore_panorama:
        Executor: render 8-view panorama -> return evidence + object lists

    4b. action=navigate_to_object("oven", view_idx=2):
        Executor: GD detect at view_idx=2 -> quality filter ->
        if GD ok -> spiral navigate -> arrival verification -> return evidence
        if GD failed -> return evidence(outcome="detection_failed")
        -> Planner re-decides

    4c. action=explore_seed(seed_id=1):
        Executor: navigate to seed -> silent_perception -> return evidence
        Notebook: add seed_visited entry

    4d. action=inspect_object("oven"):
        Executor: no movement, multi-view close inspection -> return evidence (high resolution)

    4e. action=submit_answer("white"):
        -> Episode ends

    5. Notebook: update entries based on evidence
    6. Loop detection: if 3x same (action, outcome) -> force strategy switch

  Fallback (if 10 rounds without submit):
    Planner: final reasoning based on all notebook entries -> submit_best_guess
```

---

## 7. Panorama Improvements

| Parameter | Current | New | Reason |
|---|---|---|---|
| View count | 7 | 8 | Match design doc, support view_idx reference |
| Layout | 1-row mosaic | 3x3 grid | More intuitive spatial layout |
| Resolution (target_h) | 200 | 400 | Small objects visible (oven handle, towel) |
| Direction labels | None | "Front View"/"Right View" etc. | Help VLM understand spatial relationships |
| Front view emphasis | Equal | Larger in mosaic (2 cells vs 1) | Front view has richest actionable cues |
| Object list per view | None | YOLO results as text annotation | Scene Analysis component for Planner |
| cam_pose storage | None | Store 8 cam_poses for GD use | GD can use VLM-selected view direction |

---

## 8. File Structure & Module Responsibilities

### New Files

```
src/agent_planner.py    # Planner layer: Qwen3.6-Plus API + decision logic
src/agent_notebook.py   # EvidenceNotebook class: persistent memory management
src/agent_evidence.py   # TrajectoryEvidence class: structured return format
src/agent_executor.py   # Executor interface: 6 tools structured wrapper
```

### Modified Files

```
src/agent_workflow.py   # REWRITE: from free loop to Planner-Executor two-tier loop
src/agent_tools.py      # MODIFY: GD quality filter + converge_dist + arrival verification
src/scene_aeqa.py       # MODIFY: panorama 8 views + resolution 400 + labels + cam_pose
src/agent_context.py    # MODIFY: ContextManager preserves notebook on transition
src/agent_memory.py     # MODIFY: MemoryStore key-frame index + on-demand retrieval
```

### Module Dependencies

| Module | Responsibility | Depends on |
|---|---|---|
| agent_planner.py | Qwen3.6-Plus API calls, 4-component prompt, JSON response parsing | notebook, evidence, executor |
| agent_notebook.py | 5 entry types, loop detection, belief revision, injection text generation | None |
| agent_evidence.py | TrajectoryEvidence data class, to_notebook_entry() conversion | None |
| agent_executor.py | 6 tool wrappers, calls scene_aeqa/tsdf_planner functions | scene_aeqa, tsdf_planner |
| agent_workflow.py | Main loop: Planner -> Executor -> Notebook -> loop detection -> fallback | planner, notebook, executor |

### Qwen3.6-Plus API

- Use Alibaba Cloud server (8.147.163.63) existing proxy config
- Planner prompt sent via HTTP API to Qwen3.6-Plus
- Parse JSON-format action response
- temperature=0.3, max_tokens=1024

---

## 9. Implementation Phases

### Phase 1: Evidence Notebook + Executor Interface (foundation)

- Implement `agent_notebook.py` and `agent_evidence.py` (pure data, no dependencies)
- Implement `agent_executor.py` (wraps existing functions)
- Add GD quality filtering in `agent_tools.py`
- Adjust converge_dist and arrival verification

### Phase 2: Planner + Structured Prompt

- Implement `agent_planner.py` with Qwen3.6-Plus API
- Implement 4-component prompt template
- Test Planner decisions with mock executor

### Phase 3: Workflow Rewrite + Panorama

- Rewrite `agent_workflow.py` as Planner-Executor loop
- Modify `scene_aeqa.py` for 8-view panorama + labels + resolution
- Modify `agent_context.py` to preserve notebook

### Phase 4: Integration Testing + Iteration

- End-to-end test on same scenario (oven towel)
- Compare with current results (trace.jsonl)
- Iterate on prompt quality and notebook effectiveness

---

## 10. Expected Impact

Based on Qwen-RobotNav's results (HM-EQA 76.7% vs 3D-Mem 50.4%):

| Problem | Current | Expected After Refactor |
|---|---|---|
| VLM death loop | 17min, 8+ loops | Max 10 rounds, loop detection forces exit |
| Context loss | Each call independent | Notebook persists, Planner sees full history |
| GD false arrival | 1-step "arrived", 0.1m move | converge_dist=12 voxels, Euclidean verification |
| Prompt info deficit | No objects/rooms/history | 4-component prompt with full context |
| Navigation efficiency | ~50 steps, 15 min | ~10-20 rounds, ~5-8 min (77% fewer steps target) |
