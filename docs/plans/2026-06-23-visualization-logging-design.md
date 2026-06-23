# Plan 2: 可视化日志 — RunLogger 模块

> **目标**：把机器人每一步移动、frontier、memory snapshot 都保存到 `results/` 文件夹，按阶段结构化输出可视化结果。
>
> **Worktree**：`.worktrees/add-visualization`（基于 `fix-navigation` 合并后创建）
> **创建时间**：2026-06-23
> **依赖**：Plan 1（导航修复）必须先完成并合并

---

## 1. 核心策略

新建轻量级 `src/run_logger.py` 模块，**不改原 `logger_aeqa.py`**（保留给 `run_aeqa_evaluation.py` 兼容）。

原 `logger_aeqa.py` 耦合了 `run_aeqa_evaluation.py` 的 step 计数模型，与新的 6 阶段状态机不兼容。新 `RunLogger` 按阶段+子步记录。

---

## 2. 输出目录结构

```
results/<run_timestamp>/<question_id>/
├── trace.jsonl                          # 完整 VLM 调用链（每行一次调用，含 reason）
├── panorama/                            # Stage 1: 8 视角全景
│   ├── mosaic.png                       # 3×3 拼图（含方位标签）
│   ├── view0_前.png ... view7_左前.png  # 8 张单独视角
│   └── meta.json                        # {view_idx: {angle, cam_pose, ...}}
├── stage2_decision/                     # Stage 2: 主方向决策
│   ├── input_mosaic.png                 # 给 VLM 看的拼图
│   └── response.json                    # VLM 返回的 JSON（含 reason）
├── stage2_5a_seed_selection/            # Stage 2.5a: Seed 选择（若进入）
│   ├── seed_mosaic.png                  # 所有 seed 图片拼图
│   └── response.json
├── stage3_object_selection/             # Stage 3: 物体选择
│   ├── view_idx{N}_large.png            # VLM 选中的大图
│   └── response.json                    # VLM 返回的物体名 + reason
├── stage4_navigation/                   # Stage 4: GD 导航（核心可视化）
│   ├── gd_detection.png                 # GD bbox + SAM mask 可视化
│   ├── backprojection.png              # 3D 反投影点云 + target voxel 在俯视图上
│   ├── iter1/
│   │   ├── spiral_search.png            # 螺旋搜索结果（target + 搜索路径）
│   │   ├── nav_walk/
│   │   │   ├── step01_topdown.png       # 每子步俯视图（agent 轨迹 + frontier）
│   │   │   ├── step01_views/            # 每子步 3 视角 RGB（可选，默认关）
│   │   │   ├── step02_topdown.png ...
│   │   │   └── ...
│   │   └── topdown_iter_summary.png     # 迭代总结俯视图
│   ├── iter2/ ...
│   └── final_topdown.png                # 最终俯视图（全迭代历史）
├── stage5_decision/                     # Stage 5: 到达后重新决策
│   ├── frontal_3views.png              # 正面 3 视角拼图
│   └── response.json
├── stage6_frontier_selection/           # Stage 6: Frontier 选择（若进入）
│   ├── frontier_mosaic.png
│   └── response.json
├── seed_views/                          # Seed 视角图（持续更新）
│   ├── seed{N}_current.png             # 每个 seed 当前代表图
│   └── seed{N}_history/                # 历史版本（可选，debug 用）
├── snapshot/                            # 所有 silent_perception 存档
│   ├── step{N}_view{V}.png
│   └── ...
└── summary.json                         # 最终运行总结
```

---

## 3. RunLogger 模块接口

```python
# src/run_logger.py
class RunLogger:
    def __init__(self, output_root="results"):
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_root, self.run_timestamp)
        self.episode_dir = None
        self._step_counter = 0  # 全局子步计数器（与 silent_perception 同步）
    
    def init_episode(self, question_id, question, answer):
        """创建 question_id 子目录，初始化 trace.jsonl"""
        self.episode_dir = os.path.join(self.run_dir, question_id)
        os.makedirs(self.episode_dir + "/panorama", exist_ok=True)
        # ... 创建所有子目录
        self._init_trace(question_id, question, answer)
    
    # ── Stage 级日志 ──
    def log_panorama(self, views, mosaic_img, question):
        """Stage 1: 保存 8 视角 + 拼图 + meta"""
    
    def log_vlm_decision(self, stage, input_images, response_json, parsed, reason):
        """Stage 2/2.5a/3/5/6: 保存 VLM 输入图 + 返回 JSON + reason"""
    
    def log_gd_detection(self, rgb, bbox, mask, phrase, score):
        """Stage 4: GD bbox + SAM mask 可视化"""
    
    def log_backprojection(self, tsdf_planner, target_normal, target_voxel, pcd_voxels):
        """Stage 4: 3D 反投影在俯视图上的标注"""
    
    def log_spiral_search(self, iteration, target_voxel_xy, spiral_result, tsdf_planner):
        """Stage 4: 螺旋搜索结果俯视图"""
    
    def log_nav_step(self, iteration, step, pts, angle, tsdf_planner, fig=None):
        """Stage 4: 每子步俯视图 + agent 轨迹"""
    
    def log_iter_summary(self, iteration, tsdf_planner, nav_trace, spiral_history):
        """Stage 4: 迭代总结俯视图"""
    
    def log_final_topdown(self, tsdf_planner, nav_trace, spiral_history, target_voxel):
        """Stage 4: 最终俯视图"""
    
    def log_seed_view_update(self, seed_id, image, position, reason):
        """Seed 视角图更新（register/update）"""
    
    def log_snapshot(self, snapshot_id, image, room_id, objects_in_view, position):
        """silent_perception 存档同步保存"""
    
    def log_trace(self, event_type, data, reason=None):
        """通用 trace.jsonl 追加（每行一个 JSON 事件）"""
    
    def log_summary(self, result):
        """运行结束写 summary.json"""
```

---

## 4. 插入点（最小侵入）

| 模块 | 插入位置 | 调用 |
|------|---------|------|
| `agent_workflow.py` Stage 1 | `observe_panorama` 后 | `logger.log_panorama(...)` |
| `agent_workflow.py` 每次 VLM 调用 | `call_vlm` 返回后 | `logger.log_vlm_decision(..., reason=parsed["reason"])` |
| `scene_aeqa.py` GD 检测后 | `_gd_detect` + SAM 后 | `logger.log_gd_detection(...)` |
| `scene_aeqa.py` 反投影后 | target_voxel 计算后 | `logger.log_backprojection(...)` |
| `scene_aeqa.py` 螺旋搜索后 | `spiral_search` 返回后 | `logger.log_spiral_search(...)` |
| `scene_aeqa.py` 每子步 | `silent_perception` 后 | `logger.log_nav_step(...)` |
| `scene_aeqa.py` 迭代结束 | 循环末尾 | `logger.log_iter_summary(...)` |
| `agent_tools.py:silent_perception_step` | `memory_store.add_snapshot` 后 | `logger.log_snapshot(...)` |
| `seed_views.py`（新） | register/update 时 | `logger.log_seed_view_update(...)` |

---

## 5. 俯视图渲染（复用工作脚本）

从 `MSGNav-main/tools/debug_render/debug_iterative_spiral_navigate.py:render_topdown`（line 305+）移植渲染函数到 `run_logger.py`，包含：

1. 房间分割填色 + 标签（`draw_room_overlay_on_axis`）
2. 红色轨迹折线（`nav_trace_voxels`）
3. 3D 点云红色散点（`pcd_voxels`）
4. 红色十字 + 目标标签（`target_voxel_xy`）
5. Agent 蓝圆 + heading tick
6. 螺旋搜索历史（多色标注每次迭代的 spiral point）

渲染层级（自底向上）：
1. 房间填色 + 标签
2. 红色轨迹折线
3. 3D 点云红色散点
4. 红色十字 + 目标标签
5. Agent 蓝圆 + heading tick

---

## 6. trace.jsonl 格式

每行一个 JSON 事件，按时间顺序记录，含 reason 字段：

```json
{"ts": "2026-06-23T15:30:01", "event": "panorama", "stage": 1, "view_count": 8, "pts": [...]}
{"ts": "2026-06-23T15:30:15", "event": "vlm_call", "stage": 2, "input": "mosaic.png",
 "reason": "问题问的是烤箱上的毛巾，我在 view3 右前方向看到厨房区域，可能有烤箱",
 "response": {"action": "navigate_to_object", "view_idx": 3}, "latency_ms": 3200}
{"ts": "2026-06-23T15:30:18", "event": "vlm_call", "stage": "2.5a", "input": "seed_mosaic.png",
 "reason": "seed2 朝向厨房方向，从当前视角看过去有橱柜，可能找到烤箱",
 "response": {"action": "explore_seed", "seed_id": 2}, "latency_ms": 2800}
{"ts": "2026-06-23T15:30:20", "event": "vlm_call", "stage": 3, "input": "view3_large.png",
 "reason": "图片中央有一个不锈钢家电，方形门板，符合烤箱特征",
 "response": {"object": "oven"}, "latency_ms": 1500}
{"ts": "2026-06-23T15:30:22", "event": "gd_detect", "phrase": "oven", "score": 0.78, "bbox": [...]}
{"ts": "2026-06-23T15:30:22", "event": "backproject", "target_normal": [...], "target_voxel": [42, 78], "pcd_points": 1214}
{"ts": "2026-06-23T15:30:23", "event": "spiral_search", "iter": 1, "dist": 37, "voxel": [74, 73]}
{"ts": "2026-06-23T15:30:24", "event": "nav_step", "iter": 1, "step": 1, "pts": [...], "voxel": [112, 66], "arrived": false}
{"ts": "2026-06-23T15:30:25", "event": "nav_step", "iter": 1, "step": 5, "pts": [...], "voxel": [74, 73], "arrived": true}
{"ts": "2026-06-23T15:30:26", "event": "iter_summary", "iter": 1, "arrived": true, "converged": false, "nav_steps": 5}
{"ts": "2026-06-23T15:30:27", "event": "spiral_search", "iter": 2, "dist": 4, "voxel": [43, 80]}
{"ts": "...", "event": "nav_step", "iter": 2, "step": 4, "pts": [...], "arrived": true}
{"ts": "...", "event": "iter_summary", "iter": 2, "arrived": true, "converged": true}
{"ts": "...", "event": "seed_view_update", "seed_id": 2, "reason": "dist_decreased", "new_dist": 3.2}
{"ts": "...", "event": "vlm_call", "stage": 5, "input": "frontal_3views.png",
 "reason": "已到达烤箱前方，正面视角能清晰看到烤箱把手",
 "response": {"action": "submit_answer", "answer": "..."}}
{"ts": "...", "event": "episode_end", "success": true, "total_steps": 9, "iters": 2}
```

`reason` 作为顶层字段记录，方便后续分析 VLM 决策质量。

---

## 7. 配置控制

```yaml
# cfg.yaml 新增字段
visualization:
  enabled: true
  output_root: "results"          # 结果根目录
  save_nav_topdown: true          # 每子步俯视图（最占磁盘）
  save_nav_views: false           # 每子步 3 视角 RGB（默认关，太占空间）
  save_seed_history: false        # seed 视角历史版本（debug 用）
  dpi: 110
```

---

## 8. 依赖关系

- **Plan 2 必须在 Plan 1 完成后启动**：导航不修好，没有"移动"可截图
- Plan 2 的插入点会触碰 `scene_aeqa.py`（Plan 1 也改这个文件）→ **必须在同一 worktree 或合并后进行**
- 推荐：Plan 1 完成后合并到 `fix-navigation`，Plan 2 在新 worktree `add-visualization` 基于合并后的 `fix-navigation` 创建

---

## 9. 不做的事

- 不改原 `logger_aeqa.py`（保留给 `run_aeqa_evaluation.py` 兼容）
- 不做视频生成（原 `frontier_video/` 是 ffmpeg 拼 PNG，可后续加）
- 不做实时 web dashboard（`run_logger` 只写文件，不启动服务器）
- 不存 depth .npy（磁盘太大，只存 RGB PNG）

---

## 10. 参考实现

- 原生产可视化输出：`/media/afdsafg/系统/Users/afdsafg/Downloads/exp_eval_results/exp_eval_aeqa_41_hmge_region_frontier_pred_yolov8m_samb_640/00c2be2a-1377-4fae-a889-30936b7890c3/visualization/`
  - `{N}_map.png` 每步俯视图
  - `frontier/{step}_{view}.png` frontier 可视化
  - `snapshot/{step}-view_{N}.png` snapshot
  - `chosen_snapshot/` 选中的目标 snapshot
  - `hmge_arbiter_trace.jsonl` VLM 调用链
- 俯视图渲染：`MSGNav-main/tools/debug_render/debug_iterative_spiral_navigate.py:render_topdown`（line 305+）
