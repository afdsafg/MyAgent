# Plan 1: 导航修复 — 整体移植调试脚本管线

> **目标**：修复 Habitat 中机器人无法导航的 bug。当前 GD 3D 反投影把所有物体投到同一 TSDF 边界点 `[156, 0, 2]`，agent 只移动 0.5m 就停止。
>
> **Worktree**：`.worktrees/fix-navigation`（现有分支，commit `6ab91c9`）
> **创建时间**：2026-06-23
> **依赖**：无（独立执行，Plan 2 依赖本计划完成）

---

## 1. 根因诊断

### 1.1 当前错误代码 vs 工作脚本

工作脚本 `MSGNav-main/tools/debug_render/debug_iterative_spiral_navigate.py`（990 行，已验证）的反投影组合：

```python
trans_pose = pose_habitat_to_tsdf(cam_pose)   # TSDF pose
camera_convention = "z_forward"                # 匹配 TSDF pose
# 结果 pcd 已在 normal 坐标系，直接 normal2voxel
```

当前损坏代码 `src/scene_aeqa.py:grounded_navigate_to_object` 的组合：

```python
trans_pose = cam_pose                          # raw Habitat pose
# 不传 camera_convention → 默认 "opengl"
# 然后手动 pos_habitat_to_normal(pcd) 转换
```

两种组合理论上应等价，但实际产出 `voxel=[156,0,2]`（所有物体投到同一 TSDF 边界点）。这是坐标系约定不匹配的典型症状——`detections_to_obj_pcd_and_bbox` 内部实现在某种组合下产生错误方向。

### 1.2 关键修正点

| 修正项 | 当前错误代码 | 修正后（复刻工作脚本） |
|--------|------------|---------------------|
| `trans_pose` | raw Habitat `cam_pose` | `pose_habitat_to_tsdf(cam_pose)` |
| `camera_convention` | 默认 "opengl"（不传） | 显式 `"z_forward"` |
| pcd 坐标系 | `pos_habitat_to_normal(pcd)` 转换 | 已在 normal 系，直接用 |
| target 计算 | pcd mean（已对） | pcd mean（保持） |
| Z-clip 轴 | `normal[2]`（已对） | `normal[2]`（保持） |
| Pin 轴 | `normal[2]`（已对） | `normal[2]`（保持） |

**核心修复**：前 3 行。当前代码传 raw `cam_pose` 但不传 `camera_convention`，导致 `detections_to_obj_pcd_and_bbox` 用 OpenGL 约定处理 Habitat pose，反投影方向反转。

---

## 2. 修正后的工作流（v3）

导航修复涉及整个 agent 工作流的重构，因为 VLM 必须选定视角和物体才能做 GD 检测。

### 2.1 完整工作流

**Stage 1 — 初始全景**
- Agent 渲染 8 视角全景（8 个方位：前/右前/右/右后/后/左后/左/左前）
- 存每视角 `{view_idx, angle, cam_pose, rgb}`
- 不调用 VLM

**Stage 2 — 主方向决策（VLM 调用 1）**
- 输入：8 视角拼接图 + 问题
- VLM 输出（二选一）：
  - `navigate_to_object` + `view_idx`（0-7）+ `reason`
  - `explore_other_room` + `reason`
- 不直接给 seed 选项

**Stage 2.5a — Seed 选择（VLM 调用 2，仅当 Stage 2 选 explore_other_room）**
- 输入：所有未探索 seed 的图片拼图（每个 seed 标注 seed_id）+ 问题
- VLM 输出（二选一）：
  - `explore_seed` + `seed_id` + `reason`
  - `explore_frontier` + `reason`
- 代码约束：seed 列表为空时强制只能选 `explore_frontier`

**Stage 3 — 物体选择（VLM 调用 3，仅当 Stage 2 选 navigate_to_object）**
- 输入：Stage 2 选中的 view_idx 大图 + 问题
- VLM 输出：一个物体名词 + `reason`（强约束，必须图中存在）
- 这是本轮移动的锚点

**Stage 4 — GD 导航到锚点（代码执行，无 VLM）**
- 用 Stage 2 的 `view_idx` 对应 `cam_pose` 做 GD 检测 + SAM + 3D 反投影
- 迭代螺旋搜索导航，每子步地图融合
- 抵达锚点附近为止

**Stage 5 — 到达后重新决策（VLM 调用 4）**
- 输入：agent 当前位置正面 3 视角图 + 问题
- VLM 输出（三选一）：
  - `navigate_to_object` + `view_idx`（0/1/2）+ `reason` → 回 Stage 3
  - `explore_other_room` + `reason` → 回 Stage 2.5a
  - `submit_answer` + `answer` + `reason` → 结束

**Stage 6 — Frontier 选择（VLM 调用 5，仅当 Stage 2.5a 选 explore_frontier）**
- 输入：所有 frontier 图片的拼图
- VLM 输出：`frontier_id` + `reason`
- 导航到该 frontier → 回 Stage 5 重新决策

### 2.2 VLM 输出 Schema（强制 reason 字段）

每个 stage 的 prompt 末尾加 schema 强约束：

```
你必须输出以下 JSON 格式（不要输出其他内容）：
{
  "reason": "<一句话解释为什么做这个选择，必须包含你观察到的具体线索>",
  "action": "<action_name>",
  ...其他字段...
}

reason 字段要求：
- 必须包含你从图片中观察到的具体视觉线索（如"view3 中看到不锈钢家电"）
- 必须解释该选择如何帮助回答问题
- 不允许输出"我决定..."等空泛表述，必须有具体依据
```

各 stage 的 schema：

**Stage 2**：
```json
{"reason": "...", "action": "navigate_to_object", "view_idx": 3}
{"reason": "...", "action": "explore_other_room"}
```

**Stage 2.5a**：
```json
{"reason": "...", "action": "explore_seed", "seed_id": 2}
{"reason": "...", "action": "explore_frontier"}
```

**Stage 3**：
```json
{"reason": "...", "object": "oven"}
```

**Stage 5**：
```json
{"reason": "...", "action": "navigate_to_object", "view_idx": 2}
{"reason": "...", "action": "explore_other_room"}
{"reason": "...", "action": "submit_answer", "answer": "..."}
```

**Stage 6**：
```json
{"reason": "...", "frontier_id": 0}
```

容错：如果 VLM 连续 2 次输出无 reason，强制 fallback 到 `explore_frontier` 避免死循环。

---

## 3. 8 视角全景 + 方位标注

### 3.1 视角与方位映射

参考 Spatial-X 的 `DiscreteVisualSimulatorV2`（`/home/afdsafg/spatialNav/Spatial-X/spatialx/mp3d_extensions/discrete_env.py:486-558`），实现 8 视角全景。

相对 agent 当前朝向，顺时针每 45°：

```python
DIRECTIONS = ["前", "右前", "右", "右后", "后", "左后", "左", "左前"]
# view_idx:  0     1      2     3      4     5      6     7
angles = [agent_angle + i * 2*np.pi/8 for i in range(8)]  # 顺时针
```

### 3.2 拼图布局（3×3 网格，中心是方位指南针）

```
[左前(7)] [前(0)] [右前(1)]
[左(6)]   [指南针] [右(2)]
[左后(5)] [后(4)] [右后(3)]
```

- 中心格画方位箭头（↑前 ↓后 ←左 →右），标注当前 agent 朝向
- 每个子图标题显示方位中文标签
- 保存每视角的 `{view_idx, angle, cam_pose, rgb}` 到 `panorama_views` 列表

---

## 4. Seed 视角更新策略

每个 seed 只有一张代表图，遵循"agent 不能透视、不能穿墙、不能瞬移"原则。

### 4.1 Seed 生成时

在 `tsdf_planner.update_frontier_map` / `update_room_map` 内部，新 seed 出现时：

1. 计算方向角：`angle_to_seed = atan2(seed.x - agent.x, seed.z - agent.z)`
2. 渲染：`obs = scene.get_observation(pts, angle_to_seed)`
3. 存：`seed.view_image = obs["color_sensor"]`, `seed.view_image_pos = pts`, `seed.view_image_angle = angle_to_seed`

### 4.2 导航中途更新条件

每个 agent_step 后检查每个未选 seed：

```python
for seed in unexplored_seeds:
    # 条件 1: agent 朝向 seed 方向无墙壁遮挡（高度感知）
    ray_blocked = check_ray_blocked(
        tsdf_planner, cur_pts, seed.position,
        min_blocking_height=cfg.planner.seed_occlusion_height  # 默认 1.2m
    )
    
    # 条件 2: 欧式距离比上次渲染时变小（0.1m 防抖）
    cur_dist = np.linalg.norm(cur_pts[[0,2]] - seed.position[[0,2]])
    last_dist = np.linalg.norm(seed.view_image_pos[[0,2]] - seed.position[[0,2]])
    dist_decreased = cur_dist < last_dist - 0.1
    
    if (not ray_blocked) and dist_decreased:
        angle_to_seed = atan2(seed.position[0]-cur_pts[0],
                              seed.position[2]-cur_pts[2])
        obs = scene.get_observation(cur_pts, angle_to_seed)
        seed.view_image = obs["color_sensor"]
        seed.view_image_pos = cur_pts.copy()
        seed.view_image_angle = angle_to_seed
```

### 4.3 墙壁遮挡检查：高度感知版本

用 3D TSDF 体素，只把"高于阈值的障碍物"视为遮挡。桌子(0.75m)、椅子(0.5m)不算遮挡；墙(2.4m+)、柜子(1.8m)算遮挡。

```python
def check_ray_blocked(tsdf_planner, agent_pos, target_pos,
                      min_blocking_height=1.2):
    """检查 agent→target 射线是否被高障碍物遮挡。"""
    agent_voxel = tsdf_planner.habitat2voxel(agent_pos)
    target_voxel = tsdf_planner.habitat2voxel(target_pos)
    
    # 2D Bresenham 射线
    ray_voxels = bresenham_2d(agent_voxel[:2], target_voxel[:2])
    
    # 高度阈值 → voxel z-index
    voxel_size = tsdf_planner._voxel_size
    floor_z = tsdf_planner.min_height_voxel
    min_block_z = floor_z + int(min_blocking_height / voxel_size)
    max_z = tsdf_planner._tsdf_vol_cpu.shape[2]
    
    for vy, vx in ray_voxels[1:-1]:  # 跳过端点
        if not (0 <= vy < H and 0 <= vx < W):
            return True  # 越界 = 遮挡
        # 检查该 voxel 列在 [min_block_z, max_z) 是否有表面
        column = tsdf_planner._tsdf_vol_cpu[vy, vx, min_block_z:max_z]
        if (column < -0.1).any():  # TSDF < 0 = 占据
            return True
    return False
```

### 4.4 关键参数

| 参数 | 值 | 理由 |
|------|-----|------|
| `min_blocking_height` | 1.2m | 矮于 agent 眼高(1.5m)，高于桌子(0.75m)、椅子(0.5m)。能挡住视线的典型障碍：墙(2.4m+)、柜子(1.8m)、门(2.0m)、冰箱(1.8m) |
| TSDF 阈值 | -0.1 | TSDF < 0 = 表面后方（占据）；留 0.1 余量避免数值噪声 |
| 防抖阈值 | 0.1m | 避免 agent 在小范围内来回移动时频繁重渲染 |

### 4.5 边界情况

- TSDF 未观测区域（value=0）：视为不遮挡（保守策略，鼓励更新图片）
- agent 和 seed 在同一 voxel：直接返回 False
- 射线穿过未探索区域：继续检查（TSDF 融合是渐进的，已知区域按实际占据判断）
- 不更新已选中导航的 seed（agent 正在去那里）
- 不更新 ray_blocked 的 seed（保持上次可见图）
- 不更新距离变大的 seed（agent 在远离，旧图更准确）

### 4.6 实现位置

1. **`src/agent_tools.py:observe_panorama`** 改造：7 视角 → 8 视角 + 方位标签 + 3×3 拼图布局 + 存 cam_pose
2. **`src/seed_views.py`**（新）`SeedViewManager` 类：
   - `register_seed(seed_id, position, scene, tsdf_planner, pts)` — 生成时渲染
   - `update_after_step(seeds, cur_pts, tsdf_planner, scene)` — 每步检查更新
   - `get_mosaic(question)` — 返回拼图给 VLM
3. **`src/tsdf_planner.py`**：在 `_commit_room_regions` 或 seed 生成处调用 `SeedViewManager.register_seed`
4. **`src/agent_tools.py:navigate_to_*`**：每个 agent_step 后调用 `SeedViewManager.update_after_step`

---

## 5. Stage 4 导航循环：迭代螺旋搜索

### 5.1 `grounded_navigate_to_object` 重写

```python
def grounded_navigate_to_object(
    scene, tsdf_planner, pts, angle,
    view_idx, view_angle, view_cam_pose,  # 新增：VLM 选定的视角
    object_desc,
    max_iterations=5, converge_dist_voxels=5,
    max_nav_steps_per_iter=15,
    memory_store=None, cam_intr=None, cfg=None, cfg_cg=None,
    detection_model=None, sam_predictor=None,
    clip_model=None, clip_preprocess=None, clip_tokenizer=None,
    cnt_step_base=0, step_budget=None,
    gd_model=None,  # 新增：GroundingDINO 模型
):
    """GD 导航链：VLM 选定的视角 → GD 检测 → 3D 反投影 → 迭代螺旋搜索导航。"""
```

### 5.2 Phase A：GD 检测 + SAM + 3D 反投影（一次性）

```python
# 用 VLM 选定视角的 cam_pose 渲染
obs, cam_pose_habitat = scene.get_observation(pts, view_angle)
rgb, depth = obs["color_sensor"], obs["depth_sensor"]

# GD 检测
bbox, phrase, score = _gd_detect(rgb, object_desc, gd_model)
if bbox is None:
    return pts, angle, False, f"GD no detection for '{object_desc}'", images

# SAM 分割
sam_out = scene.sam_predictor.predict(rgb, bboxes=[bbox.tolist()], verbose=False)
mask = sam_out[0].masks.data.cpu().numpy()[0].astype(bool)

# 3D 反投影 — 严格复刻工作脚本
cam_pose_tsdf = pose_habitat_to_tsdf(cam_pose_habitat)
obj_list = detections_to_obj_pcd_and_bbox(
    depth_array=depth,
    masks=mask[None, :, :].astype(np.float32),
    cam_K=cam_intr,
    image_rgb=rgb,
    trans_pose=cam_pose_tsdf,           # TSDF pose（不是 raw Habitat）
    camera_convention="z_forward",       # 匹配 TSDF pose
    ...
)
pcd_np = np.asarray(obj_list[0]["pcd"].points)  # 已在 normal 坐标系

# Z-clip
pcd_np_clipped = pcd_np.copy()
over_z = pcd_np_clipped[:, 2] > cfg.tsdf_grid_size * 30  # 3.0m
pcd_np_clipped[over_z, 2] = pts[1]  # 钉到地板

# Target 用 pcd mean
target_normal = pcd_np_clipped.mean(axis=0)
target_normal[2] = pts[1]  # pin height to floor

target_voxel = tsdf_planner.normal2voxel(target_normal)
target_voxel_xy = (int(target_voxel[0]), int(target_voxel[1]))
```

### 5.3 Phase B：迭代螺旋搜索导航循环

```python
cur_pts, cur_angle = pts.copy(), angle
converged = False
arrived_any = False

for iteration in range(1, max_iterations + 1):
    # 1. 刷新地图网格
    _refresh_planner_grids(tsdf_planner, cur_pts)
    
    # 2. 检查 target 是否已可导航
    tv_y, tv_x = target_voxel_xy
    if tsdf_planner.island[tv_y, tv_x]:
        normal_pos = tsdf_planner.voxel2normal(np.array([tv_y, tv_x]))
        normal_3d = np.array([normal_pos[0], normal_pos[1], pts[1]])
        hab_pos = pos_normal_to_habitat(normal_3d)
        snapped = scene.pathfinder.snap_point(hab_pos[:3])
        if snapped is not None and not np.isnan(snapped).any():
            spiral_result = {"habitat_pos": snapped, "voxel_xy": target_voxel_xy,
                             "search_steps": 0, "spiral_dist": 0}
            converged = True
    
    # 3. 螺旋搜索（如果未收敛）
    if not converged:
        spiral_result = _spiral_search_navigable_point(
            tsdf_planner, scene.pathfinder,
            target_voxel_xy, cur_pts,
            max_radius_voxels=max(tsdf_planner._tsdf_vol_cpu.shape[:2]),
            floor_height=pts[1])
        if spiral_result is None:
            break
    
    # 4. 收敛判断
    if spiral_result["spiral_dist"] <= converge_dist_voxels:
        converged = True
    
    # 5. 设置导航点 — 用生产代码正规路径
    tsdf_planner.max_point = None
    tsdf_planner.target_point = None
    pathfinder_target = np.array(spiral_result["habitat_pos"])
    tsdf_planner.set_next_navigation_point(
        target_type="image",
        choice=pathfinder_target,
        pts=cur_pts.tolist(),
        objects=None, obs_points=None,
        cfg=cfg.planner, pathfinder=scene.pathfinder,
        random_position=False, observe_snapshot=False)
    
    # 6. 每子步导航
    arrived = False
    for nav_step in range(1, max_nav_steps_per_iter + 1):
        result = tsdf_planner.agent_step(
            pts=cur_pts, angle=cur_angle,
            objects=scene.objects, snapshots=scene.snapshots,
            pathfinder=scene.pathfinder, cfg=cfg.planner,
            save_visualization=False)
        if result[0] is None:
            break
        cur_pts, cur_angle, _, _, _, target_arrived = result
        
        # 每子步感知
        silent_perception_step(
            scene, tsdf_planner, cur_pts, cur_angle,
            cnt_step_base + iteration * max_nav_steps_per_iter + nav_step,
            memory_store, cam_intr, cfg,
            detection_model, sam_predictor,
            clip_model, clip_preprocess, clip_tokenizer)
        
        # 刷新网格
        _refresh_planner_grids(tsdf_planner, cur_pts)
        
        # 更新 frontier map
        tsdf_planner.update_frontier_map(
            cur_pts, cfg.planner, scene,
            cnt_step_base + iteration * max_nav_steps_per_iter + nav_step,
            save_frontier_image=False)
        
        if target_arrived:
            arrived = True
            break
    
    arrived_any = arrived or arrived_any
    
    # 7. 迭代终止逻辑（关键：到达但未收敛 → 继续）
    if converged and arrived:
        break
    if arrived and not converged:
        continue  # 重新螺旋搜索
    # 未到达 → 也继续下一轮
```

### 5.4 4 个辅助函数（从工作脚本移植到 `scene_aeqa.py`）

| 函数 | 来源 | 作用 |
|------|------|------|
| `_spiral_search_navigable_point()` | debug 脚本:97-156 | 顺时针螺旋搜索最近可导航点 |
| `_check_voxel_navigable()` | debug 脚本:159-193 | 单 voxel 可达性检查 |
| `_refresh_planner_grids()` | debug 脚本:279-298 | 重算 island/unoccupied/unexplored |
| ~~`_render_and_integrate_phase1()`~~ | ~~debug 脚本:239-276~~ | ~~删除~~：`silent_perception_step` 已覆盖 3 视角融合 |

### 5.5 不修改的部分

- `agent_tools.py:_navigate_to_target_with_agent_step` — 保持原样，给 `navigate_to_seed/frontier` 用
- `tsdf_planner.py` 公共接口 — 不改
- `silent_perception_step` — 不改

---

## 6. 关键约束（来自 Obsidian 笔记）

- `explored_depth` 保持配置默认 1.7m，不覆盖
- 相机约定：传 `pose_habitat_to_tsdf(cam_pose)` 给 `detections_to_obj_pcd_and_bbox`，配 `camera_convention="z_forward"`
- Z-clip：`normal[2] > 3.0m` → 钉到地板
- 用 pcd mean (x,y) 不用 OBB center
- Pin `normal[2]`（高度）到地板
- 迭代到达但未收敛时继续下一轮螺旋搜索
- `set_next_navigation_point` 用 `target_type="image"`（生产代码 `tsdf_planner.py:570-582` 已支持）

---

## 7. 验证标准

1. GD 反投影：oven 和 door 两个不同物体应投影到**不同** voxel（不再都是 `[156, 0, 2]`）
2. 导航移动：agent 应执行多步移动（>5 步），位移 >2m
3. 迭代收敛：螺旋搜索距离应从大变小（如 37→4）
4. 地图扩展：每迭代 island/unoccupied voxels 数应增加
5. 抵达目标：至少一次 `converged and arrived`

---

## 8. 相关 Obsidian 笔记

- `HM-GE科研开发日志/MSGNav-调试笔记-20260614-迭代螺旋搜索多段导航.md`
- `HM-GE科研开发日志/MSGNav-调试笔记-20260614-GD-SAM-3D-2D-导航-完整管线技术总结.md`
- `HM-GE科研阶段笔记/05-阶段三-完整管线与螺旋搜索-0614/`
