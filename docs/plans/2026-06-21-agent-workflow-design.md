# HM-GE Agent Workflow 设计方案

> 在 3D-Mem 中实现一个面向 AEQA 场景问答的 Agent 导航工作流。
> 复用 MSGNav 中的房间分割、GD 导航链、感知管线等代码，避免重复造轮子。
>
> 创建时间: 2026-06-21
> 仓库: https://github.com/afdsafg/MyAgent.git

---

## 1. 整体架构

**核心模式**: 外层 Python 控制阶段切换，阶段内 VLM 自主决策。

```
3D-Mem/src/
├── agent_workflow.py          ← 新增：阶段控制器 + VLM 交互 + 主循环
├── agent_tools.py             ← 新增：7 个精简工具的实现
├── agent_memory.py            ← 新增：Snapshot 存储管理与自然语言检索
├── agent_context.py           ← 新增：阶段过渡摘要与上下文刷新
├── tsdf_planner.py            ← 从 MSGNav 移植房间分割(RoomRegion/gateway frontier)
├── scene_aeqa.py              ← 增加 GD 导航链方法
└── (其他 3D-Mem 现有文件不变)
```

**上下文管理**:
- 阶段内：共享完整上下文（工具调用之间有记忆）
- 阶段间：VLM 自由格式写过渡摘要，下一阶段只接收 `问题 + 摘要 + 当前阶段所需图像`
- 图像 token 不跨阶段重复上传

---

## 2. 阶段流程

```
阶段1: 初始全景
  └─ 7视角全景 + 房间分割 + 渲染种子点图像 + 渲染frontier snapshot

阶段2: 方向判断
  └─ VLM看拼接全景 → 判断目标是否在当前房间
       YES → 阶段3
       NO  → 阶段4

阶段3: GD导航链 [核心循环]
  └─ VLM从拼接图中选方向 → GD导航链走到子目标 → 3视角观测
     → VLM重新判断:
        继续该方向          → 阶段3
        换其他6张图中的方向  → 阶段3(换方向)
        探索其他房间        → 阶段4
        找到目标            → 阶段6
        无目标+当前房间无价值 → 阶段4

阶段4: 房间/前沿选择
  ├─ 有新region              → VLM看种子点图像 → 选房间 → 导航 → 阶段1
  ├─ 无新region但有未访问
  │  种子点+frontier         → VLM选择 → 导航 → 阶段1 → 阶段2
  └─ 所有region和frontier
      都探索完               → 阶段5

阶段5: 最终Fallback
  └─ 查记忆 → 能回答 → 阶段6
     └─ 记忆无信息 → 标记无法回答+记录原因 → 结束

阶段6: 提交答案 → 结束
```

**关键循环**:
- 阶段3 内循环：到达子目标→判断→继续/换方向/换房间
- 阶段1→2→3→4→1 大循环：探索完一个房间后去下一个
- 阶段3→4 跨区切换：当前房间找不到时切换到其他区域
- 阶段4→1→2 frontier fallback：无新region时退化到frontier，等同于进入新区域

---

## 3. 每 Step 静默感知（非工具，自动执行）

不论外层处于什么阶段，每个 step 底层都自动执行：

```
每个 step:
  1. 移动到目标位置
  2. 3 视角观测（正面 + 两侧各一张）
  3. YOLO + SAM + CLIP + 3D 反投影 → 更新场景图
  4. TSDF 深度集成 + 房间分割
  5. Snapshot 存档到磁盘（含结构化元数据 + CLIP embedding）
  6. 静默记忆 VLM 不会看到，除非主动调用 query_memory
```

---

## 4. 精简工具集

VLM 只需了解 7 个工具：

| 工具 | 功能 | 适用阶段 |
|------|------|---------|
| `observe_panorama` | 7 视角全景 → 返回拼接概览图（每张子图标编号） | 2 |
| `view_direction(id)` | 查看拼接图中某方向的原图大图 | 2, 3 |
| `navigate_to_object(desc)` | 自然语言描述目标物体 → GD 检测 → 导航链走到子目标 | 3 |
| `navigate_to_seed(room_id)` | 导航到指定房间种子点 | 4 |
| `navigate_to_frontier(id)` | 导航到指定 frontier | 4 |
| `query_memory(query)` | 自然语言查询记忆 → 返回匹配 snapshot 拼接图 | 3, 4, 5 |
| `submit_answer(answer, evidence)` | 提交最终答案 + 证据列表 | 6 |

**比 MSGNav 少掉的工具**: `query_scene`（俯视图由全景图替代）、`verify_grounded`（GD 检测内嵌在 navigate_to_object 中）。

---

## 5. Snapshot 存储与检索

### 存储

每张 snapshot 在存档时记录结构化元数据 + CLIP embedding：

```python
{
    "snapshot_id": "step3_view2",
    "room_id": 1,
    "objects_in_view": ["chair", "desk", "cabinet"],
    "position_3d": [6.2, 0.07, -1.5],
    "clip_embedding": tensor[...],
    "image_path": "snapshots/step3_view2.png"
}
```

### 检索流程

```
VLM: query_memory("R1里的椅子")

  1. 文本匹配过滤元数据 → 按 room_id + objects_in_view 预筛选
  2. CLIP 精排候选 → 取 top-k
  3. top-k snapshot 拼接成一张图发给 VLM
  4. VLM 可选:
     - view_snapshot(id) 查看某张大图
     - 若拼接图中都不相关 → 最多再查一次（总共2次查询配额）
```

---

## 6. 阶段过渡与上下文刷新

阶段切换时 VLM 自由格式生成过渡摘要，例如：

```
[阶段2 → 阶段3 过渡摘要]
当前位置: R1(厨房)，agent在房间中心偏东
已探索: R1 约60%，发现 object 14(oven), object 23(towel-like)
判断: 目标可能在 object 23 附近
下一阶段: GD导航到 object 23
预期: 确认 object 23 是否是目标物体
```

下一阶段开始时接收: `问题文本 + 过渡摘要 + 阶段所需图像`，不携带上一阶段的完整上下文和已用图像。

---

## 7. Fallback 机制

优先级从高到低：

1. **有新 region** → VLM 选择种子点 → 导航 → 阶段 1
2. **无新 region 但有未访问种子点 + frontier** → VLM 选择 → 导航 → 阶段 1 → 阶段 2
3. **所有 region 和 frontier 都探索完** → `query_memory` 查记忆 → 尝试回答
4. **记忆中无相关信息** → 标记"无法回答" + 记录原因和证据链

---

## 8. 实现范围

### 从 MSGNav 移植到 3D-Mem

| 模块 | 移植内容 | 目标文件 |
|------|---------|---------|
| 房间分割 | `RoomRegion` 类、`room_regions` 管理、gateway frontier、room_type 推断 | `src/tsdf_planner.py` |
| GD 导航链 | GroundingDINO 模型加载、GD→SAM→CLIP→3D 管线、navigate_to_object 逻辑 | `src/scene_aeqa.py` |
| 观察管线 | 全景拼接 `make_mosaic()`、`_observe_view()` 等辅助函数 | `agent_tools.py` |
| QUERY 基础 | 文本格式化逻辑（复用为 query_memory 的元数据构建） | `agent_memory.py` |

### 3D-Mem 中新建

| 文件 | 职责 |
|------|------|
| `agent_workflow.py` | 阶段控制器 + VLM API 调用 + 主循环 |
| `agent_tools.py` | 7 个工具的 Python 函数实现（对外暴露Agent调用接口） |
| `agent_memory.py` | Snapshot 存储 + CLIP 检索 + 元数据索引 |
| `agent_context.py` | 阶段过渡摘要管理 + 上下文刷新逻辑 |

### 复用 3D-Mem 现有

- `Scene` (`scene_aeqa.py`)
- `TSDFPlanner` (`tsdf_planner.py`)
- ConceptGraph 管线 (`conceptgraph/slam/`)
- `habitat.py`、`geom.py`、`utils.py`、`const.py`
- 配置文件 (`cfg/`)
