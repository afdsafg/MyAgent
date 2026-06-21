# HM-GE Agent Workflow 实现计划

> **For Claude:** 使用此计划逐任务实现 HM-GE Agent Workflow。每个 Task 完成后 commit。

**仓库:** https://github.com/afdsafg/MyAgent.git (本地 `3D-Mem/`，服务器 `/root/MyAgent/`)

**目标:** 在 3D-Mem 基础上构建 HM-GE Agent 工作流系统，部署到服务器运行和调试。

**架构:** 外层 Python 控制阶段切换，阶段内 VLM 自主决策。直接复用 3D-Mem 核心类（Scene、TSDFPlanner），从 MSGNav 选择性移植房间分割和 GD 导航代码。

**技术栈:** Python 3.9, Habitat-Sim, YOLO-World, SAM, CLIP, GroundingDINO, OpenAI API (mimo-v2.5)

**服��器:** root@8.147.163.63:59961, `/root/MyAgent/`, conda env `3dmem`, GPU RTX5880-Ada-16Q

---

## 环境说明

| 项目 | 值 |
|------|-----|
| SSH | `sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961` |
| 项目目录 | `/root/MyAgent/` |
| Conda 环境 | `3dmem` (`/root/miniconda3/envs/3dmem/`) |
| LLM API | `https://opencode.ai/zen/go/v1/chat/completions`, key `sk-saR5vgZjuzOpDn0wAbZnttiNvgRuoWLIok112YEWjeq1mLZvl9kFUMd88z2FpQ5Q`, model `mimo-v2.5` |
| HM3D 数据 | `/root/ContextNav/data/scene_datasets/hm3d` |
| AEQA 问题 | `/root/MyAgent/data/aeqa_questions-41.json` |
| 模型权重 | `/root/MyAgent/sam_l.pt`, `/root/MyAgent/yolov8x-world.pt` |
| GroundingDINO | `/home/afdsafg/grouding dino/GroundingDINO/` |

---

## 任务概览

```
Task 1: 服务器项目初始化（本地打包上传 → 服务器解压）
Task 2: 移植房间分割到 3D-Mem (tsdf_planner.py)
Task 3: 移植 GD 导航链到 3D-Mem (scene_aeqa.py)
Task 4: 创建图像工具模块 (image_utils.py)
Task 5: 创建 Agent 记忆模块 (agent_memory.py)
Task 6: 创建 Agent 工具模块 (agent_tools.py)
Task 7: 创建上下文管理模块 (agent_context.py)
Task 8: 创建主工作流控制器 (agent_workflow.py)
Task 9: 编写 AEQA 评估脚本 (run_hmge_evaluation.py)
Task 10: 端到端测试（Oven Towel 场景）
Task 11: 全量 AEQA 41 题评估
```

---

### Task 1: 服务器项目初始化

**目的:** 确认服务器 MyAgent 仓库已克隆，环境可用，API 配置正确。

服务器上 `/root/MyAgent/` 已从 `https://github.com/afdsafg/MyAgent.git` 克隆完成，
模型权重已通过软链接指向 `/root/3D-Mem/sam_l.pt` 和 `/root/3D-Mem/yolov8x-world.pt`。

**Step 1: 拉取最新代码到服务器**

```bash
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
```

**Step 2: 验证服务器环境**

```bash
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && source /root/miniconda3/etc/profile.d/conda.sh && conda activate 3dmem && python -c "import habitat_sim; print(\"habitat OK\"); import torch; print(f\"torch {torch.__version__}, cuda={torch.cuda.is_available()}\")"'
```

**Step 3: 更新 const.py 中的 API 配置**

编辑 `/root/MyAgent/src/const.py`，确保包含：

```python
OPENAI_API_KEY = "sk-saR5vgZjuzOpDn0wAbZnttiNvgRuoWLIok112YEWjeq1mLZvl9kFUMd88z2FpQ5Q"
OPENAI_BASE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MODEL_NAME = "mimo-v2.5"
```

**Step 4: 验证 GroundingDINO 路径**

服务器上 GroundingDINO 路径为 `/home/afdsafg/grouding dino/GroundingDINO/`，确认存在：

```bash
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'ls /home/afdsafg/grouding\ dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py 2>/dev/null && echo "GD OK" || echo "GD NOT FOUND"'
```

**Step 5: Commit 所有本地变更后推送，在服务器上验证**

后续每个 Task 完成后：本地 `git push` → 服务器 `git pull`

---

### Task 2: 移植房间分割到 tsdf_planner.py

**目的:** 给 3D-Mem 的 TSDFPlanner 添加房间分割能力。

**修改文件:** `src/tsdf_planner.py`

**Step 1: 添加导入**

在文件头部 import 区域添加（MSGNav 中这些 imports 在行 1-22）：
```python
import cv2
from collections import deque
```

**Step 2: 添加 RoomRegion 数据类**

在 Frontier 和 SnapShot 数据类之后插入（MSGNav 行 63-75）：
```python
@dataclass
class RoomRegion:
    """Geometric room region segmented from the current TSDF occupancy map."""
    room_id: int
    center: np.ndarray
    region: np.ndarray
    area: int
    room_state: str = "unknown"
    observed_ratio: float = 0.0
    hypothesis_node_id: Optional[str] = None
    frontier_ids: List[int] = field(default_factory=list)
```

**Step 3: 扩展 Frontier 数据类**

在 `Frontier` dataclass 中添加：
```python
room_id: int = -1
room_state: str = ""
```

**Step 4: 扩展 SnapShot 数据类**

添加：
```python
room_id: int = -1
```

**Step 5: 在 `TSDFPlanner.__init__` 中添加房间状态变量**

在 `self.frontier_counter = 1` 之后添加：
```python
self.room_map = np.zeros(self._vol_dim[:2], dtype=int)
self.room_regions: List[RoomRegion] = []
self.room_counter = 1
```

**Step 6: 在 `update_frontier_map` 末尾添加房间分割调用**

在方法末尾 `return True` 之前，从 MSGNav `src/tsdf_planner.py` 行 194-202 和 434-438 移植房间调用逻辑。

**Step 7: 移植房间核心方法**

从 MSGNav `src/tsdf_planner.py` 移植以下方法（按依赖顺序）：
- `_cfg_get` (行 636-646)
- `get_room_id_at` (行 1129-1152)
- `_find_geometric_room_seed_masks` (行 721-775) — 先移植这一路线，最稳定
- `_disk_seed_mask` (行 777-789)
- `_grow_regions_from_seeds` (行 791-824)
- `_make_watershed_seeds_from_grown_regions` (行 826-838)
- `_fill_watershed_boundary_gaps` (行 840-859)
- `_watershed_room_regions` (行 861-886)
- `_filter_room_regions_by_observed_adjacency` (行 888-937)
- `_room_regions_are_adjacent` (行 939-950)
- `_commit_room_regions` (行 952-996)
- `_match_previous_room_id` (行 998-1019)
- `update_room_map` (行 491-594) — 主入口
- `clear_room_map` (行 630-634)

**Step 8: 上传到服务器并测试导入**

```bash
cd /home/afdsafg/下载/new/3D-Mem && git push origin main && sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/3D-Mem && source /root/miniconda3/etc/profile.d/conda.sh && conda activate 3dmem && python -c "from src.tsdf_planner import TSDFPlanner, RoomRegion, Frontier; print(\"Import OK\")"'
```

**Step 9: Commit**

---

### Task 3: 移植 GD 导航链到 scene_aeqa.py

**目的:** 给 3D-Mem 的 Scene 类添加 GroundingDINO 开集检测和导航能力。

**修改文件:** `src/scene_aeqa.py`

**Step 1: 添加 GD 导入**

在头部添加：
```python
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
```

**Step 2: GD 模型单例加载**

添加全局函数：
```python
_gd_model = None

def _load_gd_model(gd_dir="/home/afdsafg/grouding dino/GroundingDINO"):
    """Lazy-load GroundingDINO Swin-T."""
    global _gd_model
    if _gd_model is not None:
        return _gd_model
    import sys
    sys.path.insert(0, gd_dir)
    config_path = os.path.join(gd_dir, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
    weights_path = os.path.join(gd_dir, "weights/groundingdino_swint_ogc.pth")
    args = SLConfig.fromfile(config_path)
    _gd_model = build_model(args)
    checkpoint = torch.load(weights_path, map_location="cpu")
    _gd_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _gd_model.eval()
    _gd_model = _gd_model.cuda()
    return _gd_model
```

**Step 3: 创建 GD→SAM→导航链工具函数**

添加到文件末尾：
```python
def grounded_navigate_to_object(
    scene, tsdf_planner, pts, angle, object_desc: str,
    max_steps: int = 20, gd_dir="/home/afdsafg/grouding dino/GroundingDINO"
):
    """GD 导航链：检测目标→计算导航点→移动。
    
    Returns: (new_pts, new_angle, success_bool, status_text, images_list)
    """
    # 实现 GD detect → SAM mask → 3D point → navigate loop
    pass  # 详见完整实现
```

**Step 4: 上传并测试**

```bash
cd /home/afdsafg/下载/new/3D-Mem && git push origin main && sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/3D-Mem && source /root/miniconda3/etc/profile.d/conda.sh && conda activate 3dmem && python -c "from src.scene_aeqa import Scene; print(\"Scene import OK\")"'
```

**Step 5: Commit**

---

### Task 4: 创建图像工具模块 (image_utils.py)

**创建文件:** `src/agent_image_utils.py`

从 MSGNav `mcp_server/image_utils.py` 移植：
- `numpy_to_base64` — RGB numpy → base64 JPEG
- `make_mosaic` — N 张图拼接成网格
- `fig_to_base64` — matplotlib figure → base64 PNG

**Step 1: 编写文件**

```python
"""图像工具：编码、拼接、转换。"""
import base64
import io
import numpy as np
from PIL import Image

def numpy_to_base64(img: np.ndarray, fmt="JPEG") -> str:
    """RGB uint8 numpy array → base64 string."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def make_mosaic(images, cols=None, target_h=240, padding=4):
    """拼接多张 RGB 图像为一张网格图。
    
    Args:
        images: list of (H,W,3) numpy uint8 arrays
        cols: 列数，None 则自动计算最接近正方形
        target_h: 每行统一高度
        padding: 图像间距像素
    Returns: (H,W,3) numpy uint8 mosaic
    """
    if not images:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)
    n = len(images)
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # 统一缩放到 target_h 高度
    scaled = []
    for img in images:
        h, w = img.shape[:2]
        new_w = int(target_h * w / h)
        pil = Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS)
        scaled.append(np.array(pil))
    # 取最大宽度
    max_w = max(s.shape[1] for s in scaled)
    # 构建画布
    canvas_h = rows * (target_h + padding) + padding
    canvas_w = cols * (max_w + padding) + padding
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240
    for i, img in enumerate(scaled):
        r, c = i // cols, i % cols
        y = r * (target_h + padding) + padding
        x = c * (max_w + padding) + padding
        h, w = img.shape[:2]
        x_offset = (max_w - w) // 2
        canvas[y:y+h, x+x_offset:x+x_offset+w] = img
    return canvas

def fig_to_base64(fig) -> str:
    """Matplotlib figure → base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
```

**Step 2: 上传并测试**

```bash
cd /home/afdsafg/下载/new/3D-Mem && git push origin main && sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/3D-Mem && source /root/miniconda3/etc/profile.d/conda.sh && conda activate 3dmem && python -c "from src.agent_image_utils import make_mosaic; import numpy as np; img = make_mosaic([np.zeros((100,100,3),dtype=np.uint8)+i*50 for i in range(4)]); print(f\"Mosaic shape: {img.shape}\")"'
```

**Step 3: Commit**

---

### Task 5: 创建 Agent 记忆模块 (agent_memory.py)

**创建文件:** `src/agent_memory.py`

**功能:**
- Snapshot 存储（图片 + 元数据 + CLIP embedding）
- 自然语言查询 → 文本过滤 + CLIP 精排
- 最多 2 次查询配额

**Step 1: 实现 SnapshotStore 类**

```python
"""HM-GE Agent 记忆模块。

Snapshot 存储、检索（自然语言→CLIP+元数据过滤）、查询配额管理。
"""
import json
import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from src.agent_image_utils import make_mosaic

@dataclass
class SnapshotEntry:
    """单个 snapshot 条目。"""
    snapshot_id: str
    room_id: int
    objects_in_view: List[str]
    position_3d: List[float]
    image_path: str
    clip_embedding: Optional[np.ndarray] = None

class MemoryStore:
    """Agent 静默记忆存储与检索。"""
    
    def __init__(self, output_dir: str = "/tmp/hmge_memory"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "snapshots"), exist_ok=True)
        self.snapshots: Dict[str, SnapshotEntry] = {}
        self.query_count = 0
        self.max_queries = 2
    
    def add_snapshot(
        self, snapshot_id: str, image: np.ndarray,
        room_id: int, objects_in_view: List[str],
        position_3d: List[float], clip_model, clip_preprocess, clip_tokenizer
    ):
        """存档一张 snapshot 及其元数据和 CLIP embedding。"""
        # 保存图片到磁盘
        img_path = os.path.join(self.output_dir, "snapshots", f"{snapshot_id}.png")
        from PIL import Image
        Image.fromarray(image).save(img_path)
        # 计算 CLIP embedding
        with torch.no_grad():
            img_tensor = clip_preprocess(Image.fromarray(image)).unsqueeze(0).cuda()
            embedding = clip_model.encode_image(img_tensor).cpu().numpy().flatten()
        # 存入索引
        self.snapshots[snapshot_id] = SnapshotEntry(
            snapshot_id=snapshot_id,
            room_id=room_id,
            objects_in_view=objects_in_view,
            position_3d=position_3d,
            image_path=img_path,
            clip_embedding=embedding,
        )
    
    def query(self, text_query: str, top_k: int = 8) -> Tuple[List[str], List[SnapshotEntry]]:
        """自然语言查询：文本过滤 + CLIP 精排。
        
        Returns: (image_paths, entries)
        """
        if self.query_count >= self.max_queries:
            return [], []
        self.query_count += 1
        
        # 简单文本匹配过滤：query 中的词是否匹配 objects_in_view 或 room_id
        query_words = set(text_query.lower().split())
        candidates = []
        for sid, entry in self.snapshots.items():
            text_meta = " ".join(entry.objects_in_view + [f"room{entry.room_id}"]).lower()
            if any(w in text_meta for w in query_words):
                candidates.append(entry)
        
        if not candidates:
            candidates = list(self.snapshots.values())
        
        # 取 top-k
        candidates = candidates[:top_k]
        paths = [e.image_path for e in candidates]
        return paths, candidates
    
    def make_query_mosaic(self, text_query: str, top_k: int = 8) -> Optional[np.ndarray]:
        """查询并拼接成一张图返回给 VLM。"""
        paths, _ = self.query(text_query, top_k)
        if not paths:
            return None
        from PIL import Image
        images = [np.array(Image.open(p)) for p in paths]
        return make_mosaic(images)
    
    def reset(self):
        """每个 episode 结束后重置。"""
        self.snapshots.clear()
        self.query_count = 0
```

**Step 2: 上传并测试**

```bash
cd /home/afdsafg/下载/new/3D-Mem && git push origin main && sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/3D-Mem && source /root/miniconda3/etc/profile.d/conda.sh && conda activate 3dmem && python -c "from src.agent_memory import MemoryStore; m = MemoryStore(); print(f\"MemoryStore OK, dir={m.output_dir}\")"'
```

**Step 3: Commit**

---

### Task 6: 创建 Agent 工具模块 (agent_tools.py)

**创建文件:** `src/agent_tools.py`

**功能:** 7 个工具的 Python 函数实现，阶段内 VLM 可调用。

**Step 1: 实现工具函数**

```python
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
    # 3 视角（正面 + 两侧各 60°）
    angles = [angle - np.pi/3, angle, angle + np.pi/3]
    all_added_obj_ids = []
    rgb_views = []
    
    for view_idx, ang in enumerate(angles):
        obs, cam_pose = scene.get_observation(pts, ang)
        rgb = obs["color_sensor"]
        depth = obs["depth_sensor"]
        obs_name = f"step{cnt_step}_view{view_idx}"
        
        # 场景图更新（YOLO+SAM+CLIP+3D）
        annotated_rgb, added_obj_ids, _ = scene.update_scene_graph(
            image_rgb=rgb[..., :3], depth=depth,
            intrinsics=cam_intr, cam_pos=cam_pose,
            pts=pts, pts_voxel=tsdf_planner.habitat2voxel(pts),
            img_path=obs_name, frame_idx=cnt_step * 3 + view_idx,
            target_obj_mask=None,
        )
        all_added_obj_ids += added_obj_ids
        rgb_views.append(rgb[..., :3])
        
        # TSDF 集成
        from src.habitat import pose_habitat_to_tsdf
        tsdf_planner.integrate(
            color_im=rgb, depth_im=depth, cam_intr=cam_intr,
            cam_pose=pose_habitat_to_tsdf(cam_pose),
            obs_weight=1.0,
            margin_h=int(cfg.margin_h_ratio * cfg.img_height),
            margin_w=int(cfg.margin_w_ratio * cfg.img_width),
            explored_depth=cfg.explored_depth,
        )
        
        # 定期物体清理
        scene.periodic_cleanup_objects(
            frame_idx=cnt_step * 3 + view_idx, pts=pts
        )
    
    # Snapshot 存档
    from src.hierarchy_clustering import update_snapshots
    scene.update_snapshots(obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection)
    
    # 存档到 MemoryStore
    room_id = tsdf_planner.get_room_id_at(tsdf_planner.habitat2voxel(pts)[:2])
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

# 1. observe_panorama
# 2. view_direction  
# 3. navigate_to_object
# 4. navigate_to_seed
# 5. navigate_to_frontier
# 6. query_memory
# 7. submit_answer
```

**Step 2: 上传并验证导入**

**Step 3: Commit**

---

### Task 7: 创建上下文管理模块 (agent_context.py)

**创建文件:** `src/agent_context.py`

**功能:** 阶段过渡摘要存储，上下文刷新管理。

**Step 1: 实现**

```python
"""HM-GE 上下文管理模块。

管理阶段过渡摘要与上下文刷新。阶段内共享上下文，阶段间用 VLM 自由格式摘要桥接。
图像 token 不跨阶段重复上传。
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class StageTransition:
    """阶段过渡摘要。VLM 在阶段结束时自由格式生成。"""
    from_stage: int
    to_stage: int
    summary: str  # VLM 自由格式文本
    images: List[str] = field(default_factory=list)  # 阶段专属图像路径

class ContextManager:
    """管理跨阶段的上下文传递。"""
    
    def __init__(self):
        self.transitions: List[StageTransition] = []
        self.current_stage = 0
        self.stage_messages = []  # 当前阶段对话记录
        self.stage_images = []    # 当前阶段图像（不跨阶段）
    
    def start_stage(self, stage_num: int):
        """开始新阶段，清空阶段内上下文。"""
        self.current_stage = stage_num
        self.stage_messages = []
        self.stage_images = []
    
    def add_message(self, role: str, content: str):
        """记录阶段内消息。"""
        self.stage_messages.append({"role": role, "content": content})
    
    def add_image(self, image_b64: str):
        """记录阶段内图像。"""
        self.stage_images.append(image_b64)
    
    def transition(self, to_stage: int, summary: str) -> StageTransition:
        """记录阶段过渡并保存 VLM 自由格式摘要。"""
        t = StageTransition(
            from_stage=self.current_stage,
            to_stage=to_stage,
            summary=summary,
            images=list(self.stage_images),
        )
        self.transitions.append(t)
        return t
    
    def get_stage_input(self, stage_num: int, question: str) -> str:
        """构建新阶段的输入：问题 + 上一阶段摘要。"""
        if self.transitions:
            last = self.transitions[-1]
            return (
                f"Question: {question}\n\n"
                f"[前一阶段({last.from_stage}→{last.to_stage})的过渡摘要]\n"
                f"{last.summary}\n\n"
                f"--- 当前阶段 {stage_num} 开始 ---"
            )
        return f"Question: {question}\n\n--- 阶段 {stage_num} 开始 ---"
    
    def reset(self):
        """每个 episode 结束后重置。"""
        self.transitions.clear()
        self.stage_messages.clear()
        self.stage_images.clear()
        self.current_stage = 0
```

**Step 2: 上传并验证**

**Step 3: Commit**

---

### Task 8: 创建主工作流控制器 (agent_workflow.py)

**创建文件:** `src/agent_workflow.py`

**功能:** 6 阶段主循环、VLM API 调用、阶段切换逻辑。

**这是核心文件，实现完整的 6 阶段流程：**

```
阶段1: 初始全景（7 视角 + 房间分割 + seed 点/前沿生成）
阶段2: 方向判断（VLM 看拼接图 → YES/NO）
阶段3: GD 导航链循环（选方向 → GD 导航 → 3 视角 → 重判断）
阶段4: 房间/前沿选择
阶段5: 最终 Fallback（查记忆 → 回答/标记无法回答）
阶段6: 提交答案
```

**Step 1: 编写 VLM API 调用函数**

```python
def call_vlm(messages: List[dict], image_b64: Optional[str] = None) -> str:
    """调用 mimo-v2.5 API。"""
    import requests
    payload = {
        "model": "mimo-v2.5",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    if image_b64:
        # 添加图像到消息
        payload["messages"][-1]["content"] = [
            {"type": "text", "text": payload["messages"][-1]["content"]},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    headers = {
        "Authorization": "Bearer sk-saR5vgZjuzOpDn0wAbZnttiNvgRuoWLIok112YEWjeq1mLZvl9kFUMd88z2FpQ5Q",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://opencode.ai/zen/go/v1/chat/completions",
        json=payload, headers=headers, timeout=180
    )
    return resp.json()["choices"][0]["message"]["content"]
```

**Step 2: 实现阶段逻辑**

实现 `run_episode(scene_id, question, question_id, cfg)` 主函数，包含完整的 6 阶段流程循环。

**Step 3: 上传并测试

```bash
cd /home/afdsafg/下载/new/3D-Mem && git push origin main && sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
```

**Step 4: Commit**

---

### Task 9: 编写 AEQA 评估脚本

**创建文件:** `run_hmge_evaluation.py`

以 3D-Mem 的 `run_aeqa_evaluation.py` 为模板，替换为 HM-GE workflow 调用。

**核心差异:**
- 用 `agent_workflow.run_episode()` 替代原有的 step-by-step VLM 选择循环
- 保持相同的日志和结果保存格式

**Step 1: 编写**

```python
#!/usr/bin/env python3
"""HM-GE Agent Workflow 评估脚本。在 AEQA 数据集上运行 Agent workflow。"""
# 基础结构与 run_aeqa_evaluation.py 相同
# 但用 agent_workflow.run_episode() 替代原有循环
```

**Step 2: 上传并运行首次测试**

```bash
cd /home/afdsafg/下载/new/3D-Mem && git push origin main && sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
```

**Step 3: Commit**

---

### Task 10: 端到端测试（Oven Towel 场景）

**目的:** 在单个场景上验证完整 workflow 是否跑通。

**场景:** `00824-Dd4bFSTQ8gi` — "What is hanging from the oven handle?"

**Step 1: 在服务器上启动测试**

```bash
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/3D-Mem && source /root/miniconda3/etc/profile.d/conda.sh && conda activate 3dmem && python src/agent_workflow.py --scene 00824-Dd4bFSTQ8gi --question "What is hanging from the oven handle?" 2>&1 | tee /root/MyAgent/results/hmge_test_oven.log'
```

**Step 2: 检查输出**
- 确认 6 个阶段都执行了
- 确认 VLM 返回了合理答案
- 检查阶段过渡摘要

**Step 3: Commit**

---

### Task 11: 全量 AEQA 41 题评估

**目的:** 在 41 题数据集上运行完整评估，获取准确率。

**Step 1: 运行评估**

```bash
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/3D-Mem && source /root/miniconda3/etc/profile.d/conda.sh && conda activate 3dmem && nohup python run_hmge_evaluation.py -cf cfg/eval_aeqa.yaml > /root/MyAgent/results/hmge_eval.log 2>&1 &'
```

可选分流运行：
```bash
python run_hmge_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5 &
python run_hmge_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.5 --end_ratio 1.0 &
```

**Step 2: 查看结果**

```bash
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cat /root/MyAgent/results/exp_eval_aeqa/results.json | python -m json.tool | head -50'
```

**Step 3: Commit 结果**

---

## 部署方式

**本地 git push → 服务器 git pull。** 所有代码都在仓库内，无需 scp 打包上传。

每个 Task 完成后的部署步骤：
```bash
# 本地 commit + push
cd /home/afdsafg/下载/new/3D-Mem
git add -A && git commit -m "task N: ..." && git push origin main

# 服务器拉取
sshpass -p '9a36555f-8d0f-403a-b9e9-a60b83b2ef93' ssh root@8.147.163.63 -p 59961 'cd /root/MyAgent && git pull origin main'
```
