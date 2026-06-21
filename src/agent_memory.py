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
        from PIL import Image
        img_path = os.path.join(
            self.output_dir, "snapshots", f"{snapshot_id}.png")
        Image.fromarray(image).save(img_path)

        with torch.no_grad():
            img_tensor = clip_preprocess(
                Image.fromarray(image)).unsqueeze(0).cuda()
            embedding = clip_model.encode_image(
                img_tensor).cpu().numpy().flatten()

        self.snapshots[snapshot_id] = SnapshotEntry(
            snapshot_id=snapshot_id,
            room_id=room_id,
            objects_in_view=objects_in_view,
            position_3d=position_3d,
            image_path=img_path,
            clip_embedding=embedding,
        )

    def query(
        self, text_query: str, top_k: int = 8
    ) -> Tuple[List[str], List[SnapshotEntry]]:
        """自然语言查询：文本过滤 + CLIP 精排。

        Returns: (image_paths, entries)
        """
        if self.query_count >= self.max_queries:
            return [], []

        self.query_count += 1

        query_words = set(text_query.lower().split())
        candidates = []
        for sid, entry in self.snapshots.items():
            text_meta = " ".join(
                entry.objects_in_view + [f"room{entry.room_id}"]).lower()
            if any(w in text_meta for w in query_words):
                candidates.append(entry)

        if not candidates:
            candidates = list(self.snapshots.values())

        candidates = candidates[:top_k]
        paths = [e.image_path for e in candidates]
        return paths, candidates

    def make_query_mosaic(
        self, text_query: str, top_k: int = 8
    ) -> Optional[np.ndarray]:
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
