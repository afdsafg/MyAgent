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
    summary: str
    images: List[str] = field(default_factory=list)


class ContextManager:
    """管理跨阶段的上下文传递。"""

    def __init__(self):
        self.transitions: List[StageTransition] = []
        self.current_stage = 0
        self.stage_messages = []
        self.stage_images = []

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
