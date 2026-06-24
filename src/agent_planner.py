"""Upper Planner: Qwen3.6-Plus API + 4-component structured prompt.

The Planner takes structured scene/history/progress/actions info and produces
a PlannerAction that the Executor carries out.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import openai

from src.const import (
    QWEN_PLANNER_API_KEY,
    QWEN_PLANNER_BASE_URL,
    QWEN_PLANNER_MODEL,
)

logger = logging.getLogger(__name__)


# ── Data class ────────────────────────────────────────────────────────────

@dataclass
class PlannerAction:
    """Structured decision from the Planner."""
    action_type: str
    reason: str = ""
    confidence: float = 0.0
    object_name: Optional[str] = None
    seed_id: Optional[str] = None
    frontier_id: Optional[str] = None
    view_idx: Optional[int] = None
    answer: Optional[str] = None


# ── System prompt ─────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are a high-level navigation planner for an embodied agent. \
Your job is to decide what the agent should do next to find the answer to a question. \
You receive structured information about what has been explored, what is currently visible, \
and how much progress has been made. \
You must output a JSON decision with a reason, action type, and confidence score. \
Do NOT repeat actions that have already been tried with the same outcome. \
Be strategic: use the History and Progress information to avoid redundant exploration.\
"""


# ── Planner class ─────────────────────────────────────────────────────────

class Planner:
    """Calls Qwen3.6-Plus (DashScope compatible-mode) and returns PlannerAction."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = QWEN_PLANNER_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def decide(
        self,
        question: str,
        history: str,
        scene: str,
        progress: str,
        actions: str,
        image_b64: Optional[str] = None,
    ) -> PlannerAction:
        """Call Qwen3.6-Plus with 4-component prompt, parse action."""
        prompt = self.build_prompt(question, history, scene, progress, actions)
        messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]

        if image_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            })
        else:
            messages.append({"role": "user", "content": prompt})

        response_text = self._call_api(messages)
        return self.parse_response(response_text)

    def build_prompt(
        self,
        question: str,
        history: str,
        scene: str,
        progress: str,
        actions: str,
    ) -> str:
        """Build the 4-component prompt string sent to the VLM."""
        return (
            f"You are a navigation agent. Given the information below, decide the next action.\n\n"
            f"Question: {question}\n\n"
            f"{history}\n\n"
            f"{scene}\n\n"
            f"{progress}\n\n"
            f"{actions}\n\n"
            f"Return ONLY a valid JSON object (no other text):\n"
            f'{{"reason": "why this action", "action": "explore_panorama|navigate_to_object|explore_seed|explore_frontier|inspect_object|submit_answer", "confidence": 0.8, "object_name": null, "seed_id": null, "frontier_id": null, "view_idx": null, "answer": null}}'
        )

    def parse_response(self, response: str) -> PlannerAction:
        """Parse VLM response — try JSON first, then keyword fallback."""
        if not response:
            return PlannerAction(action_type="explore_panorama", reason="Empty VLM response", confidence=0.0)
        
        import re
        raw = response.strip()
        
        # Try simple { ... } extraction first
        data = None
        if "{" in raw:
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                data = json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try ```json ... ``` code fences
        if data is None:
            m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass
        
        if data is not None:
            return PlannerAction(
                action_type=data.get("action", "explore_panorama"),
                reason=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.5)),
                object_name=data.get("object_name"),
                seed_id=str(data["seed_id"]) if data.get("seed_id") is not None else None,
                frontier_id=str(data["frontier_id"]) if data.get("frontier_id") is not None else None,
                view_idx=data.get("view_idx"),
                answer=data.get("answer"),
            )
        
        # Keyword fallback
        raw_l = raw.lower()
        for kw, action in [
            ("submit_answer", "submit_answer"),
            ("navigate_to_object", "navigate_to_object"),
            ("explore_seed", "explore_seed"),
            ("explore_frontier", "explore_frontier"),
            ("explore_panorama", "explore_panorama"),
        ]:
            if kw in raw_l:
                if action == "submit_answer":
                    # Try to extract answer from context
                    ans = raw.strip().split('\n')[-1][:300]
                    return PlannerAction(action_type=action, answer=ans, reason="Inferred", confidence=0.5)
                return PlannerAction(action_type=action, reason="Inferred", confidence=0.4,
                                    object_name="oven" if action == "navigate_to_object" else None)
        
        logger.info("Planner parse fallback, raw[:200]=%s", raw[:200])
        return PlannerAction(action_type="explore_panorama", reason="Parse failed", confidence=0.0)

    def _call_api(self, messages: list[dict]) -> str:
        """Call API via requests (matches original call_vlm in agent_workflow.py)."""
        import requests
        payload = {"model": self.model_name, "messages": messages, "max_tokens": 1024, "temperature": 0.3}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(self.base_url, json=payload, headers=headers, timeout=180)
        except Exception as e:
            logger.error(f"Planner API error: {e}")
            return ""
        if resp.status_code != 200:
            logger.error(f"Planner API error: {resp.status_code}")
            return ""
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        # Try content, then reasoning_content (mimo-v2.5 reasoning model)
        return msg.get("content") or msg.get("reasoning_content") or ""
