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

import requests

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
            f"Answer this embodied question by navigating to find the answer.\n\n"
            f"Question: {question}\n\n"
            f"{history}\n\n"
            f"{scene}\n\n"
            f"{progress}\n\n"
            f"{actions}\n\n"
            "Output your decision as JSON with these fields:\n"
            '{"reason": "...", "action": "...", "confidence": 0.0-1.0, '
            '[optional: "object_name", "seed_id", "frontier_id", "view_idx", "answer"]}'
        )

    def parse_response(self, response: str) -> PlannerAction:
        """Parse VLM JSON response into a PlannerAction."""
        import re
        data = None
        raw = response.strip()

        # Try ```json ... ``` code fences
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try raw { ... } extraction
        if data is None and "{" in raw:
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                data = json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

        if data is not None:
            return PlannerAction(
                action_type=data.get("action", "explore_panorama"),
                reason=data.get("reason", ""),
                confidence=float(data.get("confidence", 0.5)),
                object_name=data.get("object_name"),
                seed_id=str(data.get("seed_id")) if data.get("seed_id") is not None else None,
                frontier_id=str(data.get("frontier_id")) if data.get("frontier_id") is not None else None,
                view_idx=data.get("view_idx"),
                answer=data.get("answer"),
            )

        # Fallback: keyword-based action inference from natural language
        raw_l = raw.lower()
        if "submit_answer" in raw_l or "answer is" in raw_l:
            ans = raw.split("answer is")[-1].strip().strip('"').strip()
            return PlannerAction(action_type="submit_answer", answer=ans[:200], reason="Inferred", confidence=0.5)
        if "navigate_to_object" in raw_l or "go to" in raw_l or "move to" in raw_l:
            return PlannerAction(action_type="navigate_to_object", reason="Inferred", confidence=0.4, object_name="oven")
        if "explore_seed" in raw_l or "go to seed" in raw_l:
            return PlannerAction(action_type="explore_seed", reason="Inferred", confidence=0.4)
        if "explore_frontier" in raw_l or "frontier" in raw_l:
            return PlannerAction(action_type="explore_frontier", reason="Inferred", confidence=0.4)

        logger.debug("Planner parse failed, raw=%.200s", raw)
        return PlannerAction(action_type="explore_panorama", reason="Parse failed", confidence=0.0)

    def _call_api(self, messages: list[dict]) -> str:
        """Call mimo-v2.5 via requests.post (matches original call_vlm pattern)."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        proxies = {}
        proxy_http = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
        proxy_https = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        if proxy_http:
            proxies["http"] = proxy_http
        if proxy_https:
            proxies["https"] = proxy_https

        try:
            resp = requests.post(
                self.base_url, json=payload, headers=headers,
                timeout=180, proxies=proxies if proxies else None)
        except Exception as e:
            logger.error(f"Planner API request failed: {e}")
            return ""

        if resp.status_code != 200:
            logger.error(f"Planner API error: {resp.status_code} {resp.text[:500]}")
            return ""

        data = resp.json()
        return data["choices"][0]["message"].get("content", "")
