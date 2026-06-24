"""Tests for src/agent_planner.py — PlannerAction, prompt building, JSON parsing."""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import target
# ---------------------------------------------------------------------------
from src.agent_planner import Planner, PlannerAction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def planner():
    return Planner(api_key="test-key", base_url="https://example.invalid/v1")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_parse_planner_response(planner):
    response = (
        '{"reason": "oven likely in kitchen", '
        '"action": "navigate_to_object", '
        '"object_name": "oven", '
        '"confidence": 0.7}'
    )
    action = planner.parse_response(response)
    assert action.action_type == "navigate_to_object"
    assert action.object_name == "oven"
    assert pytest.approx(action.confidence) == 0.7
    assert action.reason == "oven likely in kitchen"


def test_parse_planner_response_with_markdown(planner):
    response = "```json\n{\n  \"action\": \"explore_seed\",\n  \"seed_id\": 3,\n  \"confidence\": 0.85\n}\n```"
    action = planner.parse_response(response)
    assert action.action_type == "explore_seed"
    assert action.seed_id == "3"
    assert pytest.approx(action.confidence) == 0.85


def test_parse_planner_response_bad_json_fallback(planner):
    action = planner.parse_response("This is not JSON at all.")
    assert action.action_type == "explore_panorama"
    assert "Parse failed" in action.reason


def test_build_prompt_contains_components(planner):
    history = "## History\n- [Step 5] Bedroom: no oven found"
    scene = "## Scene Analysis\n- View 0: [cabinet, door]"
    progress = "## Progress\nTarget oven not found. Kitchen to explore."
    actions = "## Actions\n1. navigate_to_object\n2. explore_seed"

    prompt = planner.build_prompt(
        question="What color is the towel?",
        history=history,
        scene=scene,
        progress=progress,
        actions=actions,
    )
    assert "History" in prompt
    assert "Scene Analysis" in prompt
    assert "Progress" in prompt
    assert "What color is the towel?" in prompt


def test_planner_action_dataclass():
    action = PlannerAction(
        action_type="explore_seed",
        seed_id="3",
        confidence=0.6,
        reason="seed 3 near kitchen",
    )
    assert action.action_type == "explore_seed"
    assert action.seed_id == "3"
    assert pytest.approx(action.confidence) == 0.6
    assert action.reason == "seed 3 near kitchen"
    assert action.object_name is None
    assert action.frontier_id is None
    assert action.answer is None


def test_planner_action_submit_answer():
    action = PlannerAction(
        action_type="submit_answer",
        answer="blue",
        confidence=1.0,
        reason="saw blue towel in kitchen",
    )
    assert action.action_type == "submit_answer"
    assert action.answer == "blue"


def test_planner_action_defaults():
    action = PlannerAction(action_type="explore_panorama")
    assert action.action_type == "explore_panorama"
    assert action.reason == ""
    assert action.confidence == 0.0
    assert action.object_name is None
    assert action.seed_id is None
    assert action.frontier_id is None
    assert action.view_idx is None
    assert action.answer is None
