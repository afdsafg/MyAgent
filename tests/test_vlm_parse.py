import json
from src.agent_workflow import _parse_vlm_response


def test_parse_with_reason():
    """Valid response with reason field."""
    resp = json.dumps({
        "reason": "I see an oven in view 3",
        "action": "navigate_to_object",
        "view_idx": 3,
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "navigate_to_object"
    assert parsed["reason"] == "I see an oven in view 3"
    assert parsed["view_idx"] == 3


def test_parse_missing_reason():
    """Response without reason -> flagged as missing_reason."""
    resp = json.dumps({"action": "explore_other_room"})
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "missing_reason"


def test_parse_submit_answer():
    resp = json.dumps({
        "reason": "I can see the towel on the oven handle",
        "action": "submit_answer",
        "answer": "yes",
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "submit_answer"
    assert parsed["answer"] == "yes"


def test_parse_explore_seed():
    resp = json.dumps({
        "reason": "seed 2 is toward the kitchen",
        "action": "explore_seed",
        "seed_id": 2,
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "explore_seed"
    assert parsed["seed_id"] == 2


def test_parse_explore_frontier():
    resp = json.dumps({
        "reason": "all seeds are bedrooms, fallback to frontier",
        "action": "explore_frontier",
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "explore_frontier"


def test_parse_frontier_selection():
    resp = json.dumps({
        "reason": "frontier 0 leads to unexplored hallway",
        "frontier_id": 0,
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "explore_frontier"
    assert parsed["frontier_id"] == 0


def test_parse_object_selection():
    resp = json.dumps({
        "reason": "stainless steel appliance with square door",
        "object": "oven",
    })
    parsed = _parse_vlm_response(resp)
    assert parsed["tool"] == "object_selected"
    assert parsed["object"] == "oven"


def test_parse_invalid_json():
    parsed = _parse_vlm_response("not json at all")
    assert parsed["tool"] == "parse_error"
