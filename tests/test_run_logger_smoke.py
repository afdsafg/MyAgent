"""Smoke test: verify RunLogger can be instantiated and all methods exist."""
import os
import tempfile
import numpy as np
from src.run_logger import RunLogger


def test_all_logging_methods_exist():
    """Verify all expected methods are defined."""
    logger = RunLogger(enabled=False)
    methods = [
        "init_episode", "log_trace", "log_summary",
        "render_topdown", "log_panorama", "log_vlm_decision",
        "log_gd_detection", "log_backprojection",
        "log_spiral_search", "log_nav_step", "log_iter_summary",
        "log_final_topdown", "log_seed_view_update", "log_snapshot",
    ]
    for m in methods:
        assert hasattr(logger, m), f"missing method: {m}"


def test_trace_writes_jsonl():
    """Verify trace.jsonl accumulates events."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        logger.init_episode("q-test", "question?", "answer")
        logger.log_trace("test_event", {"key": "value"}, reason="test reason")
        logger.log_trace("another", {"num": 42})

        trace_path = os.path.join(tmp, logger.run_timestamp, "q-test", "trace.jsonl")
        with open(trace_path) as f:
            lines = f.readlines()
        assert len(lines) >= 3  # episode_start + 2 events
        import json
        for line in lines:
            evt = json.loads(line)
            assert "ts" in evt
            assert "event" in evt


def test_disabled_logger_no_files():
    """When disabled, no files should be created."""
    logger = RunLogger(enabled=False)
    logger.init_episode("q-test", "q", "a")
    logger.log_trace("test", {})
    # No exception, no files
