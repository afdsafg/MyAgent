"""Tests for RunLogger core (Task 1)."""
import os
import tempfile
from src.run_logger import RunLogger


def test_init_creates_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        assert os.path.isdir(os.path.join(tmp, logger.run_timestamp))


def test_init_episode_creates_subdirs():
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        logger.init_episode("q-001", "what is on the oven?", "towel")
        ep_dir = os.path.join(tmp, logger.run_timestamp, "q-001")
        assert os.path.isdir(ep_dir)
        for sub in ["panorama", "stage2_decision", "stage2_5a_seed_selection",
                    "stage3_object_selection", "stage4_navigation",
                    "stage5_decision", "stage6_frontier_selection",
                    "seed_views", "snapshot"]:
            assert os.path.isdir(os.path.join(ep_dir, sub)), f"missing {sub}"


def test_trace_jsonl_exists_after_init():
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(output_root=tmp)
        logger.init_episode("q-001", "question?", "answer")
        trace_path = os.path.join(
            tmp, logger.run_timestamp, "q-001", "trace.jsonl")
        assert os.path.isfile(trace_path)
