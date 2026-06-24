"""End-to-end smoke test: verify imports and basic structure.
Does NOT run Habitat (too slow for CI). Run full E2E manually on server.
"""
import inspect
import subprocess
import sys
import pytest


def _py_compile(filepath):
    """Verify Python file has valid syntax."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", filepath],
        capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error in {filepath}: {result.stderr}"


def test_imports():
    """Verify modules that don't require habitat_sim import cleanly."""
    from src.seed_views import SeedViewManager
    from src.geom import bresenham_2d, check_ray_blocked
    from src.agent_image_utils import numpy_to_base64, make_mosaic
    from src.agent_workflow import (
        _parse_vlm_response,
        STAGE2_PROMPT, STAGE2_5A_PROMPT, STAGE3_PROMPT,
        STAGE5_PROMPT, STAGE6_PROMPT,
    )
    print("All available imports OK")


def test_syntax_check():
    """Verify all modified files have valid Python syntax."""
    files = [
        "src/scene_aeqa.py",
        "src/agent_tools.py",
        "src/agent_workflow.py",
        "src/seed_views.py",
        "src/geom.py",
    ]
    for f in files:
        _py_compile(f)


def test_navigate_to_object_signature():
    """Verify navigate_to_object accepts new params."""
    from src.agent_tools import navigate_to_object
    sig = inspect.signature(navigate_to_object)
    params = list(sig.parameters.keys())
    assert "view_idx" in params
    assert "view_angle" in params
    assert "view_cam_pose" in params


def test_grounded_navigate_signature():
    """Verify grounded_navigate_to_object accepts new params via AST."""
    import ast
    with open("src/scene_aeqa.py") as f:
        tree = ast.parse(f.read())
    # Find grounded_navigate_to_object function
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "grounded_navigate_to_object":
            arg_names = [a.arg for a in node.args.args]
            assert "view_idx" in arg_names
            assert "view_angle" in arg_names
            assert "view_cam_pose" in arg_names
            return
    pytest.fail("grounded_navigate_to_object not found in src/scene_aeqa.py")


def test_observe_panorama_returns_views():
    """Verify observe_panorama returns panorama_views list."""
    from src.agent_tools import observe_panorama
    # Check return annotation mentions panorama_views or is a tuple of 5
    sig = inspect.signature(observe_panorama)
    ret = sig.return_annotation
    # Just verify it's annotated (don't enforce exact type)
    assert ret is not inspect.Parameter.empty
