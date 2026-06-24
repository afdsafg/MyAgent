"""Tests for GD quality filter.

gd_quality_filter 是位于 src/scene_aeqa.py 顶部的纯函数。由于场景文件深层依赖
habitat_sim / numba / HiPart 等环境不可用模块，本测试通过 importlib 读取源码并
exec 出函数，无需 import 整棵模块树。
"""
import ast
import os
import sys

import numpy as np

# ── 从源码文件中提取 gd_quality_filter 函数并 exec ──────────────────

_src_path = os.path.join(os.path.dirname(__file__), "..", "src", "scene_aeqa.py")
with open(_src_path, "r") as fh:
    _src = fh.read()

_tree = ast.parse(_src)

# 收集顶层函数定义直到 _gd_detect（含 gd_quality_filter）
_lines = _src.splitlines()
_start = None
_end = None
for i, node in enumerate(_tree.body):
    if isinstance(node, ast.FunctionDef):
        if node.name == "gd_quality_filter":
            _start = node.lineno - 1  # 0-indexed
        elif node.name == "_gd_detect" and _start is not None:
            _end = node.lineno - 1
            break

# 提取函数源码（含空行分隔）
_fn_src = "\n".join(_lines[_start:_end])

# exec 出函数（只需要 numpy）
_exec_ns = {"np": np}
exec(_fn_src, _exec_ns)
gd_quality_filter = _exec_ns["gd_quality_filter"]


class TestGDQualityFilter:
    """Unit tests for gd_quality_filter."""

    def test_bbox_too_large(self):
        """bbox covering >30% of image should be rejected."""
        bbox = np.array([0, 0, 768, 768])
        result, reason = gd_quality_filter(bbox, score=0.5, image_shape=(1280, 1280))
        assert result is None
        assert reason == "bbox_too_large"

    def test_score_too_low(self):
        """Low confidence detection should be rejected."""
        bbox = np.array([100, 100, 200, 200])
        result, reason = gd_quality_filter(bbox, score=0.25, image_shape=(1280, 1280))
        assert result is None
        assert reason == "score_too_low"

    def test_good_detection(self):
        """Normal-sized bbox with decent score should pass."""
        bbox = np.array([100, 100, 300, 300])
        result, reason = gd_quality_filter(bbox, score=0.55, image_shape=(1280, 1280))
        assert result is not None
        assert reason == "ok"
        np.testing.assert_array_equal(result, bbox)
