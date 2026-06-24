"""Tests for Task 3: Panorama improvements (8-view + labels + resolution 400)."""
import numpy as np
import pytest

from src.agent_tools import observe_panorama_config, VIEW_LABELS, observe_panorama


class TestPanoramaConfig:
    """Test panorama configuration constants and config helper."""

    def test_panorama_8_views(self):
        config = observe_panorama_config()
        assert config["view_count"] == 8
        assert config["resolution"] == 400
        assert len(config["view_labels"]) == 8
        assert "Front" in config["view_labels"]

    def test_view_labels(self):
        expected = [
            "Front", "Front-Right", "Right", "Back-Right",
            "Back", "Back-Left", "Left", "Front-Left",
        ]
        assert VIEW_LABELS == expected

    def test_view_labels_count_matches_config(self):
        config = observe_panorama_config()
        assert len(VIEW_LABELS) == config["view_count"]


class TestViewAngles:
    """Test that 8 angles cover the full 360 degree circle."""

    def test_view_angles_8(self):
        angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        assert len(angles) == 8
        assert angles[0] == -np.pi
        # Last angle should be at -pi + 7*pi/4 = 3pi/4
        expected_last = -np.pi + 7 * (2 * np.pi / 8)
        assert abs(angles[-1] - expected_last) < 1e-10

    def test_angles_cover_full_circle(self):
        n_views = observe_panorama_config()["view_count"]
        angles = np.linspace(-np.pi, np.pi, n_views, endpoint=False)
        # Consecutive diffs should all equal 2*pi/n_views
        step = 2 * np.pi / n_views
        for i in range(len(angles)):
            next_ang = angles[(i + 1) % len(angles)]
            diff = next_ang - angles[i]
            if diff < 0:
                diff += 2 * np.pi
            assert abs(diff - step) < 1e-10
