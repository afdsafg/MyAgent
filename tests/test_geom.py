import numpy as np
from src.geom import bresenham_2d


def test_bresenham_horizontal():
    """Horizontal ray along x-axis."""
    pts = bresenham_2d((5, 0), (5, 10))
    assert pts[0] == (5, 0)
    assert pts[-1] == (5, 10)
    assert len(pts) == 11


def test_bresenham_vertical():
    """Vertical ray along y-axis."""
    pts = bresenham_2d((0, 3), (8, 3))
    assert pts[0] == (0, 3)
    assert pts[-1] == (8, 3)
    assert len(pts) == 9


def test_bresenham_diagonal():
    """45-degree diagonal."""
    pts = bresenham_2d((0, 0), (5, 5))
    assert (0, 0) in pts
    assert (5, 5) in pts
    # Diagonal should hit every (i, i)
    for i in range(6):
        assert (i, i) in pts


def test_bresenham_single_point():
    """Start == end."""
    pts = bresenham_2d((3, 3), (3, 3))
    assert pts == [(3, 3)]


def test_bresenham_negative_direction():
    """Ray going in negative y direction."""
    pts = bresenham_2d((10, 5), (2, 5))
    assert pts[0] == (10, 5)
    assert pts[-1] == (2, 5)
    assert len(pts) == 9


class _FakePlanner:
    """Minimal stand-in for TSDFPlanner exposing only what check_ray_blocked needs."""
    def __init__(self, tsdf_vol, voxel_size=0.05, min_height_voxel=0):
        self._tsdf_vol_cpu = tsdf_vol  # shape (H, W, Z)
        self._voxel_size = voxel_size
        self.min_height_voxel = min_height_voxel

    def habitat2voxel(self, pos):
        # Simplified: assume pos already in voxel coords (test only)
        return np.array(pos, dtype=int)


def test_check_ray_blocked_clear_path():
    """No obstacles above 1.2m -> not blocked."""
    # 10x10x40 volume, all zeros (free space)
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    blocked = check_ray_blocked(planner, [5, 0, 5], [5, 0, 1],
                                 min_blocking_height=1.2)
    assert blocked is False


def test_check_ray_blocked_wall():
    """Wall (TSDF<0) above 1.2m -> blocked."""
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    # Place a wall at voxel (5, 3) from z=12 to z=30 (1.2m to 3.0m)
    vol[5, 3, 12:30] = -0.5  # occupied
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    # Ray from (5,0) to (5,4) passes through voxel (5,3) as interior point
    # Note: _FakePlanner.habitat2voxel casts to int directly, so (y,x) come from first two elements
    blocked = check_ray_blocked(planner, [5, 0, 0], [5, 4, 0],
                                 min_blocking_height=1.2)
    assert blocked is True


def test_check_ray_blocked_low_obstacle():
    """Table (0.75m) below 1.2m threshold -> not blocked."""
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    # Table at z=0..7 (0 to 0.7m), should NOT block
    vol[5, 3, 0:8] = -0.5
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    blocked = check_ray_blocked(planner, [5, 0, 5], [5, 0, 1],
                                 min_blocking_height=1.2)
    assert blocked is False


def test_check_ray_blocked_endpoints_skipped():
    """Endpoints (agent and target voxels) should be skipped."""
    vol = np.zeros((10, 10, 40), dtype=np.float32)
    # Obstacle AT agent voxel — should be skipped
    vol[5, 0, 12:30] = -0.5
    planner = _FakePlanner(vol, voxel_size=0.1, min_height_voxel=0)
    from src.geom import check_ray_blocked
    blocked = check_ray_blocked(planner, [5, 0, 5], [5, 0, 1],
                                 min_blocking_height=1.2)
    assert blocked is False  # endpoint skipped
