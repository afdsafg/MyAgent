import numpy as np
from src.seed_views import SeedViewManager


class _FakeScene:
    def get_observation(self, pts, angle):
        # Image depends on position so moving agent changes the view
        val = int((pts[0] * 100 + pts[2] * 10 + np.degrees(angle)) % 256)
        return {"color_sensor": np.full((100, 100, 3), val, dtype=np.uint8)}, None


class _FakePlanner:
    _voxel_size = 0.1
    min_height_voxel = 0
    _tsdf_vol_cpu = np.zeros((20, 20, 40), dtype=np.float32)

    def habitat2voxel(self, pos):
        return np.array(pos, dtype=int)


def test_register_seed_renders_image():
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    mgr.register_seed(1, np.array([5.0, 0.0, 5.0]), scene, planner,
                      np.array([0.0, 0.0, 0.0]))
    assert 1 in mgr.seeds
    assert mgr.seeds[1]["image"] is not None
    assert mgr.seeds[1]["image"].shape == (100, 100, 3)
    assert np.array_equal(mgr.seeds[1]["view_image_pos"], [0.0, 0.0, 0.0])


def test_update_after_step_no_change_when_dist_increases():
    """Agent moves away from seed -> no update."""
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    mgr.register_seed(1, np.array([3.0, 0.0, 0.0]), scene, planner,
                      np.array([0.0, 0.0, 0.0]))  # dist=3
    original_image = mgr.seeds[1]["image"].copy()
    # Agent moves away (dist=8)
    mgr.update_after_step([1], np.array([8.0, 0.0, 0.0]), planner, scene)
    assert np.array_equal(mgr.seeds[1]["image"], original_image)


def test_update_after_step_updates_when_dist_decreases():
    """Agent moves closer to seed -> update."""
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    mgr.register_seed(1, np.array([5.0, 0.0, 0.0]), scene, planner,
                      np.array([0.0, 0.0, 0.0]))  # dist=5
    original_image = mgr.seeds[1]["image"].copy()
    # Agent moves closer (dist=2)
    mgr.update_after_step([1], np.array([3.0, 0.0, 0.0]), planner, scene)
    assert not np.array_equal(mgr.seeds[1]["image"], original_image)


def test_get_mosaic_returns_image():
    mgr = SeedViewManager()
    scene = _FakeScene()
    planner = _FakePlanner()
    for sid in [1, 2, 3]:
        mgr.register_seed(sid, np.array([5.0, 0.0, float(sid)]),
                          scene, planner, np.array([0.0, 0.0, 0.0]))
    mosaic = mgr.get_mosaic("test question")
    assert mosaic is not None
    assert mosaic.ndim == 3  # H, W, 3
