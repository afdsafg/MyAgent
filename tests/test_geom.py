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
