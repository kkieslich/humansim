"""Polygon and wall geometry shared by the spawner, the despawner and the lanes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from arena_humansim.utils.types import Pose2D, Segment


def point_in_polygon(px: float, py: float, vertices: Sequence[Pose2D]) -> bool:
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        vi, vj = vertices[i], vertices[j]
        if ((vi.y > py) != (vj.y > py)) and (px < (vj.x - vi.x) * (py - vi.y) / (vj.y - vi.y) + vi.x):
            inside = not inside
        j = i
    return inside


def box_wall_segments(pose: Pose2D, x_min: float, x_max: float, y_min: float, y_max: float) -> list[Segment]:
    """Convert a rotated 2D bounding box into 4 wall segments."""
    cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
    corners_local = [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (x_max, y_min),
    ]
    corners_world = []
    for lx, ly in corners_local:
        wx = pose.x + cos_t * lx - sin_t * ly
        wy = pose.y + sin_t * lx + cos_t * ly
        corners_world.append((wx, wy))

    segments = []
    for i in range(4):
        j = (i + 1) % 4
        segments.append((corners_world[i], corners_world[j]))
    return segments
