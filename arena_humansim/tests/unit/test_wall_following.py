"""Routes laid along walls (`global_planner/wall_following.py`): a route keeps an offset from a wall that runs along it,
on the preferred side, goes round what stands against the wall, and lines up before a narrow stretch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.global_planner.wall_following import PASSING_GAP, approach, follow_walls, leave, sample, thin
from arena_humansim.utils.wall_grid import WallGrid, cast_rays

Segment = tuple[tuple[float, float], tuple[float, float]]
# a 4 m corridor along x, 30 m long
CORRIDOR: list[Segment] = [((0.0, 0.0), (30.0, 0.0)), ((0.0, 4.0), (30.0, 4.0))]
MIDDLE = np.array([(1.0, 2.0), (29.0, 2.0)])
BODY, OFFSET, REACH = 0.3, 0.42, 4.5


def _world(segments: list[Segment]) -> tuple[Callable[[np.ndarray, np.ndarray, float], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    grid = WallGrid(segments)
    seg = np.array(segments, dtype=np.float64).reshape(-1, 4)
    a, ab = seg[:, :2], seg[:, 2:] - seg[:, :2]

    def cast(origins: np.ndarray, directions: np.ndarray, far: float) -> np.ndarray:
        return cast_rays(grid, origins, directions, far)

    def clearance(points: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(points)
        t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(axis=2) / (ab * ab).sum(axis=1)[None], 0.0, 1.0)
        gap = p[:, None] - (a[None] + t[..., None] * ab[None])
        return np.hypot(gap[..., 0], gap[..., 1]).min(axis=1)

    return cast, clearance


def _trail(segments: list[Segment], route: np.ndarray, **kw: float | bool) -> np.ndarray:
    cast, clearance = _world(segments)
    settings: dict = dict(body=BODY, offset=OFFSET, reach=REACH, prefer=1.0, either=True, ramp=1.0)
    settings.update(kw)
    return follow_walls(route, cast, clearance, **settings).points


def _between(points: np.ndarray, x0: float, x1: float) -> np.ndarray:
    return points[(points[:, 0] >= x0) & (points[:, 0] <= x1)]


@pytest.mark.parametrize(("prefer", "y"), [(1.0, 4.0 - OFFSET), (-1.0, OFFSET)])
def test_a_route_down_a_corridor_runs_at_the_offset_from_the_wall_on_the_preferred_side(prefer: float, y: float) -> None:
    points = _trail(CORRIDOR, MIDDLE, prefer=prefer)
    assert points[0].tolist() == [1.0, 2.0]
    assert points[-1].tolist() == [29.0, 2.0]
    assert _between(points, 4.0, 26.0)[:, 1] == pytest.approx(y)


def test_without_a_preferred_side_the_nearer_wall_is_followed() -> None:
    points = _trail(CORRIDOR, np.array([(1.0, 1.0), (29.0, 1.0)]), prefer=0.0)
    assert _between(points, 4.0, 26.0)[:, 1] == pytest.approx(OFFSET)


def test_a_wall_beyond_reach_leaves_the_route_where_it_is() -> None:
    cast, clearance = _world(CORRIDOR)
    shaped = follow_walls(MIDDLE, cast, clearance, body=BODY, offset=OFFSET, reach=1.0, prefer=1.0, either=True)
    assert not shaped.follows.any()
    assert shaped.points[:, 1] == pytest.approx(2.0)


def test_the_other_side_is_followed_where_only_it_has_a_wall_unless_one_side_is_asked_for() -> None:
    south_only = CORRIDOR[:1]
    assert _between(_trail(south_only, MIDDLE, prefer=1.0), 4.0, 26.0)[:, 1] == pytest.approx(OFFSET)
    assert _trail(south_only, MIDDLE, prefer=1.0, either=False)[:, 1] == pytest.approx(2.0)


def test_the_route_goes_round_what_stands_against_the_wall() -> None:
    box: list[Segment] = [((14.0, 4.0), (14.0, 3.4)), ((14.0, 3.4), (16.0, 3.4)), ((16.0, 3.4), (16.0, 4.0))]
    _, clearance = _world(CORRIDOR + box)
    points = _trail(CORRIDOR + box, MIDDLE)
    dense = sample(points, 0.05)[0]
    assert clearance(dense).min() >= BODY - 5e-3
    assert _between(points, 14.4, 15.6)[:, 1] == pytest.approx(3.4 - OFFSET)
    assert _between(points, 5.0, 12.0)[:, 1] == pytest.approx(4.0 - OFFSET)
    assert _between(points, 18.0, 25.0)[:, 1] == pytest.approx(4.0 - OFFSET)


def test_a_short_piece_of_wall_is_not_followed() -> None:
    pillar: list[Segment] = [((14.0, 3.0), (15.0, 3.0))]
    assert _trail(pillar, MIDDLE)[:, 1] == pytest.approx(2.0)


def test_a_wall_is_followed_only_where_it_runs_along_for_longer_than_the_way_to_it_and_back() -> None:
    route = np.array([(0.0, 0.0), (40.0, 0.0)])
    short: list[Segment] = [((18.0, 4.0), (22.0, 4.0))]
    long: list[Segment] = [((5.0, 4.0), (35.0, 4.0))]
    assert _trail(short, route)[:, 1] == pytest.approx(0.0)
    assert _between(_trail(long, route), 12.0, 28.0)[:, 1] == pytest.approx(4.0 - OFFSET)


def test_the_route_reaches_the_wall_where_its_stretch_begins_when_it_starts_early() -> None:
    wall: list[Segment] = [((10.0, 3.0), (30.0, 3.0))]
    route = np.array([(0.0, 0.0), (40.0, 0.0)])
    early = _trail(wall, route)
    late = _trail(wall, route, early=False)
    assert _between(early, 10.0, 30.0)[:, 1] == pytest.approx(3.0 - OFFSET)
    assert _between(late, 0.0, 9.9)[:, 1] == pytest.approx(0.0)
    assert _between(late, 10.0, 10.1)[:, 1].max() <= 0.5 + 1e-9
    assert _between(early, 0.0, 7.0)[:, 1] == pytest.approx(0.0)
    assert np.abs(np.diff(early[:, 1]) / np.maximum(np.hypot(*np.diff(early, axis=0).T), 1e-9)).max() <= np.sin(np.pi / 4) + 1e-6


def test_an_outer_corner_is_walked_along_the_wall() -> None:
    walls: list[Segment] = [((0.0, 0.0), (22.0, 0.0)), ((22.0, 0.0), (22.0, 20.0))]
    route = np.array([(2.0, 2.0), (20.0, 2.0), (20.0, 18.0)])
    points = _trail(walls, route, prefer=-1.0)
    corner = np.array([22.0 - OFFSET, OFFSET])
    assert np.hypot(*(points - corner).T).min() < 0.02


def test_a_door_in_the_wall_is_crossed_straight() -> None:
    wall: list[Segment] = [((0.0, 4.0), (13.0, 4.0)), ((15.0, 4.0), (30.0, 4.0))]
    kept = _trail(wall, MIDDLE, bridge=REACH)
    assert _between(kept, 5.0, 25.0)[:, 1] == pytest.approx(4.0 - OFFSET)
    left = _trail(wall, MIDDLE)
    assert _between(left, 13.5, 14.5)[:, 1].min() < 4.0 - OFFSET - 0.2


def test_offset_zero_only_measures_the_route() -> None:
    cast, clearance = _world(CORRIDOR)
    shaped = follow_walls(MIDDLE, cast, clearance, body=BODY, offset=0.0, reach=0.0, prefer=1.0, either=False, span=5.0)
    assert shaped.points[:, 1] == pytest.approx(2.0)
    assert shaped.width[1:-1] == pytest.approx(4.0)
    assert not shaped.follows.any()


def test_a_long_body_keeps_both_ends_off_the_wall_while_it_moves_to_its_lane() -> None:
    walls: list[Segment] = [((0.0, 0.0), (30.0, 0.0)), ((0.0, 2.3), (30.0, 2.3))]
    cast, clearance = _world(walls)
    route = np.array([(1.0, 1.15), (29.0, 1.15)])
    radius, axial, lane = 0.325, 0.225, 0.4575
    shaped = follow_walls(route, cast, clearance, body=radius + 0.05, offset=lane, reach=3.6, prefer=-1.0, either=False, corridor=3.6, early=False, along=0.36, axial=axial, ends=radius)
    points = shaped.points
    assert _between(points, 4.0, 26.0)[:, 1] == pytest.approx(lane)
    legs = np.diff(points, axis=0)
    legs = legs / np.hypot(legs[:, 0], legs[:, 1])[:, None]
    for along in (np.vstack([legs[:1], legs]), np.vstack([legs, legs[-1:]])):
        assert clearance(points + axial * along).min() >= radius - 1e-9
        assert clearance(points - axial * along).min() >= radius - 1e-9


def test_a_corridor_wider_than_asked_for_has_no_lane() -> None:
    cast, clearance = _world(CORRIDOR)
    shaped = follow_walls(MIDDLE, cast, clearance, body=0.375, offset=0.4575, reach=3.6, prefer=-1.0, either=False, corridor=3.6, early=False)
    assert not shaped.follows.any()


# a hall (x < 0, 10 m wide) that narrows to a passage along x from 0 to 10, and a hall beyond it
def _narrowing(width: float) -> list[Segment]:
    w = width / 2.0
    return [
        ((-12.0, -5.0), (0.0, -5.0)),
        ((0.0, -5.0), (0.0, -w)),
        ((0.0, -w), (10.0, -w)),
        ((10.0, -w), (10.0, -5.0)),
        ((10.0, -5.0), (22.0, -5.0)),
        ((-12.0, 5.0), (0.0, 5.0)),
        ((0.0, 5.0), (0.0, w)),
        ((0.0, w), (10.0, w)),
        ((10.0, w), (10.0, 5.0)),
        ((10.0, 5.0), (22.0, 5.0)),
    ]


RADIUS, AXIAL, LANE = 0.325, 0.225, 0.4575
FIT = dict(body=RADIUS + 0.05, axial=AXIAL, ends=RADIUS)
THROUGH = np.array([(-11.0, 3.0), (-0.6, 0.0), (10.6, 0.0), (21.0, -3.0)])


def _lanes(width: float, prefer: float, lane: float = LANE):  # noqa: ANN202
    cast, clearance = _world(_narrowing(width))
    return follow_walls(THROUGH, cast, clearance, offset=lane, reach=3.6, prefer=prefer, either=False, corridor=3.6, span=1.8, early=False, along=0.36, bridge=3.6, **FIT), clearance


def _inside(points: np.ndarray) -> np.ndarray:
    return points[(points[:, 0] > 0.0) & (points[:, 0] < 10.0)]


def test_a_lane_is_entered_and_left_in_line_with_it() -> None:
    lanes, clearance = _lanes(2.3, prefer=-1.0)
    y = -1.15 + LANE
    assert _inside(lanes.points)[:, 1].max() > y + 0.2
    straight = leave(lanes, lanes.follows, clearance, length=1.1, **FIT)
    straight = approach(straight, straight.follows, clearance, length=1.5, prefer=-1.0, **FIT)
    points = straight.points
    assert _inside(points)[:, 1] == pytest.approx(y)
    assert points[(points[:, 0] > -1.4) & (points[:, 0] <= 0.0)][:, 1] == pytest.approx(y)
    assert points[(points[:, 0] >= 10.0) & (points[:, 0] < 11.0)][:, 1] == pytest.approx(y)
    assert points[0].tolist() == THROUGH[0].tolist()
    assert points[-1].tolist() == THROUGH[-1].tolist()
    dense, facing = sample(points, 0.05)
    assert clearance(dense).min() >= RADIUS
    assert min(clearance(dense + AXIAL * facing).min(), clearance(dense - AXIAL * facing).min()) >= RADIUS - 5e-3


def test_before_a_narrow_stretch_the_route_lines_up_beside_its_mouth() -> None:
    lanes, clearance = _lanes(1.5, prefer=-1.0)
    shaped = approach(lanes, lanes.width < 1.8, clearance, length=4.5, lateral=LANE + RADIUS + PASSING_GAP, prefer=-1.0, waits=True, **FIT)
    waiting = shaped.points[shaped.waits]
    assert len(waiting) >= 2
    assert waiting[:, 1] == pytest.approx(-(0.75 + RADIUS + PASSING_GAP))
    assert waiting[:, 0].max() < -1.0
    assert _inside(shaped.points)[:-1, 1] == pytest.approx(-0.75 + LANE)
    dense, facing = sample(shaped.points, 0.05)
    assert clearance(dense).min() >= RADIUS
    assert min(clearance(dense + AXIAL * facing).min(), clearance(dense - AXIAL * facing).min()) >= RADIUS - 5e-3


def test_without_a_side_the_approach_lies_in_line_with_the_stretch() -> None:
    lanes, clearance = _lanes(1.5, prefer=0.0, lane=0.0)
    shaped = approach(lanes, lanes.width < 1.8, clearance, length=4.5, waits=True, **FIT)
    waiting = shaped.points[shaped.waits]
    assert len(waiting) >= 2
    assert waiting[:, 1] == pytest.approx(0.0, abs=1e-6)
    assert waiting[:, 0].max() < 0.0


def test_thin_keeps_the_points_where_something_changes() -> None:
    cast, clearance = _world(CORRIDOR)
    shaped = follow_walls(MIDDLE, cast, clearance, body=BODY, offset=OFFSET, reach=REACH, prefer=1.0, either=True, ramp=1.0)
    less = thin(shaped)
    assert len(less.points) < len(shaped.points)
    assert less.points[0].tolist() == shaped.points[0].tolist()
    assert less.points[-1].tolist() == shaped.points[-1].tolist()
    assert sample(less.points, 0.05)[0][:, 1].max() == pytest.approx(shaped.points[:, 1].max())
    assert less.follows.any() and not less.follows.all()
