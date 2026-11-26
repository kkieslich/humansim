from __future__ import annotations

import numpy as np
import pytest

from arena_humansim.utils.mesh import Mesh
from arena_humansim.utils.router import SNAP_MARGIN, VERIFY_EPS, Router
from arena_humansim.utils.types import Segments

from ._helpers import two_rooms, wall_clearance

_R = 0.3
_REACH = 1.0
_CLOSED: Segments = [*two_rooms(2.0)[:4], ((6.0, 0.0), (6.0, 6.0))]
_CLOSET: Segments = [((0.0, 0.0), (3.0, 0.0)), ((3.0, 0.0), (3.0, 0.4)), ((3.0, 0.4), (0.0, 0.4)), ((0.0, 0.4), (0.0, 0.0))]


def _router(walls: Segments, comfort: float = _R) -> Router:
    return Router(Mesh.build(np.array(walls).reshape(-1, 4), _REACH), _R, comfort)


def _samples(poly: np.ndarray) -> np.ndarray:
    return np.vstack([np.linspace(a, b, 200) for a, b in zip(poly[:-1], poly[1:], strict=True)])


def _crossing_y(poly: np.ndarray, x: float) -> float:
    for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:], strict=True):
        if (x0 - x) * (x1 - x) <= 0 and x0 != x1:
            return float(y0 + (x - x0) / (x1 - x0) * (y1 - y0))
    raise AssertionError("path does not cross x")


def _assert_route(poly: np.ndarray, start: tuple[float, float], goal: tuple[float, float], walls: Segments) -> None:
    assert poly[0].tolist() == list(start)
    assert poly[-1].tolist() == list(goal)
    assert wall_clearance(_samples(poly), walls).min() >= _R - VERIFY_EPS - 1e-6


def test_radius_above_the_mesh_cap_is_refused() -> None:
    mesh = Mesh.build(np.array(two_rooms(2.0)).reshape(-1, 4), _REACH)
    with pytest.raises(ValueError, match="width cap"):
        Router(mesh, 0.6, 0.6)


def test_line_of_sight_query_returns_the_two_terminals() -> None:
    route = _router(two_rooms(2.0)).plan(np.array([2.0, 3.0]), np.array([4.0, 3.0]))
    assert route is not None
    assert route[0].tolist() == [[2.0, 3.0], [4.0, 3.0]]
    assert len(route[1]) == 0


def test_route_through_a_door_keeps_the_radius_from_every_wall() -> None:
    walls = two_rooms(2.0)
    route = _router(walls).plan(np.array([3.0, 1.0]), np.array([9.0, 1.0]))
    assert route is not None
    poly, channel = route
    assert len(channel) > 0
    _assert_route(poly, (3.0, 1.0), (9.0, 1.0), walls)
    assert 2.0 + _R - VERIFY_EPS <= _crossing_y(poly, 6.0) <= 4.0 - _R + VERIFY_EPS


def test_door_narrower_than_the_body_has_no_route() -> None:
    assert _router(two_rooms(0.4)).plan(np.array([3.0, 3.0]), np.array([9.0, 3.0])) is None


def test_terminal_inside_the_inflation_band_has_no_route() -> None:
    router = _router(two_rooms(2.0))
    assert router.plan(np.array([0.1, 3.0]), np.array([3.0, 3.0])) is None
    assert router.plan(np.array([3.0, 3.0]), np.array([0.1, 3.0])) is None


def test_plan_many_routes_each_start_to_the_shared_goal() -> None:
    walls = two_rooms(2.0)
    goal = (9.0, 3.0)
    starts = [(3.0, 1.0), (3.0, 3.0), (1.0, 5.0), (0.1, 3.0)]
    routes = _router(walls).plan_many(np.array(starts), np.array(goal))
    assert routes[3] is None
    for start, route in zip(starts[:3], routes[:3], strict=True):
        assert route is not None
        _assert_route(route[0], start, goal, walls)


def test_blocking_a_door_edge_closes_it_until_released() -> None:
    router = _router(two_rooms(2.0))
    start, goal = np.array([3.0, 3.0]), np.array([9.0, 3.0])
    door = router.mesh.free_edges(np.array([[6.0, 2.0]]), np.array([[6.0, 4.0]]))
    assert door[0] >= 0
    router.set_blocked(door)
    assert router.plan(start, goal) is None
    router.set_blocked(np.zeros(0, np.int64))
    assert router.plan(start, goal) is not None


def test_clearance_is_exact_near_walls_and_inf_beyond_the_comfort_radius() -> None:
    router = _router(two_rooms(2.0), comfort=0.5)
    got = router.clearance(np.array([[3.0, 0.2], [3.0, 3.0]]))
    assert got[0] == pytest.approx(0.2)
    assert got[1] == np.inf


def test_line_of_sight_is_per_segment() -> None:
    router = _router(two_rooms(2.0))
    a = np.array([[3.0, 3.0], [3.0, 1.0]])
    b = np.array([[9.0, 3.0], [9.0, 1.0]])
    assert router.line_of_sight(a, b).tolist() == [True, False]


def test_snap_leaves_a_clear_point_alone() -> None:
    p = np.array([3.0, 3.0])
    q = _router(two_rooms(2.0)).snap(p)
    assert q is not None
    assert q.tolist() == p.tolist()


def test_snap_pushes_a_point_past_the_inflation_band() -> None:
    q = _router(two_rooms(2.0)).snap(np.array([3.0, 0.1]))
    assert q is not None
    assert q == pytest.approx((3.0, _R + SNAP_MARGIN))


def test_snap_resolves_a_corner_through_both_walls() -> None:
    walls = two_rooms(2.0)
    q = _router(walls).snap(np.array([0.1, 0.1]))
    assert q is not None
    assert wall_clearance(q, walls) >= _R


def test_snap_fails_in_a_gap_narrower_than_the_body() -> None:
    assert _router(_CLOSET).snap(np.array([1.5, 0.2])) is None


def test_nearest_reachable_stops_at_the_wall_of_a_closed_room() -> None:
    router = _router(_CLOSED)
    start = np.array([3.0, 3.0])
    q = router.nearest_reachable(start, np.array([9.0, 3.0]))
    assert q is not None
    assert q[0] == pytest.approx(6.0 - _R - SNAP_MARGIN, abs=0.02)
    assert abs(q[1] - 3.0) < 0.5
    assert router.plan(start, q) is not None


def test_nearest_reachable_of_a_blocked_start_is_none() -> None:
    assert _router(_CLOSED).nearest_reachable(np.array([0.1, 3.0]), np.array([9.0, 3.0])) is None


def test_components_tell_joined_rooms_from_separate_ones() -> None:
    pts = np.array([[3.0, 3.0], [9.0, 3.0]])
    joined = _router(two_rooms(2.0)).components(pts)
    assert joined[0] & joined[1]
    closed = _router(_CLOSED).components(pts)
    assert closed[0]
    assert closed[1]
    assert not closed[0] & closed[1]


def test_components_snap_a_point_in_the_band_and_drop_one_off_the_mesh() -> None:
    near, inside, off = _router(two_rooms(2.0)).components(np.array([[3.0, 0.1], [3.0, 3.0], [50.0, 50.0]]))
    assert near & inside
    assert off == frozenset()


def test_a_congestion_update_weighs_the_arcs_as_a_full_rebuild_does() -> None:
    """Only the arcs whose weight changed are written on an update; the A* rows and the tree weights are those a router
    built from scratch with the same congestion has, update after update, back to plain lengths too."""
    router = _router(two_rooms(2.0))
    rng = np.random.default_rng(2)
    E = len(router.mesh.ea)
    for slow in (np.maximum(1.0, rng.normal(1.5, 0.6, E)), np.where(rng.random(E) < 0.1, 3.0, 1.0), None, np.ones(E)):
        router.set_congestion(slow)
        fresh = _router(two_rooms(2.0))
        fresh._slow = None if slow is None else np.maximum(np.asarray(slow, dtype=np.float64), 1.0)
        fresh._topology = None
        fresh._graph()
        assert router._cst == fresh._cst
        assert router._data.tobytes() == fresh._data.tobytes()
        start, goal = np.array([2.0, 1.0]), np.array([10.0, 5.0])
        a, b = router.plan(start, goal), fresh.plan(start, goal)
        assert (a is None) == (b is None)
        assert a is None or b is None or a[0].tobytes() == b[0].tobytes()
