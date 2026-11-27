from __future__ import annotations

import inspect

import numpy as np
import pytest

from arena_humansim.core.agents.types import AgentType
from arena_humansim.global_planner import (
    GlobalPlanner,
    _registry,
    simplify_path,
)
from arena_humansim.global_planner.astar import AStarPlanner
from arena_humansim.global_planner.dijkstra import DijkstraPlanner
from arena_humansim.utils.types import Pose2D


def _poses(points: np.ndarray) -> list[Pose2D]:
    return [Pose2D(x=float(p[0]), y=float(p[1]), theta=0.0) for p in points]


def test_simplify_path_short_input_returns_as_is() -> None:
    empty = simplify_path([])
    assert empty == []

    single = _poses(np.array([[0.0, 0.0]]))
    assert simplify_path(single) == single

    pair = _poses(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert simplify_path(pair) == pair


def test_simplify_path_removes_collinear_interior() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    waypoints = _poses(pts)
    result = simplify_path(waypoints, min_area=0.01)
    assert len(result) == 2
    assert result[0] is waypoints[0]
    assert result[-1] is waypoints[-1]


def test_simplify_path_preserves_corner_above_min_area() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    waypoints = _poses(pts)
    result = simplify_path(waypoints, min_area=0.01)
    assert len(result) == 4


def test_simplify_path_mixed_keeps_sharp_drops_collinear() -> None:
    pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 2.0],
            [3.0, 4.0],
            [5.0, 4.0],
        ]
    )
    waypoints = _poses(pts)
    result = simplify_path(waypoints, min_area=0.01)
    xs_ys = [(p.x, p.y) for p in result]
    assert (0.0, 0.0) in xs_ys
    assert (3.0, 0.0) in xs_ys
    assert (3.0, 4.0) in xs_ys
    assert (5.0, 4.0) in xs_ys
    assert len(result) == 4


def test_advance_along_path_progresses_through_waypoints() -> None:
    waypoints = _poses(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))

    before = GlobalPlanner.advance_along_path(Pose2D(x=0.4, y=0.0), waypoints, 0)
    assert before == 0

    past_first = GlobalPlanner.advance_along_path(Pose2D(x=1.5, y=0.0), waypoints, 0)
    assert past_first == 1

    past_many = GlobalPlanner.advance_along_path(Pose2D(x=2.5, y=0.0), waypoints, 0)
    assert past_many == 2

    at_end = GlobalPlanner.advance_along_path(Pose2D(x=100.0, y=0.0), waypoints, 0)
    assert at_end == len(waypoints) - 1


def test_advance_along_path_reaches_a_waypoint_within_arrive_but_never_skips_the_target() -> None:
    """A corner point the wall push keeps a walker 0.1 m short of counts as reached within `arrive`; the target is
    reached only by getting there."""
    waypoints = _poses(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]))
    short = Pose2D(x=0.9, y=-0.05)
    assert GlobalPlanner.advance_along_path(short, waypoints, 0) == 0
    assert GlobalPlanner.advance_along_path(short, waypoints, 0, arrive=0.3) == 1
    near_target = Pose2D(x=1.85, y=1.0)
    assert GlobalPlanner.advance_along_path(near_target, waypoints, 2, arrive=0.3) == 2


def test_advance_along_path_respects_current_idx() -> None:
    waypoints = _poses(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    idx = GlobalPlanner.advance_along_path(Pose2D(x=0.5, y=0.0), waypoints, 1)
    assert idx == 1


def test_register_lists_builtin_planners() -> None:
    available = GlobalPlanner.list_available()
    assert "dijkstra" in available
    assert "astar" in available


def test_register_decorator_adds_new_entry() -> None:
    name = "__fake_planner_for_test__"
    assert name not in GlobalPlanner.list_available()

    try:

        @GlobalPlanner.register(name)
        def _load() -> type[GlobalPlanner]:
            from arena_humansim.global_planner.dijkstra import DijkstraPlanner

            return DijkstraPlanner

        assert name in GlobalPlanner.list_available()
    finally:
        _registry._registry.pop(name, None)

    assert name not in GlobalPlanner.list_available()


@pytest.mark.parametrize("planner_cls", [AStarPlanner, DijkstraPlanner])
def test_default_inflation_radius_covers_default_agent_radius(planner_cls: type) -> None:
    inflation_default = inspect.signature(planner_cls.__init__).parameters["inflation_radius"].default
    agent_radius_default = AgentType(name="default").agent_radius.mean
    assert inflation_default > agent_radius_default, f"{planner_cls.__name__} default inflation_radius={inflation_default} must be > default agent_radius={agent_radius_default}; otherwise planned paths can hug walls closer than a pedestrian can fit"
