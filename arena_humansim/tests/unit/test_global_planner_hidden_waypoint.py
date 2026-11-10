"""An agent pushed off its path, within the replan distance, to where a wall hides its next waypoint plans anew instead
of steering into the wall; one that still sees its next waypoint keeps its path."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import Pose2D, Segments

# two rooms, 6 m deep, joined by a 1.0 m door in the wall at x 6 (y 2.5 to 3.5)
WALL_X, DOOR = 6.0, (2.5, 3.5)
WALLS: Segments = [
    ((0.0, 0.0), (12.0, 0.0)),
    ((12.0, 0.0), (12.0, 6.0)),
    ((12.0, 6.0), (0.0, 6.0)),
    ((0.0, 6.0), (0.0, 0.0)),
    ((WALL_X, 0.0), (WALL_X, DOOR[0])),
    ((WALL_X, DOOR[1]), (WALL_X, 6.0)),
]
START, TARGET = (2.0, 3.0), (10.0, 3.0)


def _through_wall(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether the straight leg a -> b crosses the dividing wall outside the door."""
    if (a[0] - WALL_X) * (b[0] - WALL_X) >= 0.0:
        return False
    y = a[1] + (b[1] - a[1]) * (WALL_X - a[0]) / (b[0] - a[0])
    return not DOOR[0] < y < DOOR[1]


def _deviation(p: tuple[float, float], path: list[Pose2D]) -> float:
    pts = np.array([(w.x, w.y) for w in path])
    q = np.array(p)
    best = np.inf
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        d = b - a
        t = np.clip(np.dot(q - a, d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0)
        best = min(best, float(np.hypot(*(q - (a + t * d)))))
    return best


def _planner_with_path(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> tuple[NavMeshPlanner, BaseAgent, dict[int, Any], list[Pose2D]]:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6)
    planner.set_walls(WALLS)
    agent = agent_factory(agent_id=1, x=START[0], y=START[1])
    cmds = commands_factory(agent_ids=[1], target=TARGET)
    planner.compute([agent], cmds)
    return planner, agent, cmds, planner.get_cached_paths()[1]


def test_an_agent_whose_next_waypoint_a_wall_hides_plans_anew(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    planner, agent, cmds, path = _planner_with_path(agent_factory, commands_factory)
    # pushed (by the crowd) against the wall beside the door: within the replan distance of its path, but the waypoint
    # it heads for lies beyond the wall
    pushed = (WALL_X - 0.25, DOOR[1] + 0.4)
    assert _deviation(pushed, path) < 1.0
    ahead = path[planner.advance_along_path(Pose2D(x=pushed[0], y=pushed[1]), path, 0) + 1]
    assert _through_wall(pushed, (ahead.x, ahead.y))
    agent.state.pose = Pose2D(x=pushed[0], y=pushed[1])
    goal = planner.compute([agent], cmds)[1]
    assert not _through_wall(pushed, (goal.x, goal.y)), f"steers into the wall towards ({goal.x:.2f}, {goal.y:.2f})"
    assert planner.get_cached_paths()[1] is not path


def test_an_agent_that_still_sees_its_next_waypoint_keeps_its_path(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    planner, agent, cmds, path = _planner_with_path(agent_factory, commands_factory)
    nudged = (4.0, 3.6)
    assert 0.3 < _deviation(nudged, path) < 1.0
    agent.state.pose = Pose2D(x=nudged[0], y=nudged[1])
    goal = planner.compute([agent], cmds)[1]
    assert not _through_wall(nudged, (goal.x, goal.y))
    assert planner.get_cached_paths()[1] is path
