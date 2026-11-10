"""Lanes at a bend: walkers going opposite ways round one inner corner keep a walking lane apart. Both routes wrap the
same corner; the walker that has the inside of the bend on its lane side cannot move further in (the wall), so only
the other one moves, by `lane_offset`, and that offset is all the room between the two directions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner import LANE_OFFSET_M
from arena_humansim.global_planner.navmesh import NavMeshPlanner

# an L: a 3 m corridor north (x 0-3, y 0-10) and a 3 m corridor east (x 3-15, y 7-10); the inner corner at (3, 7)
WALLS = [
    ((0.0, 0.0), (3.0, 0.0)),
    ((3.0, 0.0), (3.0, 7.0)),
    ((3.0, 7.0), (15.0, 7.0)),
    ((15.0, 7.0), (15.0, 10.0)),
    ((15.0, 10.0), (0.0, 10.0)),
    ((0.0, 10.0), (0.0, 0.0)),
]
CORNER = np.array([3.0, 7.0])
SOUTH, EAST = (1.5, 1.0), (14.0, 8.5)


def _routes(lane_offset: float, side: str, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> tuple[np.ndarray, np.ndarray]:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, lane_side=side, lane_offset=lane_offset)
    planner.set_walls(WALLS)
    up = agent_factory(agent_id=1, x=SOUTH[0], y=SOUTH[1])  # north, then right round the corner
    down = agent_factory(agent_id=2, x=EAST[0], y=EAST[1])  # west, then left round the corner
    planner.compute([up], commands_factory(agent_ids=[1], target=EAST))
    planner.compute([down], commands_factory(agent_ids=[2], target=SOUTH))
    paths = planner.get_cached_paths()
    return tuple(np.array([(w.x, w.y) for w in paths[aid]]) for aid in (1, 2))  # type: ignore[return-value]


def _distance_at_corner(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart the two routes pass the corner: from each route's point nearest the corner to the other route."""

    def to_line(p: np.ndarray, line: np.ndarray) -> float:
        s, e = line[:-1], line[1:]
        d = e - s
        u = np.clip(((p - s) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
        return float(np.min(np.hypot(*(s + u[:, None] * d - p).T)))

    pa = a[np.argmin(np.hypot(*(a - CORNER).T))]
    pb = b[np.argmin(np.hypot(*(b - CORNER).T))]
    return min(to_line(pa, b), to_line(pb, a))


@pytest.mark.parametrize("side", ["left", "right"])
def test_walkers_going_opposite_ways_round_a_bend_keep_a_walking_lane_apart(side: str, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    up, down = _routes(LANE_OFFSET_M, side, agent_factory, commands_factory)
    assert _distance_at_corner(up, down) >= 0.76 - 0.05


def test_half_a_lane_leaves_opposite_walkers_less_room_at_a_bend_than_two_bodies_take(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    up, down = _routes(LANE_OFFSET_M / 2.0, "left", agent_factory, commands_factory)
    assert _distance_at_corner(up, down) < 2 * 0.25


def test_without_a_lane_side_both_directions_wrap_the_corner_on_one_line(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    up, down = _routes(LANE_OFFSET_M, "none", agent_factory, commands_factory)
    assert _distance_at_corner(up, down) < 0.05

