"""A walker the crowd pushed off its route's floor into a gap narrower than the inflation radius (a niche in a wall,
between a pillar and a wall) walks back out along the route its body fits, then on by its own route. Before, the
planner found no start there, and the walker steered straight at its target into the wall, or stood in the niche, for
good. A target behind a door narrower than the inflation radius stays out of reach."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import Pose2D

# a 20 x 4 m corridor with a 0.6 m wide, 1 m deep niche in its north wall at x 9.7-10.3
WALLS = [
    ((0.0, 0.0), (20.0, 0.0)),
    ((20.0, 0.0), (20.0, 4.0)),
    ((20.0, 4.0), (10.3, 4.0)),
    ((10.3, 4.0), (10.3, 5.0)),
    ((10.3, 5.0), (9.7, 5.0)),
    ((9.7, 5.0), (9.7, 4.0)),
    ((9.7, 4.0), (0.0, 4.0)),
    ((0.0, 4.0), (0.0, 0.0)),
]
TARGET = (18.0, 2.0)
IN_NICHE = (10.0, 4.6)  # 0.3 m from both sides of the niche: a 0.25 m body fits, the 0.38 m inflation radius does not


def _planner() -> NavMeshPlanner:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25)
    planner.set_walls(WALLS)
    return planner


def test_a_walker_pushed_into_a_niche_narrower_than_the_inflation_radius_walks_back_out(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    planner = _planner()
    agent = agent_factory(agent_id=1, x=8.0, y=2.0)
    cmds = commands_factory(agent_ids=[1], target=TARGET)
    planner.compute([agent], cmds)  # its route along the corridor
    agent.state.pose = Pose2D(x=IN_NICHE[0], y=IN_NICHE[1])  # the crowd pushes it into the niche, off its route
    assert planner._router.snap(np.array(IN_NICHE)) is None
    goal = planner.compute([agent], cmds)[1]
    path = planner.get_cached_paths()[1]
    assert (path[-1].x, path[-1].y) == pytest.approx(TARGET)
    assert goal.y < IN_NICHE[1] and abs(goal.x - IN_NICHE[0]) < 0.3  # its next step leads out of the niche, not through the wall
    out = next(w for w in path if w.y <= 4.0 - 0.38)  # back on the floor its own route keeps, within a few steps
    assert np.hypot(out.x - IN_NICHE[0], out.y - IN_NICHE[1]) < 2.0


def test_a_walker_that_never_had_a_route_gets_no_way_through_a_gap_narrower_than_the_inflation_radius(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    planner = _planner()
    agent = agent_factory(agent_id=1, x=IN_NICHE[0], y=IN_NICHE[1])
    planner.compute([agent], commands_factory(agent_ids=[1], target=TARGET))
    assert 1 not in planner.get_cached_paths()


def test_a_walker_on_open_floor_needs_no_body_route(agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    planner = _planner()
    agent = agent_factory(agent_id=1, x=5.0, y=2.0)
    planner.compute([agent], commands_factory(agent_ids=[1], target=TARGET))
    assert 1 in planner.get_cached_paths() and planner._body_routers == {}
    assert planner.body_clearance(0.25) == pytest.approx(0.25)
    assert planner.body_clearance(0.40) is None  # no narrower than the inflation radius: nothing left to try
