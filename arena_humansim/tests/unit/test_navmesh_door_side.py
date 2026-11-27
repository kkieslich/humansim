"""A walker that cannot step aside (a wheelchair user, `along_heading` kinematics) keeps to its lane side through a door
that it and an adult fit side by side (navmesh planner, `NavMeshPlanner._keep_side`), so one coming the other way can
pass; narrower doors, wider doors, other walkers and a crowd without a lane side keep their routes as they were."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import KINEMATICS, SampledLocomotion
from arena_humansim.global_planner import LANE_OFFSET_M
from arena_humansim.global_planner.navmesh import SIDE_GAP, NavMeshPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

RADIUS = 0.325


def _planner(width: float, lane_side: str = "left") -> NavMeshPlanner:
    """Two halls of 10 x 10 m on either side of a wall without thickness at x = 0, a door `width` wide in its middle."""
    w = width / 2.0
    outline = [(-10.0, -5.0), (10.0, -5.0), (10.0, 5.0), (-10.0, 5.0)]
    walls = [*zip(outline, outline[1:] + outline[:1], strict=True), ((0.0, -5.0), (0.0, -w)), ((0.0, w), (0.0, 5.0))]
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25, lane_side=lane_side, lane_offset=LANE_OFFSET_M)
    planner.set_walls(walls)
    planner.set_timing(0.05, 8.0)
    return planner


def _agent(agent_factory: Callable[..., BaseAgent], rigid: bool) -> BaseAgent:
    agent = agent_factory(agent_id=1, x=-6.0, y=0.0)
    agent.state.pose.theta = 0.0
    params = attrs.evolve(agent.params, agent_radius=RADIUS)
    if rigid:
        params = attrs.evolve(params, locomotion=SampledLocomotion(active=True, kinematics=KINEMATICS.index("along_heading")))
    agent.params = params
    return agent


def _y_at_door(planner: NavMeshPlanner, agent: BaseAgent) -> float:
    planner.compute([agent], {1: HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=6.0, y=0.0), desired_velocity=0.9)})
    points = np.array([(w.x, w.y) for w in planner.get_cached_paths()[1]])
    k = int(np.argmax(points[:, 0] >= 0.0))
    a, b = points[k - 1], points[k]
    return float(a[1] + (b[1] - a[1]) * (0.0 - a[0]) / (b[0] - a[0]))


def test_a_wheelchair_user_keeps_to_its_lane_side_through_a_door_it_and_an_adult_fit(agent_factory: Callable[..., BaseAgent]) -> None:
    y = _y_at_door(_planner(1.5), _agent(agent_factory, rigid=True))
    assert y > 0.1  # left of its way east
    assert 1.5 / 2.0 - y >= RADIUS + SIDE_GAP - 0.02  # its body clear of the jamb


@pytest.mark.parametrize(("width", "rigid", "lane_side"), [(1.2, True, "left"), (2.2, True, "left"), (1.5, False, "left"), (1.5, True, "none")])
def test_other_doors_walkers_and_crowds_keep_their_route(width: float, rigid: bool, lane_side: str, agent_factory: Callable[..., BaseAgent]) -> None:
    assert abs(_y_at_door(_planner(width, lane_side), _agent(agent_factory, rigid))) < 0.05
