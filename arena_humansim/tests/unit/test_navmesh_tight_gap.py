"""A walker whose verified route fails at a gap only a few centimetres wider than two inflation radii still walks to its
target where a route at the radius exists (the route the stair and escalator choice priced), instead of standing.

The walls are those of smu_link_full_v3 around the bibimbap shop front and the up escalator of the T junction: its mouth
is narrowed to 0.79 m (0.03 m more than 2 x 0.38 m) by a wall end at (133.9, 34.32). From the shop front the funnelled
route clips that wall end by a few centimetres, its repair fails and the gap's portal is banned; before this rule the
walker walked to `_nearest_reachable` instead (in the twin crowd: its own position, for good)."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner import LANE_OFFSET_M
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

#: smu_link_full_v3, closed and outside walls touching x 127-142, y 32.5-37.5
WALLS = [
    ((126.913, 33.24), (128.26, 33.14)),
    ((128.26, 33.14), (128.943, 33.193)),
    ((128.943, 33.193), (129.371, 32.956)),
    ((129.371, 32.956), (129.605, 33.05)),
    ((129.605, 33.05), (129.831, 32.975)),
    ((129.831, 32.975), (130.075, 32.791)),
    ((130.075, 32.791), (130.247, 32.49)),
    ((142.024, 33.69), (141.288, 33.85)),
    ((141.288, 33.85), (141.355, 34.01)),
    ((141.355, 34.01), (141.261, 34.13)),
    ((141.261, 34.13), (141.143, 33.97)),
    ((141.143, 33.97), (141.2, 33.97)),
    ((141.2, 33.97), (141.19, 33.77)),
    ((141.19, 33.77), (138.978, 33.81)),
    ((138.978, 33.81), (138.774, 33.768)),
    ((138.774, 33.768), (138.51, 33.44)),
    ((138.51, 33.44), (138.21, 33.39)),
    ((138.21, 33.39), (138.81, 33.39)),
    ((138.81, 33.39), (138.744, 33.23)),
    ((138.744, 33.23), (137.867, 33.181)),
    ((137.867, 33.181), (137.62, 33.186)),
    ((137.62, 33.186), (137.38, 33.386)),
    ((137.38, 33.386), (137.224, 33.23)),
    ((137.224, 33.23), (136.416, 33.23)),
    ((136.416, 33.23), (136.256, 33.39)),
    ((136.256, 33.39), (137.85, 33.39)),
    ((137.85, 33.39), (137.682, 33.53)),
    ((137.682, 33.53), (136.56, 33.49)),
    ((136.56, 33.49), (135.54, 33.61)),
    ((135.54, 33.61), (135.096, 33.51)),
    ((135.096, 33.51), (134.956, 33.65)),
    ((134.956, 33.65), (135.142, 33.656)),
    ((135.142, 33.656), (135.011, 33.722)),
    ((135.011, 33.722), (134.735, 34.09)),
    ((134.735, 34.09), (134.654, 34.02)),
    ((134.654, 34.02), (134.764, 33.91)),
    ((134.764, 33.91), (134.544, 33.91)),
    ((134.544, 33.91), (134.439, 33.805)),
    ((134.439, 33.805), (134.516, 34.02)),
    ((134.516, 34.02), (134.466, 34.24)),
    ((134.466, 34.24), (134.27, 34.24)),
    ((134.27, 34.24), (134.23, 33.91)),
    ((134.23, 33.91), (134.07, 33.91)),
    ((134.07, 33.91), (134.11, 34.09)),
    ((134.11, 34.09), (134.0, 34.13)),
    ((134.0, 34.13), (134.0, 34.284)),
    ((134.0, 34.284), (133.9, 34.324)),
    ((133.9, 34.324), (133.9, 33.8)),
    ((133.9, 33.8), (123.5, 33.8)),
    ((133.06, 36.74), (132.51, 36.69)),
    ((132.51, 36.69), (132.31, 36.84)),
    ((132.31, 36.84), (132.266, 37.15)),
    ((132.266, 37.15), (131.6, 37.15)),
    ((131.6, 37.15), (131.6, 38.62)),
    ((123.65, 36.55), (133.75, 36.55)),
    ((133.75, 36.55), (133.71, 36.59)),
    ((133.71, 36.59), (133.06, 36.74)),
    ((133.6, 35.4), (123.6, 35.4)),
    ((123.6, 35.05), (133.6, 35.05)),
    ((133.6, 35.05), (133.6, 35.4)),
    ((134.98, 37.35), (135.089, 37.456)),
    ((135.089, 37.456), (135.134, 37.709)),
    ((133.81, 37.69), (133.91, 37.49)),
    ((133.91, 37.49), (134.0, 37.52)),
    ((134.217, 37.51), (134.98, 37.35)),
    ((133.25, 33.8), (132.25, 33.8)),
    ((132.25, 33.8), (128.925, 33.8)),
    ((133.25, 35.05), (132.25, 35.05)),
    ((132.25, 35.05), (128.925, 35.05)),
    ((124.6, 35.05), (128.925, 35.05)),
    ((124.6, 33.8), (128.925, 33.8)),
    ((133.25, 35.4), (132.25, 35.4)),
    ((132.25, 35.4), (128.925, 35.4)),
    ((133.25, 36.55), (132.25, 36.55)),
    ((132.25, 36.55), (128.925, 36.55)),
    ((124.6, 36.55), (128.925, 36.55)),
    ((124.6, 35.4), (128.925, 35.4)),
]
#: a box round them, so the floor is bounded
BOX = [((126.0, 32.0), (143.0, 32.0)), ((143.0, 32.0), (143.0, 38.0)), ((143.0, 38.0), (126.0, 38.0)), ((126.0, 38.0), (126.0, 32.0))]
#: the up escalator's walking point short of its seam (`TeleportZones._inner`)
TARGET = (129.925, 34.3)


def _planner() -> NavMeshPlanner:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25, lane_side="left", lane_offset=LANE_OFFSET_M)
    planner.set_walls([*WALLS, *BOX])
    planner.set_timing(0.05, 8.0)
    return planner


@pytest.mark.parametrize("start", [(136.70686, 34.09209), (138.13147, 33.88908)])
def test_a_walker_routes_through_a_gap_barely_wider_than_two_inflation_radii(start: tuple[float, float], agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner()
    agent = agent_factory(agent_id=1, x=start[0], y=start[1])
    agent.params = attrs.evolve(agent.params, agent_radius=0.25)
    cmd = {1: HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=TARGET[0], y=TARGET[1]), desired_velocity=1.2)}
    goal = planner.compute([agent], cmd)[1]
    assert np.hypot(goal.x - start[0], goal.y - start[1]) > 0.5
    path = planner.get_cached_paths()[1]
    assert np.hypot(path[-1].x - TARGET[0], path[-1].y - TARGET[1]) < 0.1


def test_a_wheelchair_is_not_routed_through_a_gap_narrower_than_its_body_needs(agent_factory: Callable[..., BaseAgent]) -> None:
    """The funnel through that gap keeps 0.33 m from the wall end; a wheelchair (0.325 m) needs its radius and
    `LANE_WALL_GAP`, 0.375 m: it has no route there, and the leg is on record (`unroutable`)."""
    planner = _planner()
    start = (136.70686, 34.09209)
    agent = agent_factory(agent_id=1, x=start[0], y=start[1])
    agent.params = attrs.evolve(agent.params, agent_radius=0.325)
    cmd = {1: HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=TARGET[0], y=TARGET[1]), desired_velocity=0.9)}
    planner.compute([agent], cmd)
    path = planner.get_cached_paths().get(1)
    assert path is None or np.hypot(path[-1].x - TARGET[0], path[-1].y - TARGET[1]) > 0.5
    assert [(aid, target) for aid, target, _ in planner.unroutable()] == [(1, TARGET)]
