"""Walkers that cannot sidestep in corridors (`passage` on the agent type, navmesh planner): a lane on the crowd's side,
entered and left in line; narrow stretches one direction at a time; queueing behind another of the kind. And walkers
that trail walls (`wall_trailing`). A type without the blocks gets the route and the commands it got before."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import Passage, SampledLocomotion, WallTrailing
from arena_humansim.global_planner import LANE_OFFSET_M
from arena_humansim.global_planner.navmesh import LINE_UP, LOOK_AHEAD, LOOK_CUT, LOOK_MIN, LOOK_STEP, NavMeshPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

RADIUS, LENGTH = 0.325, 1.1
PASSAGE = Passage(lane_width=0.915, corridor_width=3.6, two_way_width=1.8, hold_back=1.5, queue_space=1.22)
WEST, EAST = (-12.0, 0.0), (12.0, 0.0)


def _walls(width: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Two halls of 10 x 10 m joined by a passage 10 m long and `width` wide along x."""
    w = width / 2.0
    outline = [(-15.0, -5.0), (-5.0, -5.0), (-5.0, -w), (5.0, -w), (5.0, -5.0), (15.0, -5.0), (15.0, 5.0), (5.0, 5.0), (5.0, w), (-5.0, w), (-5.0, 5.0), (-15.0, 5.0)]
    return list(zip(outline, outline[1:] + outline[:1], strict=True))


def _planner(width: float, side: str = "right") -> NavMeshPlanner:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25, lane_side=side, lane_offset=LANE_OFFSET_M)
    planner.set_walls(_walls(width))
    return planner


def _chair(agent_factory: Callable[..., BaseAgent], aid: int, x: float, y: float, theta: float = 0.0, passage: Passage = PASSAGE) -> BaseAgent:
    agent = agent_factory(agent_id=aid, x=x, y=y)
    agent.state.pose.theta = theta
    agent.params = attrs.evolve(agent.params, agent_radius=RADIUS, passage=passage, locomotion=SampledLocomotion(active=True, kinematics=1, footprint_length=LENGTH))
    return agent


def _go(targets: dict[int, tuple[float, float]]) -> dict[int, HighLevelCommand]:
    return {aid: HighLevelCommand(agent_id=aid, type=CommandType.NAVIGATE, target_pose=Pose2D(x=x, y=y), desired_velocity=0.9) for aid, (x, y) in targets.items()}


def _path(planner: NavMeshPlanner, aid: int) -> np.ndarray:
    return np.array([(w.x, w.y) for w in planner.get_cached_paths()[aid]])


def _at(path: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """The path's y at points every 0.1 m along it between x0 and x1."""
    legs = np.diff(path, axis=0)
    pieces = np.maximum(np.ceil(np.hypot(legs[:, 0], legs[:, 1]) / 0.1).astype(int), 1)
    dense = np.vstack([path[i] + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * legs[i] for i, n in enumerate(pieces)] + [path[-1:]])
    return dense[(dense[:, 0] >= x0) & (dense[:, 0] <= x1)][:, 1]


@pytest.mark.parametrize(("side", "sign"), [("right", -1.0), ("left", 1.0)])
def test_a_wheelchair_keeps_a_lane_on_the_crowd_s_side_and_enters_and_leaves_it_in_line(side: str, sign: float, agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(2.3, side)
    chair = _chair(agent_factory, 1, -12.0, 3.0)
    planner.compute([chair], _go({1: EAST}))
    lane = sign * (1.15 - PASSAGE.lane_width / 2.0)
    assert _at(_path(planner, 1), -5.0 - LINE_UP + 1.0, 5.0 + LINE_UP - 1.0) == pytest.approx(lane, abs=5e-3)
    assert planner.passage_commands()[1].lane is False
    chair.state.pose.x, chair.state.pose.y = 0.0, lane
    planner.compute([chair], _go({1: EAST}))
    assert planner.passage_commands()[1].lane is True


def test_without_a_lane_side_there_is_no_lane(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(2.3, "none")
    planner.compute([_chair(agent_factory, 1, -12.0, 3.0)], _go({1: EAST}))
    assert np.abs(_at(_path(planner, 1), -4.0, 4.0)).max() < 0.6


def test_a_type_without_the_blocks_gets_the_route_it_got_before(agent_factory: Callable[..., BaseAgent]) -> None:
    plain = _planner(2.3)
    walker = agent_factory(agent_id=1, x=-12.0, y=3.0)
    walker.params = attrs.evolve(walker.params, agent_radius=RADIUS)
    goals_plain = plain.compute([walker], _go({1: EAST}))
    off = _planner(2.3)
    goals_off = off.compute([_chair(agent_factory, 1, -12.0, 3.0, passage=Passage())], _go({1: EAST}))
    assert np.array_equal(_path(plain, 1), _path(off, 1))
    assert goals_plain == goals_off
    assert off.passage_commands() == {}
    assert plain.passage_commands() == {}


def test_a_wheelchair_waits_beside_the_mouth_while_one_comes_through_the_other_way(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.5)
    lane = -(0.75 - PASSAGE.lane_width / 2.0)
    inside = _chair(agent_factory, 1, 0.0, lane)
    waiting = _chair(agent_factory, 2, 8.0, 1.5, theta=np.pi)
    planner.compute([inside, waiting], _go({1: EAST, 2: WEST}))
    commands = planner.passage_commands()
    assert commands[1].hold is None
    hold = commands[2].hold
    assert hold is not None
    assert hold[0] > 5.0 + 1.0
    assert hold[1] >= 0.75 + RADIUS
    assert (hold[2], hold[3]) == pytest.approx((-1.0, 0.0), abs=1e-6)

    inside.state.pose.x = 6.5
    planner.compute([inside, waiting], _go({1: EAST, 2: WEST}))
    assert planner.passage_commands()[2].hold is None


def test_the_end_that_claimed_the_stretch_first_goes_first_with_those_behind_it(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.5)
    first = _chair(agent_factory, 5, -8.0, -1.2)
    planner.compute([first], _go({5: EAST}))
    other = _chair(agent_factory, 2, 8.0, 1.2, theta=np.pi)
    planner.compute([first, other], _go({5: EAST, 2: WEST}))
    behind = _chair(agent_factory, 9, -9.4, -1.2)
    planner.compute([first, other, behind], _go({5: EAST, 2: WEST, 9: EAST}))
    commands = planner.passage_commands()
    assert commands[5].hold is None
    assert commands[2].hold is not None
    assert commands[9].hold is None or commands[9].hold[0] < -8.0


def test_two_that_met_inside_the_one_that_came_less_far_rolls_back(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.5)
    lane = 0.75 - PASSAGE.lane_width / 2.0
    far_in = _chair(agent_factory, 1, 1.0, -lane)
    just_in = _chair(agent_factory, 2, 3.5, lane, theta=np.pi)
    planner.compute([far_in, just_in], _go({1: EAST, 2: WEST}))
    commands = planner.passage_commands()
    assert commands[1].hold is None
    assert commands[2].back is True
    assert commands[2].hold is not None and commands[2].hold[0] > 5.0


def test_a_passage_wide_enough_for_two_is_taken_without_waiting(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(2.3)
    lane = 1.15 - PASSAGE.lane_width / 2.0
    east = _chair(agent_factory, 1, 0.0, -lane)
    west = _chair(agent_factory, 2, 8.0, 1.2, theta=np.pi)
    planner.compute([east, west], _go({1: EAST, 2: WEST}))
    assert all(command.hold is None for command in planner.passage_commands().values())


def test_a_wheelchair_queues_behind_one_on_its_route_but_not_behind_one_coming_towards_it(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(2.3)
    lane = -(1.15 - PASSAGE.lane_width / 2.0)
    leader = _chair(agent_factory, 1, 0.0, lane)
    follower = _chair(agent_factory, 2, -2.5, lane)
    planner.compute([leader, follower], _go({1: EAST, 2: EAST}))
    commands = planner.passage_commands()
    assert commands[1].hold is None
    hold = commands[2].hold
    assert hold is not None
    assert hold[0] == pytest.approx(-PASSAGE.queue_space, abs=0.15)
    assert hold[1] == pytest.approx(lane, abs=0.01)

    oncoming = _chair(agent_factory, 3, 0.0, lane, theta=np.pi)
    other = _planner(2.3)
    other.compute([oncoming, follower], _go({3: WEST, 2: EAST}))
    assert other.passage_commands()[2].hold is None


def test_the_steering_point_lies_ahead_on_the_route_and_does_not_cut_a_corner(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(2.3)
    lane = -(1.15 - PASSAGE.lane_width / 2.0)
    chair = _chair(agent_factory, 1, 0.0, lane)
    goals = planner.compute([chair], _go({1: EAST}))
    assert (goals[1].x, goals[1].y) == pytest.approx((1.5, lane), abs=5e-3)
    way = planner._ways[1]
    for s in np.arange(0.0, float(way.at[-1]), 0.3):
        x, y = way.point(float(s))
        gx, gy = way.look(float(s), x, y)
        inner = way.points[(way.at > s) & (way.at < s + LOOK_AHEAD)]
        chord = np.hypot(gx - x, gy - y)
        assert chord <= LOOK_AHEAD + 1e-9
        if s + LOOK_MIN < way.at[-1]:
            assert chord >= LOOK_MIN - 1e-9
        if len(inner) and chord > LOOK_MIN + LOOK_STEP:
            off = np.abs((inner[:, 0] - x) * (gy - y) - (inner[:, 1] - y) * (gx - x)) / chord
            ahead = ((inner[:, 0] - x) * (gx - x) + (inner[:, 1] - y) * (gy - y)) / chord
            assert off[(ahead > 0.0) & (ahead < chord)].max(initial=0.0) <= LOOK_CUT + 1e-9


def test_a_walker_that_trails_walls_walks_along_the_passage_wall_and_one_that_does_not_keeps_its_route(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(2.7)
    cane = agent_factory(agent_id=1, x=-12.0, y=3.0)
    cane.params = attrs.evolve(cane.params, wall_trailing=WallTrailing(offset=0.42, reach=4.5))
    adult = agent_factory(agent_id=2, x=-12.0, y=3.0)
    planner.compute([cane, adult], _go({1: EAST, 2: EAST}))
    trailed = _at(_path(planner, 1), -4.0, 4.0)
    assert np.abs(np.abs(trailed) - (1.35 - 0.42)).max() < 1e-6
    assert np.abs(_at(_path(planner, 2), -4.0, 4.0)).max() < 1.35 - 0.5
    assert planner.passage_commands() == {}

    alone = _planner(2.7)
    alone.compute([agent_factory(agent_id=2, x=-12.0, y=3.0)], _go({2: EAST}))
    assert np.array_equal(_path(alone, 2), _path(planner, 2))
