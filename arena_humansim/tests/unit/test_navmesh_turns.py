"""Every walker without a `passage` block takes a passage too narrow to pass an oncoming walker by turns (navmesh
planner, `NavMeshPlanner._take_turns`): it holds before the stretch while one comes through the other way or one at the
other end claimed it first, two that met inside sort it out by the one that came less far rolling back, and a passage
wide enough for the two bodies is taken without waiting."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner import LANE_OFFSET_M
from arena_humansim.global_planner.navmesh import TURN_HOLD, TURN_STUCK_S, TURN_YIELD_S, NavMeshPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

WEST, EAST = (-12.0, 0.0), (12.0, 0.0)


def _walls(width: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Two halls of 10 x 10 m joined by a passage 10 m long and `width` wide along x."""
    w = width / 2.0
    outline = [(-15.0, -5.0), (-5.0, -5.0), (-5.0, -w), (5.0, -w), (5.0, -5.0), (15.0, -5.0), (15.0, 5.0), (5.0, 5.0), (5.0, w), (-5.0, w), (-5.0, 5.0), (-15.0, 5.0)]
    return list(zip(outline, outline[1:] + outline[:1], strict=True))


def _planner(width: float) -> NavMeshPlanner:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25, lane_side="left", lane_offset=LANE_OFFSET_M)
    planner.set_walls(_walls(width))
    planner.set_timing(0.05, 8.0)
    return planner


def _walker(agent_factory: Callable[..., BaseAgent], aid: int, x: float, y: float = 0.0, radius: float = 0.25) -> BaseAgent:
    agent = agent_factory(agent_id=aid, x=x, y=y)
    agent.state.pose.theta = 0.0 if x < 0.0 else np.pi
    agent.params = attrs.evolve(agent.params, agent_radius=radius)
    return agent


def _go(targets: dict[int, tuple[float, float]]) -> dict[int, HighLevelCommand]:
    return {aid: HighLevelCommand(agent_id=aid, type=CommandType.NAVIGATE, target_pose=Pose2D(x=x, y=y), desired_velocity=1.2) for aid, (x, y) in targets.items()}


def _hold(planner: NavMeshPlanner, aid: int) -> tuple[float, float, float, float] | None:
    command = planner.passage_commands().get(aid)
    return None if command is None else command.hold


def test_one_waits_before_the_passage_while_one_comes_through_the_other_way(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.0)
    inside = _walker(agent_factory, 1, 0.0)
    waiting = _walker(agent_factory, 2, 8.0)
    planner.compute([inside, waiting], _go({1: EAST, 2: WEST}))
    assert _hold(planner, 1) is None
    hold = _hold(planner, 2)
    assert hold is not None
    assert hold[0] == pytest.approx(5.0 + TURN_HOLD, abs=0.3)
    assert (hold[2], hold[3]) == pytest.approx((-1.0, 0.0), abs=1e-6)

    inside.state.pose.x = 6.5
    planner.compute([inside, waiting], _go({1: EAST, 2: WEST}))
    assert _hold(planner, 2) is None


def test_the_end_that_claimed_the_passage_first_goes_first_with_those_behind_it(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.0)
    first = _walker(agent_factory, 5, -8.0)
    planner.compute([first], _go({5: EAST}))
    other = _walker(agent_factory, 2, 8.0)
    planner.compute([first, other], _go({5: EAST, 2: WEST}))
    behind = _walker(agent_factory, 9, -9.4)
    planner.compute([first, other, behind], _go({5: EAST, 2: WEST, 9: EAST}))
    assert _hold(planner, 5) is None
    assert _hold(planner, 2) is not None
    assert _hold(planner, 9) is None


def test_two_that_met_inside_the_one_that_came_less_far_walks_back(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.0)
    far_in = _walker(agent_factory, 1, 1.0)
    just_in = _walker(agent_factory, 2, 3.5)
    planner.compute([far_in, just_in], _go({1: EAST, 2: WEST}))
    assert _hold(planner, 1) is None
    command = planner.passage_commands()[2]
    assert command.back is True
    assert command.hold is not None and command.hold[0] > 5.0


@pytest.mark.parametrize(("width", "radius", "waits"), [(1.3, 0.25, False), (1.0, 0.25, True), (1.15, 0.325, True), (1.3, 0.325, False)])
def test_a_passage_wide_enough_for_both_bodies_is_taken_without_waiting(width: float, radius: float, waits: bool, agent_factory: Callable[..., BaseAgent]) -> None:
    """Two adults (0.5 m) pass in 1.3 m, not in 1.0 m; an adult and a wheelchair user (0.65 m) not in 1.15 m."""
    planner = _planner(width)
    inside = _walker(agent_factory, 1, 0.0, radius=radius)
    coming = _walker(agent_factory, 2, 8.0)
    planner.compute([inside, coming], _go({1: EAST, 2: WEST}))
    assert (_hold(planner, 2) is not None) is waits


def test_a_walker_that_does_not_navigate_keeps_no_one_waiting(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.0)
    inside = _walker(agent_factory, 1, 0.0)
    coming = _walker(agent_factory, 2, 8.0)
    planner.compute([inside, coming], _go({1: EAST, 2: WEST}))
    assert _hold(planner, 2) is not None
    planner.compute([inside, coming], _go({2: WEST}))
    assert _hold(planner, 2) is None


def test_one_that_stands_inside_for_another_reason_stops_keeping_others_waiting(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.0)
    inside = _walker(agent_factory, 1, 0.0)
    coming = _walker(agent_factory, 2, 8.0)
    ticks = int(TURN_STUCK_S / 0.05) + 2
    held = []
    for _ in range(ticks):
        planner.compute([inside, coming], _go({1: EAST, 2: WEST}))
        held.append(_hold(planner, 2) is not None)
    assert held[0] and not held[-1]


def test_walkers_away_from_narrow_passages_get_no_commands(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(3.0)
    walkers = [_walker(agent_factory, 1, -8.0), _walker(agent_factory, 2, 8.0)]
    planner.compute(walkers, _go({1: EAST, 2: WEST}))
    assert planner.passage_commands() == {}


def _door_planner(width: float) -> NavMeshPlanner:
    """Two halls of 10 x 10 m on either side of a wall without thickness at x = 0 (as the arena-world walls are lines),
    with a door `width` wide in its middle."""
    w = width / 2.0
    outline = [(-10.0, -5.0), (10.0, -5.0), (10.0, 5.0), (-10.0, 5.0)]
    walls = [*zip(outline, outline[1:] + outline[:1], strict=True), ((0.0, -5.0), (0.0, -w)), ((0.0, w), (0.0, 5.0))]
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25, lane_side="left", lane_offset=LANE_OFFSET_M)
    planner.set_walls(walls)
    planner.set_timing(0.05, 8.0)
    return planner


@pytest.mark.parametrize(("start", "goal"), [((-6.0, 0.0), (6.0, 0.0)), ((6.0, 0.0), (-6.0, 0.0)), ((-6.0, -2.0), (6.0, 2.5))])
def test_a_door_in_a_wall_without_thickness_is_a_narrow_stretch(start: tuple[float, float], goal: tuple[float, float]) -> None:
    """No chord through a point beside such a door ends on walls at both ends; its jambs make it as narrow as it is."""
    planner = _door_planner(1.2)
    path = planner.paths([(Pose2D(x=start[0], y=start[1]), Pose2D(x=goal[0], y=goal[1]))])[0]
    way = planner._turn_way(path, 2 * 0.325 + 0.8 + 0.1)
    assert len(way.narrows) == 1
    narrow, width = way.narrows[0]
    assert width == pytest.approx(1.2, abs=0.05)
    mouth = narrow.points[len(narrow.points) // 2]
    assert abs(mouth[0]) < 0.3 and abs(mouth[1]) < 0.6


def test_a_wide_door_and_the_walls_beside_it_are_no_narrow_stretch() -> None:
    planner = _door_planner(3.0)
    path = planner.paths([(Pose2D(x=-6.0, y=0.0), Pose2D(x=6.0, y=0.0))])[0]
    assert planner._turn_way(path, 2 * 0.325 + 0.8 + 0.1).narrows == []
    along = planner.paths([(Pose2D(x=-6.0, y=4.0), Pose2D(x=-6.0, y=-4.0))])[0]
    assert planner._turn_way(along, 2 * 0.325 + 0.8 + 0.1).narrows == []


@pytest.mark.parametrize(("width", "radius", "waits"), [(1.2, 0.325, True), (1.0, 0.25, True), (1.2, 0.25, False)])
def test_one_waits_before_a_door_in_a_wall_without_thickness_while_one_comes_through(width: float, radius: float, waits: bool, agent_factory: Callable[..., BaseAgent]) -> None:
    """A wheelchair user (0.65 m) and an adult do not pass in a 1.2 m door, two adults do not in a 1.0 m one."""
    planner = _door_planner(width)
    inside = _walker(agent_factory, 1, -0.1, radius=radius)
    coming = _walker(agent_factory, 2, 3.0)
    planner.compute([inside, coming], _go({1: (6.0, 0.0), 2: (-6.0, 0.0)}))
    assert _hold(planner, 1) is None
    hold = _hold(planner, 2)
    assert (hold is not None) is waits
    if waits:
        assert hold is not None and hold[0] > 0.5


def test_a_walker_the_crowd_pushed_back_keeps_its_place_in_the_order(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.0)
    early = _walker(agent_factory, 2, 8.0)
    planner.compute([early], _go({2: WEST}))
    stamp = planner._turn_claims[2][2]
    early.state.pose.x, early.state.pose.y = 12.0, 3.0  # pushed back beyond LINE_UP: its route still leads through the passage
    later = _walker(agent_factory, 5, -8.0)
    for _ in range(5):
        planner.compute([early, later], _go({2: WEST, 5: EAST}))
    assert planner._turn_claims[2][2] == stamp and 2 not in planner._turn_held
    early.state.pose.x, early.state.pose.y = 8.0, 0.0
    planner.compute([early, later], _go({2: WEST, 5: EAST}))
    assert _hold(planner, 2) is None
    assert _hold(planner, 5) is not None


def test_the_end_that_has_the_passage_yields_once_the_other_end_waited_its_time(agent_factory: Callable[..., BaseAgent]) -> None:
    """Platoons: the end that came first keeps the passage while its walkers come, at most `TURN_YIELD_S` while the
    other end waits; then the other end gets it."""
    planner = _planner(1.0)
    east = _walker(agent_factory, 1, -7.0)
    planner.compute([east], _go({1: EAST}))
    west = _walker(agent_factory, 2, 8.0)
    held = []
    for _ in range(int(TURN_YIELD_S / 0.05) + 3):
        planner.compute([east, west], _go({1: EAST, 2: WEST}))
        held.append((_hold(planner, 1) is not None, _hold(planner, 2) is not None))
    assert held[0] == (False, True)
    assert held[-1] == (True, False)


def test_the_waiting_walkers_of_one_end_line_up_in_the_order_they_came(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner(1.0)
    inside = _walker(agent_factory, 1, 0.0)
    far = _walker(agent_factory, 3, 9.0)
    planner.compute([inside, far], _go({1: EAST, 3: WEST}))
    near = _walker(agent_factory, 4, 6.5)
    planner.compute([inside, far, near], _go({1: EAST, 3: WEST, 4: WEST}))
    spot_far, spot_near = _hold(planner, 3), _hold(planner, 4)
    assert spot_far is not None and spot_near is not None
    assert spot_far[0] < spot_near[0]  # the one that came first stands nearer the mouth (x = 5)
