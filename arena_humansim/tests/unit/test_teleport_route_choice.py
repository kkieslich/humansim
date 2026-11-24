"""Route choice over stairs: a walker whose target walking reaches also rides a stair half beside its partner where that
is quicker (zone_route_choice), timed as the stairs joined in place are, on the planner's congestion weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import attrs
import numpy as np
import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.teleport_zones import CHOICE_INTERVAL, CHOICE_STAGGER, STAIR_SPEED, SWITCH_M, TeleportZones, Zone
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segments

# two rooms joined by a 3 m wide corridor over the top (about 51 m from A's middle to B's) and by a stair whose halves are
# 4 m pockets off the facing walls (about 21 m with the climb)
WALLS: Segments = [
    ((0.0, 0.0), (10.0, 0.0)),
    ((0.0, 0.0), (0.0, 10.0)),
    ((0.0, 10.0), (3.0, 10.0)),
    ((6.0, 10.0), (10.0, 10.0)),
    ((10.0, 0.0), (10.0, 4.0)),
    ((10.0, 6.0), (10.0, 10.0)),
    ((20.0, 0.0), (30.0, 0.0)),
    ((30.0, 0.0), (30.0, 10.0)),
    ((20.0, 10.0), (23.0, 10.0)),
    ((26.0, 10.0), (30.0, 10.0)),
    ((20.0, 0.0), (20.0, 4.0)),
    ((20.0, 6.0), (20.0, 10.0)),
    ((3.0, 10.0), (3.0, 22.0)),
    ((6.0, 10.0), (6.0, 19.0)),
    ((3.0, 22.0), (26.0, 22.0)),
    ((6.0, 19.0), (23.0, 19.0)),
    ((26.0, 22.0), (26.0, 10.0)),
    ((23.0, 19.0), (23.0, 10.0)),
    ((10.0, 4.0), (14.0, 4.0)),
    ((10.0, 6.0), (14.0, 6.0)),
    ((14.0, 4.0), (14.0, 6.0)),
    ((16.0, 4.0), (20.0, 4.0)),
    ((16.0, 6.0), (20.0, 6.0)),
    ((16.0, 4.0), (16.0, 6.0)),
]
IN_B = Pose2D(x=25.0, y=5.0)
IN_A = Pose2D(x=2.0, y=8.0)
SPEED = 1.1
#: the climb of both halves at stair speed, effort 1: (13.55 - 10) + (20 - 16.45) m
RUN_S = 7.1 / STAIR_SPEED


def _box(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _stairs(kind: str = "stairs", belt: float = 0.0) -> list[Zone]:
    a_entry, b_entry = Pose2D(x=10.0, y=5.0), Pose2D(x=20.0, y=5.0, theta=math.pi)
    return [
        Zone(name="st_a", destinations=("st_b",), entry=a_entry, wait=Pose2D(x=9.2, y=5.0), hold=Pose2D(x=13.55, y=5.0), footprint=_box(10.0, 4.0, 14.0, 6.0), kind=kind, belt=belt),
        Zone(name="st_b", destinations=("st_a",), entry=b_entry, wait=Pose2D(x=20.8, y=5.0, theta=math.pi), hold=Pose2D(x=16.45, y=5.0, theta=math.pi), footprint=_box(16.0, 4.0, 20.0, 6.0), kind=kind, belt=-belt),
    ]


def _zones(zones: list[Zone], route_choice: bool = True) -> TeleportZones:
    tz = TeleportZones()
    tz.set_zones(zones, [True] * len(zones))
    tz.route_choice = route_choice
    return tz


@pytest.fixture
def planner() -> NavMeshPlanner:
    p = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6)
    p.set_walls(WALLS)
    return p


def _walker(agent_factory: Callable[..., BaseAgent], planner: object, aid: int, x: float, y: float, climbs: tuple[str, ...] = ("stairs", "escalator")) -> BaseAgent:
    agent = agent_factory(agent_id=aid, x=x, y=y)
    agent.global_planner = planner
    agent.params = attrs.evolve(agent.params, climbs=climbs, desired_velocity=SPEED, stair_effort=1.0)
    agent.state.desired_velocity = SPEED
    return agent


def _go(agents: list[BaseAgent], goal: Pose2D) -> dict[int, HighLevelCommand]:
    return {a.state.agent_id: HighLevelCommand(agent_id=a.state.agent_id, type=CommandType.NAVIGATE, target_pose=goal) for a in agents}


def test_walker_takes_the_stair_over_a_long_detour(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_B), now=0.0)
    transfer = tz.transfers[1]
    assert (transfer.zone, transfer.destination, transfer.ride) == ("st_a", "st_b", True)
    assert out[1].target_pose == tz.zones["st_a"].lane


def test_without_route_choice_a_walkable_target_is_walked_to(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs(), route_choice=False)
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


def test_without_a_time_a_walkable_target_is_walked_to(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B))
    assert 1 not in tz.transfers


def test_wheelchair_walks_the_detour(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    chair, walker = _walker(agent_factory, planner, 1, 5.0, 5.0, climbs=()), _walker(agent_factory, planner, 2, 5.0, 4.0)
    out = tz.steer([chair, walker], _go([chair, walker], IN_B), now=0.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B
    assert tz.transfers[2].zone == "st_a"


def test_type_that_climbs_only_stairs_walks_past_an_escalator(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs("escalator", 0.6))
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0, climbs=("stairs",))
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in tz.transfers


def test_a_target_close_by_is_walked_to(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_A), now=0.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_A


def test_one_way_escalator_is_only_taken_in_its_direction(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    zones = _stairs("escalator", 0.6)
    zones[1] = attrs.evolve(zones[1], destinations=())
    tz = _zones(zones)
    there, back = _walker(agent_factory, planner, 1, 5.0, 5.0), _walker(agent_factory, planner, 2, 25.0, 5.0)
    tz.steer([there], _go([there], IN_B), now=0.0)
    tz.steer([back], _go([back], Pose2D(x=5.0, y=5.0)), now=0.0)
    assert tz.transfers[1].zone == "st_a"
    assert 2 not in tz.transfers


def test_a_belt_running_back_to_its_entry_is_never_ridden(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs("escalator", -0.6))
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in tz.transfers


def test_a_crowded_stair_sends_walkers_round_once_their_choice_is_due(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    assert tz.transfers[1].zone == "st_a"
    xs, ys = np.meshgrid(np.linspace(10.3, 13.7, 10), np.linspace(4.3, 5.7, 8))
    crowd = np.column_stack([xs.ravel(), ys.ravel()])
    planner._set_slowdown(planner._slowdown_at(crowd))
    tz.steer([walker], cmds, now=1.0)
    assert tz.transfers[1].zone == "st_a"  # not due yet
    out = tz.steer([walker], cmds, now=2.0 * CHOICE_INTERVAL)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


@attrs.define(eq=False)
class _Timed:
    """A planner stand-in: one free-space component, straight paths and fixed walking costs per goal [m], per (start,
    goal) where `between` has one."""

    costs: dict[tuple[float, float], float]
    between: dict[tuple[tuple[float, float], tuple[float, float]], float] = attrs.Factory(dict)
    legs: list[tuple[Pose2D, Pose2D]] = attrs.Factory(list)

    def components(self, poses: Sequence[Pose2D]) -> list[frozenset[int]]:
        return [frozenset({0}) for _ in poses]

    def paths(self, legs: Sequence[tuple[Pose2D, Pose2D]]) -> list[list[Pose2D] | None]:
        self.legs.extend(legs)
        return [[start, goal] for start, goal in legs]

    def path_cost(self, path: Sequence[Pose2D]) -> float:
        start, goal = (round(path[0].x, 2), round(path[0].y, 2)), (round(path[-1].x, 2), round(path[-1].y, 2))
        if (start, goal) in self.between:
            return self.between[(start, goal)]
        return self.costs.get(goal, math.hypot(path[-1].x - path[0].x, path[-1].y - path[0].y))


def _priced(walk: float) -> _Timed:
    # board 9 m to st_a's mouth, on 9 m from st_b's mouth to the goal
    return _Timed({(25.0, 5.0): walk, (10.0, 5.0): 9.0, (13.55, 5.0): 3.55, (20.0, 5.0): 3.55}, {((20.0, 5.0), (25.0, 5.0)): 9.0})


def _stair_time() -> float:
    return (9.0 + 9.0) / SPEED + RUN_S


def test_a_walker_keeps_its_choice_unless_another_is_clearly_quicker(agent_factory: Callable[..., BaseAgent]) -> None:
    priced = _priced(100.0)
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    assert tz.transfers[1].zone == "st_a"
    margin = SWITCH_M / SPEED
    stair = _stair_time()
    assert margin > 0.1 * stair
    priced.costs[(25.0, 5.0)] = (stair - (margin - 0.5)) * SPEED
    tz.steer([walker], cmds, now=2.0 * CHOICE_INTERVAL)
    assert tz.transfers[1].zone == "st_a"
    priced.costs[(25.0, 5.0)] = (stair - (margin + 0.5)) * SPEED
    tz.steer([walker], cmds, now=4.0 * CHOICE_INTERVAL)
    assert 1 not in tz.transfers
    priced.costs[(25.0, 5.0)] = (stair + (margin - 0.5)) * SPEED
    tz.steer([walker], cmds, now=6.0 * CHOICE_INTERVAL)
    assert 1 not in tz.transfers


def test_a_walker_on_the_stair_is_not_asked_again(agent_factory: Callable[..., BaseAgent]) -> None:
    priced = _priced(100.0)
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    walker.state.pose = Pose2D(x=12.0, y=5.0)
    priced.costs[(25.0, 5.0)] = 1.0
    out = tz.steer([walker], cmds, now=2.0 * CHOICE_INTERVAL)
    assert tz.transfers[1].zone == "st_a"
    assert out[1].target_pose == tz.zones["st_a"].lane


def test_forget_drops_the_choice(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 in tz._next_choice
    tz.forget(1)
    assert 1 not in tz.transfers
    assert 1 not in tz._next_choice


def test_the_stair_flight_counts_in_the_ride_time(agent_factory: Callable[..., BaseAgent]) -> None:
    walk = (_stair_time() + 10.0) * SPEED
    walker = _walker(agent_factory, _priced(walk), 1, 5.0, 5.0)
    fast = _zones(_stairs())
    fast.steer([walker], _go([walker], IN_B), now=0.0)
    assert fast.transfers[1].zone == "st_a"
    zones = _stairs()
    slow = _zones([attrs.evolve(zones[0], delay=20.0), zones[1]])
    slow.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in slow.transfers


@pytest.mark.parametrize("closing", ["st_a", "st_b"])
def test_a_walker_whose_chosen_half_or_its_partner_closes_is_routed_again_at_once(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner, closing: str) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    assert tz.transfers[1].zone == "st_a"
    tz.open[closing] = False  # set_zone_access
    out = tz.steer([walker], cmds, now=1.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


def test_walkers_that_chose_together_choose_again_spread_over_the_interval(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walkers = [_walker(agent_factory, planner, aid, 5.0, 2.0 + 0.3 * (aid % 10)) for aid in range(1, 41)]
    tz.steer(walkers, _go(walkers, IN_B), now=0.0)
    due = sorted(set(tz._next_choice.values()))
    assert len(due) == CHOICE_STAGGER and min(due) == CHOICE_INTERVAL and max(due) < 2.0 * CHOICE_INTERVAL


def test_a_walker_far_from_every_stair_plans_only_its_own_walk(agent_factory: Callable[..., BaseAgent]) -> None:
    priced = _Timed({(2.0, 8.0): 3.0})
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 2.0, 5.0)
    tz.steer([walker], _go([walker], IN_A), now=0.0)
    assert 1 not in tz.transfers
    assert len(priced.legs) == 1


def test_paths_leave_the_agents_route_memory_alone(planner: NavMeshPlanner) -> None:
    planner._routed = {0: (1.0, 1.0), 1: (2.0, 2.0)}
    paths = planner.paths([(Pose2D(x=5.0, y=5.0), IN_B), (Pose2D(x=5.0, y=5.0), Pose2D(x=8.0, y=8.0))])
    assert all(p is not None for p in paths)
    assert planner._routed == {0: (1.0, 1.0), 1: (2.0, 2.0)}


def test_a_jam_weighs_the_path_cost(planner: NavMeshPlanner) -> None:
    path = [Pose2D(x=1.0, y=5.0), Pose2D(x=9.0, y=5.0)]
    assert planner.path_cost(path) == pytest.approx(8.0)
    xs, ys = np.meshgrid(np.linspace(3.0, 7.0, 10), np.linspace(4.0, 6.0, 6))
    planner._set_slowdown(planner._slowdown_at(np.column_stack([xs.ravel(), ys.ravel()])))
    assert planner.path_cost(path) > 12.0
