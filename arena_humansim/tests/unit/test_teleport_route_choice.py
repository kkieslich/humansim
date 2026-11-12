"""Route choice over stairs: a walker whose target walking reaches takes a stair where that costs less (zone_route_choice)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import attrs
import numpy as np
import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.teleport_zones import LANDING_BACKOFF, STAIR_SPEED, SWITCH_M, TeleportZones, Zone
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


def _walker(agent_factory: Callable[..., BaseAgent], planner: object, aid: int, x: float, y: float, climbs: tuple[str, ...] = ("stairs", "escalator"), speed: float = 1.1) -> BaseAgent:
    agent = agent_factory(agent_id=aid, x=x, y=y)
    agent.global_planner = planner
    agent.params = attrs.evolve(agent.params, climbs=climbs)
    agent.state.desired_velocity = speed
    return agent


def _go(agents: list[BaseAgent], goal: Pose2D) -> dict[int, HighLevelCommand]:
    return {a.state.agent_id: HighLevelCommand(agent_id=a.state.agent_id, type=CommandType.NAVIGATE, target_pose=goal) for a in agents}


def test_navmesh_prices_the_detour_and_the_stair_approach(planner: NavMeshPlanner) -> None:
    walk = planner.route_costs([Pose2D(x=5.0, y=5.0)], IN_B)
    assert walk is not None
    assert 45.0 < float(walk[0]) < 65.0
    to_stair = planner.route_costs([Pose2D(x=5.0, y=5.0)], Pose2D(x=13.55, y=5.0))
    assert to_stair is not None
    assert 8.0 < float(to_stair[0]) < 10.5


def test_walker_takes_the_stair_over_a_long_detour(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_B), now=0.0)
    transfer = tz.transfers[1]
    assert (transfer.zone, transfer.destination, transfer.chosen) == ("st_a", "st_b", True)
    assert out[1].target_pose == tz.zones["st_a"].hold


def test_without_route_choice_a_walkable_target_is_walked_to(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs(), route_choice=False)
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


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


def test_a_crowded_stair_sends_walkers_round_after_the_choice_interval(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
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
    out = tz.steer([walker], cmds, now=6.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


@attrs.define(eq=False)
class _Priced:
    """A planner stand-in: one free-space component and fixed walking costs per target."""

    costs: dict[tuple[float, float], float]
    #: (start, target) pairs priced apart from the target's cost
    legs: dict[tuple[tuple[float, float], tuple[float, float]], float] = attrs.Factory(dict)

    def components(self, poses: Sequence[Pose2D]) -> list[frozenset[int]]:
        return [frozenset({0}) for _ in poses]

    def route_costs(self, starts: Sequence[Pose2D], goal: Pose2D) -> np.ndarray:
        g = (round(goal.x, 2), round(goal.y, 2))
        return np.array([self.legs.get(((round(s.x, 2), round(s.y, 2)), g), self.costs[g]) for s in starts])


def _stair_cost(into: float, onward: float, speed: float = 1.1) -> float:
    extra = 2 * 3.55 * (speed / STAIR_SPEED - 1.0)
    return into + onward + extra


def test_a_walker_keeps_its_choice_unless_the_other_option_is_clearly_cheaper(agent_factory: Callable[..., BaseAgent]) -> None:
    landing = (round(16.45 + LANDING_BACKOFF, 2), 5.0)
    priced = _Priced({(25.0, 5.0): 100.0, (13.55, 5.0): 9.0, (16.45, 5.0): 60.0}, {(landing, (25.0, 5.0)): 9.0})
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    assert tz.transfers[1].zone == "st_a"
    stair = _stair_cost(9.0, 9.0)
    priced.costs[(25.0, 5.0)] = stair - (SWITCH_M - 1.0)
    tz.steer([walker], cmds, now=6.0)
    assert tz.transfers[1].zone == "st_a"
    priced.costs[(25.0, 5.0)] = stair - (SWITCH_M + 1.0)
    tz.steer([walker], cmds, now=12.0)
    assert 1 not in tz.transfers
    priced.costs[(25.0, 5.0)] = stair + (SWITCH_M - 1.0)
    tz.steer([walker], cmds, now=18.0)
    assert 1 not in tz.transfers


def test_a_walker_on_the_stair_is_not_asked_again(agent_factory: Callable[..., BaseAgent]) -> None:
    landing = (round(16.45 + LANDING_BACKOFF, 2), 5.0)
    priced = _Priced({(25.0, 5.0): 100.0, (13.55, 5.0): 9.0, (16.45, 5.0): 60.0}, {(landing, (25.0, 5.0)): 9.0})
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    walker.state.pose = Pose2D(x=12.0, y=5.0)
    priced.costs[(25.0, 5.0)] = 1.0
    out = tz.steer([walker], cmds, now=10.0)
    assert tz.transfers[1].zone == "st_a"
    assert out[1].target_pose == tz.zones["st_a"].hold


def test_forget_drops_the_choice(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    tz.forget(1)
    assert 1 not in tz.transfers
    assert 1 not in tz._choices
