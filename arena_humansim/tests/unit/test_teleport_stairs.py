from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.core.teleport_zones import FLIGHT_WAIT_GAP, LANDING_BACKOFF, LANDING_STEP, STAIR_SPEED, Phase, TeleportZones, Transfer, Zone
from arena_humansim.global_planner.dijkstra import DijkstraPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segments

GOAL_B = Pose2D(x=27.0, y=2.0)
GOAL_A = Pose2D(x=2.0, y=2.0)
HALF_PI = math.pi / 2.0


def _room(x0: float) -> Segments:
    corners = [(x0, 0.0), (x0 + 10.0, 0.0), (x0 + 10.0, 6.0), (x0, 6.0)]
    return list(zip(corners, corners[1:] + corners[:1], strict=True))


WALLS: Segments = [
    *_room(0.0),
    *_room(20.0),
    ((5.0, 4.4), (10.0, 4.4)),
    ((5.0, 1.6), (10.0, 1.6)),
    ((22.0, 0.0), (22.0, 3.5)),
    ((23.6, 0.0), (23.6, 3.5)),
    ((25.0, 4.4), (30.0, 4.4)),
]


def _box(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _stair(name: str, destinations: tuple[str, ...], entry: Pose2D, hold: Pose2D, footprint: tuple[tuple[float, float], ...], kind: str = "stairs", belt: float = 0.0) -> Zone:
    wait = Pose2D(x=entry.x - 0.8 * math.cos(entry.theta), y=entry.y - 0.8 * math.sin(entry.theta), theta=entry.theta)
    return Zone(name=name, destinations=destinations, entry=entry, wait=wait, hold=hold, footprint=footprint, kind=kind, belt=belt)


def _up(kind: str = "stairs", belt: float = 0.0, destinations: tuple[str, ...] = ("up_b",)) -> Zone:
    return _stair("up", destinations, Pose2D(x=5.0, y=5.2), Pose2D(x=9.55, y=5.2), _box(5.0, 4.4, 10.0, 6.0), kind, belt)


def _up_b(kind: str = "stairs", belt: float = 0.0, destinations: tuple[str, ...] = ("up",)) -> Zone:
    return _stair("up_b", destinations, Pose2D(x=22.8, y=3.5, theta=-HALF_PI), Pose2D(x=22.8, y=0.45, theta=-HALF_PI), _box(22.0, 0.0, 23.6, 3.5), kind, belt)


def _low() -> list[Zone]:
    return [
        _stair("low", ("low_b",), Pose2D(x=5.0, y=0.8), Pose2D(x=9.55, y=0.8), _box(5.0, 0.0, 10.0, 1.6)),
        _stair("low_b", ("low",), Pose2D(x=25.0, y=5.2), Pose2D(x=29.55, y=5.2), _box(25.0, 4.4, 30.0, 6.0)),
    ]


def _lift() -> list[Zone]:
    return [
        Zone(name="lift", destinations=("lift_b",), entry=Pose2D(x=1.4, y=3.0, theta=math.pi), wait=Pose2D(x=2.2, y=3.0), hold=Pose2D(x=0.8, y=3.0), footprint=_box(0.2, 2.4, 1.4, 3.6)),
        Zone(name="lift_b", destinations=("lift",), entry=Pose2D(x=28.6, y=3.0), wait=Pose2D(x=27.8, y=3.0), hold=Pose2D(x=29.2, y=3.0), footprint=_box(28.6, 2.4, 29.8, 3.6)),
    ]


def _zones(zones: list[Zone], closed: tuple[str, ...] = ()) -> TeleportZones:
    tz = TeleportZones()
    tz.set_zones(zones, [z.name not in closed for z in zones])
    return tz


@pytest.fixture(scope="module")
def planner() -> DijkstraPlanner:
    p = DijkstraPlanner(resolution=0.1)
    p.set_walls(WALLS)
    return p


def _walker(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner, aid: int, x: float, y: float, climbs: tuple[str, ...] = ("stairs", "escalator"), kind: int = 0) -> BaseAgent:
    agent = agent_factory(agent_id=aid, x=x, y=y)
    agent.global_planner = planner
    agent.params = attrs.evolve(agent.params, climbs=climbs)
    agent.state.kind = kind
    return agent


def _go(agents: list[BaseAgent], goal: Pose2D) -> dict[int, HighLevelCommand]:
    return {a.state.agent_id: HighLevelCommand(agent_id=a.state.agent_id, type=CommandType.NAVIGATE, target_pose=goal) for a in agents}


def _pool(agent_factory: Callable[..., BaseAgent], *rows: tuple[int, float, float]) -> AgentPool:
    pool = AgentPool(capacity=8)
    for aid, x, y in rows:
        i = pool.add_agent(agent_factory(agent_id=aid, x=x, y=y))
        pool.policy_idx[i] = 0
    return pool


def test_route_takes_the_stair_whose_entry_is_nearest_among_equal_teleport_counts(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up(), _up_b(), *_low()])
    top, bottom = _walker(agent_factory, planner, 1, 4.0, 5.0), _walker(agent_factory, planner, 2, 4.0, 1.0)
    out = tz.steer([top, bottom], _go([top, bottom], GOAL_B))
    assert (tz.transfers[1].zone, tz.transfers[1].destination) == ("up", "up_b")
    assert (tz.transfers[2].zone, tz.transfers[2].destination) == ("low", "low_b")
    assert out[1].target_pose == tz.zones["up"].lane
    assert out[2].target_pose == tz.zones["low"].lane


def test_route_keeps_its_stair_while_another_becomes_nearer(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up(), _up_b(), *_low()])
    walker = _walker(agent_factory, planner, 1, 4.0, 5.0)
    tz.steer([walker], _go([walker], GOAL_B))
    walker.state.pose = Pose2D(x=4.0, y=1.0)
    tz.steer([walker], _go([walker], GOAL_B))
    assert tz.transfers[1].zone == "up"


def test_type_that_climbs_nothing_takes_the_cabin_over_a_nearer_stair(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up(), _up_b(), *_lift()])
    chair = _walker(agent_factory, planner, 1, 4.0, 5.0, climbs=())
    adult = _walker(agent_factory, planner, 2, 4.0, 5.0)
    out = tz.steer([chair, adult], _go([chair, adult], GOAL_B))
    assert tz.transfers[1].zone == "lift"
    assert out[1].target_pose == tz.zones["lift"].wait
    assert tz.transfers[2].zone == "up"


def test_type_that_climbs_only_stairs_is_not_routed_onto_an_escalator(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ()), *_low()])
    walker = _walker(agent_factory, planner, 1, 4.0, 5.0, climbs=("stairs",))
    tz.steer([walker], _go([walker], GOAL_B))
    assert tz.transfers[1].zone == "low"


def test_type_that_climbs_nothing_without_a_cabin_keeps_its_command(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up(), _up_b()])
    chair = _walker(agent_factory, planner, 1, 4.0, 5.0, climbs=())
    cmds = _go([chair], GOAL_B)
    assert tz.steer([chair], cmds)[1] == cmds[1]
    assert not tz.transfers


def test_closed_boarding_half_is_skipped_but_a_pedestrian_on_it_carries_on(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up("escalator"), _up_b("escalator", destinations=()), *_low()], closed=("up",))
    outside = _walker(agent_factory, planner, 1, 4.0, 5.0)
    aboard = _walker(agent_factory, planner, 2, 7.0, 5.2)
    tz.transfers[2] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    out = tz.steer([outside, aboard], _go([outside, aboard], GOAL_B))
    assert tz.transfers[1].zone == "low"
    assert tz.transfers[2].zone == "up"
    assert out[2].target_pose == tz.zones["up"].lane


def test_exit_only_half_is_never_routed_into(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up("escalator", -0.5, ()), _up_b("escalator", 0.5)])
    below, above = _walker(agent_factory, planner, 1, 4.0, 5.0), _walker(agent_factory, planner, 2, 25.0, 2.0)
    cmds = {**_go([below], GOAL_B), **_go([above], GOAL_A)}
    out = tz.steer([below, above], cmds)
    assert 1 not in tz.transfers
    assert out[1] == cmds[1]
    assert (tz.transfers[2].zone, tz.transfers[2].destination) == ("up_b", "up")


def test_robot_is_never_routed_over_a_stair(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up(), _up_b()])
    robot = _walker(agent_factory, planner, -1, 4.0, 5.0, kind=KIND_ROBOT)
    cmds = _go([robot], GOAL_B)
    assert tz.steer([robot], cmds)[-1] == cmds[-1]
    assert not tz.transfers


def test_running_belt_carries_riders_at_belt_speed_standing_and_back_on_the_exit_half(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.4, ())])
    pool = _pool(agent_factory, (1, 6.0, 5.2), (2, 22.8, 2.0), (3, 3.0, 3.0))
    pool.vel[:3] = (1.0, 1.0)
    pool.desired_vel[:3] = 1.2
    tz.carry(pool)
    assert np.allclose(pool.vel[0], (0.5, 0.0))
    assert np.allclose(pool.vel[1], (0.0, 0.4))
    assert np.allclose(pool.vel[2], (1.0, 1.0))
    assert (pool.desired_vel[:3] == (0.0, 0.0, 1.2)).all()
    assert tz.riders == {1, 2}
    assert tz.standing(pool).tolist() == [True, True, False]


def test_stairs_cap_walking_speed_keeping_its_direction(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 6.0, 5.2), (2, 7.0, 5.2))
    pool.vel[0] = (1.2, 0.9)
    pool.vel[1] = (0.3, 0.2)
    tz.carry(pool)
    assert math.hypot(*pool.vel[0]) == pytest.approx(STAIR_SPEED)
    assert pool.vel[0, 1] / pool.vel[0, 0] == pytest.approx(0.75)
    assert np.allclose(pool.vel[1], (0.3, 0.2))
    assert not tz.riders


def test_stopped_escalator_walks_like_stairs(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up("escalator"), _up_b("escalator", destinations=())], closed=("up",))
    pool = _pool(agent_factory, (1, 6.0, 5.2))
    pool.vel[0] = (1.5, 0.0)
    tz.carry(pool)
    assert np.allclose(pool.vel[0], (STAIR_SPEED, 0.0))
    assert not tz.riders


def test_crossing_the_seam_lands_mirrored_past_the_partner_seam_keeping_side_and_relative_heading(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    pool.theta[0] = 0.1
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    [(aid, x, y, theta)] = tz.hold(pool, 0, 1000, 0.05, 0)
    assert aid == 1
    assert (x, y) == pytest.approx((22.8 - 0.2, 0.45 + LANDING_BACKOFF))
    assert theta == pytest.approx(HALF_PI + 0.1)


def test_landing_offset_outside_the_partner_footprint_falls_back_to_its_center_line(agent_factory: Callable[..., BaseAgent]) -> None:
    narrow = _stair("up_b", ("up",), Pose2D(x=22.8, y=3.5, theta=-HALF_PI), Pose2D(x=22.8, y=0.45, theta=-HALF_PI), _box(22.5, 0.0, 23.1, 3.5))
    tz = _zones([_up(), narrow])
    pool = _pool(agent_factory, (1, 9.6, 5.8))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    [(_, x, y, _)] = tz.hold(pool, 0, 1000, 0.05, 0)
    assert (x, y) == pytest.approx((22.8, 0.45 + LANDING_BACKOFF))


def test_pedestrian_short_of_the_seam_trigger_is_not_teleported(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.4, 5.2))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    assert tz.hold(pool, 0, 1000, 0.05, 0) == []


def test_landing_on_an_occupied_spot_holds_on_the_trigger_line_until_it_clears(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ())])
    pool = _pool(agent_factory, (1, 9.9, 5.3), (2, 22.8, 0.7))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    pool.vel[0] = (0.5, 0.0)
    pool.desired_vel[0] = 1.2
    assert tz.hold(pool, 0, 1000, 0.05, 0) == []
    assert tuple(pool.pos[0]) == pytest.approx((9.55, 5.3))
    tz.carry(pool)
    assert not pool.vel[0].any()
    assert pool.desired_vel[0] == 0.0
    assert tz.standing(pool)[0]
    pool.pos[1] = (22.8, 2.5)
    [(aid, *_)] = tz.hold(pool, 1, 1000, 0.05, 0)
    assert aid == 1


def test_two_crossings_in_one_tick_do_not_land_on_each_other(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ())])
    pool = _pool(agent_factory, (1, 9.6, 5.2), (2, 9.7, 5.25))
    landed = tz.hold(pool, 0, 1000, 0.05, 0)
    assert [aid for aid, *_ in landed] == [1]


def test_pedestrian_past_the_seam_without_a_transfer_crosses_unless_the_half_can_only_be_left(agent_factory: Callable[..., BaseAgent]) -> None:
    belt = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ())])
    assert [aid for aid, *_ in belt.hold(_pool(agent_factory, (1, 9.6, 5.2)), 0, 1000, 0.05, 0)] == [1]
    assert belt.hold(_pool(agent_factory, (1, 22.8, 0.3)), 0, 1000, 0.05, 0) == []
    stairs = _zones([_up(), _up_b()])
    [(aid, x, y, _)] = stairs.hold(_pool(agent_factory, (1, 22.8, 0.3)), 0, 1000, 0.05, 0)
    assert aid == 1
    assert (x, y) == pytest.approx((9.55 - LANDING_BACKOFF, 5.2))


def test_robot_on_a_stair_is_neither_carried_nor_teleported(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ())])
    pool = _pool(agent_factory, (-1, 9.6, 5.2), (-2, 6.0, 5.2))
    pool.kind[:2] = KIND_ROBOT
    pool.vel[1] = (0.0, 0.3)
    assert tz.hold(pool, 0, 1000, 0.05, 0) == []
    tz.carry(pool)
    assert np.allclose(pool.vel[1], (0.0, 0.3))
    assert not tz.riders


def test_resent_stairs_keep_their_transfers_and_changed_cabins_drop_theirs(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ()), *_lift()])
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    tz.transfers[2] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="lift", destination="lift_b")
    tz.transfers[3] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="lift", destination="lift_b")
    tz.set_zones([_up("escalator"), _up_b("escalator", destinations=()), *_lift()], [False, True, True, True])
    assert sorted(tz.transfers) == [1, 2, 3]
    moved = attrs.evolve(_lift()[0], hold=Pose2D(x=0.9, y=3.0))
    tz.set_zones([_up("escalator"), _up_b("escalator", destinations=()), moved, _lift()[1]], [False, True, True, True])
    assert sorted(tz.transfers) == [1]
    tz.set_zones([], [])
    assert not tz.transfers


def _hall(way_round: bool, *lanes: tuple[float, float, float, float]) -> Segments:
    """One floor of 30 by 32 m split by a wall along x = 15, open only in the lanes (x0, x1, y0, y1) walled along their sides, and over the last 2 m when there is a way round."""
    outer = _box(0.0, 0.0, 30.0, 32.0)
    walls = [*zip(outer, outer[1:] + outer[:1], strict=True)]
    y = 0.0
    for x0, x1, y0, y1 in sorted(lanes, key=lambda lane: lane[2]):
        walls += [((15.0, y), (15.0, y0)), ((x0, y0), (x1, y0)), ((x0, y1), (x1, y1))]
        y = y1
    return [*walls, ((15.0, y), (15.0, 30.0 if way_round else 32.0))]


ESCALATOR, STEPS = (11.0, 19.0, 1.0, 2.2), (11.0, 19.0, 3.0, 4.2)
EAST = Pose2D(x=25.0, y=1.6)
WEST = Pose2D(x=5.0, y=1.6)


def _pair(name: str, lane: tuple[float, float, float, float], seam: float, kind: str = "stairs", belt: float = 0.0, inset: float = -0.05) -> list[Zone]:
    """The two halves of a stair on one floor meeting at x = seam, hold poses as Arena sends them, an escalator boarding on the west half only."""
    x0, x1, y0, y1 = lane
    y = (y0 + y1) / 2.0
    return [
        _stair(f"{name}_b", (f"{name}_t",), Pose2D(x=x0, y=y), Pose2D(x=seam - inset, y=y), _box(x0, y0, seam, y1), kind, belt),
        _stair(f"{name}_t", () if belt else (f"{name}_b",), Pose2D(x=x1, y=y, theta=math.pi), Pose2D(x=seam + inset, y=y, theta=math.pi), _box(seam, y0, x1, y1), kind, -belt),
    ]


def _link(belt: float = 0.5) -> list[Zone]:
    return _pair("link", ESCALATOR, 15.0, "escalator", belt)


def _steps() -> list[Zone]:
    return _pair("steps", STEPS, 15.0)


def _gated(walls: Segments, zones: list[Zone]) -> DijkstraPlanner:
    """A planner over the walls and the seams of the pairs joined in place, as the engine hands them to a global planner."""
    p = DijkstraPlanner(resolution=0.1)
    p.set_walls([*walls, *_zones(zones).gates])
    return p


@pytest.fixture(scope="module")
def floor() -> DijkstraPlanner:
    """The escalator and the stairs through the wall, and a long way round it."""
    return _gated(_hall(True, ESCALATOR, STEPS), [*_link(), *_steps()])


@pytest.fixture(scope="module")
def sealed() -> DijkstraPlanner:
    """The same floor without a way round."""
    return _gated(_hall(False, ESCALATOR, STEPS), [*_link(), *_steps()])


def _length(path: list[Pose2D] | None) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(path, path[1:], strict=False)) if path else math.inf


def test_halves_that_meet_in_place_are_joined_and_their_seam_is_a_gate() -> None:
    tz = _zones([*_link(), *_steps(), _up(), _up_b()])
    assert tz._joined == {"link_b": "link_t", "link_t": "link_b", "steps_b": "steps_t", "steps_t": "steps_b"}
    assert sorted(tz.gates) == [((15.0, 1.0), (15.0, 2.2)), ((15.0, 3.0), (15.0, 4.2))]
    assert _zones([_up(), _up_b(), *_low()]).gates == []


def test_no_plan_crosses_a_joined_pair(floor: DijkstraPlanner, sealed: DijkstraPlanner) -> None:
    around, straight = floor.paths([(WEST, EAST), (Pose2D(x=5.0, y=31.0), Pose2D(x=25.0, y=31.0))])
    assert _length(around) > 2.0 * math.hypot(10.0, 29.0) - 1.0
    assert _length(straight) == pytest.approx(20.0, abs=0.5)
    assert sealed.paths([(WEST, EAST)]) == [None]


def test_a_ride_on_the_walkers_own_floor_is_taken_when_quicker_than_walking_round(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner) -> None:
    tz = _zones(_link())
    walker = _walker(agent_factory, floor, 1, WEST.x, WEST.y)
    out = tz.steer([walker], _go([walker], EAST))
    assert (tz.transfers[1].zone, tz.transfers[1].destination, tz.transfers[1].ride) == ("link_b", "link_t", True)
    half = tz.zones["link_b"]
    assert half.contains(out[1].target_pose.x, out[1].target_pose.y) and half.along(out[1].target_pose.x, out[1].target_pose.y) <= half.length - 0.5
    assert not tz.through


def test_walking_wins_where_the_ride_is_the_longer_way(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner) -> None:
    tz = _zones(_link())
    walker = _walker(agent_factory, floor, 1, 5.0, 31.0)
    goal = Pose2D(x=25.0, y=31.0)
    out = tz.steer([walker], _go([walker], goal))
    assert 1 not in tz.transfers and out[1].target_pose == goal
    assert tz._walks[1][0] == (goal.x, goal.y)


def test_an_escalator_is_ridden_only_from_its_boarding_end(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner, sealed: DijkstraPlanner) -> None:
    for planner in (floor, sealed):
        tz = _zones(_link())
        walker = _walker(agent_factory, planner, 1, EAST.x, EAST.y)
        out = tz.steer([walker], _go([walker], WEST))
        assert 1 not in tz.transfers and out[1].target_pose == WEST


def test_a_walker_down_from_the_exit_of_an_up_escalator_takes_the_stairs(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner) -> None:
    tz = _zones([*_link(), *_steps()])
    walker = _walker(agent_factory, floor, 1, EAST.x, EAST.y)
    tz.steer([walker], _go([walker], WEST))
    assert (tz.transfers[1].zone, tz.transfers[1].ride) == ("steps_t", True)


def test_a_walk_round_a_shut_stair_is_chosen_anew_when_it_opens_or_its_belt_starts(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner) -> None:
    walker = _walker(agent_factory, floor, 1, WEST.x, WEST.y)
    cmds = _go([walker], EAST)
    for reopen in (lambda tz: tz.open.update(link_b=True), lambda tz: tz.set_zones(_link(), [True, True])):
        tz = _zones(_link(0.0), closed=("link_b",))
        assert tz.steer([walker], cmds)[1].target_pose == EAST and 1 not in tz.transfers
        reopen(tz)
        tz.steer([walker], cmds)
        assert (tz.transfers[1].zone, tz.transfers[1].ride) == ("link_b", True)


@pytest.mark.parametrize(("climbs", "zone"), [(("stairs",), "steps_b"), (("escalator",), "link_b"), ((), None)])
def test_a_type_is_sent_only_over_the_kinds_it_climbs_and_plans_round_the_others(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner, sealed: DijkstraPlanner, climbs: tuple[str, ...], zone: str | None) -> None:
    for planner in (floor, sealed):
        tz = _zones([*_link(), *_steps()])
        walker = _walker(agent_factory, planner, 1, WEST.x, WEST.y, climbs=climbs)
        out = tz.steer([walker], _go([walker], EAST))
        assert (tz.transfers[1].zone if 1 in tz.transfers else None) == zone
        assert zone is not None or out[1].target_pose == EAST


@pytest.mark.parametrize(("effort", "zone"), [(1.0, "steps_b"), (2.0, "link_b")])
def test_a_walker_rides_beside_still_stairs_as_its_stair_effort_makes_the_climb_slower(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner, sealed: DijkstraPlanner, effort: float, zone: str) -> None:
    for planner in (floor, sealed):
        tz = _zones([*_link(), *_steps()])
        walker = _walker(agent_factory, planner, 1, 5.0, 2.6)
        walker.params = attrs.evolve(walker.params, stair_effort=effort)
        tz.steer([walker], _go([walker], Pose2D(x=25.0, y=2.6)))
        assert (tz.transfers[1].zone, tz.transfers[1].ride) == (zone, True)


def test_riders_are_sent_into_a_joined_half_on_their_own_line_clear_of_its_sides() -> None:
    tz = _zones([*_pair("wide", (11.0, 19.0, 4.0, 8.0), 15.0), *_steps()])
    wide, narrow = tz.zones["wide_b"], tz.zones["steps_b"]
    assert tz._inner(wide).x == pytest.approx(14.0)
    assert [tz._inner(wide, across).y for across in (-5.0, -1.0, 0.0, 1.2, 5.0)] == pytest.approx([4.5, 5.0, 6.0, 7.2, 7.5])
    assert [tz._inner(narrow, across).y for across in (-0.4, 0.4)] == pytest.approx([3.5, 3.7])


def test_a_rider_keeps_the_line_it_came_on_while_it_walks_in(agent_factory: Callable[..., BaseAgent]) -> None:
    lane = (11.0, 19.0, 4.0, 8.0)
    zones = _pair("wide", lane, 15.0)
    tz = _zones(zones)
    walker = _walker(agent_factory, _gated(_hall(False, lane), zones), 1, 5.0, 7.0)
    cmds = _go([walker], Pose2D(x=25.0, y=7.0))
    first = tz.steer([walker], cmds)[1].target_pose
    assert (first.x, first.y) == pytest.approx((14.0, 7.0))
    walker.state.pose = Pose2D(x=8.0, y=5.0)
    assert tz.steer([walker], cmds)[1].target_pose == first


def test_crossing_a_joined_pair_steers_straight_on_until_the_far_half_is_left(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner) -> None:
    tz = _zones(_link())
    walker = _walker(agent_factory, floor, 1, WEST.x, WEST.y)
    cmds = _go([walker], EAST)
    tz.steer([walker], cmds)
    for x in (13.0, 15.4, 18.5, 14.6):
        walker.state.pose = Pose2D(x=x, y=1.9)
        tz.steer([walker], cmds)
        assert tz.transfers[1].aboard
        assert (tz.through[1].x, tz.through[1].y) == pytest.approx((20.0, 1.9))
    walker.state.pose = Pose2D(x=19.2, y=1.9)
    out = tz.steer([walker], cmds)
    assert 1 not in tz.transfers and 1 not in tz.through and out[1].target_pose == EAST
    assert tz._walks[1][0] == (EAST.x, EAST.y)


def test_a_walker_pushed_back_out_of_the_mouth_boards_again(agent_factory: Callable[..., BaseAgent], floor: DijkstraPlanner) -> None:
    tz = _zones(_steps())
    walker = _walker(agent_factory, floor, 1, 5.0, 3.6)
    cmds = _go([walker], Pose2D(x=25.0, y=3.6))
    tz.steer([walker], cmds)
    walker.state.pose = Pose2D(x=11.5, y=3.6)
    tz.steer([walker], cmds)
    assert 1 in tz.through
    walker.state.pose = Pose2D(x=10.5, y=3.6)
    out = tz.steer([walker], cmds)
    assert tz.transfers[1].zone == "steps_b" and not tz.transfers[1].aboard and 1 not in tz.through
    assert tz.zones["steps_b"].contains(out[1].target_pose.x, out[1].target_pose.y)


def test_a_seam_joined_in_place_never_teleports(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones(_pair("steps", STEPS, 15.0, inset=0.45))
    pool = _pool(agent_factory, (1, 14.7, 3.6))
    assert tz.hold(pool, tick=0, timeout_ticks=100, dt=0.05, let_out_ticks=0) == []
    assert tuple(pool.pos[0]) == pytest.approx((14.7, 3.6))


def test_a_walker_whose_plan_would_end_on_a_stair_it_does_not_climb_stays_where_it_is(agent_factory: Callable[..., BaseAgent]) -> None:
    sealed = _gated(_hall(False, STEPS), _steps())
    tz = _zones(_steps())
    wheelchair = _walker(agent_factory, sealed, 1, 5.0, 3.6, climbs=())
    walker = _walker(agent_factory, sealed, 2, 5.0, 3.0, climbs=("escalator",))
    cmds = _go([wheelchair, walker], Pose2D(x=25.0, y=3.6))
    sealed.compute([wheelchair, walker], tz.steer([wheelchair, walker], cmds))
    end = sealed.get_cached_paths()[1][-1]
    assert tz.zones["steps_b"].contains(end.x, end.y)
    tz.steer([wheelchair, walker], cmds)
    assert tz.through == {1: wheelchair.state.pose, 2: walker.state.pose}


FLIGHTS = (7.0, 21.0, 3.0, 4.2)


def _flights() -> list[Zone]:
    """Two flights in one lane with a landing between them (x 15 to 17)."""
    return [*_pair("lower", (7.0, 15.0, 3.0, 4.2), 11.0), *_pair("upper", (17.0, 21.0, 3.0, 4.2), 19.0)]


@pytest.fixture(scope="module")
def flights() -> DijkstraPlanner:
    walls = [wall for wall in _hall(True, FLIGHTS) if wall[0][0] != 15.0 or wall[1][0] != 15.0]
    return _gated([*walls, ((14.0, 0.0), (14.0, 3.0)), ((14.0, 4.2), (14.0, 30.0))], _flights())


def test_two_flights_are_ridden_one_after_the_other_and_never_back_down_from_the_landing(agent_factory: Callable[..., BaseAgent], flights: DijkstraPlanner) -> None:
    tz = _zones(_flights())
    walker = _walker(agent_factory, flights, 1, 5.0, 3.6)
    cmds = _go([walker], Pose2D(x=25.0, y=3.6))
    tz.steer([walker], cmds)
    assert (tz.transfers[1].zone, tz.transfers[1].ride) == ("lower_b", True)
    walker.state.pose = Pose2D(x=9.0, y=3.6)
    tz.steer([walker], cmds)
    walker.state.pose = Pose2D(x=15.5, y=3.6)
    out = tz.steer([walker], cmds)
    assert (tz.transfers[1].zone, tz.transfers[1].ride) == ("upper_b", True)
    assert tz.zones["upper_b"].contains(out[1].target_pose.x, out[1].target_pose.y)


def test_a_rider_pushed_back_over_the_seam_it_crossed_takes_that_flight_again(agent_factory: Callable[..., BaseAgent], flights: DijkstraPlanner) -> None:
    tz = _zones(_flights())
    walker = _walker(agent_factory, flights, 1, 5.0, 3.6)
    cmds = _go([walker], Pose2D(x=25.0, y=3.6))
    for x in (5.0, 9.0, 15.5):
        walker.state.pose = Pose2D(x=x, y=3.6)
        tz.steer([walker], cmds)
    assert tz.transfers[1].zone == "upper_b"
    walker.state.pose = Pose2D(x=10.3, y=3.6)
    tz.steer([walker], cmds)
    assert (tz.transfers[1].zone, tz.transfers[1].ride) == ("lower_b", True)


def test_a_stair_half_is_walked_to_on_the_right_of_its_hold_line_and_at_its_hold_pose_when_too_narrow() -> None:
    tz = _zones([_up(), _up_b()])
    assert (tz.zones["up"].lane.x, tz.zones["up"].lane.y) == pytest.approx((9.55, 4.9))
    assert (tz.zones["up_b"].lane.x, tz.zones["up_b"].lane.y) == pytest.approx((22.5, 0.45))
    wide = _stair("wide", (), Pose2D(x=5.0, y=4.0), Pose2D(x=9.55, y=4.0), _box(5.0, 2.0, 10.0, 6.0))
    assert (wide.lane.x, wide.lane.y) == pytest.approx((9.55, 3.0))
    narrow = _stair("narrow", (), Pose2D(x=5.0, y=5.2), Pose2D(x=9.55, y=5.2), _box(5.0, 4.7, 10.0, 5.7))
    assert narrow.lane == narrow.hold


def test_the_two_ways_over_one_stair_walk_to_and_land_on_opposite_sides(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    up, down = tz.zones["up"].lane, tz.zones["up_b"].lane
    pool = _pool(agent_factory, (1, up.x, up.y), (2, down.x, down.y))
    crossed = {aid: (x, y) for aid, x, y, _ in tz.hold(pool, 0, 1000, 0.05, 0)}
    assert crossed[1] == pytest.approx((23.1, 0.45 + LANDING_BACKOFF))
    assert crossed[2] == pytest.approx((9.55 - LANDING_BACKOFF, 5.5))


def test_two_walkers_crossing_one_seam_the_opposite_ways_on_each_others_landing_change_places(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.7, 5.2), (2, 22.8, 0.3))
    pool.theta[0] = 0.1
    crossed = {aid: (x, y, theta) for aid, x, y, theta in tz.hold(pool, 0, 1000, 0.05, 0)}
    assert crossed[1] == pytest.approx((22.8, 0.3, HALF_PI + 0.1))
    assert crossed[2] == pytest.approx((9.7, 5.2, -HALF_PI))
    assert not tz._held


def test_walkers_queued_behind_the_two_do_not_stop_them_changing_places(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.6, 5.2), (2, 22.8, 0.4), (3, 9.1, 5.2), (4, 22.8, 0.9))
    assert sorted(aid for aid, *_ in tz.hold(pool, 0, 1000, 0.05, 0)) == [1, 2]


def test_a_crossing_never_lands_on_a_walker_that_stays(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.7, 5.2), (2, 22.55, 0.3), (3, 9.7, 5.7), (4, 22.3, 1.0))
    for aid, x, y, _ in tz.hold(pool, 0, 1000, 0.05, 0):
        pool.pos[pool.idx(aid)] = (x, y)
    assert tz._held == {3}
    gaps = [math.hypot(*(pool.pos[i] - pool.pos[j])) for i in range(4) for j in range(i)]
    assert min(gaps) >= 0.5 - 1e-9


def test_a_larger_walker_changes_places_only_where_it_fits(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.7, 5.2), (2, 22.8, 0.3), (3, 22.8, 0.85))
    pool.agent_radius[0] = 0.4
    assert tz.hold(pool, 0, 1000, 0.05, 0) == []
    assert tz._held == {1, 2}
    pool.pos[2] = (22.8, 1.2)
    assert sorted(aid for aid, *_ in tz.hold(pool, 1, 1000, 0.05, 0)) == [1, 2]


def test_a_walker_is_not_sent_back_from_the_half_it_landed_on_before_it_has_left_it_or_is_routed_back(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.6, 5.2), (2, 9.6, 5.8))
    [(_, x, y, _), (_, x2, y2, _)] = tz.hold(pool, 0, 1000, 0.05, 0)
    pool.pos[0], pool.pos[1] = (x, y - 0.3), (x2, y2 - 0.3)
    assert tz.hold(pool, 1, 1000, 0.05, 0) == []
    assert not tz._held and tuple(pool.pos[0]) == pytest.approx((x, y - 0.3))
    tz.transfers[2] = Transfer(goal=(GOAL_A.x, GOAL_A.y), zone="up_b", destination="up")
    assert [aid for aid, *_ in tz.hold(pool, 2, 1000, 0.05, 0)] == [2]
    pool.pos[0], pool.pos[1] = (22.8, 4.0), (2.0, 2.0)
    assert tz.hold(pool, 3, 1000, 0.05, 0) == []
    pool.pos[0] = (22.8, 0.3)
    assert [aid for aid, *_ in tz.hold(pool, 4, 1000, 0.05, 0)] == [1]


def test_a_walker_on_the_landing_that_crosses_nowhere_still_holds_the_crossing(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.7, 5.2), (2, 22.8, 0.5))
    pool.policy_idx[1] = -1
    assert tz.hold(pool, 0, 1000, 0.05, 0) == []



def test_a_half_with_a_delay_carries_the_crossing_pedestrian_along_the_flight_then_lands_it(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([attrs.evolve(_up(), delay=1.0), _up_b()])
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    end = (22.6, 0.45 + LANDING_BACKOFF)
    assert tz.hold(pool, 0, 1000, 0.05, 0) == [(1, 9.6, 5.4, pytest.approx(math.atan2(end[1] - 5.4, end[0] - 9.6)))]
    assert tz.flying == {1}
    wait = tz._flights[1].wait
    assert math.dist(wait, end) == pytest.approx(FLIGHT_WAIT_GAP)
    [(_, x, y, _)] = tz.hold(pool, 10, 1000, 0.05, 0)
    assert (x, y) == pytest.approx((9.6 + (wait[0] - 9.6) * 0.5, 5.4 + (wait[1] - 5.4) * 0.5))
    assert pool.vel[0] == pytest.approx((0.0, 0.0)) and pool.desired_vel[0] == 0.0
    [(aid, x, y, theta)] = tz.hold(pool, 20, 1000, 0.05, 0)
    assert (aid, x, y) == (1, pytest.approx(end[0]), pytest.approx(end[1]))
    assert tz.flying == frozenset()


def _fly_until(tz: TeleportZones, pool: AgentPool, ticks: range) -> list[tuple[int, int]]:
    """Run hold over `ticks`, applying its moves to the pool; the (tick, aid) of each landing."""
    landed = []
    for tick in ticks:
        before = set(tz.flying)
        for aid, x, y, theta in tz.hold(pool, tick, 400, 0.05, 0):
            i = pool.idx(aid)
            pool.pos[i] = (x, y)
            pool.theta[i] = theta
        landed += [(tick, aid) for aid in sorted(before - set(tz.flying))]
    return landed


def test_a_flight_lands_on_a_free_spot_across_the_partner_half_when_its_own_is_taken(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([attrs.evolve(_up(), delay=0.5), _up_b()])
    pool = _pool(agent_factory, (1, 9.6, 5.2), (2, 22.8, 0.55))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    assert _fly_until(tz, pool, range(0, 11)) == [(10, 1)]
    x, y = pool.pos[0]
    assert abs(x - 22.8) == pytest.approx(LANDING_STEP) and y == pytest.approx(0.45 + LANDING_BACKOFF)


def test_a_flight_waits_short_of_its_landing_while_every_spot_is_taken_and_lands_once_one_clears(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([attrs.evolve(_up(), delay=0.5), _up_b()])
    blockers = [(10 + k, 22.8 + dx, 0.55) for k, dx in enumerate((-0.6, 0.0, 0.6))]
    pool = _pool(agent_factory, (1, 9.6, 5.2), *blockers)
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    assert _fly_until(tz, pool, range(0, 30)) == []
    assert tuple(pool.pos[0]) == pytest.approx(tz._flights[1].wait)
    pool.pos[2] = (27.0, 2.0)
    assert _fly_until(tz, pool, range(30, 32)) == [(30, 1)]
    assert tuple(pool.pos[0]) == pytest.approx((22.8, 0.45 + LANDING_BACKOFF))


def test_waiting_fliers_do_not_block_each_others_landing(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([attrs.evolve(_up(), delay=0.5), _up_b()])
    blockers = [(10 + k, 22.8 + dx, 0.55) for k, dx in enumerate((-0.6, 0.0, 0.6))]
    pool = _pool(agent_factory, (1, 9.6, 5.25), (2, 9.6, 5.1), *blockers)
    for aid in (1, 2):
        tz.transfers[aid] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    assert _fly_until(tz, pool, range(0, 40)) == []
    for i in (2, 3, 4):
        pool.pos[i] = (27.0, 2.0 + i)
    landed = _fly_until(tz, pool, range(40, 42))
    assert sorted(aid for _, aid in landed) == [1, 2] and tz.flying == frozenset()
    assert math.dist(pool.pos[0], pool.pos[1]) >= 2 * 0.25 - 1e-9


def test_flights_land_in_the_order_their_time_ends(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([attrs.evolve(_up(), delay=0.5), _up_b()])
    pool = _pool(agent_factory, (2, 9.4, 5.2), (1, 9.4, 5.2), (10, 22.0, 0.55), (11, 22.8, 0.55), (12, 23.4, 0.55))
    tz.transfers[2] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    pool.pos[0] = (9.6, 5.2)
    _fly_until(tz, pool, range(0, 3))  # 2 boards at tick 0
    pool.pos[1] = (9.6, 5.2)
    _fly_until(tz, pool, range(3, 30))  # 1 boards at tick 3, both wait
    assert tz.flying == {1, 2}
    pool.pos[3] = (27.0, 2.0)  # one spot clears: the flight that ended first takes it
    assert _fly_until(tz, pool, range(30, 31)) == [(30, 2)]


def test_a_flight_lands_anyway_once_it_waited_the_timeout(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([attrs.evolve(_up(), delay=0.5), _up_b()])
    blockers = [(10 + k, 22.8 + dx, 0.55) for k, dx in enumerate((-0.6, 0.0, 0.6))]
    pool = _pool(agent_factory, (1, 9.6, 5.2), *blockers)
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    landed = _fly_until(tz, pool, range(0, 10 + 400 + 2))
    assert landed == [(410, 1)]
    assert tuple(pool.pos[0]) == pytest.approx((22.8, 0.45 + LANDING_BACKOFF))


def test_emptying_the_zones_drops_flights_and_climbs(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([attrs.evolve(_up(), delay=1.0), _up_b()])
    walker = _walker(agent_factory, planner, 1, 9.6, 5.4)
    tz.steer([walker], _go([walker], GOAL_B))
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    tz.hold(pool, 1000, 1000, 0.05, 0)
    assert tz.flying == {1}
    tz.set_zones([], [])
    assert tz.flying == frozenset() and not tz._climbs
    assert tz.hold(_pool(agent_factory, (1, 3.0, 3.0)), 0, 1000, 0.05, 0) == []


def test_a_flying_pedestrian_keeps_steering_at_its_landing_and_forget_ends_the_flight(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([attrs.evolve(_up(), delay=1.0), _up_b()])
    walker = _walker(agent_factory, planner, 1, 9.6, 5.4)
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    tz.hold(pool, 0, 1000, 0.05, 0)
    out = tz.steer([walker], _go([walker], GOAL_B))
    assert (out[1].target_pose.x, out[1].target_pose.y) == pytest.approx((22.6, 0.45 + LANDING_BACKOFF))
    tz.forget(1)
    assert tz.flying == frozenset()


def test_a_type_that_does_not_climb_the_half_is_not_carried_over_its_seam(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up(), _up_b()])
    chair = _walker(agent_factory, planner, 1, 9.6, 5.4, climbs=())
    tz.steer([chair], _go([chair], GOAL_A))
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    assert tz.hold(pool, 0, 1000, 0.05, 0) == []


def test_a_running_belt_does_not_carry_a_type_that_does_not_climb_it(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up("escalator", 0.6), _up_b("escalator", -0.6, ())])
    chair, walker = _walker(agent_factory, planner, 1, 7.0, 5.2, climbs=()), _walker(agent_factory, planner, 2, 8.0, 5.2)
    tz.steer([chair, walker], {**_go([chair], GOAL_A), **_go([walker], GOAL_B)})  # the walker's route takes the belt
    pool = _pool(agent_factory, (1, 7.0, 5.2), (2, 8.0, 5.2))
    pool.vel[0] = (-0.8, 0.0)
    tz.carry(pool)
    assert pool.vel[0] == pytest.approx((-0.8, 0.0)) and 1 not in tz.riders
    assert pool.vel[1] == pytest.approx((0.6, 0.0)) and 2 in tz.riders


def test_a_walker_on_a_half_its_route_does_not_take_walks_over_it(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    # beside its landing or on a wall-trailing line round a mouth: no belt carries it and it is not carried over the seam,
    # or it rides back, takes the first half again and goes round for good
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ())])
    walker = _walker(agent_factory, planner, 1, 9.6, 5.2)
    tz.steer([walker], _go([walker], GOAL_A))
    assert 1 not in tz.transfers
    pool = _pool(agent_factory, (1, 9.6, 5.2))
    pool.vel[0] = (0.0, -0.9)
    pool.desired_vel[0] = 1.2
    tz.carry(pool)
    assert pool.vel[0] == pytest.approx((0.0, -0.9)) and pool.desired_vel[0] == 1.2 and not tz.riders
    assert tz.hold(pool, 0, 1000, 0.05, 0) == [] and tz.flying == frozenset()


def test_a_walker_whose_route_takes_the_half_is_carried_and_crosses(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ())])
    walker = _walker(agent_factory, planner, 1, 9.6, 5.2)
    tz.steer([walker], _go([walker], GOAL_B))
    assert tz.transfers[1].zone == "up"
    pool = _pool(agent_factory, (1, 9.6, 5.2))
    tz.carry(pool)
    assert pool.vel[0] == pytest.approx((0.5, 0.0)) and tz.riders == {1}
    assert [aid for aid, *_ in tz.hold(pool, 0, 1000, 0.05, 0)] == [1]


def test_a_walker_that_landed_is_carried_off_its_landing_by_the_belt_while_walking_on(agent_factory: Callable[..., BaseAgent], planner: DijkstraPlanner) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.4, ())])
    walker = _walker(agent_factory, planner, 1, 9.6, 5.2)
    tz.steer([walker], _go([walker], GOAL_B))
    pool = _pool(agent_factory, (1, 9.6, 5.2))
    _land(tz, pool, range(0, 1))
    walker.state.pose = Pose2D(x=float(pool.pos[0, 0]), y=float(pool.pos[0, 1]))
    tz.steer([walker], _go([walker], GOAL_B))
    assert 1 not in tz.transfers
    tz.carry(pool)
    assert pool.vel[0] == pytest.approx((0.0, 0.4)) and tz.riders == {1}


def test_unreachable_names_stair_halves_whose_hold_is_on_a_wall_or_outside_the_footprint() -> None:
    tz = TeleportZones()
    on_seam = _stair("on_seam", ("up_b",), Pose2D(x=5.0, y=5.2), Pose2D(x=10.0, y=5.2), _box(5.0, 4.4, 10.0, 6.0))
    outside = _stair("outside", ("up_b",), Pose2D(x=5.0, y=5.2), Pose2D(x=10.5, y=5.2), _box(5.0, 4.4, 10.0, 6.0))
    assert tz.unreachable([_up(), _up_b(), on_seam, outside, *_lift()], WALLS) == ["on_seam", "outside"]
    assert tz.unreachable([_up()], []) == []  # no walls yet: only the footprint is checked
    assert tz.unreachable([on_seam], []) == ["on_seam"]  # a hold on the footprint edge counts as outside


def _land(tz: TeleportZones, pool: AgentPool, ticks: range) -> None:
    """Run hold over `ticks`, applying its moves and ending the transfer of each pedestrian that lands, as the node does."""
    for tick in ticks:
        for aid, x, y, theta in tz.hold(pool, tick, 1000, 0.05, 0):
            i = pool.idx(aid)
            pool.pos[i] = (x, y)
            pool.theta[i] = theta
            if aid not in tz.flying:
                tz.forget(aid)


def test_a_pedestrian_drifting_back_past_the_seam_it_landed_on_is_not_carried_back_until_it_steps_off(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    _land(tz, pool, range(0, 1))
    assert tuple(pool.pos[0]) == pytest.approx((22.6, 0.45 + LANDING_BACKOFF))
    pool.pos[0] = (22.6, 0.3)  # pushed past up_b's seam, no route over it
    assert tz.hold(pool, 1, 1000, 0.05, 0) == []
    pool.pos[0] = (21.0, 3.0)  # off up_b
    assert tz.hold(pool, 2, 1000, 0.05, 0) == []
    pool.pos[0] = (22.6, 0.3)
    assert [aid for aid, *_ in tz.hold(pool, 3, 1000, 0.05, 0)] == [1]


def test_a_pedestrian_whose_route_takes_the_half_it_landed_on_back_crosses_it(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up(), _up_b()])
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    _land(tz, pool, range(0, 1))
    tz.transfers[1] = Transfer(goal=(GOAL_A.x, GOAL_A.y), zone="up_b", destination="up")
    pool.pos[0] = (22.6, 0.3)
    assert [aid for aid, *_ in tz.hold(pool, 1, 1000, 0.05, 0)] == [1]


def test_a_flight_landing_is_not_flown_back_by_a_drift_past_the_partner_seam(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([attrs.evolve(_up(), delay=0.5), attrs.evolve(_up_b(), delay=0.5)])
    pool = _pool(agent_factory, (1, 9.6, 5.4))
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    _land(tz, pool, range(0, 11))
    assert tz.flying == frozenset() and tuple(pool.pos[0]) == pytest.approx((22.6, 0.45 + LANDING_BACKOFF))
    pool.pos[0] = (22.6, 0.3)
    assert tz.hold(pool, 11, 1000, 0.05, 0) == [] and tz.flying == frozenset()
    tz.clear_transfers()
    assert [aid for aid, *_ in tz.hold(pool, 12, 1000, 0.05, 0)] == [1] and tz.flying == {1}


def test_a_pedestrian_held_on_a_seam_for_the_wait_timeout_lands_anyway(agent_factory: Callable[..., BaseAgent]) -> None:
    tz = _zones([_up("escalator", 0.5), _up_b("escalator", -0.5, ())])
    pool = _pool(agent_factory, (1, 9.6, 5.2), (2, 22.8, 0.55))  # 2 stands on 1's landing and is not held
    tz.transfers[1] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone="up", destination="up_b")
    for tick in range(10):
        assert tz.hold(pool, tick, 10, 0.05, 0) == []
    [(aid, x, y, _)] = tz.hold(pool, 10, 10, 0.05, 0)
    assert (aid, x, y) == (1, pytest.approx(22.8), pytest.approx(0.45 + LANDING_BACKOFF))
    assert tz._held == set() and tz._held_since == {}


def test_alighting_finds_whoever_stands_in_a_zone_as_testing_every_zone_for_everyone(agent_factory: Callable[..., BaseAgent]) -> None:
    """The zones someone stands in who is not waiting for, boarding or holding in that zone: the same as testing every
    pedestrian against every zone (a cabin's hold disc reaches past its footprint)."""
    zones = [*_low(), *_lift(), _up(), _up_b()]
    tz = _zones(zones)
    rng = np.random.default_rng(11)
    spots = [(float(rng.uniform(-0.5, 30.5)), float(rng.uniform(-0.5, 6.5))) for _ in range(150)]
    spots += [(z.hold.x + 0.29, z.hold.y) for z in zones] + [(x, y) for z in zones for x, y in z.footprint]
    pool = AgentPool(capacity=len(spots))
    for aid, (x, y) in enumerate(spots, start=1):
        pool.add_agent(agent_factory(agent_id=aid, x=x, y=y))
    for aid in range(1, len(spots) + 1, 3):
        x, y = spots[aid - 1]
        inside = [z.name for z in zones if z.contains(x, y)]
        if inside:
            tz.transfers[aid] = Transfer(goal=(GOAL_B.x, GOAL_B.y), zone=inside[0], destination=tz.zones[inside[0]].destinations[0], phase=Phase.WAIT if aid % 2 else Phase.WALK)
    every = set()
    for i in range(pool.n):
        transfer = tz.transfers.get(int(pool.agent_ids[i]))
        x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
        every.update(z.name for z in zones if not (transfer is not None and transfer.zone == z.name and transfer.phase != Phase.WALK) and z.contains(x, y))
    assert tz._alighting(pool) == every
    assert every
