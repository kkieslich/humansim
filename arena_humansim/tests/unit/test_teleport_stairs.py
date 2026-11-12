from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.core.teleport_zones import FLIGHT_WAIT_GAP, LANDING_BACKOFF, LANDING_STEP, STAIR_SPEED, TeleportZones, Transfer, Zone
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
    assert out[1].target_pose == tz.zones["up"].hold
    assert out[2].target_pose == tz.zones["low"].hold


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
    assert out[2].target_pose == tz.zones["up"].hold


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
    tz.steer([chair, walker], _go([chair, walker], GOAL_A))
    pool = _pool(agent_factory, (1, 7.0, 5.2), (2, 8.0, 5.2))
    pool.vel[0] = (-0.8, 0.0)
    tz.carry(pool)
    assert pool.vel[0] == pytest.approx((-0.8, 0.0)) and 1 not in tz.riders
    assert pool.vel[1] == pytest.approx((0.6, 0.0)) and 2 in tz.riders


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
