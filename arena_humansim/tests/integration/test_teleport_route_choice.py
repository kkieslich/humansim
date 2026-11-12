"""zone_route_choice in the node: walkers take a stair over a long walkable detour, wheelchairs walk, same seed same run."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams, WallConfig
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import TeleportZone
from arena_humansim_msgs.srv import SetTeleportZones
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.parameter import Parameter

# rooms A (x 0-10) and B (x 20-30) joined by a 3 m corridor over the top and by a stair whose halves are pockets off the facing walls
SEGMENTS = [
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
GOAL = Pose2D(x=25.0, y=5.0)


def _half(name: str, destination: str, entry: tuple[float, float, float], hold: tuple[float, float], box: tuple[float, float, float, float]) -> TeleportZone:
    z = TeleportZone()
    z.name, z.destinations, z.kind, z.open, z.delay = name, [destination], "stairs", True, 0.0
    z.entry = Pose2DMsg(x=entry[0], y=entry[1], theta=entry[2])
    z.wait = Pose2DMsg(x=entry[0] - 0.8 * math.cos(entry[2]), y=entry[1] - 0.8 * math.sin(entry[2]), theta=entry[2])
    z.hold = Pose2DMsg(x=hold[0], y=hold[1], theta=entry[2])
    x0, y0, x1, y1 = box
    z.footprint = [Point32(x=x, y=y, z=0.0) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return z


def _manager(manager_factory: Callable[..., AgentManager], name: str, choice: bool, agents: list[AgentConfig]) -> AgentManager:
    walls = [WallConfig(name=f"w{i}", start=Pose2D(x=a[0], y=a[1]), end=Pose2D(x=b[0], y=b[1])) for i, (a, b) in enumerate(SEGMENTS)]
    scenario = ScenarioConfig(name=name, simulation=SimulationParams(seed=5, dt=0.05), modules=ModuleConfig(global_planner="navmesh"), walls=walls, agents=agents)
    mgr = manager_factory(scenario, node_name=f"test_{name}", extra_params=[Parameter("zone_route_choice", Parameter.Type.BOOL, choice)])
    zones = [_half("st_a", "st_b", (10.0, 5.0, 0.0), (13.55, 5.0), (10.0, 4.0, 14.0, 6.0)), _half("st_b", "st_a", (20.0, 5.0, math.pi), (16.45, 5.0), (16.0, 4.0, 20.0, 6.0))]
    assert mgr._set_teleport_zones_callback(SetTeleportZones.Request(zones=zones), SetTeleportZones.Response()).success
    return mgr


def _pos(mgr: AgentManager, aid: int) -> tuple[float, float]:
    i = mgr._pool.idx(aid)
    return float(mgr._pool.pos[i, 0]), float(mgr._pool.pos[i, 1])


def _pair() -> list[AgentConfig]:
    return [
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=5.0, y=5.0), goal_sequence=[GOAL], desired_velocity=1.2),
        AgentConfig(agent_id=2, agent_type="wheelchair_manual", spawn_pose=Pose2D(x=5.0, y=2.5), goal_sequence=[GOAL], desired_velocity=0.9),
    ]


def test_walker_takes_the_stair_and_a_wheelchair_walks_the_detour(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "route_choice_on", True, _pair())
    took_stair = wheelchair_round = False
    adult_at, chair_at = None, None
    for n in range(2400):
        mgr.tick()
        if 1 in mgr._zones.transfers:
            took_stair |= mgr._zones.transfers[1].zone == "st_a"
        assert 2 not in mgr._zones.transfers
        if 2 in mgr._agents:
            wheelchair_round |= _pos(mgr, 2)[1] > 15.0
        if adult_at is None and 1 in mgr._agents and math.dist(_pos(mgr, 1), (GOAL.x, GOAL.y)) < 1.0:
            adult_at = n
        if chair_at is None and 2 in mgr._agents and math.dist(_pos(mgr, 2), (GOAL.x, GOAL.y)) < 1.0:
            chair_at = n
        if adult_at is not None and chair_at is not None:
            break
    assert took_stair and wheelchair_round
    assert adult_at is not None and chair_at is not None
    assert adult_at < chair_at


def test_without_route_choice_the_walker_walks_the_detour(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "route_choice_off", False, _pair()[:1])
    went_round = False
    for _ in range(1200):
        mgr.tick()
        assert 1 not in mgr._zones.transfers
        went_round |= _pos(mgr, 1)[1] > 15.0
        if went_round:
            break
    assert went_round


def test_same_seed_same_run_with_route_choice(manager_factory: Callable[..., AgentManager]) -> None:
    def crowd() -> list[AgentConfig]:
        out = []
        for k in range(8):
            out.append(AgentConfig(agent_id=k + 1, agent_type="wheelchair_manual" if k % 4 == 3 else "adult", spawn_pose=Pose2D(x=2.0 + 1.0 * k, y=1.5 + 0.9 * (k % 3)), goal_sequence=[GOAL], desired_velocity=1.0 + 0.05 * k))
            out.append(AgentConfig(agent_id=k + 21, spawn_pose=Pose2D(x=22.0 + 0.9 * k, y=2.0 + 0.8 * (k % 4)), goal_sequence=[Pose2D(x=5.0, y=5.0)], desired_velocity=1.1))
        return out

    runs = []
    for r in range(2):
        mgr = _manager(manager_factory, f"route_choice_seed_{r}", True, crowd())
        trace = []
        for n in range(700):
            mgr.tick()
            if n % 50 == 49:
                ids = sorted(mgr._agents)
                trace.append((tuple(ids), np.array([_pos(mgr, a) for a in ids]), tuple(sorted((a, t.zone) for a, t in mgr._zones.transfers.items()))))
        runs.append(trace)
    used = {zone for _, _, ts in runs[0] for _, zone in ts}
    assert used  # some walkers took the stair
    for (ids_a, pos_a, tr_a), (ids_b, pos_b, tr_b) in zip(runs[0], runs[1], strict=True):
        assert ids_a == ids_b
        assert np.array_equal(pos_a, pos_b)
        assert tr_a == tr_b


def test_a_stair_half_whose_hold_lies_on_the_seam_wall_is_refused(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "route_choice_refuse", True, [])
    bad = _half("st_a", "st_b", (10.0, 5.0, 0.0), (14.0, 5.0), (10.0, 4.0, 14.0, 6.0))
    good = _half("st_b", "st_a", (20.0, 5.0, math.pi), (16.45, 5.0), (16.0, 4.0, 20.0, 6.0))
    response = mgr._set_teleport_zones_callback(SetTeleportZones.Request(zones=[bad, good]), SetTeleportZones.Response())
    assert not response.success
    assert "st_a" in response.message and "st_b" not in response.message.split("[")[1].split("]")[0]
    assert set(mgr._zones.zones) == {"st_a", "st_b"}  # the zones set before stay


def test_a_stair_flight_takes_its_delay_between_the_seam_and_the_landing(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "route_choice_flight", True, _pair()[:1])
    zones = [_half("st_a", "st_b", (10.0, 5.0, 0.0), (13.55, 5.0), (10.0, 4.0, 14.0, 6.0)), _half("st_b", "st_a", (20.0, 5.0, math.pi), (16.45, 5.0), (16.0, 4.0, 20.0, 6.0))]
    zones[0].delay = 3.0
    assert mgr._set_teleport_zones_callback(SetTeleportZones.Request(zones=zones), SetTeleportZones.Response()).success
    boarded = landed = None
    between = []
    for n in range(1200):
        mgr.tick()
        if 1 in mgr._zones.flying and boarded is None:
            boarded = n
        x, y = _pos(mgr, 1)
        if boarded is not None and landed is None and 14.0 < x < 16.0:
            between.append(x)
        if boarded is not None and landed is None and 1 not in mgr._zones.flying:
            landed = n
            assert 16.0 <= x <= 20.0
            break
    assert boarded is not None and landed is not None
    assert (landed - boarded) * 0.05 >= 3.0 - 1e-9
    assert len(between) > 10 and between == sorted(between)
