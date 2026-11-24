from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import attrs
import py_trees
import pytest
import yaml
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.animation_kinds import AnimationType
from arena_humansim.core.teleport_zones import HOLD_RADIUS, LANDING_BACKOFF, STAIR_SPEED, WAIT_RADIUS, Phase
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.scenario import (
    AgentConfig,
    AgentTemplateModel,
    FlowScenarioConfig,
    ModuleConfig,
    RateKeyframeModel,
    ScenarioConfig,
    ShapeModel,
    SimulationParams,
    SinkAffinityModel,
    SinkScenarioConfig,
    SourceScenarioConfig,
    WallConfig,
    load_scenario,
)
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import TeleportZone
from arena_humansim_msgs.srv import ResetSimulation, SetTeleportZones, SetZoneAccess, TeleportAgents
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.parameter import Parameter

ROOM = 6.0
ROOM_B_X = 20.0
SINK = Pose2D(x=ROOM_B_X + 1.0, y=3.0)
PLANNERS = {
    "navmesh": [],
    "dijkstra": [Parameter("global_planner.resolution", Parameter.Type.DOUBLE, 0.1)],
}


def _room(name: str, x0: float) -> list[WallConfig]:
    corners = [(x0, 0.0), (x0 + ROOM, 0.0), (x0 + ROOM, ROOM), (x0, ROOM)]
    return [WallConfig(name=f"{name}{i}", start=Pose2D(x=ax, y=ay), end=Pose2D(x=bx, y=by)) for i, ((ax, ay), (bx, by)) in enumerate(zip(corners, corners[1:] + corners[:1], strict=True))]


def _scenario(name: str, planner: str, riders: int = 1) -> ScenarioConfig:
    flow = FlowScenarioConfig(
        sources=[
            SourceScenarioConfig(
                pose=Pose2D(x=1.0, y=3.0),
                shape=ShapeModel(type="circle", radius=0.1),
                type="poisson",
                rate_profile=[RateKeyframeModel(t=0.0, rate=20.0)],
                max_concurrent=riders,
                max_total=riders,
                agent_template=AgentTemplateModel(desired_velocity_min=1.2, desired_velocity_max=1.2, agent_radius=0.25, sink_affinity=[SinkAffinityModel(sink_idx=0, weight=1.0)]),
            ),
        ],
        sinks=[SinkScenarioConfig(pose=SINK, shape=ShapeModel(type="circle", radius=0.5), absorption_radius=0.5)],
    )
    return ScenarioConfig(
        name=name,
        simulation=SimulationParams(seed=3, dt=0.05),
        modules=ModuleConfig(global_planner=planner),
        walls=_room("a", 0.0) + _room("b", ROOM_B_X),
        flow=flow,
    )


def _zone(name: str, destination: str, x0: float, y: float, is_open: bool = False, half: float = 0.6, delay: float = -1.0) -> TeleportZone:
    z = TeleportZone()
    z.name = name
    z.destinations = [destination] if destination else []
    z.hold = Pose2DMsg(x=x0 + 4.6 + half, y=y, theta=0.0)
    z.entry = Pose2DMsg(x=x0 + 4.0, y=y, theta=0.0)
    z.wait = Pose2DMsg(x=x0 + 3.4, y=y, theta=0.0)
    z.footprint = [Point32(x=x0 + px, y=y + py, z=0.0) for px, py in ((4.6, -half), (4.6 + 2 * half, -half), (4.6 + 2 * half, half), (4.6, half))]
    z.open = is_open
    z.delay = delay
    return z


def _set_zones(mgr: AgentManager, zones: list[TeleportZone]) -> SetTeleportZones.Response:
    return mgr._set_teleport_zones_callback(SetTeleportZones.Request(zones=zones), SetTeleportZones.Response())


def _set_access(mgr: AgentManager, name: str, is_open: bool) -> SetZoneAccess.Response:
    return mgr._set_zone_access_callback(SetZoneAccess.Request(name=[name], open=[is_open]), SetZoneAccess.Response())


def _teleport(mgr: AgentManager, ids: list[int], poses: list[tuple[float, float]]) -> TeleportAgents.Response:
    request = TeleportAgents.Request(agent_id=ids, pose=[Pose2DMsg(x=x, y=y, theta=0.0) for x, y in poses])
    return mgr._teleport_agents_callback(request, TeleportAgents.Response())


def _tick_until(mgr: AgentManager, done: Callable[[], bool], limit: int) -> int:
    for n in range(limit):
        if done():
            return n
        mgr.tick()
    raise AssertionError(f"condition not met in {limit} ticks")


def _rider(mgr: AgentManager) -> int:
    _tick_until(mgr, lambda: bool(mgr._agents), 40)
    return next(iter(mgr._agents))


def _zone_pair(name: str, delay: float = -1.0) -> list[TeleportZone]:
    return [_zone(f"{name}_a", f"{name}_b", 0.0, 4.6, delay=delay), _zone(f"{name}_b", f"{name}_a", ROOM_B_X, 4.6, delay=delay)]


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_pedestrian_waits_at_closed_zone_holds_inside_and_reaches_sink_in_other_room_after_teleport(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    mgr = manager_factory(_scenario(f"zone_pass_{planner}", planner), node_name=f"test_zone_pass_{planner}", extra_params=PLANNERS[planner])
    assert _set_zones(mgr, _zone_pair("pair")).success
    aid = _rider(mgr)
    pool = mgr._pool

    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    assert mgr._zones.transfers[aid].zone == "pair_a"
    for _ in range(60):
        mgr.tick()
    i = pool.idx(aid)
    assert mgr._zones.transfers[aid].phase == Phase.WAIT
    assert math.hypot(pool.pos[i, 0] - 3.4, pool.pos[i, 1] - 4.6) <= WAIT_RADIUS + 0.3
    assert math.hypot(*pool.vel[i]) < 0.05

    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: mgr._zones.transfers[aid].phase == Phase.HOLD, 200)
    for _ in range(40):
        mgr.tick()
    i = pool.idx(aid)
    assert mgr._zones.transfers[aid].phase == Phase.HOLD
    assert math.hypot(*pool.vel[i]) < 0.05

    x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
    assert _teleport(mgr, [aid], [(x + ROOM_B_X, y)]).success
    assert aid not in mgr._zones.transfers
    i = pool.idx(aid)
    assert (pool.pos[i, 0], pool.pos[i, 1]) == (x + ROOM_B_X, y)
    assert not pool.vel[i].any()

    last: list[tuple[float, float]] = []

    def absorbed() -> bool:
        if aid not in mgr._agents:
            return True
        j = pool.idx(aid)
        last[:] = [(float(pool.pos[j, 0]), float(pool.pos[j, 1]))]
        assert aid not in mgr._zones.transfers
        return False

    _tick_until(mgr, absorbed, 400)
    assert math.hypot(last[0][0] - SINK.x, last[0][1] - SINK.y) < 1.0


def _riders(mgr: AgentManager, n: int) -> list[int]:
    _tick_until(mgr, lambda: len(mgr._agents) >= n, 200)
    return sorted(mgr._agents)[:n]


def test_second_rider_waits_clear_of_the_opening_while_the_only_spot_is_taken(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_queue", "navmesh", riders=2), node_name="test_zone_queue")
    assert _set_zones(mgr, _zone_pair("pair")).success
    assert len(mgr._zones.zones["pair_a"].slots) == 1
    first, second = _riders(mgr, 2)
    transfers = mgr._zones.transfers

    _tick_until(mgr, lambda: all(a in transfers and transfers[a].phase == Phase.WAIT for a in (first, second)), 600)
    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: any(transfers[a].phase == Phase.HOLD for a in (first, second)), 300)
    holder, queued = (first, second) if transfers[first].phase == Phase.HOLD else (second, first)
    for _ in range(40):
        mgr.tick()
    assert transfers[queued].phase == Phase.WAIT
    j = mgr._pool.idx(queued)
    assert math.hypot(mgr._pool.pos[j, 0] - 3.4, mgr._pool.pos[j, 1] - 4.6) <= WAIT_RADIUS + 0.3

    i = mgr._pool.idx(holder)
    assert _teleport(mgr, [holder], [(float(mgr._pool.pos[i, 0]) + ROOM_B_X, float(mgr._pool.pos[i, 1]))]).success
    _tick_until(mgr, lambda: transfers[queued].phase == Phase.HOLD, 300)


def test_riders_of_one_zone_hold_on_distinct_spots(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_spots", "navmesh", riders=2), node_name="test_zone_spots")
    zones = [_zone("big_a", "big_b", 0.0, 3.0, half=1.0), _zone("big_b", "big_a", ROOM_B_X, 3.0, half=1.0)]
    assert _set_zones(mgr, zones).success
    assert len(mgr._zones.zones["big_a"].slots) == 4
    riders = _riders(mgr, 2)
    transfers = mgr._zones.transfers

    _tick_until(mgr, lambda: all(a in transfers and transfers[a].phase == Phase.WAIT for a in riders), 600)
    assert _set_access(mgr, "big_a", True).success
    _tick_until(mgr, lambda: all(transfers[a].phase == Phase.HOLD for a in riders), 400)
    assert transfers[riders[0]].slot != transfers[riders[1]].slot


def test_holding_rider_pushed_out_of_an_open_zone_boards_again(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_reboard", "navmesh"), node_name="test_zone_reboard")
    assert _set_zones(mgr, _zone_pair("pair")).success
    aid = _rider(mgr)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: aid in transfers and transfers[aid].phase == Phase.WAIT, 300)
    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)

    i = mgr._pool.idx(aid)
    mgr._pool.pos[i] = (3.9, 4.6)
    mgr.tick()
    assert transfers[aid].phase == Phase.BOARD
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_wait_timeout_moves_waiting_rider_to_the_next_zone_then_back_to_the_first(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    params = [*PLANNERS[planner], Parameter("zone_wait_timeout", Parameter.Type.DOUBLE, 1.0)]
    mgr = manager_factory(_scenario(f"zone_timeout_{planner}", planner), node_name=f"test_zone_timeout_{planner}", extra_params=params)
    zones = _zone_pair("a") + [_zone("b_a", "b_b", 0.0, 1.0), _zone("b_b", "b_a", ROOM_B_X, 1.0)]
    assert _set_zones(mgr, zones).success
    aid = _rider(mgr)

    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    assert mgr._zones.transfers[aid].zone == "a_a"
    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].zone == "b_a", 60)
    _tick_until(mgr, lambda: mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    i = mgr._pool.idx(aid)
    assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 1.0) <= WAIT_RADIUS + 0.3
    _tick_until(mgr, lambda: aid in mgr._zones.transfers and mgr._zones.transfers[aid].zone == "a_a", 60)
    _tick_until(mgr, lambda: mgr._zones.transfers[aid].phase == Phase.WAIT, 300)
    i = mgr._pool.idx(aid)
    assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 4.6) <= WAIT_RADIUS + 0.3


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_wait_timeout_at_the_only_zone_requeues_there_and_never_targets_the_other_room(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    params = [*PLANNERS[planner], Parameter("zone_wait_timeout", Parameter.Type.DOUBLE, 1.0)]
    mgr = manager_factory(_scenario(f"zone_requeue_{planner}", planner), node_name=f"test_zone_requeue_{planner}", extra_params=params)
    assert _set_zones(mgr, _zone_pair("only")).success
    aid = _rider(mgr)
    transfers = mgr._zones.transfers
    global_planner = mgr._agents[aid].global_planner

    _tick_until(mgr, lambda: aid in transfers and transfers[aid].phase == Phase.WAIT, 300)
    first = transfers[aid]
    requeued = False
    for _ in range(120):
        mgr.tick()
        transfer = transfers.get(aid)
        if transfer is not None:
            assert transfer.zone == "only_a"
            requeued = requeued or transfer is not first
        planned = global_planner._path_cache.get(aid)
        assert planned is None or planned[0][0] < ROOM
        subgoal = mgr._cached_intermediate_goals.get(aid)
        assert subgoal is None or subgoal.x < ROOM
        i = mgr._pool.idx(aid)
        assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 4.6) <= WAIT_RADIUS + 0.3
    assert requeued


def _boarding_manager(manager_factory: Callable[..., AgentManager], name: str, delay: float) -> tuple[AgentManager, int]:
    mgr = manager_factory(_scenario(name, "navmesh"), node_name=f"test_{name}")
    zones = [_zone("hop_a", "hop_b", 0.0, 4.6, is_open=True, delay=delay), _zone("hop_b", "hop_a", ROOM_B_X, 4.6, delay=delay)]
    assert _set_zones(mgr, zones).success
    aid = _rider(mgr)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: aid in transfers and transfers[aid].phase == Phase.BOARD, 300)
    return mgr, aid


def _pos(mgr: AgentManager, aid: int) -> tuple[float, float]:
    i = mgr._pool.idx(aid)
    return float(mgr._pool.pos[i, 0]), float(mgr._pool.pos[i, 1])


def test_zone_delay_teleports_a_holding_rider_to_the_destination_at_the_same_offset_after_the_delay(manager_factory: Callable[..., AgentManager]) -> None:
    mgr, aid = _boarding_manager(manager_factory, "zone_delay_two", 2.0)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)

    held = 0
    before = _pos(mgr, aid)
    while aid in transfers:
        assert transfers[aid].phase == Phase.HOLD
        assert held <= 41
        before = _pos(mgr, aid)
        mgr.tick()
        held += 1
    assert abs(held - 40) <= 1
    x, y = _pos(mgr, aid)
    assert math.hypot(x - before[0] - ROOM_B_X, y - before[1]) < 0.05
    assert not mgr._pool.vel[mgr._pool.idx(aid)].any()


def test_zero_zone_delay_teleports_a_rider_on_its_first_holding_tick(manager_factory: Callable[..., AgentManager]) -> None:
    mgr, aid = _boarding_manager(manager_factory, "zone_delay_zero", 0.0)
    transfers = mgr._zones.transfers
    spot = mgr._zones.zones["hop_a"].slots[transfers[aid].slot]

    before, at_spot = _pos(mgr, aid), False
    for _ in range(200):
        before = _pos(mgr, aid)
        at_spot = math.hypot(before[0] - spot.x, before[1] - spot.y) <= HOLD_RADIUS
        mgr.tick()
        if aid not in transfers:
            break
        assert transfers[aid].phase == Phase.BOARD
        assert not at_spot
    else:
        raise AssertionError("rider never teleported")
    assert at_spot
    x, y = _pos(mgr, aid)
    assert math.hypot(x - before[0] - ROOM_B_X, y - before[1]) < 0.05


def test_negative_zone_delay_never_teleports_a_holding_rider(manager_factory: Callable[..., AgentManager]) -> None:
    mgr, aid = _boarding_manager(manager_factory, "zone_delay_never", -1.0)
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: transfers[aid].phase == Phase.HOLD, 200)
    for _ in range(200):
        mgr.tick()
        assert transfers[aid].phase == Phase.HOLD
        assert _pos(mgr, aid)[0] < ROOM


def test_set_teleport_zones_replaces_and_clears_the_zone_set_and_survives_soft_reset(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_set", "navmesh"), node_name="test_zone_set")
    assert _set_zones(mgr, _zone_pair("x") + [_zone("solo", "", 0.0, 1.4, is_open=True)]).success
    assert sorted(mgr._zones.zones) == ["solo", "x_a", "x_b"]
    assert mgr._zones.open == {"x_a": False, "x_b": False, "solo": True}

    assert _set_zones(mgr, _zone_pair("y")).success
    assert sorted(mgr._zones.zones) == ["y_a", "y_b"]
    assert not _set_access(mgr, "x_a", True).success
    assert _set_access(mgr, "y_b", True).success
    assert mgr._zones.open["y_b"]

    rejected = _set_zones(mgr, [_zone("z", "nowhere", 0.0, 1.4), _zone("z", "", 0.0, 1.4)])
    assert not rejected.success
    assert "nowhere" in rejected.message and "'z'" in rejected.message
    assert sorted(mgr._zones.zones) == ["y_a", "y_b"]

    assert mgr._reset_callback(ResetSimulation.Request(soft=True), ResetSimulation.Response()).success
    assert sorted(mgr._zones.zones) == ["y_a", "y_b"]

    assert _set_zones(mgr, []).success
    assert mgr._zones.zones == {}

    assert _set_zones(mgr, _zone_pair("y")).success
    assert mgr._reset_callback(ResetSimulation.Request(soft=False), ResetSimulation.Response()).success
    assert mgr._zones.zones == {}


def test_teleport_agents_rejects_unknown_ids_and_moves_nobody(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_teleport", "navmesh"), node_name="test_zone_teleport")
    aid = _rider(mgr)
    i = mgr._pool.idx(aid)
    before = mgr._pool.pos[i].copy()

    response = _teleport(mgr, [aid, 999], [(2.0, 2.0), (3.0, 3.0)])
    assert not response.success
    assert "999" in response.message
    assert (mgr._pool.pos[i] == before).all()
    assert not _teleport(mgr, [aid], []).success
    assert _teleport(mgr, [aid], [(2.0, 2.0)]).success
    assert tuple(mgr._pool.pos[mgr._pool.idx(aid)]) == (2.0, 2.0)


def _go_to_scenario(tmp_path: Path, planner: str) -> ScenarioConfig:
    walls = [{"name": w.name, "start": {"x": w.start.x, "y": w.start.y}, "end": {"x": w.end.x, "y": w.end.y}} for w in _room("a", 0.0) + _room("b", ROOM_B_X)]
    raw = {
        "name": f"zone_go_to_{planner}",
        "simulation": {"seed": 42, "dt": 0.05, "bt_tick_interval": 1},
        "modules": {"global_planner": planner},
        "walls": walls,
        "agent_types": {
            "go_to_walker": {
                "extends": "adult",
                "mode": "behavior_tree",
                "initial_sequence": "walk",
                "sequences": {"walk": {"steps": {"to_other_room": {"kind": "go_to", "target_pose": {"x": SINK.x, "y": SINK.y}, "patience": {"mean": 120.0, "std": 0.0, "clip_low": 120.0, "clip_high": 120.0}}}}},
            },
        },
        "agents": [{"agent_id": 1, "agent_type": "go_to_walker", "spawn_pose": {"x": 1.0, "y": 3.0, "theta": 0.0}, "desired_velocity": 1.2}],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_scenario(str(path))


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_behavior_tree_go_to_rider_is_teleported_and_reaches_target_in_other_room(manager_factory: Callable[..., AgentManager], tmp_path: Path, planner: str) -> None:
    mgr = manager_factory(_go_to_scenario(tmp_path, planner), node_name=f"test_zone_go_to_{planner}", extra_params=PLANNERS[planner])
    assert _set_zones(mgr, _zone_pair("pair")).success
    bt = mgr._behavior_trees[1]
    pool = mgr._pool

    _tick_until(mgr, lambda: 1 in mgr._zones.transfers and mgr._zones.transfers[1].phase == Phase.WAIT, 300)
    assert mgr._zones.transfers[1].zone == "pair_a"
    assert _set_access(mgr, "pair_a", True).success
    _tick_until(mgr, lambda: mgr._zones.transfers[1].phase == Phase.HOLD, 200)
    i = pool.idx(1)
    assert _teleport(mgr, [1], [(float(pool.pos[i, 0]) + ROOM_B_X, float(pool.pos[i, 1]))]).success

    _tick_until(mgr, lambda: bt.root.status == py_trees.common.Status.SUCCESS, 600)
    assert 1 not in mgr._zones.transfers
    pose = mgr._agents[1].state.pose
    assert math.hypot(pose.x - SINK.x, pose.y - SINK.y) <= DISTANCE_TOLERANCE


def test_robots_with_a_goal_in_the_other_room_never_transfer(manager_factory: Callable[..., AgentManager]) -> None:
    goal = [Pose2D(x=SINK.x, y=SINK.y)]
    agents = [
        AgentConfig(agent_id=-1, kind=1, policy="sfm", agent_type="robot", spawn_pose=Pose2D(x=1.0, y=1.5), goal_sequence=goal, desired_velocity=1.0),
        AgentConfig(agent_id=-2, kind=1, policy="", agent_type="robot", spawn_pose=Pose2D(x=1.0, y=4.5), goal_sequence=goal, desired_velocity=1.0),
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=goal, desired_velocity=1.2),
    ]
    scenario = attrs.evolve(_scenario("zone_robots", "navmesh"), flow=None, agents=agents)
    mgr = manager_factory(scenario, node_name="test_zone_robots")
    assert _set_zones(mgr, _zone_pair("pair")).success
    assert {-1, -2, 1} <= set(mgr._agents)

    phases: set[Phase] = set()
    for _ in range(300):
        mgr.tick()
        assert not {-1, -2} & mgr._zones.transfers.keys()
        assert all(mgr._high_level_cmds[r].target_pose.x == SINK.x for r in (-1, -2))
        if 1 in mgr._zones.transfers:
            phases.add(mgr._zones.transfers[1].phase)
    assert Phase.WAIT in phases


def _robot_in_zone_manager(manager_factory: Callable[..., AgentManager], name: str) -> AgentManager:
    center = Pose2D(x=5.6, y=3.0)
    agents = [
        AgentConfig(agent_id=-1, kind=1, policy="", agent_type="robot", agent_radius=0.267, spawn_pose=center, goal_sequence=[center], desired_velocity=1.0),
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=[Pose2D(x=SINK.x, y=SINK.y)], desired_velocity=1.2, agent_radius=0.25),
    ]
    mgr = manager_factory(attrs.evolve(_scenario(name, "navmesh"), flow=None, agents=agents), node_name=f"test_{name}")
    zones = [_zone("big_a", "big_b", 0.0, 3.0, half=1.0), _zone("big_b", "big_a", ROOM_B_X, 3.0, half=1.0)]
    assert _set_zones(mgr, zones).success
    return mgr


def test_rider_waits_outside_while_a_robot_in_the_zone_covers_every_spot(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _robot_in_zone_manager(manager_factory, "zone_robot_inside")
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: 1 in transfers and transfers[1].phase == Phase.WAIT, 300)
    assert _set_access(mgr, "big_a", True).success
    for _ in range(100):
        mgr.tick()
        assert transfers[1].phase == Phase.WAIT
    i = mgr._pool.idx(1)
    assert math.hypot(mgr._pool.pos[i, 0] - 3.4, mgr._pool.pos[i, 1] - 3.0) <= WAIT_RADIUS + 0.3

    assert _teleport(mgr, [-1], [(5.6 + ROOM_B_X, 3.0)]).success
    _tick_until(mgr, lambda: transfers[1].phase == Phase.HOLD, 300)


def test_boarding_rider_steps_back_to_waiting_when_a_robot_takes_the_zone(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _robot_in_zone_manager(manager_factory, "zone_robot_enters")
    assert _teleport(mgr, [-1], [(5.6 + ROOM_B_X, 3.0)]).success
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: 1 in transfers and transfers[1].phase == Phase.WAIT, 300)
    assert _set_access(mgr, "big_a", True).success
    _tick_until(mgr, lambda: transfers[1].phase == Phase.BOARD, 50)

    assert _teleport(mgr, [-1], [(5.6, 3.0)]).success
    mgr.tick()
    assert transfers[1].phase == Phase.WAIT
    assert transfers[1].slot == -1


def _alighting_manager(manager_factory: Callable[..., AgentManager], name: str, let_out: float) -> AgentManager:
    agents = [
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=[Pose2D(x=SINK.x, y=SINK.y)], desired_velocity=1.2, agent_radius=0.25),
        AgentConfig(agent_id=2, spawn_pose=Pose2D(x=1.0, y=1.0), goal_sequence=[Pose2D(x=1.0, y=1.0)], desired_velocity=1.2, agent_radius=0.25),
    ]
    params = [Parameter("zone_let_out_timeout", Parameter.Type.DOUBLE, let_out)]
    mgr = manager_factory(attrs.evolve(_scenario(name, "navmesh"), flow=None, agents=agents), node_name=f"test_{name}", extra_params=params)
    zones = [_zone("big_a", "big_b", 0.0, 3.0, half=1.0), _zone("big_b", "big_a", ROOM_B_X, 3.0, half=1.0)]
    assert _set_zones(mgr, zones).success
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: 1 in transfers and transfers[1].phase == Phase.WAIT, 300)
    assert _teleport(mgr, [2], [(5.6, 3.0)]).success
    assert _set_access(mgr, "big_a", True).success
    return mgr


def _inside(mgr: AgentManager, zone: str, aid: int) -> bool:
    return mgr._zones.zones[zone].contains(*_pos(mgr, aid))


def test_waiting_rider_boards_only_once_the_arrived_pedestrian_has_left_the_zone(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _alighting_manager(manager_factory, "zone_let_out", 10.0)
    transfers = mgr._zones.transfers
    for _ in range(200):
        if transfers[1].phase == Phase.BOARD:
            break
        assert transfers[1].phase == Phase.WAIT
        mgr.tick()
    else:
        raise AssertionError("rider never boarded")
    assert not _inside(mgr, "big_a", 2)


def test_zero_let_out_timeout_boards_while_the_arrived_pedestrian_is_still_inside(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _alighting_manager(manager_factory, "zone_no_let_out", 0.0)
    mgr.tick()
    assert mgr._zones.transfers[1].phase == Phase.BOARD
    assert _inside(mgr, "big_a", 2)


def test_waiting_rider_boards_past_a_robot_that_stays_in_the_zone_after_the_let_out_timeout(manager_factory: Callable[..., AgentManager]) -> None:
    center = Pose2D(x=5.6, y=3.0)
    agents = [
        AgentConfig(agent_id=-1, kind=1, policy="", agent_type="robot", agent_radius=0.1, spawn_pose=center, goal_sequence=[center], desired_velocity=1.0),
        AgentConfig(agent_id=1, spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=[Pose2D(x=SINK.x, y=SINK.y)], desired_velocity=1.2, agent_radius=0.25),
    ]
    params = [Parameter("zone_let_out_timeout", Parameter.Type.DOUBLE, 1.0)]
    mgr = manager_factory(attrs.evolve(_scenario("zone_let_out_robot", "navmesh"), flow=None, agents=agents), node_name="test_zone_let_out_robot", extra_params=params)
    assert _set_zones(mgr, [_zone("big_a", "big_b", 0.0, 3.0, half=1.0), _zone("big_b", "big_a", ROOM_B_X, 3.0, half=1.0)]).success
    transfers = mgr._zones.transfers
    _tick_until(mgr, lambda: 1 in transfers and transfers[1].phase == Phase.WAIT, 300)
    assert _set_access(mgr, "big_a", True).success
    assert _tick_until(mgr, lambda: transfers[1].phase == Phase.BOARD, 100) == round(1.0 / 0.05) + 1
    assert _inside(mgr, "big_a", -1)


def test_teleport_zones_from_scenario_yaml_reach_the_manager_open_with_zero_delay(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    walls = [{"name": w.name, "start": {"x": w.start.x, "y": w.start.y}, "end": {"x": w.end.x, "y": w.end.y}} for w in _room("a", 0.0) + _room("b", ROOM_B_X)]
    footprint = [{"x": 4.6, "y": 4.0}, {"x": 5.8, "y": 4.0}, {"x": 5.8, "y": 5.2}, {"x": 4.6, "y": 5.2}]
    raw = {
        "name": "zone_yaml",
        "simulation": {"seed": 3, "dt": 0.05},
        "modules": {"global_planner": "navmesh"},
        "walls": walls,
        "teleport_zones": [
            {"name": "up", "destinations": ["down"], "entry": {"x": 4.0, "y": 4.6}, "wait": {"x": 3.4, "y": 4.6}, "hold": {"x": 5.2, "y": 4.6, "theta": 1.5}, "footprint": footprint, "kind": "escalator", "belt": 0.5},
            {"name": "down", "entry": {"x": ROOM_B_X + 4.0, "y": 4.6}, "wait": {"x": ROOM_B_X + 3.4, "y": 4.6}, "hold": {"x": ROOM_B_X + 5.2, "y": 4.6}, "open": False, "delay": -1.0},
        ],
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    mgr = manager_factory(load_scenario(str(path)), node_name="test_zone_yaml")

    zones = mgr._zones.zones
    assert sorted(zones) == ["down", "up"]
    assert mgr._zones.open == {"up": True, "down": False}
    assert (zones["up"].delay, zones["down"].delay) == (0.0, -1.0)
    assert (zones["up"].destinations, zones["down"].destinations) == (("down",), ())
    assert zones["up"].entry == Pose2D(x=4.0, y=4.6)
    assert zones["up"].hold == Pose2D(x=5.2, y=4.6, theta=1.5)
    assert zones["up"].footprint == ((4.6, 4.0), (5.8, 4.0), (5.8, 5.2), (4.6, 5.2))
    assert zones["down"].footprint == ()
    assert (zones["up"].kind, zones["up"].belt, zones["down"].kind, zones["down"].belt) == ("escalator", 0.5, "", 0.0)


def test_set_teleport_zones_rejects_an_unknown_kind_and_a_stair_without_a_footprint(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(_scenario("zone_kinds", "navmesh"), node_name="test_zone_kinds")
    ladder = _zone("ladder", "", 0.0, 1.4)
    ladder.kind = "ladder"
    bare = _zone("bare", "", 0.0, 4.6)
    bare.kind = "stairs"
    bare.footprint = []
    rejected = _set_zones(mgr, [ladder, bare])
    assert not rejected.success
    assert "'ladder'" in rejected.message and "'bare'" in rejected.message
    assert mgr._zones.zones == {}


STAIRS_WALLS = [
    WallConfig(name="stair_a", start=Pose2D(x=2.5, y=4.4), end=Pose2D(x=ROOM, y=4.4)),
    WallConfig(name="stair_b", start=Pose2D(x=ROOM_B_X, y=4.4), end=Pose2D(x=ROOM_B_X + 3.5, y=4.4)),
]


def _stair_half(name: str, destination: str, kind: str, belt: float, b_side: bool, is_open: bool = True) -> TeleportZone:
    z = TeleportZone()
    z.name = name
    z.destinations = [destination] if destination else []
    if b_side:
        z.entry = Pose2DMsg(x=ROOM_B_X + 3.5, y=5.2, theta=math.pi)
        z.wait = Pose2DMsg(x=ROOM_B_X + 4.3, y=5.2, theta=math.pi)
        z.hold = Pose2DMsg(x=ROOM_B_X + 0.45, y=5.2, theta=math.pi)
        corners = ((ROOM_B_X, 4.4), (ROOM_B_X + 3.5, 4.4), (ROOM_B_X + 3.5, ROOM), (ROOM_B_X, ROOM))
    else:
        z.entry = Pose2DMsg(x=2.5, y=5.2, theta=0.0)
        z.wait = Pose2DMsg(x=1.7, y=5.2, theta=0.0)
        z.hold = Pose2DMsg(x=ROOM - 0.45, y=5.2, theta=0.0)
        corners = ((2.5, 4.4), (ROOM, 4.4), (ROOM, ROOM), (2.5, ROOM))
    z.footprint = [Point32(x=x, y=y, z=0.0) for x, y in corners]
    z.open = is_open
    z.delay = 0.0
    z.kind = kind
    z.belt = belt
    return z


def _stair_manager(manager_factory: Callable[..., AgentManager], name: str, planner: str, zones: list[TeleportZone], agents: list[AgentConfig] | None = None) -> AgentManager:
    scenario = _scenario(name, planner)
    scenario = attrs.evolve(scenario, walls=scenario.walls + STAIRS_WALLS)
    if agents is not None:
        scenario = attrs.evolve(scenario, flow=None, agents=agents)
    mgr = manager_factory(scenario, node_name=f"test_{name}", extra_params=PLANNERS[planner])
    assert _set_zones(mgr, zones).success
    return mgr


def _ride(mgr: AgentManager, aid: int, on_stair: Callable[[int], None], limit: int = 600) -> tuple[tuple[float, float], tuple[float, float, float]]:
    """Tick until the rider lands on the other level, return its last pose on the stair and its first one after."""
    zone = mgr._zones.zones["st_a"]
    last = (0.0, 0.0)
    for _ in range(limit):
        i = mgr._pool.idx(aid)
        x, y = float(mgr._pool.pos[i, 0]), float(mgr._pool.pos[i, 1])
        if x > ROOM_B_X:
            return last, (x, y, float(mgr._pool.theta[i]))
        last = (x, y)
        if zone.contains(x, y):
            on_stair(i)
        mgr.tick()
    raise AssertionError("rider never crossed the stair")


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_pedestrian_walks_a_stair_stair_capped_lands_past_the_partner_seam_and_reaches_the_sink(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    zones = [_stair_half("st_a", "st_b", "stairs", 0.0, b_side=False), _stair_half("st_b", "st_a", "stairs", 0.0, b_side=True)]
    mgr = _stair_manager(manager_factory, f"stairs_walk_{planner}", planner, zones)
    aid = _rider(mgr)
    pool = mgr._pool
    _tick_until(mgr, lambda: aid in mgr._zones.transfers, 40)
    assert mgr._zones.transfers[aid].zone == "st_a"
    speeds: list[float] = []

    def on_stairs(i: int) -> None:
        speeds.append(math.hypot(*pool.vel[i]))

    (x, y), (lx, ly, theta) = _ride(mgr, aid, on_stairs)
    assert len(speeds) > 10
    assert max(speeds[1:]) <= STAIR_SPEED + 1e-9
    assert x >= ROOM - 0.45 - 0.2
    assert lx == pytest.approx(ROOM_B_X + 0.45 + LANDING_BACKOFF)
    assert ly == pytest.approx(y, abs=0.05)
    assert abs(math.remainder(theta, math.tau)) < 0.5
    assert aid not in mgr._zones.transfers

    _tick_until(mgr, lambda: aid not in mgr._agents, 400)


def test_escalator_rider_stands_on_the_belt_is_carried_over_both_halves_and_reports_idle(manager_factory: Callable[..., AgentManager]) -> None:
    zones = [_stair_half("st_a", "st_b", "escalator", 0.6, b_side=False), _stair_half("st_b", "", "escalator", -0.6, b_side=True)]
    mgr = _stair_manager(manager_factory, "escalator_ride", "navmesh", zones)
    aid = _rider(mgr)
    pool = mgr._pool
    carried: list[int] = []

    def on_belt(i: int) -> None:
        carried.append(i)
        if len(carried) > 1:
            assert pool.vel[i] == pytest.approx((0.6, 0.0))
            assert pool.desired_vel[i] == 0.0
            assert pool.animation_state[i] == AnimationType.IDLE

    _ride(mgr, aid, on_belt)
    assert len(carried) > 10
    exit_half = mgr._zones.zones["st_b"]
    on_exit = 0
    for _ in range(100):
        mgr.tick()
        if aid not in mgr._agents:
            break
        j = pool.idx(aid)
        if exit_half.contains(*pool.pos[j]):
            on_exit += 1
            assert pool.vel[j] == pytest.approx((0.6, 0.0))
            assert pool.animation_state[j] == AnimationType.IDLE
    assert on_exit > 0
    _tick_until(mgr, lambda: aid not in mgr._agents, 400)


def test_manual_wheelchair_never_boards_a_stair_stair_and_hard_reset_clears_stair_transfers(manager_factory: Callable[..., AgentManager]) -> None:
    goal = [Pose2D(x=SINK.x, y=SINK.y)]
    agents = [
        AgentConfig(agent_id=1, agent_type="wheelchair_manual", spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=goal, desired_velocity=0.9),
        AgentConfig(agent_id=2, spawn_pose=Pose2D(x=1.0, y=2.0), goal_sequence=goal, desired_velocity=1.2),
    ]
    zones = [_stair_half("st_a", "st_b", "stairs", 0.0, b_side=False), _stair_half("st_b", "st_a", "stairs", 0.0, b_side=True)]
    mgr = _stair_manager(manager_factory, "stairs_wheelchair", "navmesh", zones, agents)
    assert mgr._agents[1].params.climbs == ()
    stair = mgr._zones.zones["st_a"]
    for _ in range(200):
        mgr.tick()
        assert 1 not in mgr._zones.transfers
        assert not stair.contains(*mgr._pool.pos[mgr._pool.idx(1)])
        if 2 in mgr._zones.transfers:
            break
    assert mgr._zones.transfers[2].zone == "st_a"
    assert mgr._reset_callback(ResetSimulation.Request(soft=False), ResetSimulation.Response()).success
    assert not mgr._zones.transfers
    assert mgr._zones.zones == {}


SEAM_X = 13.0
LANE = (9.5, 16.5, 4.4, ROOM)


def _floor_walls(way_round: bool) -> list[WallConfig]:
    """One floor of 26 by 6 m split by a wall along x = 13, open only in a walled lane along its top wall and, with a way round, over 1.4 m at the bottom."""
    x0, x1, y0, _ = LANE
    segments = [
        ((0.0, 0.0), (26.0, 0.0)),
        ((26.0, 0.0), (26.0, ROOM)),
        ((26.0, ROOM), (0.0, ROOM)),
        ((0.0, ROOM), (0.0, 0.0)),
        ((SEAM_X, 1.4 if way_round else 0.0), (SEAM_X, y0)),
        ((x0, y0), (x1, y0)),
    ]
    return [WallConfig(name=f"floor{i}", start=Pose2D(x=ax, y=ay), end=Pose2D(x=bx, y=by)) for i, ((ax, ay), (bx, by)) in enumerate(segments)]


def _floor_half(name: str, destination: str, kind: str, belt: float, top: bool) -> TeleportZone:
    """A half of a stair on one floor as Arena sends it: the footprint up to the seam, the hold pose just past it."""
    x0, x1, y0, y1 = LANE
    y = (y0 + y1) / 2.0
    z = TeleportZone()
    z.name = name
    z.destinations = [destination] if destination else []
    if top:
        z.entry = Pose2DMsg(x=x1, y=y, theta=math.pi)
        z.wait = Pose2DMsg(x=x1 + 0.8, y=y, theta=math.pi)
        z.hold = Pose2DMsg(x=SEAM_X - 0.05, y=y, theta=math.pi)
        corners = ((SEAM_X, y0), (x1, y0), (x1, y1), (SEAM_X, y1))
    else:
        z.entry = Pose2DMsg(x=x0, y=y, theta=0.0)
        z.wait = Pose2DMsg(x=x0 - 0.8, y=y, theta=0.0)
        z.hold = Pose2DMsg(x=SEAM_X + 0.05, y=y, theta=0.0)
        corners = ((x0, y0), (SEAM_X, y0), (SEAM_X, y1), (x0, y1))
    z.footprint = [Point32(x=x, y=y, z=0.0) for x, y in corners]
    z.open = True
    z.delay = 0.0
    z.kind = kind
    z.belt = belt
    return z


def _floor_manager(manager_factory: Callable[..., AgentManager], name: str, planner: str, zones: list[TeleportZone], agents: list[AgentConfig] | None = None, way_round: bool = False) -> AgentManager:
    scenario = attrs.evolve(_scenario(name, planner), walls=_floor_walls(way_round))
    if agents is not None:
        scenario = attrs.evolve(scenario, flow=None, agents=agents)
    mgr = manager_factory(scenario, node_name=f"test_{name}", extra_params=PLANNERS[planner])
    assert _set_zones(mgr, zones).success
    return mgr


def _cross(mgr: AgentManager, aid: int, limit: int = 1500) -> tuple[float, dict[str, list[tuple[float, float]]]]:
    """Tick until the pedestrian is gone, return its longest step and its velocities on each stair half."""
    pool = mgr._pool
    longest, on = 0.0, {name: [] for name in mgr._zones.zones}
    last = None
    for _ in range(limit):
        if aid not in mgr._agents:
            return longest, on
        i = pool.idx(aid)
        x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
        if last is not None:
            longest = max(longest, math.hypot(x - last[0], y - last[1]))
        last = (x, y)
        for name, zone in mgr._zones.zones.items():
            if zone.contains(x, y):
                on[name].append((float(pool.vel[i, 0]), float(pool.vel[i, 1])))
        mgr.tick()
    raise AssertionError("pedestrian never reached the sink")


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_pedestrian_walks_a_stair_on_its_own_floor_stair_capped_without_a_teleport_and_reaches_the_sink(manager_factory: Callable[..., AgentManager], planner: str) -> None:
    zones = [_floor_half("fl_b", "fl_t", "stairs", 0.0, top=False), _floor_half("fl_t", "fl_b", "stairs", 0.0, top=True)]
    mgr = _floor_manager(manager_factory, f"floor_walk_{planner}", planner, zones)
    assert mgr._zones.gates == [((SEAM_X, LANE[2]), (SEAM_X, LANE[3]))]
    aid = _rider(mgr)
    _tick_until(mgr, lambda: aid in mgr._zones.transfers, 40)
    assert mgr._zones.transfers[aid].zone == "fl_b"
    longest, on = _cross(mgr, aid)
    assert longest < 0.2
    assert len(on["fl_b"]) > 10 and len(on["fl_t"]) > 10
    assert max(math.hypot(*vel) for half in on.values() for vel in half[1:]) <= STAIR_SPEED + 1e-9


def test_escalator_on_one_floor_carries_its_rider_over_both_halves_without_a_teleport(manager_factory: Callable[..., AgentManager]) -> None:
    zones = [_floor_half("fl_b", "fl_t", "escalator", 0.6, top=False), _floor_half("fl_t", "", "escalator", -0.6, top=True)]
    mgr = _floor_manager(manager_factory, "floor_escalator", "navmesh", zones)
    aid = _rider(mgr)
    longest, on = _cross(mgr, aid)
    assert longest < 0.2
    assert len(on["fl_b"]) > 10 and len(on["fl_t"]) > 10
    assert all(vel == pytest.approx((0.6, 0.0)) for half in on.values() for vel in half[2:])


@pytest.mark.parametrize("way_round", [False, True])
def test_manual_wheelchair_never_stands_on_a_stair_on_its_own_floor_and_takes_the_way_round(manager_factory: Callable[..., AgentManager], way_round: bool) -> None:
    goal = [Pose2D(x=SINK.x, y=SINK.y)]
    agents = [
        AgentConfig(agent_id=1, agent_type="wheelchair_manual", spawn_pose=Pose2D(x=1.0, y=3.0), goal_sequence=goal, desired_velocity=0.9),
        AgentConfig(agent_id=2, spawn_pose=Pose2D(x=1.0, y=5.2), goal_sequence=goal, desired_velocity=1.2),
    ]
    zones = [_floor_half("fl_b", "fl_t", "escalator", 0.6, top=False), _floor_half("fl_t", "", "escalator", -0.6, top=True)]
    mgr = _floor_manager(manager_factory, f"floor_wheelchair_{way_round}", "navmesh", zones, agents, way_round)
    pool = mgr._pool
    rode = False
    for _ in range(900):
        mgr.tick()
        assert 1 not in mgr._zones.transfers
        x, y = (float(v) for v in pool.pos[pool.idx(1)])
        assert not any(zone.contains(x, y) for zone in mgr._zones.zones.values())
        rode = rode or (2 in mgr._zones.transfers and mgr._zones.transfers[2].zone == "fl_b")
    assert rode or way_round
    assert (float(pool.pos[pool.idx(1), 0]) > SEAM_X) == way_round


def _two_way(riders: int) -> ScenarioConfig:
    """Riders released in each room with their sink in the other one."""
    scenario = _scenario("stairs_two_way", "navmesh", riders)
    [source], [sink] = scenario.flow.sources, scenario.flow.sinks
    back = attrs.evolve(source, pose=Pose2D(x=ROOM_B_X + 5.0, y=3.0), agent_template=attrs.evolve(source.agent_template, sink_affinity=[SinkAffinityModel(sink_idx=1, weight=1.0)]))
    flow = attrs.evolve(scenario.flow, sources=[source, back], sinks=[sink, attrs.evolve(sink, pose=Pose2D(x=1.0, y=1.0))])
    return attrs.evolve(scenario, walls=scenario.walls + STAIRS_WALLS, flow=flow)


def test_two_way_traffic_over_one_stair_gets_everyone_across_once(manager_factory: Callable[..., AgentManager]) -> None:
    riders = 8
    mgr = manager_factory(_two_way(riders), node_name="test_stairs_two_way")
    assert _set_zones(mgr, [_stair_half("st_a", "st_b", "stairs", 0.0, b_side=False), _stair_half("st_b", "st_a", "stairs", 0.0, b_side=True)]).success
    pool = mgr._pool
    last: dict[int, float] = {}
    crossings: dict[int, int] = {}

    def everyone_arrived() -> bool:
        for i in range(pool.n):
            aid, x = int(pool.agent_ids[i]), float(pool.pos[i, 0])
            if abs(x - last.setdefault(aid, x)) > 5.0:
                crossings[aid] = crossings.get(aid, 0) + 1
            last[aid] = x
        return len(last) == 2 * riders and not mgr._agents

    _tick_until(mgr, everyone_arrived, 4000)
    assert crossings == dict.fromkeys(last, 1)
