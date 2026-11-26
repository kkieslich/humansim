"""A source of wheelchair users added before the clock builds their walled-off mesh and their local planner there and
then: the first route of the first wheelchair user builds nothing within a tick.

Before, both were built on first use, within the tick that spawned the first wheelchair user: on the walkway world
(smu_link_full_v2__layout_d837888b) the walled-off mesh took 2.6 s, the crowd fell 2.35 s behind its clock and the
episode was flagged. The world is the north stair of the T junction of smu_link_full_v3 (as test_navmesh_stair_halves);
every mesh and router build here sleeps, so one within a tick shows as a tick over its budget."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.agents.types import AgentType
from arena_humansim.global_planner import navmesh
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim_msgs.msg import AgentTemplate, RateKeyframe, Shape, SinkAffinity, SinkConfig, SourceConfig, TeleportZone
from arena_humansim_msgs.srv import AddSink, AddSource, AddWalls, RemoveWalls, ResetSimulation, SetTeleportZones
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg

from tests._lane_messages import walls_request
from tests.unit.test_navmesh_stair_halves import BOX, GATES, HALVES, START, WALLS

#: a wheelchair user's type (as config/agent_types/wheelchair_manual.yaml: no stairs, its own local planner)
WHEELCHAIR = "wheelchair_test"
#: every mesh and router build takes this much longer [s]: more than a tick's budget
SLOW_BUILD = 0.08
GOAL = (119.0, 46.0)
PAIRS = {
    "stairs_t_junction_north_upper_top": "stairs_t_junction_north_upper_bottom",
    "stairs_t_junction_north_upper_bottom": "stairs_t_junction_north_upper_top",
    "stairs_t_junction_north_lower_top": "stairs_t_junction_north_lower_bottom",
    "stairs_t_junction_north_lower_bottom": "stairs_t_junction_north_lower_top",
}


class _Builds:
    """Counts the mesh and router builds, those within a tick apart, each slowed by `SLOW_BUILD`."""

    def __init__(self) -> None:
        self.total = 0
        self.in_tick = 0
        self.ticking = False

    def note(self) -> None:
        self.total += 1
        self.in_tick += self.ticking
        time.sleep(SLOW_BUILD)


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Builds]:
    counter = _Builds()
    build, router = navmesh.Mesh.build.__func__, navmesh.Router  # pyright: ignore[reportFunctionMemberAccess]

    def slow_build(cls: type, *args: object, **kwargs: object) -> object:
        counter.note()
        return build(cls, *args, **kwargs)

    class SlowRouter(router):  # pyright: ignore[reportGeneralTypeIssues, reportUntypedBaseClass]
        def __init__(self, *args: object, **kwargs: object) -> None:
            counter.note()
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(navmesh.Mesh, "build", classmethod(slow_build))
    monkeypatch.setattr(navmesh, "Router", SlowRouter)
    yield counter


def _zone(name: str, footprint: tuple[tuple[float, float], ...]) -> TeleportZone:
    xs, ys = [p[0] for p in footprint], [p[1] for p in footprint]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    zone = TeleportZone()
    zone.name, zone.destinations, zone.kind, zone.open = name, [PAIRS[name]], "stairs", True
    zone.entry = Pose2DMsg(x=cx, y=cy, theta=0.0)
    zone.wait = Pose2DMsg(x=cx, y=cy, theta=0.0)
    zone.hold = Pose2DMsg(x=cx, y=cy, theta=0.0)
    zone.footprint = [Point32(x=x, y=y, z=0.0) for x, y in footprint]
    return zone


def _circle(x: float, y: float, radius: float) -> tuple[Pose2DMsg, Shape]:
    shape = Shape()
    shape.type = Shape.CIRCLE
    shape.radius = radius
    return Pose2DMsg(x=x, y=y, theta=0.0), shape


def _source(agent_type: str) -> SourceConfig:
    source = SourceConfig()
    source.name = f"{agent_type}_source"
    source.pose, source.shape = _circle(START[0], START[1], 0.2)
    rate = RateKeyframe()
    rate.t, rate.rate = 0.0, 20.0
    source.rate_profile.append(rate)
    source.max_concurrent = source.max_total = 1
    source.agent = AgentTemplate()
    source.agent.desired_velocity_min = source.agent.desired_velocity_max = 0.9
    source.agent.agent_type = agent_type
    affinity = SinkAffinity()
    affinity.sink_name, affinity.weight = "west", 1.0
    source.agent.sink_affinity.append(affinity)
    return source


def _sink() -> SinkConfig:
    sink = SinkConfig()
    sink.name = "west"
    sink.pose, sink.shape = _circle(GOAL[0], GOAL[1], 0.5)
    sink.absorption_radius = 0.5
    sink.capacity = -1
    return sink


@pytest.fixture
def stair_world(rclpy_context: object, manager_factory: Callable[..., AgentManager]) -> AgentManager:  # noqa: ARG001
    scenario = ScenarioConfig(name="prepare_walled_off", simulation=SimulationParams(seed=7, dt=0.05), modules=ModuleConfig(global_planner="navmesh"))
    mgr = manager_factory(scenario, node_name="test_prepare_walled_off")
    mgr._agent_types[WHEELCHAIR] = AgentType(name=WHEELCHAIR, climbs=(), local_planner="hsfm", agent_radius=0.325)
    walls = [(f"w{i}", a, b) for i, (a, b) in enumerate([*WALLS, *GATES, *BOX])]
    assert mgr._add_walls_callback(walls_request(walls), AddWalls.Response()).success
    zones = [_zone(name, footprint) for name, (_, footprint) in HALVES.items()]
    assert mgr._set_teleport_zones_callback(SetTeleportZones.Request(zones=zones), SetTeleportZones.Response()).success
    assert mgr._add_sink_callback(AddSink.Request(sink=_sink()), AddSink.Response()).success
    return mgr


def _navmesh(mgr: AgentManager) -> navmesh.NavMeshPlanner:
    planner = mgr._global_planner
    assert isinstance(planner, navmesh.NavMeshPlanner)
    return planner


def _nogo_builds(mgr: AgentManager) -> int:
    return len(_navmesh(mgr).nogo_builds)


def _tick_until_routed(mgr: AgentManager, builds: _Builds, ticks: int = 60) -> float:
    """Tick until a wheelchair user walks its route; the longest tick [s]."""
    longest = 0.0
    for _ in range(ticks):
        builds.ticking = True
        began = time.perf_counter()
        mgr.tick()
        longest = max(longest, time.perf_counter() - began)
        builds.ticking = False
    assert any(a.params.name == WHEELCHAIR for a in mgr._agents.values())
    assert mgr._global_planner.get_cached_paths()
    return longest


def test_a_wheelchair_source_builds_the_walled_off_mesh_before_the_clock(stair_world: AgentManager, builds: _Builds) -> None:
    mgr = stair_world
    assert _nogo_builds(mgr) == 0
    assert mgr._add_source_callback(AddSource.Request(source=_source(WHEELCHAIR)), AddSource.Response()).success
    assert _nogo_builds(mgr) == 1
    assert "hsfm" in mgr._policy_name_to_idx
    longest = _tick_until_routed(mgr, builds)
    assert builds.in_tick == 0
    assert _nogo_builds(mgr) == 1
    assert longest < mgr._dt


def test_without_the_source_the_first_wheelchair_route_builds_within_a_tick(stair_world: AgentManager, builds: _Builds) -> None:
    """The check above can fail: a wheelchair user that comes without its source prepared builds its mesh in the tick."""
    mgr = stair_world
    assert mgr._add_source_callback(AddSource.Request(source=_source(WHEELCHAIR)), AddSource.Response()).success
    mgr._prepared_types.clear()
    _navmesh(mgr).forget_prepared()
    _navmesh(mgr)._nogo.clear()
    longest = _tick_until_routed(mgr, builds)
    assert builds.in_tick >= 1
    assert longest > mgr._dt


def test_the_prepared_world_stays_warm_through_a_soft_reset_and_new_walls(stair_world: AgentManager, builds: _Builds) -> None:
    """The reset's "Parameters applied" keeps the prepared local planner; walls the main mesh is meshed again for (here
    the world loses a wall) build the walled-off mesh again within that call, before the clock."""
    mgr = stair_world
    assert mgr._add_source_callback(AddSource.Request(source=_source(WHEELCHAIR)), AddSource.Response()).success
    mgr._reset_callback(ResetSimulation.Request(soft=True), ResetSimulation.Response())
    assert "hsfm" in mgr._policy_name_to_idx
    assert _nogo_builds(mgr) == 1
    assert mgr._remove_walls_callback(RemoveWalls.Request(names=["w0"]), RemoveWalls.Response()).success
    assert _nogo_builds(mgr) == 2
    longest = _tick_until_routed(mgr, builds)
    assert builds.in_tick == 0
    assert longest < mgr._dt


def test_a_hard_reset_forgets_what_was_prepared(stair_world: AgentManager) -> None:
    mgr = stair_world
    assert mgr._add_source_callback(AddSource.Request(source=_source(WHEELCHAIR)), AddSource.Response()).success
    mgr._reset_callback(ResetSimulation.Request(soft=False), ResetSimulation.Response())
    assert mgr._prepared_types == {}
    assert _navmesh(mgr)._prepared == {}
