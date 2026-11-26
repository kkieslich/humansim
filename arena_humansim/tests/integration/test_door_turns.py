"""Two-way crowds through a narrow door in a wall without thickness (the arena-world walls are lines) take it by turns
and never plug it: adults, wheelchair users and long-cane users from both rooms all come through (navmesh, sfm and the
types' own local planners, study node settings, lane_side left).

Before the door counted as a narrow stretch, the same crowd plugged a 1.0 m door for good within the first minutes."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import (
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
)
from arena_humansim.utils.types import Pose2D
from rclpy.parameter import Parameter

DT = 0.05
SECONDS = 300.0
#: every walker that spawned this long before the end has come through
CLEARS_S = 120.0
#: two rooms of 8 x 8 m on either side of the wall y = 8 with the door at x = 4
TOP, WALL_Y, DOOR_X = 16.0, 8.0, 4.0


def _scenario(width: float, seed: int) -> ScenarioConfig:
    w = width / 2.0
    segs = [((0, 0), (8, 0)), ((8, 0), (8, TOP)), ((8, TOP), (0, TOP)), ((0, TOP), (0, 0)), ((0, WALL_Y), (DOOR_X - w, WALL_Y)), ((DOOR_X + w, WALL_Y), (8, WALL_Y))]
    walls = [WallConfig(name=f"w{i}", start=Pose2D(x=float(a[0]), y=float(a[1])), end=Pose2D(x=float(b[0]), y=float(b[1]))) for i, (a, b) in enumerate(segs)]
    sinks = [
        SinkScenarioConfig(pose=Pose2D(x=DOOR_X, y=TOP - 1.0), shape=ShapeModel(type="circle", radius=0.8), absorption_radius=0.8),
        SinkScenarioConfig(pose=Pose2D(x=DOOR_X, y=1.0), shape=ShapeModel(type="circle", radius=0.8), absorption_radius=0.8),
    ]

    def source(y: float, sink: int, rate: float, kind: str) -> SourceScenarioConfig:
        return SourceScenarioConfig(
            pose=Pose2D(x=DOOR_X, y=y),
            shape=ShapeModel(type="circle", radius=1.5),
            rate_profile=[RateKeyframeModel(t=0.0, rate=rate)],
            agent_template=AgentTemplateModel(
                agent_type=kind,
                desired_velocity_min=1.0 if kind == "adult" else 0.7,
                desired_velocity_max=1.3 if kind == "adult" else 0.9,
                agent_radius=0.25 if kind == "adult" else 0.0,
                sink_affinity=[SinkAffinityModel(sink_idx=sink, weight=1.0)],
            ),
        )

    sources = []
    for kind, rate in (("adult", 0.12), ("wheelchair_manual", 0.03), ("blind_white_cane", 0.03)):
        sources += [source(2.5, 0, rate, kind), source(TOP - 2.5, 1, rate, kind)]
    return ScenarioConfig(
        name="door_turns",
        simulation=SimulationParams(seed=seed, dt=DT, max_ticks=int(SECONDS / DT)),
        modules=ModuleConfig(local_planner="sfm", global_planner="navmesh"),
        walls=walls,
        flow=FlowScenarioConfig(sources=sources, sinks=sinks),
    )


@pytest.mark.slow
@pytest.mark.parametrize("width", [1.0, 1.2])
def test_two_way_crowds_take_a_narrow_door_by_turns_and_all_come_through(width: float, manager_factory: Callable[..., AgentManager]) -> None:
    mgr = manager_factory(
        _scenario(width, seed=1),
        node_name=f"door_turns_{int(width * 10)}",
        extra_params=[
            Parameter("global_planner.lane_side", Parameter.Type.STRING, "left"),
            Parameter("global_planner.stall_time", Parameter.Type.DOUBLE, 8.0),
            Parameter("global_planner.congestion_interval", Parameter.Type.DOUBLE, 5.0),
        ],
    )
    pool = mgr._pool
    born: dict[int, tuple[float, bool, str]] = {}
    gone: dict[int, float] = {}
    for k in range(int(SECONDS / DT)):
        mgr.tick()
        t = (k + 1) * DT
        for aid, p in zip(pool.agent_ids[: pool.n].tolist(), pool.pos[: pool.n].tolist(), strict=True):
            if aid not in born:
                born[aid] = (t, p[1] < WALL_Y, mgr._agents[aid].params.name)
        for aid in born:
            if aid not in gone and aid not in mgr._agents:
                gone[aid] = t
    due = [aid for aid, (t, _, _) in born.items() if t < SECONDS - CLEARS_S]
    late = sorted((born[a][2], "south" if born[a][1] else "north", round(born[a][0])) for a in due if a not in gone)
    assert late == [], f"not through within {CLEARS_S} s: {late}"
    for kind in ("adult", "wheelchair_manual", "blind_white_cane"):
        for south in (True, False):
            assert any(born[a][2] == kind and born[a][1] is south for a in gone), f"no {kind} came through from the {'south' if south else 'north'}"
