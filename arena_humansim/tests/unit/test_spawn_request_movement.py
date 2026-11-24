from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.spawn_scheduler import SpawnScheduler
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentTemplate,
    Pose2D,
    Shape,
    ShapeType,
    SinkAffinity,
    SinkConfig,
    SourceConfig,
    SpawnRequest,
    WaypointMode,
)


def test_spawn_request_movement_keeps_radii_and_once_mode() -> None:
    request = SpawnRequest(
        waypoints=[Pose2D(x=1.0, y=0.0), Pose2D(x=2.0, y=0.0)],
        waypoint_radii=[0.7, 0.4],
        waypoint_mode=WaypointMode.ONCE,
    )
    movement = request.waypoint_movement()
    assert movement.radii == [0.7, 0.4]
    assert movement.mode is WaypointMode.ONCE
    assert [p.x for p in movement.waypoints] == [1.0, 2.0]


def test_sink_spawn_request_has_no_radii_and_repeat_mode() -> None:
    scheduler = SpawnScheduler(RNG(42))
    scheduler.set_sinks({"end": SinkConfig(name="end", pose=Pose2D(x=10.0, y=0.0))})
    source = SourceConfig(
        name="start",
        pose=Pose2D(x=0.0, y=0.0),
        shape=Shape(type=ShapeType.CIRCLE, radius=0.5),
        agent=AgentTemplate(
            desired_velocity_min=1.0,
            desired_velocity_max=1.2,
            agent_radius=0.25,
            sink_affinity=[SinkAffinity(sink_name="end", weight=1.0)],
        ),
    )
    scheduler.add_source(source)
    request = scheduler._sample_spawn_request(source, tick_count=0, seq=0)
    assert request.waypoint_radii == []
    assert request.waypoint_mode is WaypointMode.REPEAT
