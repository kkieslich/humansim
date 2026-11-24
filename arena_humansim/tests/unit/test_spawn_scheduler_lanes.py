from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.spawn_scheduler import SpawnScheduler
from arena_humansim.utils.geometry import point_in_polygon
from arena_humansim.utils.lanes import resolve_clearance, resolve_lane
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import (
    AgentTemplate,
    LaneConfig,
    LaneSide,
    Pose2D,
    RateKeyframe,
    Shape,
    ShapeType,
    SinkAffinity,
    SinkConfig,
    SourceConfig,
    SourceType,
    SpawnRequest,
    WaypointMode,
)

from tests._lane_layouts import corridor, straight

CLEARANCE = resolve_clearance(0.38, 0.2)
END_SINK = SinkConfig(
    name="end",
    pose=Pose2D(x=0.0, y=0.0),
    shape=Shape(
        type=ShapeType.POLYGON,
        vertices=[Pose2D(x=18.5, y=-2.0), Pose2D(x=20.5, y=-2.0), Pose2D(x=20.5, y=2.0), Pose2D(x=18.5, y=2.0)],
    ),
)


def lane_source(*, sinks: tuple[str, ...] = ("end",), side: LaneSide = LaneSide.RIGHT) -> SourceConfig:
    return SourceConfig(
        name="lane",
        pose=Pose2D(x=0.0, y=0.0),
        shape=Shape(
            type=ShapeType.POLYGON,
            vertices=[Pose2D(x=-0.5, y=-2.0), Pose2D(x=1.0, y=-2.0), Pose2D(x=1.0, y=2.0), Pose2D(x=-0.5, y=2.0)],
        ),
        type=SourceType.POISSON,
        rate_profile=[RateKeyframe(t=0.0, rate=1.0)],
        agent=AgentTemplate(
            desired_velocity_min=1.0,
            desired_velocity_max=1.2,
            agent_radius=0.25,
            sink_affinity=[SinkAffinity(sink_name=name, weight=1.0) for name in sinks],
        ),
        lane_config=LaneConfig(centerlines=straight(20.0), side=side),
    )


def resolved(source: SourceConfig) -> SourceConfig:
    source.lane = resolve_lane(source, corridor(4.0, 20.0), {"end": END_SINK}, CLEARANCE)
    return source


def spawn_request(rng: RNG, source: SourceConfig, seq: int) -> SpawnRequest:
    scheduler = SpawnScheduler(rng)
    scheduler.set_sinks({"end": END_SINK})
    scheduler.add_source(source)
    return scheduler._sample_spawn_request(source, tick_count=0, seq=seq)


def test_lane_spawn_track_stays_in_the_band_and_ends_in_the_sink(rng: RNG) -> None:
    source = resolved(lane_source())
    lane = source.lane
    assert lane is not None
    request = spawn_request(rng, source, seq=0)
    points = request.waypoints
    assert request.waypoint_mode is WaypointMode.ONCE
    assert request.waypoint_radii == [lane.settings.arrival_radius] * len(points)
    assert (request.pose.x, request.pose.y) == pytest.approx((points[0].x, points[0].y))
    assert request.pose.theta == pytest.approx(0.0)
    assert points[-1].x == pytest.approx(20.0)
    assert point_in_polygon(points[-1].x, points[-1].y, END_SINK.shape.vertices)
    for point in points:
        center = float(np.interp(point.x, lane.s, lane.center))
        half = float(np.interp(point.x, lane.s, lane.half_width))
        assert center - half - 1e-9 <= point.y <= center + half + 1e-9


def test_lane_flow_with_two_sink_affinities_is_rejected() -> None:
    source = lane_source(sinks=("end", "other"))
    with pytest.raises(ValueError, match="exactly one sink affinity"):
        SpawnScheduler.validate(source)


def test_lane_spawns_from_one_seed_repeat_their_track() -> None:
    source = resolved(lane_source())
    first = spawn_request(RNG(42), source, seq=3).waypoints
    second = spawn_request(RNG(42), source, seq=3).waypoints
    assert first == second


def test_two_lane_spawns_get_different_lateral_positions(rng: RNG) -> None:
    source = resolved(lane_source())
    first = spawn_request(rng, source, seq=0).waypoints
    second = spawn_request(rng, source, seq=1).waypoints
    assert first[0].y != second[0].y
