from __future__ import annotations

from collections.abc import Sequence

from arena_humansim_msgs.msg import AgentTemplate, LaneConfig, LaneSegment, RateKeyframe, Shape, SinkAffinity, SinkConfig, SourceConfig
from arena_humansim_msgs.srv import AddWalls
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg

CORRIDOR_WALLS = [("north", (0.0, 2.0), (20.0, 2.0)), ("south", (0.0, -2.0), (20.0, -2.0))]
DEFAULT_CENTERLINE = [(0.0, 0.0), (20.0, 0.0)]


def walls_request(walls: Sequence[tuple[str, tuple[float, float], tuple[float, float]]]) -> AddWalls.Request:
    request = AddWalls.Request()
    for name, start, end in walls:
        request.names.append(name)
        request.starts.append(Point32(x=float(start[0]), y=float(start[1]), z=0.0))
        request.ends.append(Point32(x=float(end[0]), y=float(end[1]), z=0.0))
    return request


def end_sink_msg(name: str = "end") -> SinkConfig:
    sink = SinkConfig()
    sink.name = name
    sink.pose = Pose2DMsg(x=0.0, y=0.0, theta=0.0)
    sink.shape = _polygon([(18.5, -2.0), (20.5, -2.0), (20.5, 2.0), (18.5, 2.0)])
    sink.absorption_radius = 0.5
    sink.capacity = -1
    return sink


def lane_source_msg(
    name: str,
    *,
    side: str = "right",
    segments: Sequence[Sequence[tuple[float, float]]] = (DEFAULT_CENTERLINE,),
    sink: str = "end",
) -> SourceConfig:
    source = SourceConfig()
    source.name = name
    source.pose = Pose2DMsg(x=0.0, y=0.0, theta=0.0)
    source.shape = _polygon([(-0.5, -2.0), (1.0, -2.0), (1.0, 2.0), (-0.5, 2.0)])
    rate = RateKeyframe()
    rate.t = 0.0
    rate.rate = 1.0
    source.rate_profile.append(rate)
    source.agent = AgentTemplate()
    source.agent.desired_velocity_min = 1.0
    source.agent.desired_velocity_max = 1.2
    source.agent.agent_radius = 0.25
    source.agent.agent_type = "adult"
    affinity = SinkAffinity()
    affinity.sink_name = sink
    affinity.weight = 1.0
    source.agent.sink_affinity.append(affinity)
    source.lane = _lane(side, segments)
    return source


def _lane(side: str, segments: Sequence[Sequence[tuple[float, float]]]) -> LaneConfig:
    lane = LaneConfig()
    lane.side = side
    for centerline in segments:
        segment = LaneSegment()
        segment.centerline = [Pose2DMsg(x=float(x), y=float(y), theta=0.0) for x, y in centerline]
        lane.segments.append(segment)
    return lane


def _polygon(vertices: Sequence[tuple[float, float]]) -> Shape:
    shape = Shape()
    shape.type = Shape.POLYGON
    shape.vertices = [Point32(x=float(x), y=float(y), z=0.0) for x, y in vertices]
    return shape
