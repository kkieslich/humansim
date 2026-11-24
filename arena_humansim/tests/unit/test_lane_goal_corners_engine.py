from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.msg import Shape, SinkConfig, SourceConfig
from arena_humansim_msgs.srv import AddSink, AddSource, AddWalls, RemoveSource
from geometry_msgs.msg import Point32

from tests._lane_messages import end_sink_msg, lane_source_msg, walls_request
from tests.integration._helpers import build_manager

L_WALLS = [
    ("south", (-1.0, -2.0), (11.04, -2.0)),
    ("east", (11.04, -2.0), (11.04, 8.0)),
    ("top", (11.04, 8.0), (7.04, 8.0)),
    ("west", (7.04, 8.0), (7.04, 2.0)),
    ("inner", (7.04, 2.0), (-1.0, 2.0)),
    ("back", (-1.0, 2.0), (-1.0, -2.0)),
]
FORWARD = [[(0.0, 0.0), (9.04, 0.0)], [(9.04, 0.0), (9.04, 6.0)]]
BACKWARD = [[(9.04, 6.0), (9.04, 0.0)], [(9.04, 0.0), (0.0, 0.0)]]


def polygon(vertices: list[tuple[float, float]]) -> Shape:
    shape = Shape()
    shape.type = Shape.POLYGON
    shape.vertices = [Point32(x=x, y=y, z=0.0) for x, y in vertices]
    return shape


def two_way_source(name: str, segments: list[list[tuple[float, float]]], sink: str, box: list[tuple[float, float]]) -> SourceConfig:
    source = lane_source_msg(name, side="right", segments=segments, sink=sink)
    source.lane.two_way = True
    source.lane.track_spacing = 1.65
    source.shape = polygon(box)
    return source


def forward_source() -> SourceConfig:
    return two_way_source("forward", FORWARD, "top", [(-0.5, -2.0), (1.0, -2.0), (1.0, 2.0), (-0.5, 2.0)])


def backward_source() -> SourceConfig:
    return two_way_source("backward", BACKWARD, "end", [(7.04, 4.5), (11.04, 4.5), (11.04, 6.5), (7.04, 6.5)])


def top_sink_msg() -> SinkConfig:
    sink = end_sink_msg("top")
    sink.shape = polygon([(7.04, 5.0), (11.04, 5.0), (11.04, 8.0), (7.04, 8.0)])
    return sink


def start_sink_msg() -> SinkConfig:
    sink = end_sink_msg("end")
    sink.shape = polygon([(-1.0, -2.0), (1.0, -2.0), (1.0, 2.0), (-1.0, 2.0)])
    return sink


@pytest.fixture
def l_manager(rclpy_context: object, minimal_scenario: ScenarioConfig) -> Iterator[AgentManager]:  # noqa: ARG001
    manager = build_manager(minimal_scenario, node_name="lane_goal_corners")
    manager._add_walls_callback(walls_request(L_WALLS), AddWalls.Response())
    manager._add_sink_callback(AddSink.Request(sink=top_sink_msg()), AddSink.Response())
    manager._add_sink_callback(AddSink.Request(sink=start_sink_msg()), AddSink.Response())
    yield manager
    manager.destroy_node()


def add_source(manager: AgentManager, source: SourceConfig) -> None:
    response = manager._add_source_callback(AddSource.Request(source=source), AddSource.Response())
    assert response.success is True, response.message


def goal_count(manager: AgentManager, name: str) -> int:
    lane = manager._lane_sources[name].lane
    assert lane is not None
    return len(lane.goal_s)


def test_lone_two_way_lane_source_keeps_no_goal_corner(l_manager: AgentManager) -> None:
    add_source(l_manager, forward_source())
    assert goal_count(l_manager, "forward") == 0


def test_opposing_lane_inside_the_turn_gives_the_outer_lane_its_goal_corner(l_manager: AgentManager) -> None:
    add_source(l_manager, forward_source())
    add_source(l_manager, backward_source())
    assert goal_count(l_manager, "forward") == 1
    assert goal_count(l_manager, "backward") == 0


def test_removing_the_opposing_lane_source_drops_the_goal_corner(l_manager: AgentManager) -> None:
    add_source(l_manager, forward_source())
    add_source(l_manager, backward_source())
    l_manager._remove_source_callback(RemoveSource.Request(name="backward"), RemoveSource.Response())
    assert goal_count(l_manager, "forward") == 0
