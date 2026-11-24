from __future__ import annotations

import pytest
from arena_humansim_msgs.srv import AddSink, AddSource, AddWalls

from tests._lane_messages import CORRIDOR_WALLS, end_sink_msg, lane_source_msg, walls_request
from tests.ros._helpers import ResetSimulation, RosTestSystem, make_reset_request

pytestmark = pytest.mark.ros


@pytest.fixture
def corridor_with_end_sink(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(ResetSimulation, "reset", make_reset_request())
    ros_system.call(AddWalls, "add_walls", walls_request(CORRIDOR_WALLS))
    ros_system.call(AddSink, "add_sink", AddSink.Request(sink=end_sink_msg()))
    return ros_system


def test_add_source_with_a_lane_returns_its_derived_numbers(corridor_with_end_sink: RosTestSystem) -> None:
    response = corridor_with_end_sink.call(AddSource, "add_source", AddSource.Request(source=lane_source_msg("lane")))
    assert response.success is True
    assert response.track_spacing == pytest.approx(3.0)
    assert response.arrival_radius == pytest.approx(1.0)
    assert response.spawn_length == pytest.approx(0.5)
    assert response.sigma_m == pytest.approx(0.3409, abs=1e-3)


def test_lane_whose_segments_do_not_connect_is_refused_with_its_flow_name(corridor_with_end_sink: RosTestSystem) -> None:
    source = lane_source_msg("lane_gap", segments=[[(0.0, 0.0), (10.0, 0.0)], [(12.0, 0.0), (20.0, 0.0)]])
    response = corridor_with_end_sink.call(AddSource, "add_source", AddSource.Request(source=source))
    assert response.success is False
    assert "lane_gap" in response.message


def test_lane_without_a_side_is_refused_with_its_flow_name(corridor_with_end_sink: RosTestSystem) -> None:
    response = corridor_with_end_sink.call(AddSource, "add_source", AddSource.Request(source=lane_source_msg("lane_nosided", side="")))
    assert response.success is False
    assert "lane_nosided" in response.message
