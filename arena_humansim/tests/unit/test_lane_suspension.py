from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.srv import AddSink, AddSource, AddWalls, RemoveSink, RemoveWalls

from tests._lane_messages import CORRIDOR_WALLS, end_sink_msg, lane_source_msg, walls_request
from tests.integration._helpers import build_manager

PILLAR = ("pillar", (10.0, -2.0), (10.0, 2.0))
CRATE = ("crate", (10.0, 1.6), (10.5, 1.6))
FAR_WALL = ("far", (10.0, 40.0), (10.0, 44.0))
LONG_TICK_S = 100.0


@pytest.fixture
def lane_manager(rclpy_context: object, minimal_scenario: ScenarioConfig) -> Iterator[AgentManager]:  # noqa: ARG001
    manager = build_manager(minimal_scenario, node_name="lane_suspension")
    manager._add_walls_callback(walls_request(CORRIDOR_WALLS), AddWalls.Response())
    manager._add_sink_callback(AddSink.Request(sink=end_sink_msg()), AddSink.Response())
    response = manager._add_source_callback(AddSource.Request(source=lane_source_msg("lane")), AddSource.Response())
    assert response.success is True
    yield manager
    manager.destroy_node()


def spawned(manager: AgentManager) -> int:
    return len(manager._spawn_scheduler.tick(1, LONG_TICK_S))


def add_walls(manager: AgentManager, walls: list[tuple[str, tuple[float, float], tuple[float, float]]]) -> AddWalls.Response:
    return manager._add_walls_callback(walls_request(walls), AddWalls.Response())


def remove_walls(manager: AgentManager, names: list[str]) -> RemoveWalls.Response:
    return manager._remove_walls_callback(RemoveWalls.Request(names=names), RemoveWalls.Response())


def test_lane_source_spawns_while_its_lane_builds(lane_manager: AgentManager) -> None:
    assert spawned(lane_manager) > 0


def test_blocking_wall_is_applied_and_answered_with_success(lane_manager: AgentManager) -> None:
    response = add_walls(lane_manager, [PILLAR])
    assert response.success is True
    assert "pillar" in lane_manager._walls


def test_lane_source_whose_lane_breaks_spawns_nothing(lane_manager: AgentManager) -> None:
    add_walls(lane_manager, [PILLAR])
    assert spawned(lane_manager) == 0


def test_suspended_lane_source_resumes_when_the_blocking_wall_is_removed(lane_manager: AgentManager) -> None:
    add_walls(lane_manager, [PILLAR])
    response = remove_walls(lane_manager, ["pillar"])
    assert response.success is True
    assert spawned(lane_manager) > 0


def test_wall_change_keeping_the_band_leaves_the_source_spawning(lane_manager: AgentManager) -> None:
    assert add_walls(lane_manager, [CRATE]).success is True
    assert spawned(lane_manager) > 0


def test_wall_change_after_the_lane_sink_is_removed_is_not_refused(lane_manager: AgentManager) -> None:
    lane_manager._remove_sink_callback(RemoveSink.Request(name="end"), RemoveSink.Response())
    response = add_walls(lane_manager, [CRATE])
    assert response.success is True
    assert spawned(lane_manager) == 0


def test_removing_every_wall_is_not_refused_and_stops_the_lane_source(lane_manager: AgentManager) -> None:
    response = remove_walls(lane_manager, [])
    assert response.success is True
    assert spawned(lane_manager) == 0


def test_wall_added_far_from_the_lane_leaves_the_lane_object_untouched(lane_manager: AgentManager) -> None:
    before = lane_manager._lane_sources["lane"].lane
    assert before is not None
    add_walls(lane_manager, [FAR_WALL])
    assert lane_manager._lane_sources["lane"].lane is before


def test_wall_added_across_the_lane_suspends_it(lane_manager: AgentManager) -> None:
    add_walls(lane_manager, [PILLAR])
    assert lane_manager._lane_sources["lane"].lane is None


def test_removing_the_lane_sink_suspends_the_source_at_once(lane_manager: AgentManager) -> None:
    lane_manager._remove_sink_callback(RemoveSink.Request(name="end"), RemoveSink.Response())
    assert lane_manager._lane_sources["lane"].lane is None
    assert spawned(lane_manager) == 0


def test_adding_the_lane_sink_back_resumes_the_source(lane_manager: AgentManager) -> None:
    lane_manager._remove_sink_callback(RemoveSink.Request(name="end"), RemoveSink.Response())
    lane_manager._add_sink_callback(AddSink.Request(sink=end_sink_msg()), AddSink.Response())
    assert lane_manager._lane_sources["lane"].lane is not None
    assert spawned(lane_manager) > 0
