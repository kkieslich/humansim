from __future__ import annotations

from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim_msgs.srv import AddSink, AddSource, AddWalls

from tests._lane_messages import CORRIDOR_WALLS, end_sink_msg, lane_source_msg, walls_request


def _lane_flow(manager_factory: Callable[..., AgentManager], scenario: ScenarioConfig) -> AgentManager:
    manager = manager_factory(scenario)
    manager._add_walls_callback(walls_request(CORRIDOR_WALLS), AddWalls.Response())
    manager._add_sink_callback(AddSink.Request(sink=end_sink_msg()), AddSink.Response())
    return manager


def test_lane_source_is_accepted_with_its_derived_track_spacing(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    manager = _lane_flow(manager_factory, minimal_scenario)
    response = manager._add_source_callback(AddSource.Request(source=lane_source_msg("lane")), AddSource.Response())
    assert response.success is True
    assert response.track_spacing == pytest.approx(3.0)


def test_wall_change_that_leaves_no_band_applies_and_suspends_the_lane(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    manager = _lane_flow(manager_factory, minimal_scenario)
    manager._add_source_callback(AddSource.Request(source=lane_source_msg("lane")), AddSource.Response())
    response = manager._add_walls_callback(walls_request([("pillar", (10.0, -2.0), (10.0, 2.0))]), AddWalls.Response())
    assert response.success is True
    assert "pillar" in manager._walls
    assert manager._lane_sources["lane"].lane is None


def test_wall_change_that_keeps_every_band_is_accepted(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    manager = _lane_flow(manager_factory, minimal_scenario)
    manager._add_source_callback(AddSource.Request(source=lane_source_msg("lane")), AddSource.Response())
    response = manager._add_walls_callback(walls_request([("crate", (10.0, 1.6), (10.5, 1.6))]), AddWalls.Response())
    assert response.success is True
    assert "crate" in manager._walls
