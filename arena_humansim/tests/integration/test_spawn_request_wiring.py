from __future__ import annotations

from collections.abc import Callable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim.utils.types import Pose2D, SpawnRequest, WaypointMode, WaypointMovement


def test_spawned_agent_movement_takes_radii_and_mode_from_its_request(
    manager_factory: Callable[..., AgentManager],
    minimal_scenario: ScenarioConfig,
) -> None:
    manager = manager_factory(minimal_scenario)
    request = SpawnRequest(
        agent_id=7,
        pose=Pose2D(x=0.0, y=0.0),
        agent_radius=0.25,
        agent_type="adult",
        waypoints=[Pose2D(x=1.0, y=0.0), Pose2D(x=2.0, y=0.0)],
        waypoint_radii=[0.7, 0.4],
        waypoint_mode=WaypointMode.ONCE,
    )
    agent = manager._build_base_agent_from_spawn(7, request)
    assert isinstance(agent.movement, WaypointMovement)
    assert agent.movement.mode is WaypointMode.ONCE
    assert agent.movement.radii == [0.7, 0.4]
    assert [p.x for p in agent.movement.waypoints] == [1.0, 2.0]
