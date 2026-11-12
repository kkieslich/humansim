from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.core.behavior.nodes import HoldNode
from arena_humansim.core.despawn_monitor import DespawnMonitor
from arena_humansim.utils.types import (
    AgentLifetime,
    BehaviorTreeMovement,
    CommandType,
    HighLevelCommand,
    Pose2D,
    Shape,
    ShapeType,
    SinkConfig,
    WaypointMovement,
)


def _make_monitor() -> DespawnMonitor:
    return DespawnMonitor()


def _circle_sink(name: str = "exit", x: float = 0.0, y: float = 0.0, radius: float = 0.5, capacity: int = -1) -> SinkConfig:
    return SinkConfig(
        name=name,
        pose=Pose2D(x=x, y=y),
        shape=Shape(type=ShapeType.CIRCLE, radius=radius),
        absorption_radius=radius,
        capacity=capacity,
    )


def _polygon_sink(name: str = "room", cx: float = 0.0, cy: float = 0.0, vertices: list[Pose2D] | None = None) -> SinkConfig:
    verts = (
        vertices
        if vertices is not None
        else [
            Pose2D(x=-1.0, y=-1.0),
            Pose2D(x=1.0, y=-1.0),
            Pose2D(x=1.0, y=1.0),
            Pose2D(x=-1.0, y=1.0),
        ]
    )
    return SinkConfig(
        name=name,
        pose=Pose2D(x=cx, y=cy),
        shape=Shape(type=ShapeType.POLYGON, vertices=verts),
        absorption_radius=0.0,
    )


def _no_interaction(_aid: int) -> bool:
    return False


def _always_interaction(_aid: int) -> bool:
    return True


def test_tick_empty_returns_no_requests() -> None:
    m = _make_monitor()
    assert m.tick({}, _no_interaction, tick_count=0, dt=0.1) == []


def test_circle_sink_proximity_triggers_despawn(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_circle_sink("exit", x=5.0, y=0.0, radius=0.5))
    agent = agent_factory(agent_id=1, x=5.1, y=0.0)
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="exit", spawn_tick=0))
    requests = m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1)
    assert len(requests) == 1
    assert requests[0].agent_id == 1
    assert requests[0].reason == "sink"
    assert requests[0].force is False


def test_polygon_sink_proximity_triggers_despawn(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_polygon_sink("room", cx=2.0, cy=2.0))
    agent = agent_factory(agent_id=7, x=2.3, y=2.1)
    m.register(7, AgentLifetime(agent_id=7, target_sink_name="room"))
    requests = m.tick({7: agent}, _no_interaction, tick_count=1, dt=0.1)
    assert len(requests) == 1
    assert requests[0].agent_id == 7
    assert requests[0].reason == "sink"


def test_in_interaction_defers_despawn(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_circle_sink("exit", x=0.0, y=0.0, radius=0.5))
    agent = agent_factory(agent_id=1, x=0.1, y=0.0)
    lifetime = AgentLifetime(agent_id=1, target_sink_name="exit")
    m.register(1, lifetime)
    requests = m.tick({1: agent}, _always_interaction, tick_count=1, dt=0.1)
    assert requests == []
    assert lifetime.pending_despawn is True


def test_pending_despawn_fires_after_interaction_ends(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_circle_sink("exit", x=0.0, y=0.0, radius=0.5))
    agent = agent_factory(agent_id=1, x=5.0, y=5.0)
    lifetime = AgentLifetime(agent_id=1, target_sink_name="exit", pending_despawn=True)
    m.register(1, lifetime)
    requests = m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1)
    assert len(requests) == 1
    assert requests[0].reason == "deferred"
    assert requests[0].force is False


def test_ttl_expired_forces_despawn_in_interaction(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    agent = agent_factory(agent_id=1, x=50.0, y=50.0)
    m.register(1, AgentLifetime(agent_id=1, spawn_tick=0, max_lifetime_s=1.0))
    requests = m.tick({1: agent}, _always_interaction, tick_count=100, dt=0.1)
    assert len(requests) == 1
    assert requests[0].reason == "ttl"
    assert requests[0].force is True


def test_ttl_not_expired_does_nothing(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    agent = agent_factory(agent_id=1, x=50.0, y=50.0)
    m.register(1, AgentLifetime(agent_id=1, spawn_tick=0, max_lifetime_s=100.0))
    requests = m.tick({1: agent}, _no_interaction, tick_count=5, dt=0.1)
    assert requests == []


def test_sink_capacity_blocks_absorption(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_circle_sink("exit", x=0.0, y=0.0, radius=0.5, capacity=1))
    m._sink_occupancy["exit"] = 1
    agent = agent_factory(agent_id=1, x=0.1, y=0.0)
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="exit"))
    requests = m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1)
    assert requests == []


def test_register_unregister_and_clear() -> None:
    m = _make_monitor()
    m.register(1, AgentLifetime(agent_id=1))
    m.register(2, AgentLifetime(agent_id=2))
    assert set(m._lifetimes.keys()) == {1, 2}
    m.unregister(1)
    assert set(m._lifetimes.keys()) == {2}
    m.unregister(999)
    assert set(m._lifetimes.keys()) == {2}
    m.add_sink(_circle_sink("s"))
    m.clear()
    assert m._lifetimes == {}
    assert m._sink_occupancy == {}


def test_sink_registration_set_and_remove() -> None:
    m = _make_monitor()
    s1 = _circle_sink("a")
    s2 = _circle_sink("b")
    m.add_sink(s1)
    m.add_sink(s2)
    assert set(m.sinks.keys()) == {"a", "b"}
    m.remove_sink("a")
    assert set(m.sinks.keys()) == {"b"}
    m.set_sinks({"c": _circle_sink("c")})
    assert set(m.sinks.keys()) == {"c"}
    m.clear_sinks()
    assert m.sinks == {}


@pytest.mark.parametrize(
    "px, py, expected",
    [
        (0.0, 0.0, True),
        (2.0, 0.0, False),
        (-2.0, -2.0, False),
        (0.99, 0.99, True),
        (1.0, 0.0, False),
        (0.0, 1.0, False),
    ],
)
def test_point_in_polygon_square(px: float, py: float, expected: bool) -> None:
    square = [
        Pose2D(x=-1.0, y=-1.0),
        Pose2D(x=1.0, y=-1.0),
        Pose2D(x=1.0, y=1.0),
        Pose2D(x=-1.0, y=1.0),
    ]
    assert DespawnMonitor._point_in_polygon(px, py, square) is expected


def _door_strip() -> SinkConfig:
    """A sink strip 0.1 to 0.6 m inside a door on the line y = 0, like the one a FLOWEDIT entrance writes."""
    return _polygon_sink(
        "door_sink",
        cx=0.0,
        cy=0.35,
        vertices=[Pose2D(x=-0.6, y=-0.25), Pose2D(x=0.6, y=-0.25), Pose2D(x=0.6, y=0.25), Pose2D(x=-0.6, y=0.25)],
    )


def _navigating(agent: BaseAgent, x: float, y: float, kind: CommandType = CommandType.NAVIGATE) -> BaseAgent:
    agent.movement = BehaviorTreeMovement(command=HighLevelCommand(agent_id=agent.state.agent_id, type=kind, target_pose=Pose2D(x=x, y=y)))
    return agent


def test_behavior_tree_agent_passing_through_its_sink_stays(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_door_strip())
    agent = _navigating(agent_factory(agent_id=1, x=0.3, y=0.4), x=6.0, y=5.0)
    lifetime = AgentLifetime(agent_id=1, target_sink_name="door_sink")
    m.register(1, lifetime)
    assert m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1) == []
    assert lifetime.pending_despawn is False


def test_behavior_tree_agent_walking_into_its_sink_is_taken(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_door_strip())
    agent = _navigating(agent_factory(agent_id=1, x=0.3, y=0.4), x=0.0, y=0.35)
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="door_sink"))
    requests = m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1)
    assert [(r.agent_id, r.reason) for r in requests] == [(1, "sink")]


def test_released_visitor_stays_on_its_way_in_and_is_taken_on_its_way_out(agent_factory: Callable[..., BaseAgent]) -> None:
    """A visitor released behind its door's sink walks through it to its first stop, then comes back to leave."""
    m = _make_monitor()
    m.add_sink(_door_strip())
    agent = agent_factory(agent_id=1, x=0.0, y=1.3)
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="door_sink", spawn_tick=0))
    agent.movement = BehaviorTreeMovement()
    tick = 0
    for y in (1.3, 0.9, 0.5, 0.3, 0.5, 0.9):
        _navigating(agent, x=4.0, y=0.4)
        agent.state.pose = Pose2D(x=min(4.0, 1.3 - y), y=y)
        tick += 1
        assert m.tick({1: agent}, _no_interaction, tick_count=tick, dt=0.1) == [], f"taken at y={y} on its way to its stop"
    _navigating(agent, x=0.0, y=0.35)
    agent.state.pose = Pose2D(x=0.2, y=0.8)
    assert m.tick({1: agent}, _no_interaction, tick_count=tick + 1, dt=0.1) == []
    agent.state.pose = Pose2D(x=0.1, y=0.5)
    requests = m.tick({1: agent}, _no_interaction, tick_count=tick + 2, dt=0.1)
    assert [(r.agent_id, r.reason) for r in requests] == [(1, "sink")]


@pytest.mark.parametrize("command", [None, CommandType.STOP, CommandType.SEEK])
def test_behavior_tree_agent_in_its_sink_without_a_way_out_stays(agent_factory: Callable[..., BaseAgent], command: CommandType | None) -> None:
    m = _make_monitor()
    m.add_sink(_door_strip())
    agent = agent_factory(agent_id=1, x=0.0, y=0.35)
    agent.movement = BehaviorTreeMovement() if command is None else BehaviorTreeMovement(command=HighLevelCommand(agent_id=1, type=command, target_pose=Pose2D(x=0.0, y=0.35)))
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="door_sink"))
    assert m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1) == []


def test_behavior_tree_agent_holding_in_its_sink_stays(agent_factory: Callable[..., BaseAgent], rng_np: np.random.Generator) -> None:
    m = _make_monitor()
    m.add_sink(_door_strip())
    agent = agent_factory(agent_id=1, x=0.0, y=0.35)
    agent.movement = BehaviorTreeMovement()
    hold = HoldNode("hold", agent, duration_source=ParamDist(5.0), rng=rng_np, dt=0.1)
    hold.setup()
    hold.tick_once()
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="door_sink"))
    assert m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1) == []


def test_passing_through_in_interaction_does_not_defer_a_despawn(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_door_strip())
    agent = _navigating(agent_factory(agent_id=1, x=0.0, y=0.35), x=6.0, y=5.0)
    lifetime = AgentLifetime(agent_id=1, target_sink_name="door_sink")
    m.register(1, lifetime)
    assert m.tick({1: agent}, _always_interaction, tick_count=1, dt=0.1) == []
    assert lifetime.pending_despawn is False


def test_waypoint_walker_heading_for_its_sink_is_taken(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_circle_sink("exit", x=5.0, y=0.0, radius=0.5))
    agent = agent_factory(agent_id=1, x=5.1, y=0.0)
    agent.movement = WaypointMovement(waypoints=[Pose2D(x=5.0, y=0.0)])
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="exit"))
    assert [r.reason for r in m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1)] == ["sink"]


def test_waypoint_walker_rerouted_past_its_sink_stays(agent_factory: Callable[..., BaseAgent]) -> None:
    m = _make_monitor()
    m.add_sink(_circle_sink("exit", x=5.0, y=0.0, radius=0.5))
    agent = agent_factory(agent_id=1, x=5.1, y=0.0)
    agent.movement = WaypointMovement(waypoints=[Pose2D(x=5.0, y=0.0), Pose2D(x=12.0, y=0.0)], index=1)
    m.register(1, AgentLifetime(agent_id=1, target_sink_name="exit"))
    assert m.tick({1: agent}, _no_interaction, tick_count=1, dt=0.1) == []


def test_point_in_polygon_concave_shape() -> None:
    concave = [
        Pose2D(x=0.0, y=0.0),
        Pose2D(x=4.0, y=0.0),
        Pose2D(x=4.0, y=4.0),
        Pose2D(x=2.0, y=4.0),
        Pose2D(x=2.0, y=2.0),
        Pose2D(x=0.0, y=2.0),
    ]
    assert DespawnMonitor._point_in_polygon(1.0, 1.0, concave) is True
    assert DespawnMonitor._point_in_polygon(3.0, 1.0, concave) is True
    assert DespawnMonitor._point_in_polygon(3.0, 3.0, concave) is True
    assert DespawnMonitor._point_in_polygon(1.0, 3.0, concave) is False
    assert DespawnMonitor._point_in_polygon(5.0, 5.0, concave) is False
