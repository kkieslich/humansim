import math
from collections.abc import Callable, Sequence

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import (
    AgentLifetime,
    AgentState,
    BehaviorTreeMovement,
    CommandType,
    DespawnRequest,
    Pose2D,
    ShapeType,
    SinkConfig,
    WaypointMovement,
)


class DespawnMonitor(Loggable):
    def __init__(self):
        self._sinks: dict[str, SinkConfig] = {}
        self._sink_occupancy: dict[str, int] = {}
        self._lifetimes: dict[int, AgentLifetime] = {}

    def tick(
        self,
        agents: dict[int, BaseAgent],
        interaction_check: Callable[[int], bool],
        tick_count: int,
        dt: float,
    ) -> list[DespawnRequest]:
        to_remove: list[DespawnRequest] = []

        for aid, lifetime in list(self._lifetimes.items()):
            agent = agents.get(aid)
            if agent is None:
                continue

            age_s = (tick_count - lifetime.spawn_tick) * dt
            # inside its sink and heading for it: an agent on its way to a point outside the sink only passes through
            at_sink = self._check_sink_proximity(agent.state, lifetime.target_sink_name) and self._heading_for_sink(agent, lifetime.target_sink_name)
            ttl_expired = lifetime.max_lifetime_s > 0 and age_s >= lifetime.max_lifetime_s
            in_interaction = interaction_check(aid)

            if at_sink and not in_interaction:
                self._logger.debug(f"Agent {aid} reached sink {lifetime.target_sink_name}")
                to_remove.append(DespawnRequest(agent_id=aid, force=False, reason="sink"))
            elif at_sink and in_interaction:
                self._logger.debug(f"Agent {aid} at sink but in interaction, deferring despawn")
                lifetime.pending_despawn = True
            elif lifetime.pending_despawn and not in_interaction:
                self._logger.debug(f"Agent {aid} deferred despawn triggered")
                to_remove.append(DespawnRequest(agent_id=aid, force=False, reason="deferred"))
            elif ttl_expired:
                self._logger.debug(f"Agent {aid} TTL expired (age={age_s:.1f}s)")
                to_remove.append(DespawnRequest(agent_id=aid, force=in_interaction, reason="ttl"))

        return to_remove

    def register(self, agent_id: int, lifetime: AgentLifetime) -> None:
        self._lifetimes[agent_id] = lifetime

    def unregister(self, agent_id: int) -> None:
        self._lifetimes.pop(agent_id, None)

    def lifetime(self, agent_id: int) -> AgentLifetime | None:
        """Lifetime of a source-released agent, None for any other agent."""
        return self._lifetimes.get(agent_id)

    def clear(self) -> None:
        self._lifetimes.clear()
        self._sink_occupancy.clear()

    @property
    def sinks(self) -> dict[str, SinkConfig]:
        return dict(self._sinks)

    def add_sink(self, config: SinkConfig) -> str:
        self._sinks[config.name] = config
        self._sink_occupancy.setdefault(config.name, 0)
        self._logger.info(f"Sink added: {config.name}")
        return config.name

    def remove_sink(self, name: str) -> None:
        self._sinks.pop(name, None)
        self._sink_occupancy.pop(name, None)
        self._logger.info(f"Sink removed: {name}")

    def clear_sinks(self) -> None:
        self._sinks.clear()
        self._sink_occupancy.clear()

    def set_sinks(self, sinks: dict[str, SinkConfig]) -> None:
        self._sinks = dict(sinks)
        self._sink_occupancy = {name: 0 for name in sinks}

    def _check_sink_proximity(self, agent_state: AgentState, target_sink_name: str) -> bool:
        if not target_sink_name or target_sink_name not in self._sinks:
            return False

        sink = self._sinks[target_sink_name]

        if sink.capacity > 0 and self._sink_occupancy.get(sink.name, 0) >= sink.capacity:
            return False

        return self._contains(sink, agent_state.pose.x, agent_state.pose.y)

    def _heading_for_sink(self, agent: BaseAgent, target_sink_name: str) -> bool:
        """Whether the agent is on its way out through its sink, not on its way somewhere else.

        A waypoint walker heads for its current waypoint, and one without waypoints left has nowhere else to go. A behavior-tree
        agent heads for the target of a navigate command that moves it; without one (standing, which navigates to its own pose
        at zero speed, seeking an object, between steps) it is busy with its behavior, not leaving.
        """
        sink = self._sinks.get(target_sink_name)
        if sink is None:
            return False
        mv = agent.movement
        if isinstance(mv, BehaviorTreeMovement):
            cmd = mv.command
            if cmd is None or cmd.type != CommandType.NAVIGATE or cmd.desired_velocity <= 0.0:
                return False
            return self._contains(sink, cmd.target_pose.x, cmd.target_pose.y)
        if isinstance(mv, WaypointMovement) and 0 <= mv.index < len(mv.waypoints):
            goal = mv.waypoints[mv.index]
            return self._contains(sink, goal.x, goal.y)
        return True

    def _contains(self, sink: SinkConfig, x: float, y: float) -> bool:
        if sink.shape.type == ShapeType.POLYGON and sink.shape.vertices:
            return self._point_in_polygon(x - sink.pose.x, y - sink.pose.y, sink.shape.vertices)
        return math.hypot(x - sink.pose.x, y - sink.pose.y) < sink.absorption_radius

    @staticmethod
    def _point_in_polygon(px: float, py: float, vertices: Sequence[Pose2D]) -> bool:
        n = len(vertices)
        inside = False
        j = n - 1
        for i in range(n):
            vi, vj = vertices[i], vertices[j]
            if ((vi.y > py) != (vj.y > py)) and (px < (vj.x - vi.x) * (py - vi.y) / (vj.y - vi.y) + vi.x):
                inside = not inside
            j = i
        return inside
