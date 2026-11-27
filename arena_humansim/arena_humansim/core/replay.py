from __future__ import annotations

import json
from typing import TYPE_CHECKING

import attrs

from arena_humansim.core.agents import SampledParams
from arena_humansim.core.agents.types import SampledLocomotion, SampledPerception
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.types import AgentState, CommandType, HighLevelCommand, Pose2D

if TYPE_CHECKING:
    from rclpy.impl.rcutils_logger import RcutilsLogger

    from arena_humansim.core.agent_manager import AgentManager


@attrs.frozen
class AgentStateDivergence:
    tick: int
    agent_id: int
    detail: str


# A replay compares which agents exist exactly and their continuous state within an absolute tolerance. libm and numpy's
# SIMD kernels differ in the last bits between CPU families, and the crowd feeds those bits back tick after tick: the
# 300-tick goldens drift by up to ~1e-12 m / rad on another family, and scaling an SFM parameter by 1 to 256 ulp
# leaves them 2e-14 to 2e-11 off. A relative change of 1e-9 to an SFM parameter already moves them past 1e-9 within
# 85 to 156 ticks, a change of 1e-6 at the first tick. 1e-9 keeps a margin of ~1000 over the CPU drift and still fails
# on the first tick a behaviour change shows.
POSITION_TOL_M = 1e-9
HEADING_TOL_RAD = 1e-9
VELOCITY_TOL_MPS = 1e-9


@attrs.define
class StateError:
    """The largest per-axis pose, heading and velocity error of a replay so far."""

    position_m: float = 0.0
    heading_rad: float = 0.0
    velocity_mps: float = 0.0

    def add(self, position_m: float, heading_rad: float, velocity_mps: float) -> None:
        self.position_m = max(self.position_m, position_m)
        self.heading_rad = max(self.heading_rad, heading_rad)
        self.velocity_mps = max(self.velocity_mps, velocity_mps)


@attrs.define
class ReplayResult:
    total_ticks: int
    success: bool = True
    first_divergence: AgentStateDivergence | None = None
    max_error: StateError = attrs.Factory(StateError)


class ReplayManager(Loggable):
    def __init__(self):
        self._ticks: list[dict] = []
        self._tick_index: dict[int, int] = {}
        self._spawns: dict[int, dict] = {}
        self._log_path: str | None = None

    def load(self, log_path: str) -> None:
        self._log_path = log_path
        self._ticks = []
        self._tick_index = {}
        self._spawns = {}

        tick_idx = 0
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("event") == "spawn":
                    self._spawns[record["agent_id"]] = record
                else:
                    self._ticks.append(record)
                    self._tick_index[record["tick"]] = tick_idx
                    tick_idx += 1

        self._logger.info(f"Loaded {len(self._ticks)} tick(s), {len(self._spawns)} spawn(s) from {log_path}")

    @property
    def tick_count(self) -> int:
        return len(self._ticks)

    def get_tick(self, n: int) -> dict | None:
        idx = self._tick_index.get(n)
        if idx is None:
            return None
        return self._ticks[idx]

    def get_spawn_params(self, agent_id: int) -> SampledParams | None:
        spawn = self._spawns.get(agent_id)
        if spawn is None:
            return None
        p = spawn["params"]
        perc = p.get("perception", {})
        lp = p.get("local_planner_params", {})
        loc = p.get("locomotion", {})
        return SampledParams(
            name=p["name"],
            desired_velocity=p["desired_velocity"],
            agent_radius=p["agent_radius"],
            max_velocity=p.get("max_velocity", 2.0),
            max_acceleration=p.get("max_acceleration", 1.5),
            max_deceleration=p.get("max_deceleration", 2.5),
            min_turning_radius=p.get("min_turning_radius", 0.5),
            pivot_angular_velocity=p.get("pivot_angular_velocity", 2.0),
            reaction_time=p.get("reaction_time", 0.4),
            personal_space_min=p.get("personal_space_min", 0.6),
            perception=SampledPerception(
                vision_range=perc.get("vision_range", p.get("vision_range", 5.0)),
                vision_fov=perc.get("vision_fov", p.get("vision_fov", 180.0)),
            ),
            local_planner_params={
                "relaxation_time": lp.get("relaxation_time", p.get("relaxation_time", 0.5)),
                "repulsion_strength": lp.get("repulsion_strength", p.get("repulsion_strength", 2.1)),
                "repulsion_range": lp.get("repulsion_range", p.get("repulsion_range", 0.3)),
                "anisotropy": lp.get("anisotropy", p.get("anisotropy", 0.5)),
            },
            locomotion=SampledLocomotion(**loc),
            interaction_class=p.get("interaction_class", ""),
            stair_effort=p.get("stair_effort", 1.0),
            perception_stack=tuple(p["perception_stack"]),
            local_planner=p["local_planner"],
            global_planner=p["global_planner"],
            animation=p["animation"],
        )

    @property
    def spawned_agent_ids(self) -> list[int]:
        return list(self._spawns.keys())

    def get_agents_at_tick(self, n: int) -> dict[int, AgentState]:
        record = self.get_tick(n)
        if record is None:
            return {}
        agents = {}
        for aid_str, adata in record.get("agents", {}).items():
            aid = int(aid_str)
            pose = adata["pose"]
            vel = adata["velocity"]
            agents[aid] = AgentState(
                agent_id=aid,
                pose=Pose2D(x=pose["x"], y=pose["y"], theta=pose["theta"]),
                velocity=(vel["vx"], vel["vy"]),
            )
        return agents

    def get_commands_at_tick(self, n: int) -> dict[int, HighLevelCommand]:
        record = self.get_tick(n)
        if record is None:
            return {}
        commands = {}
        for aid_str, cdata in record.get("commands", {}).items():
            aid = int(aid_str)
            tp = cdata.get("target_pose", {})
            commands[aid] = HighLevelCommand(
                agent_id=aid,
                type=CommandType(cdata.get("type", 0)),
                target_pose=Pose2D(
                    x=tp.get("x", 0.0),
                    y=tp.get("y", 0.0),
                    theta=tp.get("theta", 0.0),
                ),
                interaction_target=cdata.get("interaction_target", -1),
            )
        return commands

    def replay(self, agent_manager: AgentManager, logger: RcutilsLogger | None = None) -> ReplayResult:
        """Run the logged ticks on `agent_manager` and stop at the first tick whose state leaves the replay tolerances."""
        result = ReplayResult(total_ticks=len(self._ticks))

        for record in self._ticks:
            tick_n = record["tick"]

            commands = self.get_commands_at_tick(tick_n)
            agent_manager._high_level_cmds = commands

            agent_manager.tick()

            actual_states = {aid: agent.state for aid, agent in agent_manager._agents.items()}
            expected_agents = self.get_agents_at_tick(tick_n)
            divergence = _compare_agent_states(actual_states, expected_agents, tick_n, result.max_error)

            if divergence is not None:
                result.success = False
                if result.first_divergence is None:
                    result.first_divergence = divergence
                    msg = f"Replay divergence at tick {tick_n}: agent {divergence.agent_id} - {divergence.detail}"
                    if logger:
                        logger.warning(msg)
                break

        return result


def _compare_agent_states(
    actual: dict[int, AgentState],
    expected: dict[int, AgentState],
    tick: int,
    max_error: StateError | None = None,
) -> AgentStateDivergence | None:
    """The first agent whose state differs from the logged one: which agents exist must match exactly, pose and velocity
    within POSITION_TOL_M, HEADING_TOL_RAD and VELOCITY_TOL_MPS. `max_error` collects the largest errors seen."""
    all_ids = set(actual.keys()) | set(expected.keys())
    for aid in sorted(all_ids):
        if aid not in actual:
            return AgentStateDivergence(tick=tick, agent_id=aid, detail="agent missing from actual state")
        if aid not in expected:
            return AgentStateDivergence(tick=tick, agent_id=aid, detail="agent missing from expected (logged) state")
        a = actual[aid]
        e = expected[aid]

        position = max(abs(a.pose.x - e.pose.x), abs(a.pose.y - e.pose.y))
        heading = abs(a.pose.theta - e.pose.theta)
        velocity = max(abs(a.velocity[0] - e.velocity[0]), abs(a.velocity[1] - e.velocity[1]))
        if max_error is not None:
            max_error.add(position, heading, velocity)

        if not (position <= POSITION_TOL_M and heading <= HEADING_TOL_RAD):
            return AgentStateDivergence(
                tick=tick,
                agent_id=aid,
                detail=f"pose mismatch: off by {position:.3g} m / {heading:.3g} rad (tolerance {POSITION_TOL_M:g} m / {HEADING_TOL_RAD:g} rad), actual=({a.pose.x}, {a.pose.y}, {a.pose.theta}) expected=({e.pose.x}, {e.pose.y}, {e.pose.theta})",
            )

        if not velocity <= VELOCITY_TOL_MPS:
            return AgentStateDivergence(
                tick=tick,
                agent_id=aid,
                detail=f"velocity mismatch: off by {velocity:.3g} m/s (tolerance {VELOCITY_TOL_MPS:g} m/s), actual=({a.velocity[0]}, {a.velocity[1]}) expected=({e.velocity[0]}, {e.velocity[1]})",
            )

    return None
