"""The stall and congestion windows hold in sim seconds whatever the behaviour-tree tick interval: the agent manager
gives the global planner the time between its calls (dt x bt_tick_interval)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import Pose2D

WALLS = [((0.0, 0.0), (20.0, 0.0)), ((20.0, 0.0), (20.0, 6.0)), ((20.0, 6.0), (0.0, 6.0)), ((0.0, 6.0), (0.0, 0.0))]


@pytest.mark.parametrize("call_interval", [0.05, 0.25])
def test_an_agent_that_makes_no_progress_replans_after_the_stall_time_in_sim_seconds(call_interval: float, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6)
    planner.set_walls(WALLS)
    planner.set_timing(call_interval, 2.0)
    agent = agent_factory(agent_id=1, x=2.0, y=3.0)
    cmds = commands_factory(agent_ids=[1], target=(18.0, 3.0))
    planner.compute([agent], cmds)
    agent.state.pose = Pose2D(x=2.5, y=3.4)  # off the path line, so a replan gives a new path; then it stands still
    path = planner.get_cached_paths()[1]
    replanned_at = None
    for call in range(1, round(4.0 / call_interval)):
        planner.compute([agent], cmds)
        if planner.get_cached_paths()[1] is not path:
            replanned_at = call * call_interval
            break
    assert replanned_at == pytest.approx(2.0, abs=call_interval + 1e-9)


@pytest.mark.parametrize("call_interval", [0.05, 0.25])
def test_the_congestion_weights_update_every_interval_in_sim_seconds(call_interval: float, agent_factory: Callable[..., BaseAgent], commands_factory: Callable[..., dict[int, Any]]) -> None:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, congestion_interval=1.0)
    planner.set_walls(WALLS)
    planner.set_timing(call_interval, 0.0)
    agents = [agent_factory(agent_id=i, x=5.0 + 0.3 * i, y=3.0) for i in range(1, 6)]
    cmds = commands_factory(agent_ids=[a.state.agent_id for a in agents], target=(18.0, 3.0))
    updated_at = None
    for call in range(1, round(3.0 / call_interval)):
        planner.compute(agents, cmds)
        if planner._slowdown is not None:
            updated_at = call * call_interval
            break
    assert updated_at == pytest.approx(1.0, abs=call_interval + 1e-9)
