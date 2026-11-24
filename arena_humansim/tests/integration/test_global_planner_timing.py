"""The agent manager gives the global planner the sim time between its calls (dt x bt_tick_interval), so the stall
and congestion windows hold in sim seconds whatever the behaviour-tree tick interval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


def test_the_agent_manager_gives_the_planner_its_call_interval(manager_factory: Callable[..., Any]) -> None:
    from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams

    for interval in (1, 5):
        mgr = manager_factory(ScenarioConfig(name="timing", simulation=SimulationParams(dt=0.05, bt_tick_interval=interval), modules=ModuleConfig(global_planner="navmesh")), node_name=f"timing_{interval}")
        assert mgr._global_planner._dt == pytest.approx(0.05 * interval)
