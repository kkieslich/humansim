from __future__ import annotations

from collections.abc import Iterable

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.params import MODE_SUBSYSTEM
from arena_humansim.utils.scenario import ScenarioConfig
from rclpy.node import Node
from rclpy.parameter import Parameter


def build_manager(
    scenario: ScenarioConfig,
    node_name: str = "test_manager",
    extra_params: Iterable[Parameter] | None = None,
) -> AgentManager:
    sim = scenario.simulation
    mods = scenario.modules

    overrides: list[Parameter] = [
        Parameter("seed", Parameter.Type.INTEGER, int(sim.seed)),
        Parameter("dt", Parameter.Type.DOUBLE, float(sim.dt)),
        Parameter("bt_tick_interval", Parameter.Type.INTEGER, int(sim.bt_tick_interval)),
        Parameter("perception", Parameter.Type.STRING, str(mods.perception)),
        Parameter("global_planner", Parameter.Type.STRING, str(mods.global_planner)),
        Parameter("local_planner", Parameter.Type.STRING, str(mods.local_planner)),
        Parameter("animation", Parameter.Type.STRING, str(mods.animation)),
        Parameter("mode", Parameter.Type.STRING, MODE_SUBSYSTEM),
        Parameter("publish_markers", Parameter.Type.INTEGER, 0),
        Parameter("profile_phases", Parameter.Type.BOOL, True),
    ]
    if extra_params:
        overrides.extend(extra_params)

    orig_init = Node.__init__

    def patched(self, _name: str, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        kwargs.setdefault("parameter_overrides", overrides)
        orig_init(self, node_name, *args, **kwargs)

    Node.__init__ = patched
    try:
        mgr = AgentManager()
    finally:
        Node.__init__ = orig_init

    mgr._apply_scenario(scenario)
    return mgr
