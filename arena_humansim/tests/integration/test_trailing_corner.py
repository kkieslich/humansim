"""A long-cane user trailing a wall turns an outer corner where a box stands flush in it (two robot bays boxed off for
walkers at a hall corner) without stopping short of its route's corner point (navmesh, sfm, the type's own settings).

Before the corner points kept only the body clearance and the waypoint advanced only once passed, the walker spent
10-14 s within 1.5 m of the box corner, hovering short of a point the wall push kept it from reaching."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams, WallConfig
from arena_humansim.utils.types import Pose2D
from rclpy.parameter import Parameter

DT = 0.05
#: the hall's west wall x = 0 runs north from its corner (0, 0), the side corridor (2 m wide) runs west from it, and a
#: 1.2 x 2 m box stands flush in the corner
WALLS = [
    ((0.0, 0.0), (0.0, 20.0)),
    ((0.0, 0.0), (-12.0, 0.0)),
    ((-12.0, -2.0), (0.0, -2.0)),
    ((0.0, -2.0), (0.0, -8.0)),
    ((0.0, -8.0), (8.0, -8.0)),
    ((8.0, -8.0), (8.0, 20.0)),
    ((0.0, 20.0), (8.0, 20.0)),
    ((-12.0, 0.0), (-12.0, -2.0)),
    ((0.0, 0.0), (1.2, 0.0)),
    ((1.2, 0.0), (1.2, 2.0)),
    ((1.2, 2.0), (0.0, 2.0)),
]
BOX_CORNER = (1.2, 0.0)
GOAL = Pose2D(x=-11.0, y=-1.0)


def test_a_long_cane_user_turns_a_boxed_corner_without_hovering_at_it(manager_factory: Callable[..., AgentManager]) -> None:
    walls = [WallConfig(name=f"w{i}", start=Pose2D(x=a[0], y=a[1]), end=Pose2D(x=b[0], y=b[1])) for i, (a, b) in enumerate(WALLS)]
    agent = AgentConfig(agent_id=1, agent_type="blind_white_cane", spawn_pose=Pose2D(x=3.0, y=18.0, theta=-np.pi / 2), goal_sequence=[GOAL])
    scenario = ScenarioConfig(name="trailing_corner", simulation=SimulationParams(seed=1, dt=DT), modules=ModuleConfig(global_planner="navmesh", local_planner="sfm"), walls=walls, agents=[agent])
    mgr = manager_factory(scenario, node_name="test_trailing_corner", extra_params=[Parameter("global_planner.lane_side", Parameter.Type.STRING, "left")])
    near = 0.0
    arrived = False
    for _ in range(int(120.0 / DT)):
        mgr.tick()
        i = mgr._pool._id_to_idx.get(1)
        if i is None:
            arrived = True
            break
        x, y = float(mgr._pool.pos[i, 0]), float(mgr._pool.pos[i, 1])
        if np.hypot(x - BOX_CORNER[0], y - BOX_CORNER[1]) < 1.5:
            near += DT
        if np.hypot(x - GOAL.x, y - GOAL.y) < 0.5:
            arrived = True
            break
    assert arrived
    assert 0.0 < near <= 8.0, f"{near:.1f} s within 1.5 m of the box corner"
