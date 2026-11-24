"""Two escalators side by side: a walker that lands beside the mouth of the escalator going back walks past it.

Rooms A (x 0-10) and B (x 20-30) are joined only by an escalator down (A to B, north) and one up (B to A, south), as the
tj escalators of smu_link_full_v2. The walker's goal in B lies south, past the up escalator's mouth. Before, its route
from the landing crossed the up escalator past its seam, which carried it back to A; it took the down escalator again and
rode round for good (long-cane walkers: their wall-trailing line runs along the wall into the mouth)."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams, WallConfig
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import TeleportZone
from arena_humansim_msgs.srv import SetTeleportZones
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Pose2D as Pose2DMsg
from rclpy.parameter import Parameter

GOAL = Pose2D(x=21.2, y=1.0)


def _room(x0: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    corners = [(x0, 0.0), (x0 + 10.0, 0.0), (x0 + 10.0, 10.0), (x0, 10.0)]
    return list(zip(corners, corners[1:] + corners[:1], strict=True))


def _half(name: str, destinations: list[str], entry: tuple[float, float, float], hold: tuple[float, float], box: tuple[float, float, float, float], belt: float, delay: float) -> TeleportZone:
    z = TeleportZone()
    z.name, z.destinations, z.kind, z.open, z.delay, z.belt = name, destinations, "escalator", True, delay, belt
    z.entry = Pose2DMsg(x=entry[0], y=entry[1], theta=entry[2])
    z.wait = Pose2DMsg(x=entry[0] - 0.8 * math.cos(entry[2]), y=entry[1] - 0.8 * math.sin(entry[2]), theta=entry[2])
    z.hold = Pose2DMsg(x=hold[0], y=hold[1], theta=entry[2])
    x0, y0, x1, y1 = box
    z.footprint = [Point32(x=x, y=y, z=0.0) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return z


def _manager(manager_factory: Callable[..., AgentManager], name: str, agent_type: str, west: float) -> AgentManager:
    """The up escalator's B half spans x `west` to `west` + 1.5: flush with B's west wall (20) or 1 m out on the floor."""
    walls = [WallConfig(name=f"w{i}", start=Pose2D(x=a[0], y=a[1]), end=Pose2D(x=b[0], y=b[1])) for i, (a, b) in enumerate(_room(0.0) + _room(20.0))]
    walker = AgentConfig(agent_id=1, agent_type=agent_type, spawn_pose=Pose2D(x=3.0, y=6.5), goal_sequence=[GOAL])
    scenario = ScenarioConfig(name=name, simulation=SimulationParams(seed=5, dt=0.05), modules=ModuleConfig(global_planner="navmesh"), walls=walls, agents=[walker])
    mgr = manager_factory(scenario, node_name=f"test_{name}", extra_params=[Parameter("zone_route_choice", Parameter.Type.BOOL, True), Parameter("global_planner.lane_side", Parameter.Type.STRING, "left")])
    zones = [
        _half("down_top", ["down_bottom"], (8.5, 6.5, 0.0), (9.2, 6.5), (8.5, 6.0, 10.0, 7.0), 0.5, 3.0),
        _half("down_bottom", [], (21.5, 6.5, math.pi), (20.8, 6.5), (20.0, 6.0, 21.5, 7.0), -0.5, 0.0),
        _half("up_bottom", ["up_top"], (west + 1.5, 5.3, math.pi), (west + 0.8, 5.3), (west, 4.8, west + 1.5, 5.8), 0.5, 3.0),
        _half("up_top", [], (8.5, 5.3, 0.0), (9.2, 5.3), (8.5, 4.8, 10.0, 5.8), -0.5, 0.0),
    ]
    assert mgr._set_teleport_zones_callback(SetTeleportZones.Request(zones=zones), SetTeleportZones.Response()).success
    return mgr


@pytest.mark.parametrize(("agent_type", "west"), [("blind_white_cane", 20.0), ("blind_white_cane", 21.0), ("adult", 21.0), ("adult", 20.0)])
def test_walker_landing_beside_the_mouth_of_the_escalator_back_walks_past_it(manager_factory: Callable[..., AgentManager], agent_type: str, west: float) -> None:
    mgr = _manager(manager_factory, f"escalator_pair_{agent_type}_{int(west)}", agent_type, west)
    flights, flying, arrived = 0, False, False
    for _ in range(3000):
        mgr.tick()
        if 1 not in mgr._agents:
            arrived = True
            break
        now = 1 in mgr._zones.flying
        flights += now and not flying
        flying = now
        i = mgr._pool.idx(1)
        if math.hypot(float(mgr._pool.pos[i, 0]) - GOAL.x, float(mgr._pool.pos[i, 1]) - GOAL.y) < 0.5:
            arrived = True
            break
    assert arrived
    assert flights == 1
