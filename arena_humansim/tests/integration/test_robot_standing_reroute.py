"""Walkers blocked by a robot that stands still go round it; robots that move stay left to the local forces.

A corridor 3 m wide, the robot fed in through world_state as Arena feeds its robots: waiting at a locker flush against
the south wall, or stopped (jammed) in the corridor 0.9 m off that wall. A long-cane user trails the south wall straight
at the robot, an adult walks at it. Before the global planner knew standing robots, both stopped at the robot for good;
now each passes within seconds and gets back onto its line. A robot driving along the corridor stays off the
walkers' maps: they keep clear of it by the local forces as before and do not swing back and forth.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams, WallConfig
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg

DT = 0.05
LENGTH, WIDTH = 15.0, 3.0
GOAL_X = 12.0
#: The robot waits at a locker flush against the south wall, 6 cm off it; or it stopped mid-corridor
PARKED = (0.0, 0.36)
JAMMED = (0.0, 0.9)
#: A divider splits the corridor (4 m wide) into a 1.5 m lane and a 2.5 m lane over x -5..5; a robot jammed in the
#: middle of the narrow lane closes it
DIVIDER = (WallConfig(name="divider", start=Pose2D(x=-5.0, y=1.5), end=Pose2D(x=5.0, y=1.5)),)
IN_LANE = (0.0, 0.75)
ROBOT_RADIUS = 0.3
ROBOT_ID = 424242


def _scenario(walkers: list[AgentConfig], seed: int = 7, width: float = WIDTH, inner: tuple[WallConfig, ...] = ()) -> ScenarioConfig:
    return ScenarioConfig(
        name="robot_standing_reroute",
        simulation=SimulationParams(seed=seed, dt=DT, bt_tick_interval=5, max_ticks=0),
        modules=ModuleConfig(global_planner="navmesh"),
        agents=walkers,
        walls=[
            WallConfig(name="south", start=Pose2D(x=-LENGTH, y=0.0), end=Pose2D(x=LENGTH, y=0.0)),
            WallConfig(name="north", start=Pose2D(x=-LENGTH, y=width), end=Pose2D(x=LENGTH, y=width)),
            WallConfig(name="west", start=Pose2D(x=-LENGTH, y=0.0), end=Pose2D(x=-LENGTH, y=width)),
            WallConfig(name="east", start=Pose2D(x=LENGTH, y=0.0), end=Pose2D(x=LENGTH, y=width)),
            *inner,
        ],
    )


def _robot_at(tick: int, x: float, y: float, theta: float = 0.0) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    stamp_ms = round(tick * DT * 1000)
    msg.header.stamp.sec = stamp_ms // 1000
    msg.header.stamp.nanosec = stamp_ms % 1000 * 1_000_000
    robot = AgentStateMsg()
    robot.agent_id = ROBOT_ID
    robot.name = "robot_0"
    robot.pose = Pose2DMsg(x=x, y=y, theta=theta)
    robot.radius = ROBOT_RADIUS
    msg.agents.append(robot)
    return msg


def run(
    manager_factory: Callable[..., AgentManager],
    walkers: list[AgentConfig],
    robot: Callable[[float], tuple[float, float]],
    seconds: float,
    node_name: str,
    width: float = WIDTH,
    inner: tuple[WallConfig, ...] = (),
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Tick the corridor (`width` wide, with `inner` walls) for `seconds` with the robot at `robot(t)`; each walker's
    track (t, x, y) and the robot's."""
    mgr = manager_factory(_scenario(walkers, width=width, inner=inner), node_name=node_name)
    ids = [w.agent_id for w in walkers]
    tracks: dict[int, list[tuple[float, float, float]]] = {aid: [] for aid in ids}
    robot_track: list[tuple[float, float, float]] = []
    for tick in range(round(seconds / DT)):
        t = tick * DT
        rx, ry = robot(t)
        mgr._world_state_callback(_robot_at(tick, rx, ry))
        mgr.tick()
        robot_track.append((t, rx, ry))
        for aid in ids:
            agent = mgr._agents.get(aid)
            if agent is not None:
                tracks[aid].append((t, agent.state.pose.x, agent.state.pose.y))
    return {aid: np.array(tr) for aid, tr in tracks.items()}, np.array(robot_track)


def _walker(aid: int, agent_type: str, y: float, x: float = -10.0, goal_x: float = GOAL_X) -> AgentConfig:
    return AgentConfig(agent_id=aid, agent_type=agent_type, spawn_pose=Pose2D(x=x, y=y), goal_sequence=[Pose2D(x=goal_x, y=y)])


def _passed_at(track: np.ndarray, x: float) -> float | None:
    """When the walker first got past `x`, None if never."""
    beyond = np.flatnonzero(track[:, 1] > x)
    return float(track[beyond[0], 0]) if len(beyond) else None


@pytest.mark.parametrize("where", ["locker", "jammed"])
def test_long_cane_user_goes_round_a_robot_standing_on_its_wall_line(manager_factory: Callable[..., AgentManager], where: str) -> None:
    spot = PARKED if where == "locker" else JAMMED
    tracks, _ = run(manager_factory, [_walker(2, "blind_white_cane", 0.5)], lambda _t: spot, 60.0, f"test_reroute_cane_{where}")
    track = tracks[2]
    passed = _passed_at(track, spot[0] + 1.0)
    assert passed is not None, f"long-cane walker never got past the parked robot (last at x={track[-1, 1]:.2f})"
    # it reaches the robot after ~13 s at 0.75 m/s; robot still 3 s, walker waits 2 s: past it well within 10 s more
    assert passed < 30.0
    # it trailed the wall before the robot and trails it again behind
    before = track[(track[:, 1] > -6.0) & (track[:, 1] < -3.0)]
    after = track[(track[:, 1] > 3.0) & (track[:, 1] < 6.0)]
    assert len(before) and len(after)
    assert abs(float(np.median(before[:, 2])) - 0.42) < 0.15
    assert abs(float(np.median(after[:, 2])) - 0.42) < 0.15
    # never through the robot's body
    gap = np.hypot(track[:, 1] - spot[0], track[:, 2] - spot[1]) - ROBOT_RADIUS
    assert float(gap.min()) > 0.0


@pytest.mark.parametrize("where", ["locker", "jammed"])
def test_adult_goes_round_a_standing_robot(manager_factory: Callable[..., AgentManager], where: str) -> None:
    spot, y = (PARKED, 0.6) if where == "locker" else (JAMMED, 0.9)
    tracks, _ = run(manager_factory, [_walker(3, "adult", y)], lambda _t: spot, 45.0, f"test_reroute_adult_{where}")
    passed = _passed_at(tracks[3], spot[0] + 1.0)
    assert passed is not None, f"adult never got past the parked robot (last at x={tracks[3][-1, 1]:.2f})"
    assert passed < 25.0


def test_walkers_keep_off_a_moving_robot_without_swinging(manager_factory: Callable[..., AgentManager]) -> None:
    """A robot driving west along the south wall at 0.5 m/s, at the walkers: not on their maps (it moves), they pass it
    by the local forces, never inside its body, and do not swing back and forth across the corridor."""

    def driving(t: float) -> tuple[float, float]:
        return (8.0 - 0.5 * t, 0.45)

    walkers = [_walker(2, "blind_white_cane", 0.5), _walker(3, "adult", 0.6, x=-12.0)]
    tracks, robot = run(manager_factory, walkers, driving, 40.0, "test_reroute_moving")
    for aid, track in tracks.items():
        n = min(len(track), len(robot))
        gap = np.hypot(track[:n, 1] - robot[:n, 1], track[:n, 2] - robot[:n, 2]) - ROBOT_RADIUS
        assert float(gap.min()) > 0.0, f"walker {aid} inside the moving robot"
        # no swinging: the walker's heading across the corridor turns over at most a few times
        dy = np.sign(np.diff(track[::10, 2]))
        dy = dy[dy != 0]
        flips = int(np.count_nonzero(np.diff(dy)))
        assert flips <= 6, f"walker {aid} swings across the corridor ({flips} turns)"


def test_robot_standing_then_driving_off_leaves_routes_alone(manager_factory: Callable[..., AgentManager]) -> None:
    """A robot parked for 20 s, then driving off north-east: the walker went round it while it stood; once it moves it is
    off the walker's map again, and the walker does not stop or turn back."""

    def leaves(t: float) -> tuple[float, float]:
        if t < 20.0:
            return PARKED
        k = min(t - 20.0, 6.0)
        return (PARKED[0] + 0.5 * k, PARKED[1] + 0.3 * k)

    tracks, _ = run(manager_factory, [_walker(2, "blind_white_cane", 0.5)], leaves, 45.0, "test_reroute_leaves")
    track = tracks[2]
    assert _passed_at(track, 8.0) is not None
    x = track[:, 1]
    assert float(np.maximum.accumulate(x)[-1] - x[-1]) < 0.5
    back = np.maximum.accumulate(x) - x
    assert float(back.max()) < 1.0, "walker turned back"


def test_planner_marks_only_walkers_a_standing_robot_blocks(manager_factory: Callable[..., AgentManager]) -> None:
    """A walker far from the parked robot is never marked blocked; one that waits at it is."""
    mgr = manager_factory(_scenario([_walker(2, "blind_white_cane", 0.5), _walker(4, "adult", 2.4, x=-12.0, goal_x=-11.0)]), node_name="test_reroute_marks")
    marked: set[int] = set()
    for tick in range(round(25.0 / DT)):
        mgr._world_state_callback(_robot_at(tick, *PARKED))
        mgr.tick()
        marked |= set(mgr._global_planner.blockers())
    assert 4 not in marked
    assert 2 in marked
    assert math.isfinite(float(mgr._agents[2].state.pose.x))
