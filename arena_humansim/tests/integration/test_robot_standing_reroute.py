"""Walkers blocked by a robot that stands still go round it; robots that move stay left to the local forces.

A corridor 3 m wide, the robot fed in through world_state as Arena feeds its robots, standing (waiting at a locker,
jammed) on the line long-cane users trail along the south wall, or in a lane too narrow to pass it. Before the global
planner knew standing robots, a walker pinned at the robot, or one whose lane it closed, stood there for good; now the
first goes round it and back onto its line, the second takes the other lane, and one with no way at all waits. A robot driving along the corridor stays off the
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
#: A robot standing on the long-cane users' trailing line, 0.67 m off the wall
ON_LINE = (0.0, 0.67)
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


def _line(track: np.ndarray, lo: float, hi: float) -> float:
    """The walker's median distance from the south wall while it is between x `lo` and `hi`."""
    part = track[(track[:, 1] > lo) & (track[:, 1] < hi)]
    assert len(part), f"walker never between x {lo} and {hi}"
    return float(np.median(part[:, 2]))


def test_long_cane_users_trailing_into_a_standing_robot_go_round_it(manager_factory: Callable[..., AgentManager]) -> None:
    """Two long-cane users trail the south wall, an adult follows; the robot stands on their trailing line (waiting at a
    locker, or jammed). The second long-cane user ends up pinned between robot and wall: before, it stood there for good;
    now it steps off the wall line, passes, and trails the wall again behind the robot."""
    walkers = [_walker(2, "blind_white_cane", 0.5), _walker(6, "blind_white_cane", 0.5, x=-12.0), _walker(7, "adult", 0.6, x=-13.0)]
    tracks, _ = run(manager_factory, walkers, lambda _t: ON_LINE, 60.0, "test_reroute_cane_queue")
    for aid, track in tracks.items():
        passed = _passed_at(track, ON_LINE[0] + 1.5)
        assert passed is not None, f"walker {aid} never got past the standing robot (last at x={track[-1, 1]:.2f}, y={track[-1, 2]:.2f})"
        assert passed < 45.0
    for aid in (2, 6):
        assert abs(_line(tracks[aid], 3.0, 8.0) - _line(tracks[aid], -6.0, -3.0)) < 0.15, f"walker {aid} not back on its wall line"


def test_wheelchair_takes_the_other_lane_when_a_robot_closes_its_lane(manager_factory: Callable[..., AgentManager]) -> None:
    """A robot stands in the middle of the 1.5 m lane, too narrow to pass it; the wheelchair user turns back and takes
    the 2.5 m lane instead (before: it stood at the robot for good)."""
    tracks, _ = run(manager_factory, [_walker(5, "wheelchair_manual", 0.75)], lambda _t: IN_LANE, 60.0, "test_reroute_bypass", width=4.0, inner=DIVIDER)
    track = tracks[5]
    assert _passed_at(track, 6.0) is not None, f"wheelchair never got past the closed lane (last at x={track[-1, 1]:.2f}, y={track[-1, 2]:.2f})"
    middle = track[(track[:, 1] > -4.0) & (track[:, 1] < 4.0)]
    assert float(middle[:, 2].max()) > 1.5 and _passed_at(track, IN_LANE[0]) is not None
    assert float(track[(np.abs(track[:, 1] - IN_LANE[0]) < 0.5), 2].min()) > 1.5, "wheelchair went through the closed lane"


def test_walker_with_no_way_round_waits(manager_factory: Callable[..., AgentManager]) -> None:
    """The robot closes the only way (a 1.5 m corridor): the walker waits before it, as before, and does not go through."""
    tracks, _ = run(manager_factory, [_walker(5, "wheelchair_manual", 0.75)], lambda _t: IN_LANE, 40.0, "test_reroute_no_way", width=1.5)
    track = tracks[5]
    assert float(track[:, 1].max()) < IN_LANE[0]
    assert float(track[-1, 1]) > -4.0


def test_walkers_keep_off_a_moving_robot_without_swinging(manager_factory: Callable[..., AgentManager]) -> None:
    """A robot driving west through the corridor at 0.5 m/s, past the walkers: never on their maps (it moves), so their
    routes stay as the static map makes them and the local forces alone keep them off it, as before."""

    def driving(t: float) -> tuple[float, float]:
        return (10.0 - 0.5 * t, 1.6)

    walkers = [_walker(2, "blind_white_cane", 0.5), _walker(3, "adult", 1.4, x=-12.0)]
    mgr = manager_factory(_scenario(walkers), node_name="test_reroute_moving")
    for tick in range(round(40.0 / DT)):
        rx, ry = driving(tick * DT)
        mgr._world_state_callback(_robot_at(tick, rx, ry))
        mgr.tick()
        assert not mgr._global_planner.still_robots(), "a moving robot counted as standing"
        assert not mgr._global_planner.blockers(), "a moving robot blocked a walker's route"
        for aid in (2, 3):
            pose = mgr._agents[aid].state.pose
            assert math.hypot(pose.x - rx, pose.y - ry) > ROBOT_RADIUS, f"walker {aid} inside the moving robot"
    for aid in (2, 3):
        assert mgr._agents[aid].state.pose.x > 5.0


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
    """A walker far from the standing robot is never marked blocked; one pinned at it is, and only by that robot."""
    walkers = [_walker(2, "blind_white_cane", 0.5), _walker(6, "blind_white_cane", 0.5, x=-12.0), _walker(4, "adult", 2.4, x=-12.0, goal_x=-11.0)]
    mgr = manager_factory(_scenario(walkers), node_name="test_reroute_marks")
    marked: dict[int, set[int]] = {}
    for tick in range(round(30.0 / DT)):
        mgr._world_state_callback(_robot_at(tick, *ON_LINE))
        mgr.tick()
        for aid, robots in mgr._global_planner.blockers().items():
            marked.setdefault(aid, set()).update(robots)
    assert 4 not in marked
    assert marked.get(6) == {mgr._external_entities[ROBOT_ID].agent_id}
