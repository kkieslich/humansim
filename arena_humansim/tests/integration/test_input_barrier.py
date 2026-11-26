"""`input_barrier`: an orchestrator that decides its inputs on the engine's frames gets the same crowd however its
messages race the clock.

Under lockstep the orchestrator sees the frame of tick k, decides its inputs (a walker released, a new leg, the robot's
pose) and sends them while the clock already moves on to tick k + 1. Without the barrier an input lands before tick k + 1
or before tick k + 2, whichever message the engine takes first, and a tick earlier or later is enough for a crowd to
take another path: the same world and seed gave walker tracks 2-10 cm apart. With the barrier tick k + 1 waits for the
world_state stamped with tick k's stamp, which the orchestrator sends once every input for tick k is in.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig, load_scenario
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.msg import Waypoint as WaypointMsg
from arena_humansim_msgs.msg import Waypoints as WaypointsMsg
from arena_humansim_msgs.srv import RemoveAgents, SetWaypoints, SpawnAgents, StartClock
from geometry_msgs.msg import Pose2D as RosPose2D
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock

NS = 1_000_000_000
DT_NS = 50_000_000
EPOCH_NS = 12 * NS + 350_000_000
SCENARIOS = Path(__file__).resolve().parents[2] / "config" / "scenarios"


def _clock(ns: int) -> Clock:
    msg = Clock()
    msg.clock.sec, msg.clock.nanosec = divmod(ns, NS)
    return msg


def _subsystem(manager_factory: Callable[..., AgentManager], scenario: ScenarioConfig, name: str, barrier: bool) -> AgentManager:
    mgr = manager_factory(
        scenario,
        node_name=name,
        extra_params=[
            Parameter("mode", Parameter.Type.STRING, "subsystem"),
            Parameter("start_on_request", Parameter.Type.BOOL, True),
            Parameter("input_barrier", Parameter.Type.BOOL, barrier),
        ],
    )
    return mgr


def _start(mgr: AgentManager, epoch_ns: int = EPOCH_NS) -> None:
    request = StartClock.Request()
    request.epoch.sec, request.epoch.nanosec = divmod(epoch_ns, NS)
    assert mgr._start_clock_callback(request, StartClock.Response()).success


def _frame_stamp_ns(mgr: AgentManager) -> int:
    assert mgr._last_frame is not None
    stamp = mgr._last_frame.header.stamp
    return stamp.sec * NS + stamp.nanosec


def _world_state(stamp_ns: int, robot: tuple[float, float] | None) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, NS)
    if robot is not None:
        agent = AgentStateMsg()
        agent.agent_id = 9001
        agent.name = "jackal"
        agent.pose = RosPose2D(x=robot[0], y=robot[1], theta=0.0)
        agent.radius = 0.31
        msg.agents.append(agent)
    return msg


def _walker(x: float, y: float, gx: float, gy: float) -> AgentStateMsg:
    msg = AgentStateMsg()
    msg.pose = RosPose2D(x=x, y=y, theta=0.0)
    msg.desired_velocity = 1.2
    msg.agent_type = "adult"
    msg.waypoints = WaypointsMsg(points=[WaypointMsg(pose=RosPose2D(x=gx, y=gy, theta=0.0))], mode=WaypointsMsg.MODE_ONCE)
    return msg


class _Orchestrator:
    """The adapter's part: on the frame of tick k it decides inputs from that frame's stamp alone and sends them."""

    def __init__(self, mgr: AgentManager, robots: bool, epoch_ns: int = EPOCH_NS) -> None:
        self.mgr = mgr
        self.robots = robots
        self.epoch_ns = epoch_ns

    def robot_pose(self, t: float) -> tuple[float, float] | None:
        # parked beside the door, then driving through it at 0.4 m/s
        return None if not self.robots else (-1.5 + 0.4 * max(t - 1.0, 0.0), -0.5)

    def inputs(self, stamp_ns: int) -> list[Callable[[], None]]:
        k = (stamp_ns - self.epoch_ns) // DT_NS
        calls: list[Callable[[], None]] = []
        if k == 7:  # a delayed release
            request = SpawnAgents.Request()
            request.agents = [_walker(-6.0, 1.5, 6.0, 1.5), _walker(6.0, -1.5, -6.0, -1.5), _walker(-7.0, 0.0, 7.0, 0.0)]
            calls.append(lambda: self.mgr._spawn_agents_callback(request, SpawnAgents.Response()))
        if k == 23:  # the next leg of an itinerary
            leg = SetWaypoints.Request()
            leg.agent_id = int(self.mgr._pool.agent_ids[0])
            leg.waypoints = WaypointsMsg(points=[WaypointMsg(pose=RosPose2D(x=-6.0, y=-2.0, theta=0.0))], mode=WaypointsMsg.MODE_ONCE)
            calls.append(lambda: self.mgr._set_waypoints_callback(leg, SetWaypoints.Response()))
        return calls


def _run(manager_factory: Callable[..., AgentManager], scenario: ScenarioConfig, name: str, *, barrier: bool, robots: bool, order_seed: int, ticks: int, epoch_ns: int = EPOCH_NS) -> np.ndarray:
    """Tracks (tick, id, x, y) of a run whose clock messages come in before or after the orchestrator's inputs, as `order_seed` draws."""
    mgr = _subsystem(manager_factory, scenario, name, barrier)
    mgr._remove_agents_callback(RemoveAgents.Request(), RemoveAgents.Response())
    mgr._apply_scenario(scenario)
    _start(mgr, epoch_ns)
    orchestrator = _Orchestrator(mgr, robots, epoch_ns)
    order = random.Random(order_seed)
    clock_ns = epoch_ns
    mgr._subsystem_timer_callback(_clock(clock_ns))
    seen = -1
    rows: list[np.ndarray] = []
    for _ in range(4 * ticks):
        if mgr._tick_count >= ticks:
            break
        stamp_ns = _frame_stamp_ns(mgr)
        if stamp_ns > seen:
            seen = stamp_ns
            n = mgr._pool.n
            rows.append(np.column_stack([np.full(n, mgr._tick_count - 1), mgr._pool.agent_ids[:n], mgr._pool.pos[:n, 0], mgr._pool.pos[:n, 1]]))
            # lockstep moves the clock on once the frame is out; the orchestrator's inputs race that clock
            calls = [*orchestrator.inputs(stamp_ns), lambda s=stamp_ns: mgr._world_state_callback(_world_state(s, orchestrator.robot_pose((s - epoch_ns) // DT_NS * 0.05)))]
            clock_ns = stamp_ns + DT_NS
            clock_first = order.random() < 0.5
            if clock_first:
                mgr._subsystem_timer_callback(_clock(clock_ns))
            for call in calls:
                call()
            if not clock_first:
                mgr._subsystem_timer_callback(_clock(clock_ns))
        else:
            mgr._subsystem_timer_callback(_clock(clock_ns))
    assert mgr._tick_count >= ticks, f"stalled at tick {mgr._tick_count}"
    return np.concatenate(rows)


@pytest.mark.parametrize("scenario_name", ["bottleneck", "corridor"])
@pytest.mark.parametrize("robots", [True, False])
def test_with_the_barrier_racing_inputs_give_the_same_crowd(manager_factory: Callable[..., AgentManager], scenario_name: str, robots: bool) -> None:
    scenario = load_scenario(str(SCENARIOS / f"{scenario_name}.yaml"))
    runs = [_run(manager_factory, scenario, f"barrier_{scenario_name}_{robots}_{i}", barrier=True, robots=robots, order_seed=i, ticks=80) for i in range(3)]
    for other in runs[1:]:
        assert other.shape == runs[0].shape
        assert np.array_equal(other, runs[0]), f"max deviation {np.abs(other - runs[0]).max():.3g} m"


def test_the_crowd_does_not_depend_on_the_epoch(manager_factory: Callable[..., AgentManager]) -> None:
    """A moving robot's velocity comes from whole-ns stamp differences: two epochs, the same crowd bit for bit."""
    scenario = load_scenario(str(SCENARIOS / "bottleneck.yaml"))
    a = _run(manager_factory, scenario, "epoch_a", barrier=True, robots=True, order_seed=0, ticks=80)
    b = _run(manager_factory, scenario, "epoch_b", barrier=True, robots=True, order_seed=1, ticks=80, epoch_ns=987 * NS + 123_450_000)
    assert a.shape == b.shape
    assert np.array_equal(a, b), f"max deviation {np.abs(a - b).max():.3g} m"


def test_without_the_barrier_the_same_inputs_land_a_tick_apart(manager_factory: Callable[..., AgentManager]) -> None:
    """The cause the barrier removes: with the old protocol the race decides the tick an input lands before."""
    scenario = load_scenario(str(SCENARIOS / "bottleneck.yaml"))
    runs = [_run(manager_factory, scenario, f"nobarrier_{i}", barrier=False, robots=True, order_seed=i, ticks=80) for i in range(4)]
    assert any(r.shape != runs[0].shape or not np.array_equal(r, runs[0]) for r in runs[1:])


def test_the_barrier_holds_a_tick_until_its_acknowledgement(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = _subsystem(manager_factory, minimal_scenario, "barrier_holds", barrier=True)
    mgr._remove_agents_callback(RemoveAgents.Request(), RemoveAgents.Response())
    _start(mgr)
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + 4 * DT_NS))
    assert mgr._tick_count == 1, "tick 0 needs no acknowledgement, tick 1 waits for one"
    mgr._world_state_callback(_world_state(EPOCH_NS + 3 * DT_NS, None))
    assert mgr._tick_count == 1, "a world_state stamped ahead of the ticks that ran acknowledges nothing"
    mgr._world_state_callback(_world_state(EPOCH_NS, None))
    assert mgr._tick_count == 2
    mgr._world_state_callback(_world_state(EPOCH_NS + DT_NS, None))
    assert mgr._tick_count == 3
    mgr._world_state_callback(_world_state(EPOCH_NS + 2 * DT_NS, None))
    mgr._world_state_callback(_world_state(EPOCH_NS + 3 * DT_NS, None))
    assert mgr._tick_count == 5, "the clock covers 5 ticks, the barrier no more"
    mgr._world_state_callback(_world_state(EPOCH_NS + 4 * DT_NS, None))
    assert mgr._tick_count == 5


def test_a_missing_acknowledgement_runs_the_ticks_after_the_timeout(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = _subsystem(manager_factory, minimal_scenario, "barrier_timeout", barrier=True)
    mgr.set_parameters([Parameter("input_barrier_timeout", Parameter.Type.DOUBLE, 0.01)])
    mgr._remove_agents_callback(RemoveAgents.Request(), RemoveAgents.Response())
    _start(mgr)
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + 2 * DT_NS))
    assert mgr._tick_count == 1
    assert mgr._barrier_wait_since is not None
    mgr._barrier_wait_since -= 1.0
    published: list[int] = []
    mgr._barrier_timeouts_pub = type("Pub", (), {"publish": lambda self, msg: published.append(msg.data)})()
    mgr._barrier_watchdog()
    assert mgr._tick_count == 3
    assert mgr._barrier_timeouts == 1
    assert published == [1], "the count goes out on input_barrier_timeouts"
    mgr._remove_agents_callback(RemoveAgents.Request(), RemoveAgents.Response())
    assert mgr._barrier_timeouts == 0 and published == [1, 0], "a new crowd starts from 0"


def test_a_remove_all_drops_the_old_crowds_robot_poses(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = _subsystem(manager_factory, minimal_scenario, "barrier_stale_robot", barrier=True)
    mgr._world_state_callback(_world_state(EPOCH_NS - 5 * DT_NS, (1.0, 2.0)))
    mgr._remove_agents_callback(RemoveAgents.Request(), RemoveAgents.Response())
    assert mgr._latest_world_state is None
