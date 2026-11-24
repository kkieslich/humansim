from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.types import BehaviorTreeMovement, GestureIntent
from arena_humansim_msgs.msg import AgentFrame as AgentFrameMsg
from arena_humansim_msgs.msg import AgentGestures as AgentGesturesMsg
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from arena_humansim_msgs.srv import RemoveAgents, ResetSimulation, SpawnAgents
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose2D as RosPose2D
from rclpy.executors import SingleThreadedExecutor
from rosgraph_msgs.msg import Clock

from ._capture import LATCHED, STREAM, settle, spin_until, subscribe

BT_INTERVAL = 5
EPOCH_NS = 7_000_000_000
DT_NS = 50_000_000


def _manager(manager_factory: Callable[..., AgentManager], name: str) -> AgentManager:
    scenario = ScenarioConfig(name=name, simulation=SimulationParams(seed=1, dt=0.05, max_ticks=0, bt_tick_interval=BT_INTERVAL), modules=ModuleConfig())
    return manager_factory(scenario, node_name=f"test_{name}")


def _spawn(mgr: AgentManager, x: float, name: str = "") -> int:
    req = SpawnAgents.Request()
    msg = AgentStateMsg()
    msg.name = name
    msg.pose = RosPose2D(x=x, y=0.0, theta=0.0)
    msg.desired_velocity = 1.3
    msg.radius = 0.3
    msg.agent_type = "adult"
    req.agents.append(msg)
    resp = SpawnAgents.Response()
    mgr._spawn_agents_callback(req, resp)
    return resp.spawned_ids[0]


def _remove(mgr: AgentManager, aid: int) -> None:
    req = RemoveAgents.Request()
    req.agent_ids = [aid]
    mgr._remove_agents_callback(req, RemoveAgents.Response())


def _clock_tick(mgr: AgentManager) -> Time:
    """One tick driven by the subsystem clock, returns the stamp its frame carries."""
    ns = EPOCH_NS + mgr._tick_count * DT_NS
    stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
    mgr._subsystem_timer_callback(Clock(clock=stamp))
    return stamp


def _tick_to_bt_tick(mgr: AgentManager) -> None:
    """Tick until the next tick is a behavior tree tick."""
    while mgr._tick_count % BT_INTERVAL != 0:
        mgr.tick()


def test_meta_published_only_on_change(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "meta_on_change")
    metas = subscribe(mgr, AgentMetaMsg, "agent_meta", LATCHED)
    frames = subscribe(mgr, AgentFrameMsg, "agent_states", STREAM)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)

    def expect_meta(count: int, stamp: Time) -> AgentMetaMsg:
        spin_until(executor, lambda: len(metas) == count and bool(frames) and frames[-1].header.stamp == stamp)
        assert metas[-1].header.stamp == stamp
        return metas[-1]

    meta = expect_meta(1, _clock_tick(mgr))
    assert list(meta.policies) == [mgr._policy_names[0]]
    assert list(meta.agent_id) == []

    first = _spawn(mgr, 0.0, name="first")
    second = _spawn(mgr, 2.0)
    meta = expect_meta(2, _clock_tick(mgr))
    assert list(meta.agent_id) == [first, second]
    assert list(meta.name) == ["first", ""]

    for _ in range(2 * BT_INTERVAL):
        _clock_tick(mgr)
    settle(executor)
    assert len(metas) == 2

    world = AgentStatesMsg()
    world.agents.append(AgentStateMsg(agent_id=second, name="tracked", pose=RosPose2D(x=2.0, y=0.0, theta=0.0), radius=0.3))
    mgr._world_state_callback(world)
    meta = expect_meta(3, _clock_tick(mgr))
    assert list(meta.agent_id) == [first, second]
    assert list(meta.name) == ["first", "tracked"]

    mgr._world_state_callback(world)
    _clock_tick(mgr)
    settle(executor)
    assert len(metas) == 3

    _remove(mgr, first)
    meta = expect_meta(4, _clock_tick(mgr))
    assert list(meta.agent_id) == [second]
    assert list(meta.name) == ["tracked"]

    mgr._reset_callback(ResetSimulation.Request(soft=True), ResetSimulation.Response())
    meta = expect_meta(5, _clock_tick(mgr))
    assert (list(meta.agent_id), list(meta.name)) == ([second], ["tracked"])

    mgr._reset_callback(ResetSimulation.Request(), ResetSimulation.Response())
    meta = expect_meta(6, _clock_tick(mgr))
    assert list(meta.agent_id) == []
    mgr._reset_callback(ResetSimulation.Request(), ResetSimulation.Response())
    meta = expect_meta(7, _clock_tick(mgr))
    assert list(meta.agent_id) == []
    executor.remove_node(mgr)


def test_gestures_published_only_on_change(manager_factory: Callable[..., AgentManager]) -> None:
    mgr = _manager(manager_factory, "gestures_on_change")
    gestures = subscribe(mgr, AgentGesturesMsg, "agent_gestures", LATCHED)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)
    mgr.tick()
    spin_until(executor, lambda: len(gestures) == 1)
    assert list(gestures[0].gestures) == []

    ped = _spawn(mgr, 0.0)
    other = _spawn(mgr, 2.0)
    for _ in range(BT_INTERVAL + 1):
        mgr.tick()
    settle(executor)
    assert len(gestures) == 1

    mv = BehaviorTreeMovement(gestures=(GestureIntent("arm", 1.0, 2.0, 3.0, hand="l"),))
    mgr._agents[ped].movement = mv
    _tick_to_bt_tick(mgr)
    settle(executor)
    assert len(gestures) == 1
    mgr.tick()
    spin_until(executor, lambda: len(gestures) == 2)
    assert list(gestures[1].agent_id) == [ped]
    assert [(g.slot, g.hand, g.at.x) for g in gestures[1].gestures] == [("arm", "l", 1.0)]

    for _ in range(2 * BT_INTERVAL):
        mgr.tick()
    settle(executor)
    assert len(gestures) == 2

    mv.gestures = ()
    _tick_to_bt_tick(mgr)
    mgr.tick()
    spin_until(executor, lambda: len(gestures) == 3)
    assert list(gestures[2].agent_id) == []

    mv.gestures = (GestureIntent("head", 4.0, 5.0, 6.0),)
    _tick_to_bt_tick(mgr)
    mgr.tick()
    spin_until(executor, lambda: len(gestures) == 4)
    assert list(gestures[3].agent_id) == [ped]

    assert mgr._tick_count % BT_INTERVAL != 0
    _remove(mgr, ped)
    mgr.tick()
    spin_until(executor, lambda: len(gestures) == 5)
    assert list(gestures[4].agent_id) == []

    _remove(mgr, other)
    mgr.tick()
    settle(executor)
    assert len(gestures) == 5
    executor.remove_node(mgr)


def test_a_repeated_clock_republishes_the_last_frame(manager_factory: Callable[..., AgentManager]) -> None:
    """A gate holding the clock repeats it to recover a frame it missed: the engine answers with its last frame, and a clock that advances without owing a tick stays silent."""
    mgr = _manager(manager_factory, "repeat_clock")
    frames = subscribe(mgr, AgentFrameMsg, "agent_states", STREAM)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)

    _clock_tick(mgr)
    stamp = _clock_tick(mgr)
    spin_until(executor, lambda: len(frames) == 2)
    ticks = mgr._tick_count

    mgr._subsystem_timer_callback(Clock(clock=stamp))
    spin_until(executor, lambda: len(frames) == 3)
    assert frames[-1].header.stamp == stamp
    assert mgr._tick_count == ticks

    ns = stamp.sec * 1_000_000_000 + stamp.nanosec + DT_NS // 2
    mgr._subsystem_timer_callback(Clock(clock=Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)))
    settle(executor)
    assert len(frames) == 3
    assert mgr._tick_count == ticks
