"""`start_on_request`: after a reset or remove-all the clock waits for start_clock, so a crowd started at the same epoch is
the same crowd at the same stamps, however far and in what steps the clock moved before."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import ScenarioConfig
from arena_humansim.utils.types import AgentTemplate, Pose2D, RateKeyframe, Shape, ShapeType, SourceConfig, SourceType
from arena_humansim_msgs.srv import RemoveAgents, ResetSimulation, StartClock
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock

NS = 1_000_000_000
DT_NS = 50_000_000
EPOCH_NS = 31 * NS + 7_000_000


def _clock(ns: int) -> Clock:
    msg = Clock()
    msg.clock.sec, msg.clock.nanosec = divmod(ns, NS)
    return msg


def _start(mgr: AgentManager, epoch_ns: int) -> StartClock.Response:
    request = StartClock.Request()
    request.epoch.sec, request.epoch.nanosec = divmod(epoch_ns, NS)
    return mgr._start_clock_callback(request, StartClock.Response())


def _remove_all(mgr: AgentManager) -> None:
    mgr._remove_agents_callback(RemoveAgents.Request(), RemoveAgents.Response())


def _stamp_ns(mgr: AgentManager) -> int:
    stamp = mgr._build_agent_frame().header.stamp
    return stamp.sec * NS + stamp.nanosec


def _subsystem(manager_factory: Callable[..., AgentManager], scenario: ScenarioConfig, name: str, start_on_request: bool = True) -> AgentManager:
    return manager_factory(scenario, node_name=name, extra_params=[Parameter("mode", Parameter.Type.STRING, "subsystem"), Parameter("start_on_request", Parameter.Type.BOOL, start_on_request)])


def _source() -> SourceConfig:
    return SourceConfig(
        name="src",
        pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        shape=Shape(type=ShapeType.CIRCLE, radius=2.0),
        type=SourceType.POISSON,
        rate_profile=[RateKeyframe(t=0.0, rate=3.0)],
        max_concurrent=-1,
        max_total=-1,
        agent=AgentTemplate(),
    )


def test_a_held_clock_runs_no_tick_until_start_clock_and_then_ticks_from_its_epoch(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = _subsystem(manager_factory, minimal_scenario, "test_start_clock_held")
    _remove_all(mgr)
    for ns in (EPOCH_NS - 3 * DT_NS, EPOCH_NS - 17_000_000, EPOCH_NS + 2 * DT_NS):
        mgr._subsystem_timer_callback(_clock(ns))
    assert mgr._tick_count == 0
    assert _start(mgr, EPOCH_NS).success
    mgr._subsystem_timer_callback(_clock(EPOCH_NS - 1))
    assert mgr._tick_count == 0, "a clock before the epoch runs no tick"
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + 2 * DT_NS + 13_000_000))
    assert (mgr._subsystem_epoch_ns, mgr._tick_count) == (EPOCH_NS, 3)
    assert _stamp_ns(mgr) == EPOCH_NS + 2 * DT_NS
    assert not _start(mgr, EPOCH_NS).success, "a running clock is not restarted"
    mgr._reset_callback(ResetSimulation.Request(), ResetSimulation.Response())
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + 9 * DT_NS))
    assert mgr._tick_count == 0, "a plain reset holds the clock again"


def test_without_start_on_request_the_first_clock_after_a_remove_all_is_the_epoch(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    mgr = _subsystem(manager_factory, minimal_scenario, "test_start_clock_off", start_on_request=False)
    _remove_all(mgr)
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + 3_000_000))
    assert (mgr._subsystem_epoch_ns, mgr._tick_count) == (EPOCH_NS + 3_000_000, 1)


def _released(mgr: AgentManager, before: list[int], after: list[int]) -> dict[int, dict[int, tuple[float, float]]]:
    """Feed the clock `before` (held), start at EPOCH_NS, feed `after`; the crowd (id -> x, y) at the stamp of each callback."""
    mgr._spawn_scheduler.add_source(_source())
    _remove_all(mgr)
    for ns in before:
        mgr._subsystem_timer_callback(_clock(ns))
    assert mgr._tick_count == 0
    assert _start(mgr, EPOCH_NS).success
    frames: dict[int, dict[int, tuple[float, float]]] = {}
    for ns in after:
        mgr._subsystem_timer_callback(_clock(ns))
        n = mgr._pool.n
        frames[_stamp_ns(mgr)] = {int(aid): (x, y) for aid, (x, y) in zip(mgr._pool.agent_ids[:n].tolist(), mgr._pool.pos[:n].tolist(), strict=True)}
    return frames


def test_a_crowd_started_at_the_same_epoch_is_the_same_crowd_whatever_the_clock_did_before(manager_factory: Callable[..., AgentManager], minimal_scenario: ScenarioConfig) -> None:
    """The episode boundary of a lockstep run moves the clock by a varying amount while a reset holds it (robot placement
    steps, bringup windows) and lands at any phase of the engine's tick: with start_on_request neither moves the crowd."""
    rng = np.random.default_rng(3)
    end = EPOCH_NS + 80 * DT_NS
    fine = list(range(EPOCH_NS, end, 10_000_000))
    coarse = sorted({int(t) for t in rng.integers(EPOCH_NS, end, 30)} | {end})
    a = _released(_subsystem(manager_factory, minimal_scenario, "test_start_clock_a"), before=[EPOCH_NS - 5 * DT_NS], after=fine + [end])
    b = _released(_subsystem(manager_factory, minimal_scenario, "test_start_clock_b"), before=[EPOCH_NS - 11 * DT_NS, EPOCH_NS - 33_000_000, EPOCH_NS - 1], after=coarse)
    assert all((stamp - EPOCH_NS) % DT_NS == 0 for stamp in a | b), "a tick off the epoch's grid"
    common = sorted(set(a) & set(b))
    assert len(common) >= 10 and max(common) == EPOCH_NS + 80 * DT_NS
    assert len(a[max(common)]) >= 5, "the source released too few agents to compare"
    for stamp in common:
        assert a[stamp] == b[stamp], f"the crowds differ at {stamp} ns"
