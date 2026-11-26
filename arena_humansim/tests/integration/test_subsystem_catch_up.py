"""Without lockstep a crowd that fell behind its clock catches up in slices: every owed tick runs, in order, and the node
serves its services between the slices instead of going silent for the whole catch-up."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

pytest.importorskip("rclpy")

import arena_humansim.core.agent_manager as agent_manager
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock

from tests.integration._golden import FIXTURES, spawn
from tests.integration._helpers import build_manager

NS = 1_000_000_000
EPOCH_NS = 3 * NS
OWED = 120


def _clock(ns: int) -> Clock:
    msg = Clock()
    msg.clock.sec = ns // NS
    msg.clock.nanosec = ns % NS
    return msg


_BUILT: list = []


@pytest.fixture(autouse=True)
def _destroy_managers():  # noqa: ANN202
    """Destroy the managers a test built, so none of them publishes into the next test's topics."""
    yield
    while _BUILT:
        _BUILT.pop().destroy_node()


def _crowd(name: str, barrier: bool = False):  # noqa: ANN202 - an AgentManager
    fixture = FIXTURES[0]
    mgr = build_manager(fixture.scenario(), node_name=name, extra_params=[Parameter("input_barrier", Parameter.Type.BOOL, barrier)])
    _BUILT.append(mgr)
    spawn(mgr, fixture.spawns)
    return mgr


def _state(mgr) -> bytes:  # noqa: ANN001
    n = mgr._pool.n
    return b"".join(np.ascontiguousarray(a[:n]).tobytes() for a in (mgr._pool.pos, mgr._pool.vel, mgr._pool.theta))


def _behind(mgr) -> int:  # noqa: ANN001
    """Start the clock, then a /clock `OWED` ticks ahead; the ticks the first catch-up slice ran."""
    mgr._subsystem_timer_callback(_clock(EPOCH_NS))
    dt_ns = int(mgr._dt * NS)
    before = mgr._tick_count
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + OWED * dt_ns))
    return mgr._tick_count - before


def test_a_long_catch_up_runs_every_owed_tick_in_slices_as_one_batch_would(monkeypatch: pytest.MonkeyPatch) -> None:
    whole = _crowd("catch_up_whole")
    monkeypatch.setattr(agent_manager, "CATCH_UP_SLICE_S", 1e9)
    assert _behind(whole) == OWED
    assert whole._catch_up_timer is None

    sliced = _crowd("catch_up_sliced")
    monkeypatch.setattr(agent_manager, "CATCH_UP_SLICE_S", 0.0)
    assert _behind(sliced) == 1
    timer = sliced._catch_up_timer
    assert timer is not None and not timer.is_canceled()
    slices = 1
    while sliced._owed() > 0:
        sliced._catch_up_callback()
        slices += 1
    sliced._catch_up_callback()
    assert slices == OWED
    assert timer.is_canceled()
    assert sliced._tick_count == whole._tick_count == OWED + 1
    assert _state(sliced) == _state(whole)


def test_a_clock_that_waited_behind_a_slice_does_not_set_the_catch_up_back(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _crowd("catch_up_stale")
    monkeypatch.setattr(agent_manager, "CATCH_UP_SLICE_S", 0.0)
    _behind(mgr)
    dt_ns = int(mgr._dt * NS)
    newest = mgr._newest_covered
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + 10 * dt_ns))
    assert mgr._newest_covered == newest == OWED + 1
    assert mgr._owed() == OWED - 2


def test_the_catch_up_is_sent_with_its_peaks_and_once_it_caught_up(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _crowd("catch_up_report")
    sent: list[dict] = []
    monkeypatch.setattr(mgr._catch_up_pub, "publish", lambda msg: sent.append(json.loads(msg.data)))
    monkeypatch.setattr(agent_manager, "CATCH_UP_SLICE_S", 0.0)
    _behind(mgr)
    while mgr._owed() > 0:
        mgr._catch_up_callback()
    assert sent[0]["owed"] == OWED and sent[0]["left"] == OWED - 1 and sent[0]["peak_owed"] == OWED
    assert sent[0]["lag_s"] == pytest.approx((OWED - 1) * mgr._dt)
    assert sent[-1]["left"] == 0 and sent[-1]["peak_owed"] == OWED
    assert sent[-1]["peak_lag_s"] == pytest.approx((OWED - 1) * mgr._dt)
    count = len(sent)
    dt_ns = int(mgr._dt * NS)
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + (mgr._newest_covered) * dt_ns))
    assert len(sent) == count, "a crowd on time sends nothing"


def test_a_reset_forgets_the_catch_up(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _crowd("catch_up_reset")
    monkeypatch.setattr(agent_manager, "CATCH_UP_SLICE_S", 0.0)
    _behind(mgr)
    mgr._hold_clock()
    assert mgr._catch_up_peaks == {}
    assert mgr._catch_up_timer is not None and mgr._catch_up_timer.is_canceled()


def _forced(name: str, slice_s: float, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202 - an AgentManager
    """A crowd whose input barrier held `OWED` ticks until its watchdog forced them, with a later /clock coming in while
    the forced catch-up runs (between its slices when it is sliced)."""
    mgr = _crowd(name, barrier=True)
    monkeypatch.setattr(agent_manager, "CATCH_UP_SLICE_S", slice_s)
    assert _behind(mgr) == 0, "the barrier holds the clock without a world_state"
    mgr._barrier_wait_since = time.perf_counter() - 1000.0
    mgr._barrier_watchdog()
    dt_ns = int(mgr._dt * NS)
    mgr._subsystem_timer_callback(_clock(EPOCH_NS + (OWED + 30) * dt_ns))
    while mgr._catch_up_timer is not None and not mgr._catch_up_timer.is_canceled():
        mgr._catch_up_callback()
    return mgr


def test_a_forced_catch_up_in_slices_ends_where_the_watchdog_forced_it_as_one_batch_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """A catch-up the barrier watchdog forced stays forced across its slices up to the ticks it was forced to; the
    ticks of a clock that came in meanwhile wait for the barrier, as after one forced batch."""
    whole = _forced("catch_up_forced_whole", 1e9, monkeypatch)
    sliced = _forced("catch_up_forced_sliced", 0.0, monkeypatch)
    assert whole._tick_count == sliced._tick_count == OWED + 1
    assert _state(sliced) == _state(whole)
    assert sliced._forced_covered is None and sliced._barrier_covered == OWED + 31
