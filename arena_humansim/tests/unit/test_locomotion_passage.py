"""Lane and hold commands of the planner on the locomotion extension (`LocomotionExtension.set_passage`): a held row
stops at its hold point, rolls back behind it when told to, makes no stall recovery there, and a row in a lane is
flagged for the planner that owns its heading."""

from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import SampledLocomotion
from arena_humansim.core.locomotion import MODE_NORMAL, REVERSE_SPEED, LocomotionExtension
from arena_humansim.core.pool import AgentPool
from arena_humansim.global_planner import PassageCommand
from arena_humansim.utils.types import Pose2D

DT = 0.05


def _chair(agent_factory: Callable[..., BaseAgent], aid: int, x: float = 0.0, **loc: object) -> BaseAgent:
    agent = agent_factory(agent_id=aid, x=x)
    profile = SampledLocomotion(active=True, kinematics=1, cadence_base=1.0, cadence_per_speed=0.0, cadence_min=1.0, cadence_max=1.0, footprint_length=1.1, **loc)
    agent.params = attrs.evolve(agent.params, locomotion=profile, max_acceleration=1.2)
    return agent


def _pool(*agents: BaseAgent) -> tuple[AgentPool, LocomotionExtension]:
    pool = AgentPool(capacity=8)
    ext = LocomotionExtension()
    ext.attach(pool)
    for agent in agents:
        pool.add_agent(agent)
    return pool, ext


def _drive(pool: AgentPool, ext: LocomotionExtension, steps: int, speed: float = 0.9) -> list[float]:
    """Each tick a planner asks for `speed` along +x; the extension constrains it and the row moves."""
    speeds = []
    heading = np.ones(pool.n, dtype=np.bool_)
    for _ in range(steps):
        pool.vel[: pool.n] = (speed, 0.0)
        ext.constrain(pool, DT, heading)
        pool.pos[: pool.n] += pool.vel[: pool.n] * DT
        speeds.append(float(pool.vel[0, 0]))
    return speeds


def test_a_held_row_slows_down_and_stops_at_its_hold_point(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(_chair(agent_factory, 1))
    ext.set_passage(pool, {1: PassageCommand(hold=(2.0, 0.0, 1.0, 0.0))})
    speeds = _drive(pool, ext, 200)
    assert speeds[:30] == [0.9] * 30
    assert pool.pos[0, 0] == pytest.approx(2.0, abs=0.02)
    assert pool.pos[0, 0] <= 2.0 + 1e-9
    assert speeds[-1] < 0.02
    assert all(b <= a + 1e-12 for a, b in zip(speeds[:-1], speeds[1:], strict=True))


def test_a_row_released_drives_on(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(_chair(agent_factory, 1))
    ext.set_passage(pool, {1: PassageCommand(hold=(1.0, 0.0, 1.0, 0.0))})
    _drive(pool, ext, 100)
    assert pool.pos[0, 0] == pytest.approx(1.0, abs=0.02)
    ext.set_passage(pool, {})
    assert _drive(pool, ext, 20)[-1] == 0.9
    assert pool.pos[0, 0] > 1.5


def test_without_commands_nothing_changes(agent_factory: Callable[..., BaseAgent]) -> None:
    plain, ext_plain = _pool(_chair(agent_factory, 1))
    told, ext_told = _pool(_chair(agent_factory, 1))
    ext_told.set_passage(told, {})
    assert _drive(plain, ext_plain, 50) == _drive(told, ext_told, 50)
    assert np.array_equal(plain.pos[:1], told.pos[:1])
    assert not told.steer_to_goal[:1].any()


def test_a_row_told_to_roll_back_does_so_until_it_is_behind_its_hold_point(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(_chair(agent_factory, 1, x=1.0))
    ext.set_passage(pool, {1: PassageCommand(hold=(0.4, 0.0, 1.0, 0.0), back=True)})
    speeds = _drive(pool, ext, 60)
    assert speeds[0] == -REVERSE_SPEED
    assert pool.theta[0] == 0.0
    assert pool.pos[0, 0] == pytest.approx(0.4, abs=REVERSE_SPEED * DT + 1e-9)
    assert abs(speeds[-1]) < 1e-9

    standing, ext_standing = _pool(_chair(agent_factory, 1, x=1.0))
    ext_standing.set_passage(standing, {1: PassageCommand(hold=(0.4, 0.0, 1.0, 0.0))})
    assert _drive(standing, ext_standing, 60) == [0.0] * 60
    assert standing.pos[0, 0] == 1.0


def test_a_held_row_makes_no_stall_recovery(agent_factory: Callable[..., BaseAgent]) -> None:
    def stalled(hold: bool) -> int:
        pool, ext = _pool(_chair(agent_factory, 1, recovery_stall_after_s=0.5, recovery_reverse_m=0.3))
        pool.set_goals({1: Pose2D(x=10.0, y=0.0)})
        pool.desired_vel[0] = 0.9
        if hold:
            ext.set_passage(pool, {1: PassageCommand(hold=(0.0, 0.0, 1.0, 0.0))})
        heading = np.ones(1, dtype=np.bool_)
        modes = set()
        for _ in range(60):
            pool.vel[0] = (0.0, 0.0)
            ext.constrain(pool, DT, heading)
            modes.add(int(ext.mode[0]))
        return len(modes - {MODE_NORMAL})

    assert stalled(hold=False) > 0
    assert stalled(hold=True) == 0


def test_a_lane_command_flags_the_row_until_the_next_call(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(_chair(agent_factory, 1), _chair(agent_factory, 2, x=3.0))
    ext.set_passage(pool, {2: PassageCommand(lane=True)})
    assert pool.steer_to_goal[:2].tolist() == [False, True]
    ext.set_passage(pool, {1: PassageCommand(lane=True)})
    assert pool.steer_to_goal[:2].tolist() == [True, False]
    ext.set_passage(pool, {})
    assert pool.steer_to_goal[:2].tolist() == [False, False]


def test_rows_without_a_locomotion_profile_take_holds_but_no_lanes(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(agent_factory(agent_id=1), agent_factory(agent_id=2, x=3.0))
    ext.set_passage(pool, {1: PassageCommand(lane=True, hold=(0.0, 0.0, 1.0, 0.0)), 2: PassageCommand(lane=True), 99: PassageCommand(lane=True)})
    assert not pool.steer_to_goal[0] and not pool.steer_to_goal[1]
    assert ext.hold[0] and not ext.hold[1]


def test_a_walker_without_a_locomotion_profile_stops_at_its_hold_point(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(agent_factory(agent_id=1, x=-2.0))
    ext.set_passage(pool, {1: PassageCommand(hold=(0.0, 0.0, 1.0, 0.0))})
    speeds = _drive(pool, ext, 200)
    assert pool.pos[0, 0] <= 1e-9
    assert pool.pos[0, 0] > -0.05
    assert speeds[-1] == pytest.approx(0.0, abs=1e-9)


def test_a_walker_without_a_locomotion_profile_walks_back_against_its_route(agent_factory: Callable[..., BaseAgent]) -> None:
    """It walks back along the route's direction, not its heading: a walker's heading follows its velocity, so backing
    along the heading would turn it round and send it forward again."""
    pool, ext = _pool(agent_factory(agent_id=1, x=1.0))
    pool.theta[0] = math.pi  # faces back the way it came
    ext.set_passage(pool, {1: PassageCommand(hold=(0.0, 0.0, 1.0, 0.0), back=True)})
    for _ in range(100):
        pool.vel[:1] = (0.9, 0.0)
        ext.constrain(pool, DT, np.ones(1, dtype=np.bool_))
        assert pool.vel[0, 0] <= 0.0
        pool.pos[:1] += pool.vel[:1] * DT
        pool.theta[0] = math.pi
    assert pool.pos[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_hold_rows_follow_swap_remove(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(_chair(agent_factory, 1), _chair(agent_factory, 2, x=3.0))
    ext.set_passage(pool, {2: PassageCommand(lane=True, hold=(5.0, 0.0, 1.0, 0.0), back=True)})
    pool.swap_remove(1)
    assert pool.idx(2) == 0
    assert ext.hold[0] and ext.back[0] and pool.steer_to_goal[0]
    assert ext.hold_at[0].tolist() == [5.0, 0.0, 1.0, 0.0]
    assert math.isclose(pool.pos[0, 0], 3.0)
