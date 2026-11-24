from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np
import pytest

from arena_humansim.core import interaction_classes
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import SampledLocomotion
from arena_humansim.core.locomotion import COOLDOWN_S, MIN_SPEED_FACTOR, MODE_NORMAL, MODE_REVERSE, MODE_ROTATE, REVERSE_SPEED, LocomotionExtension
from arena_humansim.core.pool import AgentPool

DT = 0.01
TWO_PI = 2.0 * math.pi


def _loc(**kw: object) -> SampledLocomotion:
    return SampledLocomotion(active=True, cadence_base=1.0, cadence_per_speed=0.0, cadence_min=1.0, cadence_max=1.0, **kw)


def _agent(agent_factory: Callable[..., BaseAgent], aid: int, loc: SampledLocomotion | None = None, **params: object) -> BaseAgent:
    agent = agent_factory(agent_id=aid)
    if loc is not None:
        params["locomotion"] = loc
    if params:
        agent.params = attrs.evolve(agent.params, **params)
    return agent


def _pool(capacity: int = 8) -> tuple[AgentPool, LocomotionExtension]:
    pool = AgentPool(capacity=capacity)
    ext = LocomotionExtension()
    ext.attach(pool)
    return pool, ext


def _no_heading(pool: AgentPool) -> np.ndarray:
    return np.zeros(pool.n, dtype=np.bool_)


def test_phase_advances_at_cadence_cycles_per_second(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc()))
    pool.vel[0] = (1.0, 0.0)
    start = ext.phase[0]
    assert start == pytest.approx((1 % 360) * math.pi / 180.0)
    for _ in range(100):
        ext.overlay(pool, DT)
    assert ext.phase[0] - start == pytest.approx(TWO_PI)
    assert ext.cadence[0] == 1.0
    assert pool.vel[0].tolist() == [1.0, 0.0]
    assert not ext.overlay_stored[0]


def test_cadence_follows_speed_within_its_clip(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, SampledLocomotion(active=True)))
    pool.vel[0] = (2.0, 0.0)
    ext.overlay(pool, DT)
    assert ext.cadence[0] == pytest.approx(0.4 + 0.55 * 2.0)
    pool.vel[0] = (10.0, 0.0)
    ext.overlay(pool, DT)
    assert ext.cadence[0] == 2.2


SURGE = ((0.12, 0.0), (0.05, 1.1))


@pytest.mark.parametrize("dt", [0.05, 0.025])
def test_speed_profile_keeps_mean_speed_and_surge_amplitude_with_velocity_as_planner_state(agent_factory: Callable[..., BaseAgent], dt: float) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 360, _loc(speed_profile=SURGE, speed_amplitude_scale=1.0)))
    v_cmd = 1.0
    direction = np.array([0.6, 0.8])
    warmup = int(round(5.0 / dt))
    cycles = 4
    measured = int(round(cycles / dt))
    speeds = []
    clean = []
    start = pool.pos[0].copy()
    for step in range(warmup + measured):
        if step == warmup:
            start = pool.pos[0].copy()
        ext.restore(pool)
        pool.vel[0] += (v_cmd * direction - pool.vel[0]) * dt / 0.5
        clean.append(math.hypot(*pool.vel[0]))
        ext.overlay(pool, dt)
        pool.pos[0] += pool.vel[0] * dt
        speeds.append(math.hypot(*pool.vel[0]))
    speeds = np.asarray(speeds[warmup:])
    assert np.asarray(clean[warmup:]) == pytest.approx(v_cmd, rel=1e-3)
    assert speeds.mean() == pytest.approx(v_cmd, rel=0.01)
    assert math.hypot(*(pool.pos[0] - start)) / cycles == pytest.approx(v_cmd, rel=0.01)
    w = np.linspace(0.0, TWO_PI, 3601)
    amplitude = max(sum(amp * np.sin((k + 1) * w + phase) for k, (amp, phase) in enumerate(SURGE)))
    assert speeds.max() / speeds.mean() - 1.0 == pytest.approx(amplitude, rel=0.1)


def test_speed_factor_is_floored(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 360, _loc(speed_profile=((1.0, 0.0),), speed_amplitude_scale=1.5)))
    pool.vel[0] = (1.0, 0.0)
    ext.phase[0] = 1.5 * math.pi - TWO_PI * DT
    ext.overlay(pool, DT)
    assert pool.vel[0] == pytest.approx((MIN_SPEED_FACTOR, 0.0))
    ext.restore(pool)
    assert pool.vel[0] == pytest.approx((1.0, 0.0), abs=1e-12)


def test_restore_takes_the_overlay_back_out(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(speed_profile=SURGE, lateral_profile=((0.1, 0.4),))))
    pool.add_agent(_agent(agent_factory, 2, _loc()))
    pool.theta[:2] = 0.3
    pool.vel[:2] = (0.7, -0.2)
    ext.overlay(pool, DT)
    assert ext.overlay_stored[:2].tolist() == [True, False]
    assert not np.allclose(pool.vel[0], (0.7, -0.2))
    ext.restore(pool)
    assert pool.vel[0] == pytest.approx((0.7, -0.2), abs=1e-12)
    assert pool.vel[1].tolist() == [0.7, -0.2]
    assert not ext.overlay_stored[0]
    ext.restore(pool)
    assert pool.vel[0] == pytest.approx((0.7, -0.2), abs=1e-12)


def test_restore_keeps_a_velocity_zeroed_after_the_overlay(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(speed_profile=SURGE, lateral_profile=((0.1, 0.4),))))
    pool.vel[0] = (0.7, -0.2)
    ext.overlay(pool, DT)
    pool.vel[0] = 0.0
    ext.restore(pool)
    assert pool.vel[0].tolist() == [0.0, 0.0]


def test_phase_warp_stretches_first_half_cycle(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 360, _loc(phase_warp_split=0.25)))
    rows = np.array([0])
    ext.phase[0] = 0.25 * TWO_PI
    assert ext.warped_phase(rows)[0] == pytest.approx(math.pi)
    ext.phase[0] = 0.625 * TWO_PI
    assert ext.warped_phase(rows)[0] == pytest.approx(1.5 * math.pi)
    ext.warp_split[0] = 0.5
    ext.phase[0] = 0.3 * TWO_PI
    assert ext.warped_phase(rows)[0] == pytest.approx(0.3 * TWO_PI)


def test_along_heading_rows_have_no_lateral_velocity_and_turn_at_the_clamped_rate(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1), min_turning_radius=0.3, pivot_angular_velocity=2.0))
    pool.theta[0] = 0.0
    pool.vel[0] = (1.0, 1.0)
    ext.constrain(pool, DT, _no_heading(pool))
    speed = math.hypot(1.0, 1.0)
    assert pool.theta[0] == pytest.approx(max(2.0, speed / 0.3) * DT)
    f = np.array([math.cos(pool.theta[0]), math.sin(pool.theta[0])])
    assert abs(pool.vel[0, 0] * f[1] - pool.vel[0, 1] * f[0]) < 1e-12
    assert pool.vel[0] @ f == pytest.approx(np.array([1.0, 1.0]) @ f)

    pool.vel[0] = (-1.0, 0.0)
    pool.theta[0] = 0.0
    ext.constrain(pool, DT, _no_heading(pool))
    assert pool.vel[0] == pytest.approx((0.0, 0.0))


def test_along_heading_rows_keep_the_planner_heading_when_it_provides_one(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1)))
    pool.theta[0] = 0.5
    pool.vel[0] = (1.0, 1.0)
    ext.constrain(pool, DT, np.array([True]))
    assert pool.theta[0] == 0.5
    f = np.array([math.cos(0.5), math.sin(0.5)])
    assert pool.vel[0] == pytest.approx((np.array([1.0, 1.0]) @ f) * f)


def test_holonomic_active_rows_keep_their_velocity_under_constrain(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=0)))
    pool.vel[0] = (1.0, 1.0)
    ext.constrain(pool, DT, _no_heading(pool))
    assert pool.vel[0].tolist() == [1.0, 1.0]
    assert pool.theta[0] == 0.0


def _stalled(pool: AgentPool, ext: LocomotionExtension, seconds: float, provides_heading: np.ndarray | None = None) -> None:
    for _ in range(int(round(seconds / DT))):
        ext.restore(pool)
        pool.vel[0] = (0.0, 0.0)
        ext.constrain(pool, DT, provides_heading if provides_heading is not None else _no_heading(pool))


def test_recovery_rotates_toward_the_goal_then_reverses_on_a_second_stall(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1, recovery_stall_after_s=0.5, recovery_reverse_m=0.3), pivot_angular_velocity=2.0))
    pool.goal_pos[0] = (5.0, 0.0)
    pool.has_goal[0] = True
    pool.desired_vel[0] = 1.0
    pool.theta[0] = math.pi

    _stalled(pool, ext, 0.49)
    assert ext.mode[0] == MODE_NORMAL
    _stalled(pool, ext, 0.02)
    assert ext.mode[0] == MODE_ROTATE
    ext.restore(pool)
    pool.vel[0] = (1.0, 0.0)
    ext.constrain(pool, DT, np.array([True]))
    assert pool.vel[0].tolist() == [0.0, 0.0]
    assert math.pi - 4.0 * 2.0 * DT <= abs(pool.theta[0]) < math.pi

    _stalled(pool, ext, 1.6, provides_heading=np.array([True]))
    assert ext.mode[0] == MODE_NORMAL
    assert abs(pool.theta[0]) < 0.15
    assert 0.0 < ext.cooldown[0] <= COOLDOWN_S

    _stalled(pool, ext, 0.55)
    assert ext.mode[0] == MODE_REVERSE
    ext.restore(pool)
    pool.vel[0] = (1.0, 0.0)
    ext.constrain(pool, DT, _no_heading(pool))
    f = np.array([math.cos(pool.theta[0]), math.sin(pool.theta[0])])
    assert pool.vel[0] == pytest.approx(-REVERSE_SPEED * f)
    _stalled(pool, ext, 0.3 / REVERSE_SPEED)
    assert ext.mode[0] in (MODE_ROTATE, MODE_NORMAL)
    assert ext.reversed_m[0] == 0.0


def _fought(pool: AgentPool, ext: LocomotionExtension, pivot: float) -> None:
    """One tick of a heading planner that turns a recovering row at full rate away from where recovery turns it."""
    ext.restore(pool)
    if ext.mode[0] == MODE_ROTATE:
        bearing = math.atan2(pool.goal_pos[0, 1] - pool.pos[0, 1], pool.goal_pos[0, 0] - pool.pos[0, 0])
        err = math.atan2(math.sin(bearing - pool.theta[0]), math.cos(bearing - pool.theta[0]))
        pool.theta[0] -= math.copysign(pivot * DT, err)
    elif ext.mode[0] == MODE_REVERSE:
        pool.theta[0] += pivot * DT
    pool.vel[0] = (0.0, 0.0)
    ext.constrain(pool, DT, np.array([True]))


def test_recovery_owns_the_heading_against_a_planner_turning_the_other_way(agent_factory: Callable[..., BaseAgent]) -> None:
    pivot = 2.0
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1, recovery_stall_after_s=0.5, recovery_reverse_m=0.3), pivot_angular_velocity=pivot))
    pool.goal_pos[0] = (5.0, 0.0)
    pool.has_goal[0] = True
    pool.desired_vel[0] = 1.0
    pool.theta[0] = 3.0

    while ext.mode[0] == MODE_NORMAL:
        _fought(pool, ext, pivot)
    assert ext.mode[0] == MODE_ROTATE
    ticks = 0
    while ext.mode[0] == MODE_ROTATE and ticks < 1000:
        _fought(pool, ext, pivot)
        ticks += 1
    assert ext.mode[0] == MODE_NORMAL
    assert ticks * DT <= math.pi / pivot + 0.1
    assert abs(pool.theta[0]) < 0.15

    while ext.mode[0] == MODE_NORMAL:
        _fought(pool, ext, pivot)
    assert ext.mode[0] == MODE_REVERSE
    held = float(pool.theta[0])
    ticks = 0
    while ext.mode[0] == MODE_REVERSE:
        assert pool.vel[0] == pytest.approx((-REVERSE_SPEED * math.cos(held), -REVERSE_SPEED * math.sin(held)))
        _fought(pool, ext, pivot)
        ticks += 1
        assert ext.mode[0] != MODE_REVERSE or pool.theta[0] == held
    assert ticks == pytest.approx(0.3 / REVERSE_SPEED / DT, abs=1)


def test_recovery_resets_when_the_goal_distance_improves_or_the_goal_changes(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(recovery_stall_after_s=0.5)))
    pool.goal_pos[0] = (5.0, 0.0)
    pool.has_goal[0] = True
    pool.desired_vel[0] = 1.0
    _stalled(pool, ext, 0.4)
    pool.pos[0] = (0.1, 0.0)
    _stalled(pool, ext, 0.4)
    assert ext.mode[0] == MODE_NORMAL
    assert ext.stall_timer[0] == pytest.approx(0.4, abs=DT)
    pool.goal_pos[0] = (-5.0, 0.0)
    _stalled(pool, ext, 0.3)
    assert ext.stall_timer[0] == pytest.approx(0.3, abs=DT)
    pool.has_goal[0] = False
    _stalled(pool, ext, 1.0)
    assert ext.mode[0] == MODE_NORMAL
    assert ext.stall_timer[0] == 0.0


def test_recovery_ignores_near_goals_and_parked_rows(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(recovery_stall_after_s=0.5)))
    pool.goal_pos[0] = (0.1, 0.0)
    pool.has_goal[0] = True
    pool.desired_vel[0] = 1.0
    _stalled(pool, ext, 1.0)
    assert ext.mode[0] == MODE_NORMAL
    pool.goal_pos[0] = (5.0, 0.0)
    pool.desired_vel[0] = 0.0
    _stalled(pool, ext, 1.0)
    assert ext.mode[0] == MODE_NORMAL


def test_overlay_adds_lateral_sway_to_holonomic_rows_only(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(lateral_profile=((0.1, 0.0),))))
    pool.add_agent(_agent(agent_factory, 2, _loc()))
    pool.add_agent(_agent(agent_factory, 3, _loc(kinematics=1, lateral_profile=((0.1, 0.0),))))
    pool.add_agent(_agent(agent_factory, 4, _loc(lateral_profile=((0.1, 0.0),), speed_profile=((0.2, 0.0),))))
    pool.theta[:4] = 0.0
    pool.vel[:4] = (1.0, 0.0)
    ext.phase[:4] = math.pi / 2 - TWO_PI * DT
    ext.mode[3] = MODE_ROTATE
    ext.overlay(pool, DT)
    assert pool.vel[0] == pytest.approx((1.0, 0.1))
    assert pool.vel[1].tolist() == [1.0, 0.0]
    assert pool.vel[2].tolist() == [1.0, 0.0]
    assert pool.vel[3].tolist() == [1.0, 0.0]
    assert ext.overlay_stored[:4].tolist() == [True, False, False, False]


def test_heading_owned_mask_covers_along_heading_and_recovering_rows(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1)))
    pool.add_agent(_agent(agent_factory, 2, _loc()))
    pool.add_agent(_agent(agent_factory, 3, _loc()))
    pool.add_agent(_agent(agent_factory, 4))
    pool.add_agent(_agent(agent_factory, 5, SampledLocomotion(active=False, kinematics=1)))
    ext.mode[2] = MODE_REVERSE
    assert ext.heading_owned_mask(pool.n).tolist() == [True, False, True, False, False]


def test_inactive_rows_are_untouched_bit_for_bit(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1, speed_profile=((0.2, 0.0),), recovery_stall_after_s=0.1)))
    pool.add_agent(_agent(agent_factory, 2))
    pool.add_agent(_agent(agent_factory, 3, SampledLocomotion(active=False, kinematics=1, speed_profile=((0.2, 0.0),), lateral_profile=((0.1, 0.0),), recovery_stall_after_s=0.1)))
    pool.vel[:3] = (0.7, 0.3)
    pool.theta[:3] = 0.1
    pool.pos[:3] = (0.0, 0.0)
    pool.goal_pos[:3] = (5.0, 0.0)
    pool.has_goal[:3] = True
    pool.desired_vel[:3] = 1.0
    before = {name: getattr(pool, name).copy() for name in ("pos", "vel", "theta", "prev_vel", "goal_pos", "has_goal", "axial_offset", "interaction_class")}
    changed = False
    for _ in range(50):
        ext.restore(pool)
        ext.constrain(pool, DT, _no_heading(pool))
        ext.overlay(pool, DT)
        changed = changed or not np.array_equal(pool.vel[0], before["vel"][0])
    for name, old in before.items():
        new = getattr(pool, name)
        assert np.array_equal(new[1:3], old[1:3]), name
    assert changed
    assert pool.theta[0] != before["theta"][0]
    assert not ext.overlay_stored[1:3].any()
    phase, cadence = ext.frame_arrays(np.arange(3))
    assert phase[0] != 0.0 and cadence[0] != 0.0
    assert phase[1:].tolist() == [0.0, 0.0]
    assert cadence[1:].tolist() == [0.0, 0.0]


def test_all_methods_leave_a_pool_without_active_rows_alone(pool_with_agents: Callable[..., AgentPool]) -> None:
    pool = pool_with_agents(n=3)
    ext = LocomotionExtension()
    pool.attach_late(ext, [])
    pool.vel[:3] = (0.5, 0.5)
    before = pool.vel.copy()
    ext.restore(pool)
    ext.constrain(pool, DT, _no_heading(pool))
    ext.overlay(pool, DT)
    assert np.array_equal(pool.vel, before)
    assert not ext.heading_owned_mask(pool.n).any()


def test_pool_add_fills_axial_offset_and_interaction_class(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(footprint_length=1.1), agent_radius=0.25, interaction_class="wheelchair"))
    pool.add_agent(_agent(agent_factory, 2, _loc(footprint_length=0.2), agent_radius=0.25))
    pool.add_agent(_agent(agent_factory, 3, interaction_class="robot"))
    assert pool.axial_offset[0] == pytest.approx(0.3)
    assert pool.axial_offset[1] == 0.0
    assert pool.axial_offset[2] == 0.0
    wheelchair = int(pool.interaction_class[0])
    assert wheelchair >= 2
    assert interaction_classes.name(wheelchair) == "wheelchair"
    assert interaction_classes.index("wheelchair") == wheelchair
    assert pool.interaction_class[1] == interaction_classes.HUMAN
    assert pool.interaction_class[2] == interaction_classes.ROBOT
    assert interaction_classes.count() >= 3
    assert interaction_classes.index("") == interaction_classes.HUMAN


def test_extension_rows_follow_swap_remove_and_grow(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool(capacity=2)
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1, footprint_length=1.0)))
    pool.add_agent(_agent(agent_factory, 2))
    pool.add_agent(_agent(agent_factory, 3, _loc(speed_profile=((0.2, 0.5),), recovery_stall_after_s=2.0)))
    assert pool.capacity >= 3
    assert ext.active.shape[0] == pool.capacity
    assert ext.speed_amp.shape == (pool.capacity, 3)
    assert ext.overlay_sway.shape == (pool.capacity, 2)
    assert ext.active[:3].tolist() == [True, False, True]
    assert ext.kinematics[0] == 1
    ext.phase[2] = 7.0
    pool.vel[2] = (1.0, 0.0)
    ext.overlay(pool, DT)
    overlaid = pool.vel[2].copy()
    assert ext.overlay_stored[:3].tolist() == [False, False, True]
    pool.swap_remove(1)
    assert pool.n == 2
    assert ext.active[0]
    assert ext.phase[0] == pytest.approx(7.0 + TWO_PI * DT)
    assert ext.overlay_stored[0]
    assert ext.overlay_vel[0].tolist() == overlaid.tolist()
    ext.restore(pool)
    assert pool.vel[0] == pytest.approx((1.0, 0.0), abs=1e-12)
    assert ext.speed_amp[0].tolist() == [0.2, 0.0, 0.0]
    assert ext.speed_phase[0].tolist() == [0.5, 0.0, 0.0]
    assert ext.stall_after[0] == 2.0
    assert ext.kinematics[0] == 0


def test_warped_speed_and_sway_profiles_average_to_zero_over_a_cycle(agent_factory: Callable[..., BaseAgent]) -> None:
    pool, ext = _pool()
    limp = _loc(phase_warp_split=0.58, speed_profile=((0.10, 1.6), (0.03, 0.4), (0.02, 2.0)), lateral_profile=((0.04, 1.6),))
    even = attrs.evolve(limp, phase_warp_split=0.5)
    pool.add_agent(_agent(agent_factory, 1, limp))
    pool.add_agent(_agent(agent_factory, 2, even))
    assert ext.speed_bias[1] == 0.0 and ext.lateral_bias[1] == 0.0
    assert ext.speed_bias[0] != 0.0 and ext.lateral_bias[0] != 0.0
    steps = round(1.0 / DT)
    factors = np.zeros((steps, 2))
    sways = np.zeros((steps, 2))
    for i in range(steps):
        ext.restore(pool)
        pool.vel[:2] = (1.0, 0.0)
        ext.overlay(pool, DT)
        factors[i] = pool.vel[:2, 0]
        sways[i] = pool.vel[:2, 1]
    assert np.abs(factors.mean(axis=0) - 1.0).max() < 2e-3
    assert np.abs(sways.mean(axis=0)).max() < 2e-3
    assert np.ptp(factors[:, 0]) > 0.15 and np.ptp(sways[:, 0]) > 0.06


def test_progress_after_a_reverse_counts_from_where_the_reverse_left_the_row(agent_factory: Callable[..., BaseAgent]) -> None:
    """A row that backed off and then rolls forward again makes progress: it must not count as stalled against its best
    distance from before the reverse (which made it reverse every RESTALL_AFTER_S while walkers blocked it: a loop)."""
    pool, ext = _pool()
    pool.add_agent(_agent(agent_factory, 1, _loc(kinematics=1, recovery_stall_after_s=0.5, recovery_reverse_m=0.3), pivot_angular_velocity=2.0))
    pool.goal_pos[0] = (5.0, 0.0)
    pool.has_goal[0] = True
    pool.desired_vel[0] = 1.0
    pool.theta[0] = 0.0

    _stalled(pool, ext, 0.55)
    _stalled(pool, ext, 2 * DT)
    assert ext.mode[0] == MODE_NORMAL and ext.cooldown[0] > 0.0
    _stalled(pool, ext, 0.55)
    assert ext.mode[0] == MODE_REVERSE
    pool.pos[0] = (-0.3, 0.0)
    _stalled(pool, ext, 0.3 / REVERSE_SPEED + 2 * DT)
    assert ext.mode[0] == MODE_NORMAL and ext.cooldown[0] > 0.0

    pool.pos[0] = (-0.2, 0.0)
    _stalled(pool, ext, 0.45)
    assert ext.mode[0] == MODE_NORMAL
    assert ext.stall_timer[0] == pytest.approx(0.45, abs=2 * DT)
    _stalled(pool, ext, 0.1)
    assert ext.mode[0] == MODE_REVERSE
