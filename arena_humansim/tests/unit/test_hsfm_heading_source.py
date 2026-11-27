from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.utils.types import AgentState, BeliefState, Pose2D

from .test_planner_golden import GOLDEN_DIR, assert_matches_golden, max_abs_diff, run_trajectory

BLOCKER_X = 2.0
GOAL_X = 6.0
STEPS = 400
DT = 0.05


def test_heading_from_attraction_reproduces_golden() -> None:
    want = json.loads((GOLDEN_DIR / "hsfm_pool.json").read_text())
    assert_matches_golden(run_trajectory(HSFMPlanner(), {"heading_source": 0.0}), want)
    assert max_abs_diff(run_trajectory(HSFMPlanner(), {"heading_source": 1.0}), want) > 1e-3


def _blocker_pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], heading_source: float, blocker: tuple[float, float]) -> tuple[AgentPool, HSFMPlanner]:
    pool = pool_empty(capacity=8)
    planner = HSFMPlanner()
    planner.attach(pool)
    walker = agent_factory(agent_id=1, x=0.0, y=0.0)
    walker.params.local_planner_params["lateral_gain"] = 0.0
    walker.params.local_planner_params["heading_source"] = heading_source
    pool.add_agent(walker)
    pool.add_agent(agent_factory(agent_id=2, x=blocker[0], y=blocker[1]))
    pool.kind[1] = KIND_ROBOT
    pool.interaction_class[1] = KIND_ROBOT
    pool.policy_idx[1] = -1
    pool.policy_idx[0] = 0
    pool.set_goals({1: Pose2D(x=GOAL_X, y=0.0)})
    pool.set_neighbor_csr(np.array([0, 1, 1], dtype=np.int32), np.array([1], dtype=np.int32))
    return pool, planner


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _run_blocker(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], heading_source: float) -> tuple[float, float]:
    pool, planner = _blocker_pool(pool_empty, agent_factory, heading_source, (BLOCKER_X, 0.0))
    max_x = 0.0
    min_goal_dist = GOAL_X
    for _ in range(STEPS):
        planner.compute_pool(pool, dt=DT)
        pool.vel[1] = 0.0
        pool.pos[0] += pool.vel[0] * DT
        max_x = max(max_x, float(pool.pos[0, 0]))
        min_goal_dist = min(min_goal_dist, float(np.hypot(GOAL_X - pool.pos[0, 0], pool.pos[0, 1])))
    assert np.array_equal(pool.pos[1], [BLOCKER_X, 0.0])
    return max_x, min_goal_dist


def test_heading_from_total_force_passes_static_blocker(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    max_x, min_goal_dist = _run_blocker(pool_empty, agent_factory, 1.0)
    assert max_x > BLOCKER_X + 0.3, max_x
    assert min_goal_dist < 0.5, min_goal_dist


def test_heading_from_attraction_stalls_at_static_blocker(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    max_x, _ = _run_blocker(pool_empty, agent_factory, 0.0)
    assert max_x <= BLOCKER_X + 0.3, max_x


@pytest.mark.parametrize("offset", [0.02, 0.05, 0.1, 0.3])
def test_heading_from_total_force_stays_within_a_quarter_turn_of_the_goal_bearing(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], offset: float) -> None:
    pool, planner = _blocker_pool(pool_empty, agent_factory, 1.0, (0.8, offset))
    worst_unlimited = 0.0
    for step in range(200):
        f_att, f_rep, f_wall, at_goal = planner._compute_forces_pool(pool)
        total = f_att + f_rep + f_wall
        bearing = math.atan2(-pool.pos[0, 1], GOAL_X - pool.pos[0, 0])
        desired = float(planner._desired_heading(pool, f_att, total, at_goal)[0])
        assert abs(_wrap(desired - bearing)) <= math.pi / 2.0 + 1e-9, step
        worst_unlimited = max(worst_unlimited, abs(_wrap(math.atan2(total[0, 1], total[0, 0]) - bearing)))
        planner.compute_pool(pool, dt=DT)
        pool.vel[1] = 0.0
        pool.pos[0] += pool.vel[0] * DT
    assert worst_unlimited > math.pi / 2.0, math.degrees(worst_unlimited)


def test_heading_from_total_force_falls_back_to_the_total_direction_without_a_goal(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _blocker_pool(pool_empty, agent_factory, 1.0, (0.45, -0.05))
    pool.has_goal[0] = False
    f_att, f_rep, f_wall, at_goal = planner._compute_forces_pool(pool)
    total = f_att + f_rep + f_wall
    assert at_goal[0]
    total_dir = math.atan2(total[0, 1], total[0, 0])
    assert abs(total_dir) > math.pi / 2.0
    assert planner._desired_heading(pool, f_att, total, at_goal)[0] == total_dir


def test_scalar_path_limits_the_total_force_heading_to_a_quarter_turn(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = HSFMPlanner()
    walker = agent_factory(agent_id=1, x=0.0, y=0.0)
    walker.params.local_planner_params["lateral_gain"] = 0.0
    walker.params.local_planner_params["heading_source"] = 1.0
    walker.belief = BeliefState(agent_id=1)
    walker.belief.observed_agents = [AgentState(agent_id=2, pose=Pose2D(x=0.45, y=-0.05), kind=KIND_ROBOT)]
    goal = Pose2D(x=GOAL_X, y=0.0)
    forces = planner._compute_forces_scalar(walker, goal)
    assert forces is not None
    f_att, f_rep, f_wall, _ = forces
    total_dir = math.atan2(f_att[1] + f_rep[1] + f_wall[1], f_att[0] + f_rep[0] + f_wall[0])
    assert math.atan2(f_att[1], f_att[0]) == 0.0
    assert total_dir > math.radians(120.0)

    planner.compute([walker], {1: goal}, dt=DT)

    gain = walker.params.local_planner_params["angular_gain"]
    assert walker.state.pose.theta == pytest.approx(gain * (math.pi / 2.0) * DT * DT)
    assert walker.state.pose.theta < 0.9 * gain * total_dir * DT * DT
