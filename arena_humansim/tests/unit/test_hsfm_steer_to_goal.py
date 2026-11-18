"""HSFM heading of a row flagged `pool.steer_to_goal` (a walker in a lane): it heads straight for its goal whatever
pushes it; rows without the flag turn as before."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.utils.types import Pose2D

GOAL_X = 6.0
DT = 0.05


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> tuple[AgentPool, HSFMPlanner]:
    pool = pool_empty(capacity=8)
    planner = HSFMPlanner()
    planner.attach(pool)
    walker = agent_factory(agent_id=1, x=0.0, y=0.4)
    walker.params.local_planner_params["lateral_gain"] = 0.0
    walker.params.local_planner_params["heading_source"] = 1.0
    pool.add_agent(walker)
    pool.add_agent(agent_factory(agent_id=2, x=0.6, y=0.3))
    pool.kind[1] = KIND_ROBOT
    pool.interaction_class[1] = KIND_ROBOT
    pool.policy_idx[1] = -1
    pool.policy_idx[0] = 0
    pool.set_goals({1: Pose2D(x=GOAL_X, y=0.0)})
    pool.set_neighbor_csr(np.array([0, 1, 1], dtype=np.int32), np.array([1], dtype=np.int32))
    return pool, planner


def test_a_row_in_a_lane_heads_for_its_goal_whatever_pushes_it(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _pool(pool_empty, agent_factory)
    f_att, f_rep, f_wall, at_goal = planner._compute_forces_pool(pool)
    total = f_att + f_rep + f_wall
    bearing = math.atan2(-0.4, GOAL_X)
    free = planner._desired_heading(pool, f_att, total, at_goal)
    assert abs(_wrap(float(free[0]) - bearing)) > 0.3
    pool.steer_to_goal[0] = True
    steered = planner._desired_heading(pool, f_att, total, at_goal)
    assert steered[0] == pytest.approx(bearing)
    assert steered[1] == free[1]


def test_rows_without_the_flag_move_as_before_bit_for_bit(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    plain, planner_plain = _pool(pool_empty, agent_factory)
    other, planner_other = _pool(pool_empty, agent_factory)
    other.steer_to_goal[1] = True
    for _ in range(100):
        for pool, planner in ((plain, planner_plain), (other, planner_other)):
            planner.compute_pool(pool, dt=DT)
            pool.vel[1] = 0.0
            pool.pos[0] += pool.vel[0] * DT
    assert np.array_equal(plain.pos[:1], other.pos[:1])
    assert np.array_equal(plain.theta[:1], other.theta[:1])


def test_a_row_without_a_goal_is_not_steered(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool, planner = _pool(pool_empty, agent_factory)
    pool.has_goal[0] = False
    f_att, f_rep, f_wall, at_goal = planner._compute_forces_pool(pool)
    total = f_att + f_rep + f_wall
    free = planner._desired_heading(pool, f_att, total, at_goal)[0]
    pool.steer_to_goal[0] = True
    assert planner._desired_heading(pool, f_att, total, at_goal)[0] == free
