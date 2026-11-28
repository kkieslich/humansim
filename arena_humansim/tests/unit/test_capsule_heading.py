from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.locomotion import GOAL_NEAR_M, body_heading_target
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Pose2D


def _pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], offset: float, goal: tuple[float, float] | None) -> AgentPool:
    pool = pool_empty(capacity=4)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.axial_offset[0] = offset
    if goal is not None:
        pool.set_goals({1: Pose2D(x=goal[0], y=goal[1])})
    return pool


@pytest.mark.parametrize("push", [math.pi, 0.75 * math.pi, -0.6 * math.pi])
def test_a_capsule_pushed_back_keeps_facing_its_goal(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], push: float) -> None:
    pool = _pool(pool_empty, agent_factory, 0.225, (5.0, 5.0))
    target = body_heading_target(pool, np.arange(1), np.array([push + math.pi / 4]))
    assert target[0] == pytest.approx(math.pi / 4)


@pytest.mark.parametrize("vel", [0.0, 0.4 * math.pi, -0.45 * math.pi])
def test_a_capsule_moving_towards_its_goal_side_follows_its_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], vel: float) -> None:
    pool = _pool(pool_empty, agent_factory, 0.225, (5.0, 0.0))
    assert body_heading_target(pool, np.arange(1), np.array([vel]))[0] == pytest.approx(vel)


def test_a_round_body_follows_its_velocity_even_backwards(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(pool_empty, agent_factory, 0.0, (5.0, 0.0))
    assert body_heading_target(pool, np.arange(1), np.array([math.pi]))[0] == pytest.approx(math.pi)


def test_a_capsule_without_goal_or_at_its_goal_follows_its_velocity(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(pool_empty, agent_factory, 0.225, None)
    assert body_heading_target(pool, np.arange(1), np.array([math.pi]))[0] == pytest.approx(math.pi)
    near = _pool(pool_empty, agent_factory, 0.225, (0.5 * GOAL_NEAR_M, 0.0))
    assert body_heading_target(near, np.arange(1), np.array([math.pi]))[0] == pytest.approx(math.pi)
