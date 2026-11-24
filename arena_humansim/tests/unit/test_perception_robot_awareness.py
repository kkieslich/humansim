from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import KIND_HUMAN, KIND_ROBOT, AgentPool
from arena_humansim.perception.default import DefaultPerception


def _neighbors(pool: AgentPool, idx: int) -> list[int]:
    start = int(pool.neighbor_indptr[idx])
    stop = int(pool.neighbor_indptr[idx + 1])
    return [int(i) for i in pool.neighbor_indices[start:stop]]


def _add_far_humans(pool: AgentPool, agent_factory: Callable[..., BaseAgent], count: int) -> None:
    for k in range(count):
        pool.add_agent(agent_factory(agent_id=100 + k, x=1000.0 + 10.0 * k, y=1000.0))
        pool.kind[pool.n - 1] = KIND_HUMAN


def test_blind_human_does_not_perceive_robot(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_blind[0] = True
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.add_agent(agent_factory(agent_id=3, x=0.0, y=1.0))
    pool.kind[2] = KIND_HUMAN

    DefaultPerception().compute_pool(pool)

    neighbors = _neighbors(pool, 0)
    assert 1 not in neighbors
    assert 2 in neighbors


def test_aware_human_perceives_robot_and_human(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=8)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.add_agent(agent_factory(agent_id=3, x=0.0, y=1.0))
    pool.kind[2] = KIND_HUMAN

    DefaultPerception().compute_pool(pool)

    neighbors = _neighbors(pool, 0)
    assert 1 in neighbors
    assert 2 in neighbors


@pytest.mark.parametrize("far_humans", [0, 70], ids=["dense", "kdtree"])
def test_detection_shift_shortens_the_robot_range_only(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], far_humans: int) -> None:
    pool = pool_empty(capacity=96)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.add_agent(agent_factory(agent_id=2, x=3.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.add_agent(agent_factory(agent_id=3, x=3.0 * np.cos(np.radians(60.0)), y=3.0 * np.sin(np.radians(60.0))))
    pool.kind[2] = KIND_HUMAN
    pool.robot_range_offset[0] = -3.5

    _add_far_humans(pool, agent_factory, far_humans)
    DefaultPerception().compute_pool(pool)

    neighbors = _neighbors(pool, 0)
    assert 1 not in neighbors
    assert 2 in neighbors


@pytest.mark.parametrize("far_humans", [0, 70], ids=["dense", "kdtree"])
def test_detection_shift_applies_to_every_robot(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], far_humans: int) -> None:
    pool = pool_empty(capacity=96)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.add_agent(agent_factory(agent_id=2, x=3.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    pool.add_agent(agent_factory(agent_id=3, x=3.0 * np.cos(np.radians(60.0)), y=3.0 * np.sin(np.radians(60.0))))
    pool.kind[2] = KIND_ROBOT
    pool.robot_range_offset[0] = -3.5

    _add_far_humans(pool, agent_factory, far_humans)
    DefaultPerception().compute_pool(pool)

    assert _neighbors(pool, 0) == []


@pytest.mark.parametrize("far_humans", [0, 70], ids=["dense", "kdtree"])
def test_fov_shift_narrows_the_robot_cone_only(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], far_humans: int) -> None:
    pool = pool_empty(capacity=96)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.add_agent(agent_factory(agent_id=2, x=3.0 * np.cos(np.radians(40.0)), y=3.0 * np.sin(np.radians(40.0))))
    pool.kind[1] = KIND_ROBOT
    pool.add_agent(agent_factory(agent_id=3, x=3.0 * np.cos(np.radians(80.0)), y=3.0 * np.sin(np.radians(80.0))))
    pool.kind[2] = KIND_HUMAN
    pool.robot_fov_offset[0] = -144.0

    _add_far_humans(pool, agent_factory, far_humans)
    DefaultPerception().compute_pool(pool)

    neighbors = _neighbors(pool, 0)
    assert 1 not in neighbors
    assert 2 in neighbors


@pytest.mark.parametrize("far_humans", [0, 70], ids=["dense", "kdtree"])
def test_positive_fov_shift_widens_the_robot_cone_only(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], far_humans: int) -> None:
    pool = pool_empty(capacity=96)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.vision_fov[0] = 90.0
    pool.add_agent(agent_factory(agent_id=2, x=3.0 * np.cos(np.radians(60.0)), y=3.0 * np.sin(np.radians(60.0))))
    pool.kind[1] = KIND_ROBOT
    pool.add_agent(agent_factory(agent_id=3, x=3.0 * np.cos(np.radians(65.0)), y=3.0 * np.sin(np.radians(65.0))))
    pool.kind[2] = KIND_HUMAN
    pool.robot_fov_offset[0] = 60.0

    _add_far_humans(pool, agent_factory, far_humans)
    DefaultPerception().compute_pool(pool)

    neighbors = _neighbors(pool, 0)
    assert 1 in neighbors
    assert 2 not in neighbors


def test_kdtree_path_gates_blind_humans(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = pool_empty(capacity=80)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    pool.kind[0] = KIND_HUMAN
    pool.robot_blind[0] = True
    pool.add_agent(agent_factory(agent_id=2, x=1.0, y=0.0))
    pool.kind[1] = KIND_ROBOT
    for i in range(2, 70):
        pool.add_agent(agent_factory(agent_id=i + 1, x=float(i % 10) * 0.5, y=float(i // 10) * 0.5))
        pool.kind[i] = KIND_HUMAN

    DefaultPerception().compute_pool(pool)

    neighbors = _neighbors(pool, 0)
    assert 1 not in neighbors
    assert len(neighbors) > 0


def test_blind_flag_changes_nothing_in_an_all_human_crowd(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    def build(blind: bool) -> AgentPool:
        pool = pool_empty(capacity=16)
        for i in range(10):
            pool.add_agent(agent_factory(agent_id=i + 1, x=float(i % 4) * 0.8, y=float(i // 4) * 0.8))
            pool.kind[i] = KIND_HUMAN
            pool.robot_blind[i] = blind
        DefaultPerception().compute_pool(pool)
        return pool

    normal = build(blind=False)
    blind = build(blind=True)
    assert np.array_equal(normal.neighbor_indptr, blind.neighbor_indptr)
    assert np.array_equal(normal.neighbor_indices, blind.neighbor_indices)
