from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.local_planner.sfm import SFMPlanner, _occluded_kernel
from arena_humansim.utils.types import Pose2D

GOAL_X = 20.0


def _csr_first_sees_all(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row 0 sees agents 1..n-1, the others see no one."""
    indptr = np.array([0] + [n - 1] * n, dtype=np.int32)
    return indptr, np.arange(1, n, dtype=np.int32)


def _repulsion_on_first(
    pool_empty: Callable[..., AgentPool],
    agent_factory: Callable[..., BaseAgent],
    others: list[tuple[float, float]],
    occlusion: bool,
    offsets: list[float] | None = None,
    thetas: list[float] | None = None,
    planner_cls: type[SFMPlanner] = SFMPlanner,
) -> np.ndarray:
    pool = pool_empty(capacity=16)
    planner = planner_cls(occlusion=occlusion)
    planner.attach(pool)
    pool.add_agent(agent_factory(agent_id=1, x=0.0, y=0.0))
    for k, (x, y) in enumerate(others):
        pool.add_agent(agent_factory(agent_id=k + 2, x=x, y=y))
    n = len(others) + 1
    pool.theta[:n] = 0.0 if thetas is None else thetas
    if offsets is not None:
        pool.axial_offset[:n] = offsets
    pool.set_goals({1: Pose2D(x=GOAL_X, y=0.0)})
    pool.set_neighbor_csr(*_csr_first_sees_all(n))
    f_att, f_rep, f_wall, at_goal = planner._compute_forces_pool(pool)
    return f_rep[0]


def test_a_walker_hidden_behind_a_nearer_one_does_not_push(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    front = _repulsion_on_first(pool_empty, agent_factory, [(0.7, 0.0)], occlusion=True)
    queue = _repulsion_on_first(pool_empty, agent_factory, [(0.7, 0.0), (1.3, 0.0), (1.9, 0.0)], occlusion=True)
    np.testing.assert_allclose(queue, front, atol=1e-12)
    blind = _repulsion_on_first(pool_empty, agent_factory, [(0.7, 0.0), (1.3, 0.0), (1.9, 0.0)], occlusion=False)
    assert blind[0] < front[0] - 0.05, (blind, front)


def test_a_walker_beside_the_sight_line_still_pushes(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    seen = _repulsion_on_first(pool_empty, agent_factory, [(0.7, 0.0), (1.3, 0.9)], occlusion=True)
    blind = _repulsion_on_first(pool_empty, agent_factory, [(0.7, 0.0), (1.3, 0.9)], occlusion=False)
    np.testing.assert_allclose(seen, blind, atol=1e-12)


def test_occlusion_off_reproduces_the_sum_over_all_neighbours(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pts = [(0.7, 0.0), (1.3, 0.0), (1.0, 0.8), (-0.9, 0.2)]
    total = _repulsion_on_first(pool_empty, agent_factory, pts, occlusion=False)
    parts = sum(_repulsion_on_first(pool_empty, agent_factory, [p], occlusion=False) for p in pts)
    np.testing.assert_allclose(total, parts, atol=1e-12)


def test_a_capsule_lying_across_the_sight_line_hides_what_is_behind_it(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    # a wheelchair (axis 0.45 m) across the line at x 1.0 with its centre 0.35 m off the line: a disc there would not hide
    # the walker at 2.0, its body does
    others = [(1.0, 0.35), (2.0, 0.0)]
    capsule = _repulsion_on_first(pool_empty, agent_factory, others, occlusion=True, offsets=[0.0, 0.225, 0.0], thetas=[0.0, math.pi / 2, 0.0])
    alone = _repulsion_on_first(pool_empty, agent_factory, others[:1], occlusion=True, offsets=[0.0, 0.225], thetas=[0.0, math.pi / 2])
    np.testing.assert_allclose(capsule, alone, atol=1e-12)
    disc = _repulsion_on_first(pool_empty, agent_factory, others, occlusion=True)
    disc_alone = _repulsion_on_first(pool_empty, agent_factory, others[:1], occlusion=True)
    assert disc[0] < disc_alone[0] - 1e-6


def test_a_long_body_in_a_queue_is_pushed_back_by_the_first_row_only(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    # observer: wheelchair (half axis 0.225) facing +x, a column of 1-3 standing walkers ahead of its front end
    rows = [(1.05 + 0.6 * k, 0.0) for k in range(3)]
    with_occ = [_repulsion_on_first(pool_empty, agent_factory, rows[: m + 1], occlusion=True, offsets=[0.225] + [0.0] * (m + 1), planner_cls=HSFMPlanner)[0] for m in range(3)]
    without = [_repulsion_on_first(pool_empty, agent_factory, rows[: m + 1], occlusion=False, offsets=[0.225] + [0.0] * (m + 1), planner_cls=HSFMPlanner)[0] for m in range(3)]
    assert max(with_occ) - min(with_occ) < 1e-12
    assert without[2] < without[0] - 0.05


@pytest.mark.parametrize("gap", [2.5, 4.0])
def test_far_neighbours_skip_the_test(gap: float) -> None:
    # a neighbour whose surface is farther than _OCCLUSION_MAX_GAP is never marked hidden (its push is negligible)
    pos = np.array([[0.0, 0.0], [0.8, 0.0], [0.6 + gap, 0.0]])
    out = np.zeros(2, dtype=np.bool_)
    _occluded_kernel(pos, np.zeros(3), np.zeros(3), np.full(3, 0.3), np.array([0, 2, 2, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64), out)
    assert not out[0]
    assert not out[1]


def test_kernel_marks_only_the_hidden_neighbour() -> None:
    pos = np.array([[0.0, 0.0], [0.8, 0.0], [1.6, 0.05], [1.6, 1.0]])
    out = np.zeros(3, dtype=np.bool_)
    _occluded_kernel(pos, np.zeros(4), np.zeros(4), np.full(4, 0.25), np.array([0, 3, 3, 3, 3], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64), out)
    assert out.tolist() == [False, True, False]
