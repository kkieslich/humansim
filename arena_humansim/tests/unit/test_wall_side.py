"""Wall projection keeps every walker on the side of each wall its step started on (`AgentPool.begin_step`).

A wall is a zero-thickness segment and contact projection pushes along the direction from the closest wall point. Once
crowd pressure moves a center past the wall's line, that direction points out of the room and projection used to finish
the crossing: a walker held at radius + margin in front of an end wall reappeared radius + margin behind it."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from arena_humansim.collision.wall_projection import WallProjectionResolver, _resolve_kernel
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.benchmark import generate_maze
from arena_humansim.utils.wall_grid import WallGrid

_MARGIN = 0.01
_RADIUS = 0.271
_THRESHOLD = _RADIUS + _MARGIN
_END_WALL = [((160.0, -5.0), (160.0, 5.0))]
_DT = 0.05


def _pool(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], rows: list[tuple[float, float, float, float, float]]) -> AgentPool:
    """Rows of (x, y, theta, radius, axial offset)."""
    pool = pool_empty(capacity=max(8, len(rows)))
    for i, (x, y, theta, radius, offset) in enumerate(rows):
        pool.add_agent(agent_factory(i + 1, x=x, y=y))
        pool.theta[i] = theta
        pool.agent_radius[i] = radius
        pool.axial_offset[i] = offset
        pool.policy_idx[i] = 0
    return pool


def _resolver(walls: list) -> WallProjectionResolver:
    resolver = WallProjectionResolver(margin=_MARGIN)
    resolver.set_walls(walls)
    return resolver


def _side(pool: AgentPool, i: int, wall: tuple) -> float:
    (ax, ay), (bx, by) = wall
    return (bx - ax) * (pool.pos[i, 1] - ay) - (by - ay) * (pool.pos[i, 0] - ax)


# --- the point ---------------------------------------------------------------------------------------------------------


def test_walker_pushed_past_an_end_wall_is_put_back_on_its_side(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(pool_empty, agent_factory, [(160.0 - _THRESHOLD, 1.0, 0.0, _RADIUS, 0.0)])
    pool.begin_step()
    pool.pos[0, 0] = 160.05  # the crowd shoved its center 5 cm past the wall's line
    pool.vel[0] = (0.4, 0.3)

    corrected = _resolver(_END_WALL).resolve(pool)

    assert corrected == {1}
    assert pool.pos[0, 0] == pytest.approx(160.0 - _THRESHOLD, abs=1e-9)
    assert pool.pos[0, 1] == 1.0
    assert pool.vel[0].tolist() == [0.0, 0.3]  # into the wall dropped, along it kept


def test_without_a_step_start_the_same_push_still_tunnels(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """The mechanism the step start removes: the closest-point normal points out of the room once the center is past the line."""
    pool = _pool(pool_empty, agent_factory, [(160.05, 1.0, 0.0, _RADIUS, 0.0)])

    _resolver(_END_WALL).resolve(pool)

    assert pool.pos[0, 0] == pytest.approx(160.0 + _THRESHOLD, abs=1e-9)


@pytest.mark.parametrize("depth", [1e-7, 1e-3, 0.1, _THRESHOLD, 0.6])
def test_any_depth_past_the_line_comes_back(depth: float, pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(pool_empty, agent_factory, [(160.0 - _THRESHOLD, 0.0, 0.0, _RADIUS, 0.0)])
    pool.begin_step()
    pool.pos[0, 0] = 160.0 + depth

    _resolver(_END_WALL).resolve(pool)

    assert pool.pos[0, 0] == pytest.approx(160.0 - _THRESHOLD, abs=1e-9)


def test_walker_already_behind_a_wall_stays_behind_it(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(pool_empty, agent_factory, [(160.0 + _THRESHOLD, 0.0, 0.0, _RADIUS, 0.0)])
    pool.begin_step()
    pool.pos[0, 0] = 159.9

    _resolver(_END_WALL).resolve(pool)

    assert pool.pos[0, 0] == pytest.approx(160.0 + _THRESHOLD, abs=1e-9)


def test_sliding_along_a_wall_is_unchanged(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """Pressed into the wall at an angle tick after tick, never past its line: the same bytes as without a step start."""
    walls = [((-5.0, 0.0), (5.0, 0.0))]
    runs = []
    for sweep in (False, True):
        pool = _pool(pool_empty, agent_factory, [(-3.0, _THRESHOLD, 0.0, _RADIUS, 0.0), (-3.0, 1.0, 0.0, 0.3, 0.3)])
        resolver = _resolver(walls)
        trace = []
        for _ in range(100):
            if sweep:
                pool.begin_step()
            pool.vel[0] = (1.2, -0.25)
            pool.vel[1] = (1.0, -0.4)
            pool.theta[1] += 0.01
            pool.pos[:2] += pool.vel[:2] * _DT
            resolver.resolve(pool)
            trace.append(np.concatenate([pool.pos[:2].ravel(), pool.vel[:2].ravel(), pool.theta[:2]]))
        runs.append(np.array(trace))
    assert np.array_equal(runs[0], runs[1])
    assert runs[1][-1, 1] == pytest.approx(_THRESHOLD, abs=1e-12)
    assert runs[1][-1, 0] > 2.0


def test_crowd_scene_without_crossings_is_byte_identical(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """A walking crowd in a maze, steps short enough that no center ever reaches a wall's line: sweep or not, the same bytes."""
    walls = [((float(x0), float(y0)), (float(x1), float(y1))) for (x0, y0), (x1, y1) in generate_maze(6, seed=3).walls]
    rng = np.random.default_rng(7)
    n = 40
    start = 2.0 * rng.integers(0, 6, size=(n, 2)) + 1.0 + rng.uniform(-0.3, 0.3, size=(n, 2))
    heading = rng.uniform(-np.pi, np.pi, n)
    offsets = np.where(np.arange(n) % 3 == 0, 0.2, 0.0)
    runs = []
    for sweep in (False, True):
        pool = _pool(pool_empty, agent_factory, [(float(x), float(y), float(h), 0.25, float(o)) for (x, y), h, o in zip(start, heading, offsets, strict=True)])
        resolver = _resolver(walls)
        trace = []
        for k in range(200):
            if sweep:
                pool.begin_step()
            turn = 0.3 * np.sin(0.05 * k + np.arange(n))
            pool.vel[:n] = 0.8 * np.column_stack([np.cos(heading + turn), np.sin(heading + turn)])
            pool.theta[:n] = heading + turn
            pool.pos[:n] += pool.vel[:n] * _DT
            resolver.resolve(pool)
            trace.append(np.concatenate([pool.pos[:n].ravel(), pool.vel[:n].ravel(), pool.theta[:n]]))
        runs.append(np.array(trace))
    assert np.array_equal(runs[0], runs[1])


def test_inside_corner_pushed_through_its_vertex_stays_inside(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    walls = [((0.0, 0.0), (10.0, 0.0)), ((0.0, 0.0), (0.0, 10.0))]
    pool = _pool(pool_empty, agent_factory, [(_THRESHOLD, _THRESHOLD, 0.0, _RADIUS, 0.0)])
    pool.begin_step()
    pool.pos[0] = (-0.1, -0.1)  # straight through the corner point

    _resolver(walls).resolve(pool)

    assert pool.pos[0].tolist() == pytest.approx([_THRESHOLD, _THRESHOLD], abs=1e-9)


@pytest.mark.parametrize("push", [(-0.05, -0.4), (-0.4, -0.05), (-0.3, -0.3)])
def test_inside_corner_pushed_through_either_wall_stays_inside(push: tuple[float, float], pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    walls = [((0.0, 0.0), (10.0, 0.0)), ((0.0, 0.0), (0.0, 10.0))]
    pool = _pool(pool_empty, agent_factory, [(_THRESHOLD + 0.02, _THRESHOLD + 0.02, 0.0, _RADIUS, 0.0)])
    pool.begin_step()
    pool.pos[0] += push

    _resolver(walls).resolve(pool)

    assert pool.pos[0, 0] >= _THRESHOLD - 1e-9
    assert pool.pos[0, 1] >= _THRESHOLD - 1e-9


def test_walking_round_a_free_wall_end_is_unchanged(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """A walker turning round a door jamb (a wall ending in the room) in contact with its end: no side to keep, the same bytes."""
    walls = [((0.0, -5.0), (0.0, 0.0))]
    runs = []
    for sweep in (False, True):
        pool = _pool(pool_empty, agent_factory, [(-0.3, -0.5, 0.0, _RADIUS, 0.0)])
        resolver = _resolver(walls)
        path = []
        for k in range(100):
            if sweep:
                pool.begin_step()
            if k < 20:
                v = (0.0, 0.5)
            elif k < 80:
                phi = math.pi - (k - 20) * math.pi / 60.0
                v = (math.pi / 3.0 * 0.27 * math.sin(phi), -math.pi / 3.0 * 0.27 * math.cos(phi))
            else:
                v = (0.0, -0.5)
            pool.vel[0] = v
            pool.pos[0] += pool.vel[0] * _DT
            resolver.resolve(pool)
            path.append(pool.pos[0].copy())
        runs.append(np.array(path))
    assert np.array_equal(runs[0], runs[1])
    assert runs[1][-1, 0] > 0.0  # round to the far side of the jamb
    assert runs[1][-1, 1] < -0.3


def test_step_past_a_wall_end_is_not_held(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    walls = [((0.0, -5.0), (0.0, 0.0))]
    pool = _pool(pool_empty, agent_factory, [(-0.3, 0.5, 0.0, _RADIUS, 0.0)])
    pool.begin_step()
    pool.pos[0] = (0.3, 0.4)  # crosses the wall's line 0.45 m beyond its end

    corrected = _resolver(walls).resolve(pool)

    assert corrected == set()
    assert pool.pos[0].tolist() == [0.3, 0.4]


def test_walker_walks_through_a_doorway_brushing_a_jamb(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    walls = [((0.0, -5.0), (0.0, -0.45)), ((0.0, 0.45), (0.0, 5.0))]
    runs = []
    for sweep in (False, True):
        pool = _pool(pool_empty, agent_factory, [(-1.0, 0.25, 0.0, _RADIUS, 0.0)])
        resolver = _resolver(walls)
        path = []
        for _ in range(40):
            if sweep:
                pool.begin_step()
            pool.vel[0] = (1.2, 0.0)
            pool.pos[0] += pool.vel[0] * _DT
            resolver.resolve(pool)
            path.append(pool.pos[0].copy())
        runs.append(np.array(path))
    assert np.array_equal(runs[0], runs[1])
    assert runs[1][-1, 0] > 1.0
    assert runs[1][:, 1].min() < 0.25  # the jamb's end pushed it aside on the way through


def test_doorway_closed_behind_a_walker_keeps_it_on_its_side(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """A door shuts (a wall appears) across where a walker stands: it stays on the side it stood on when the step began."""
    pool = _pool(pool_empty, agent_factory, [(-0.1, 0.0, 0.0, _RADIUS, 0.0)])
    resolver = _resolver([((0.0, -0.45), (0.0, 0.45))])
    pool.begin_step()
    pool.pos[0, 0] = 0.05

    resolver.resolve(pool)

    assert pool.pos[0, 0] == pytest.approx(-_THRESHOLD, abs=1e-9)


def test_walker_starting_on_a_wall_line_is_projected_as_before(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """A step that starts on the wall's line has no side to keep: plain contact projection."""
    runs = []
    for sweep in (False, True):
        pool = _pool(pool_empty, agent_factory, [(0.0, 0.0, 0.0, _RADIUS, 0.0)])
        if sweep:
            pool.begin_step()
        pool.pos[0, 0] = 0.02
        _resolver([((0.0, -5.0), (0.0, 5.0))]).resolve(pool)
        runs.append(pool.pos[0].copy())
    assert np.array_equal(runs[0], runs[1])
    assert runs[1][0] == pytest.approx(_THRESHOLD, abs=1e-9)


# --- the capsule -------------------------------------------------------------------------------------------------------


def test_capsule_pushed_with_an_end_past_a_wall_stays_on_its_side(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(pool_empty, agent_factory, [(160.0 - _THRESHOLD - 0.3, 0.0, 0.0, _RADIUS, 0.3)])
    pool.begin_step()
    pool.pos[0, 0] += 0.4  # front end 0.4 m past the wall

    _resolver(_END_WALL).resolve(pool)

    assert pool.pos[0, 0] + 0.3 == pytest.approx(160.0 - _THRESHOLD, abs=1e-9)
    assert pool.theta[0] == 0.0


def test_capsule_rotating_an_end_through_a_wall_is_shifted_back(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """Center still, heading turned so one end swings past the wall: the capsule moves off the wall, the heading stays."""
    wall = ((-5.0, 0.0), (5.0, 0.0))
    pool = _pool(pool_empty, agent_factory, [(0.0, _THRESHOLD + 0.05, 0.0, _RADIUS, 0.4)])
    pool.begin_step()
    pool.theta[0] = -math.pi / 2  # the -end swings from (-0.4, .) to above, the + end to (0, y - 0.4): past the line

    _resolver([wall]).resolve(pool)

    assert pool.theta[0] == -math.pi / 2
    assert pool.pos[0, 1] - 0.4 == pytest.approx(_THRESHOLD, abs=1e-9)
    assert pool.pos[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_capsule_rotation_without_a_step_start_flips_across(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """The wheelchair flip: an end rotated past the line is pushed further out."""
    pool = _pool(pool_empty, agent_factory, [(0.0, _THRESHOLD + 0.05, -math.pi / 2, _RADIUS, 0.4)])

    _resolver([((-5.0, 0.0), (5.0, 0.0))]).resolve(pool)

    assert pool.pos[0, 1] - 0.4 < 0.0


def test_capsule_turned_across_a_corridor_too_narrow_for_it_goes_back_to_its_start_pose(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    """Turned square in a corridor narrower than its length, both ends cross a wall: no shift fixes that, it keeps its start pose."""
    walls = [((-5.0, -0.6), (5.0, -0.6)), ((-5.0, 0.6), (5.0, 0.6))]
    pool = _pool(pool_empty, agent_factory, [(0.0, 0.0, 0.0, 0.3, 0.7)])
    pool.begin_step()
    pool.theta[0] = math.pi / 2
    pool.pos[0, 0] = 0.05

    _resolver(walls).resolve(pool)

    assert pool.theta[0] == 0.0
    assert pool.pos[0].tolist() == pytest.approx([0.0, 0.0], abs=1e-12)


def test_capsule_through_a_doorway_aligned_passes(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    walls = [((0.0, -5.0), (0.0, -0.45)), ((0.0, 0.45), (0.0, 5.0))]
    pool = _pool(pool_empty, agent_factory, [(-1.0, 0.0, 0.0, 0.325, 0.225)])
    resolver = _resolver(walls)
    for _ in range(40):
        pool.begin_step()
        pool.vel[0] = (1.0, 0.0)
        pool.pos[0] += pool.vel[0] * _DT
        resolver.resolve(pool)
    assert pool.pos[0, 0] == pytest.approx(1.0, abs=1e-9)


# --- the crowd ---------------------------------------------------------------------------------------------------------


def _jam(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent], sweep: bool, ticks: int = 400) -> AgentPool:
    """A jammed corridor end: 45 walkers driven at the end wall x = 160 of a 2.4 m corridor, pressed by those behind."""
    walls = [((150.0, -1.2), (160.0, -1.2)), ((150.0, 1.2), (160.0, 1.2)), ((160.0, -1.2), (160.0, 1.2))]
    rng = np.random.default_rng(11)
    rows = [(float(rng.uniform(154.0, 159.5)), float(rng.uniform(-0.9, 0.9)), 0.0, _RADIUS, 0.0) for _ in range(45)]
    pool = _pool(pool_empty, agent_factory, rows)
    resolver = _resolver(walls)
    for _ in range(ticks):
        if sweep:
            pool.begin_step()
        pool.vel[: pool.n] = (1.6, 0.0)
        pool.pos[: pool.n] += pool.vel[: pool.n] * _DT
        resolver.resolve(pool)
    return pool


def test_jammed_crowd_never_leaves_through_the_walls(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    before = _jam(pool_empty, agent_factory, sweep=False)
    after = _jam(pool_empty, agent_factory, sweep=True)
    out_before = (before.pos[: before.n, 0] > 160.0) | (np.abs(before.pos[: before.n, 1]) > 1.2)
    assert int(out_before.sum()) > 0  # the mechanism reproduces without the step start
    assert np.all(after.pos[: after.n, 0] < 160.0)
    assert np.all(np.abs(after.pos[: after.n, 1]) < 1.2)


# --- kernel and pool ---------------------------------------------------------------------------------------------------


def test_sweep_kernel_matches_its_python_reference_on_a_maze(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    walls = [((float(x0), float(y0)), (float(x1), float(y1))) for (x0, y0), (x1, y1) in generate_maze(6, seed=3).walls]
    grid = WallGrid(walls)
    rng = np.random.default_rng(5)
    n = 60
    start = 2.0 * rng.integers(0, 6, size=(n, 2)) + 1.0 + rng.uniform(-0.6, 0.6, size=(n, 2))
    start_theta = rng.uniform(-np.pi, np.pi, n)
    radius = rng.uniform(0.2, 0.4, n)
    offset = np.where(np.arange(n) % 2 == 0, 0.0, rng.uniform(0.05, 0.4, n))
    pos = start + rng.normal(0.0, 0.5, size=(n, 2))
    vel = rng.normal(0.0, 1.0, size=(n, 2))
    theta = start_theta + rng.normal(0.0, 0.5, n)
    out = []
    for kernel in (_resolve_kernel, _resolve_kernel.py_func):
        p, v, th = pos.copy(), vel.copy(), theta.copy()
        corrected = np.empty(n, dtype=np.bool_)
        kernel(p, v, radius, grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, _MARGIN, offset, th, start, start_theta, True, corrected)
        out.append((p, v, th, corrected))
    for a, b in zip(out[0], out[1], strict=True):
        assert np.array_equal(a, b)
    assert out[0][3].any()


def test_pool_step_start_follows_rows(pool_empty: Callable[..., AgentPool], agent_factory: Callable[..., BaseAgent]) -> None:
    pool = _pool(pool_empty, agent_factory, [(1.0, 2.0, 0.5, 0.25, 0.0), (3.0, 4.0, 1.5, 0.25, 0.0), (5.0, 6.0, 2.5, 0.25, 0.0)])
    assert not pool.has_step_start
    assert pool.step_start_pos[1].tolist() == [3.0, 4.0]  # a new row starts where it is added
    pool.begin_step()
    pool.pos[: pool.n] += 10.0
    pool.swap_remove(2)  # agent id 2, row 1: the last row moves in
    assert pool.step_start_pos[1].tolist() == [5.0, 6.0]
    assert pool.step_start_theta[1] == 2.5
    for k in range(20):
        pool.add_agent(agent_factory(10 + k, x=float(k), y=0.0))
    assert pool.step_start_pos[0].tolist() == [1.0, 2.0]
    assert pool.step_start_pos[pool.n - 1].tolist() == [19.0, 0.0]
    pool.reset()
    assert not pool.has_step_start

