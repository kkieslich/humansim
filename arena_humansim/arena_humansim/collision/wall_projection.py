from __future__ import annotations

import math

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from arena_humansim.collision import CollisionResolver
from arena_humansim.core.pool import AgentPool
from arena_humansim.utils.types import Segments
from arena_humansim.utils.wall_grid import WallGrid, query_walls

_QUERY_CAPACITY = 64
_CONTACT_PASSES = 3
#: A point closer than this to a wall's line is on neither side of it [m]
_SIDE_EPS = 1e-6
#: Side corrections per call before a walker that still crosses a wall goes back to its step's start pose
_SIDE_PASSES = 4


@njit(cache=True)
def _contact(segments: np.ndarray, w: int, x: float, y: float) -> tuple[float, float, float]:
    ax = segments[w, 0]
    ay = segments[w, 1]
    abx = segments[w, 2] - ax
    aby = segments[w, 3] - ay
    ab_sq = max(abx * abx + aby * aby, 1e-12)
    t = min(max(((x - ax) * abx + (y - ay) * aby) / ab_sq, 0.0), 1.0)
    dx = x - (ax + t * abx)
    dy = y - (ay + t * aby)
    dist = math.sqrt(dx * dx + dy * dy)
    safe_dist = dist if dist > 1e-9 else 1e-9
    return dist, dx / safe_dist, dy / safe_dist


@njit(cache=True)
def _first_crossing(
    segments: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    buf: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    """The first wall the step from (ax, ay) to (bx, by) crosses: its chord parameter (2.0 for none), its unit normal
    towards the side the step starts on, and how far the step's end lies on that side of the wall's line (< _SIDE_EPS).
    A step that starts within _SIDE_EPS of a wall's line has no side of it to keep."""
    best_t = 2.0
    best_nx = 0.0
    best_ny = 0.0
    best_db = 0.0
    if ax == bx and ay == by:
        return best_t, best_nx, best_ny, best_db, buf
    mx = 0.5 * (ax + bx)
    my = 0.5 * (ay + by)
    reach = 0.5 * math.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay)) + 2.0 * _SIDE_EPS
    k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, mx, my, reach, buf)
    if k > buf.shape[0]:
        buf = np.empty(k, dtype=np.int64)
        k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, mx, my, reach, buf)
    for c in range(k):
        w = buf[c]
        cx = segments[w, 0]
        cy = segments[w, 1]
        sx = segments[w, 2] - cx
        sy = segments[w, 3] - cy
        length = math.sqrt(sx * sx + sy * sy)
        if length < 1e-12:
            continue
        ux = -sy / length
        uy = sx / length
        da = (ax - cx) * ux + (ay - cy) * uy
        if abs(da) <= _SIDE_EPS:
            continue
        if da < 0.0:
            ux = -ux
            uy = -uy
            da = -da
        db = (bx - cx) * ux + (by - cy) * uy
        if db >= _SIDE_EPS:
            continue
        t = da / (da - db) if db < 0.0 else 1.0
        qx = ax + t * (bx - ax)
        qy = ay + t * (by - ay)
        u = ((qx - cx) * sx + (qy - cy) * sy) / (length * length)
        pad = _SIDE_EPS / length
        if u < -pad or u > 1.0 + pad:
            continue
        if t < best_t:
            best_t = t
            best_nx = ux
            best_ny = uy
            best_db = db
    return best_t, best_nx, best_ny, best_db, buf


@njit(cache=True)
def _keep_side(
    i: int,
    pos: np.ndarray,
    vel: np.ndarray,
    theta: np.ndarray,
    offset: np.ndarray,
    start_pos: np.ndarray,
    start_theta: np.ndarray,
    segments: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    buf: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """Keep row i on the side of every wall its step started on: each disk center (one, or a capsule's two ends) moves
    from its start-of-step place to its place now, and where that step crosses a wall the agent goes back along the
    wall's normal to just short of the wall's line, its velocity into the wall dropped; contact projection then pushes it
    out on that side. One still crossing after _SIDE_PASSES corrections goes back to its start-of-step pose."""
    moved = False
    ends = 2 if offset[i] > 0.0 else 1
    s0x = start_pos[i, 0]
    s0y = start_pos[i, 1]
    o0x = offset[i] * math.cos(start_theta[i]) if ends == 2 else 0.0
    o0y = offset[i] * math.sin(start_theta[i]) if ends == 2 else 0.0
    for _ in range(_SIDE_PASSES):
        ox = offset[i] * math.cos(theta[i]) if ends == 2 else 0.0
        oy = offset[i] * math.sin(theta[i]) if ends == 2 else 0.0
        best_t = 2.0
        best_nx = 0.0
        best_ny = 0.0
        best_db = 0.0
        for end in range(ends):
            sign = 2.0 * end - 1.0 if ends == 2 else 0.0
            t, ux, uy, db, buf = _first_crossing(
                segments,
                cell_start,
                cell_walls,
                wall_cx0,
                wall_cy0,
                origin_x,
                origin_y,
                cell,
                nx,
                ny,
                s0x + sign * o0x,
                s0y + sign * o0y,
                pos[i, 0] + sign * ox,
                pos[i, 1] + sign * oy,
                buf,
            )
            if t < best_t:
                best_t = t
                best_nx = ux
                best_ny = uy
                best_db = db
        if best_t > 1.0:
            return moved, buf
        moved = True
        shift = 2.0 * _SIDE_EPS - best_db
        pos[i, 0] += shift * best_nx
        pos[i, 1] += shift * best_ny
        into = vel[i, 0] * best_nx + vel[i, 1] * best_ny
        if into < 0.0:
            vel[i, 0] -= into * best_nx
            vel[i, 1] -= into * best_ny
    still = False
    ox = offset[i] * math.cos(theta[i]) if ends == 2 else 0.0
    oy = offset[i] * math.sin(theta[i]) if ends == 2 else 0.0
    for end in range(ends):
        sign = 2.0 * end - 1.0 if ends == 2 else 0.0
        t, _, _, _, buf = _first_crossing(
            segments, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny,
            s0x + sign * o0x, s0y + sign * o0y, pos[i, 0] + sign * ox, pos[i, 1] + sign * oy, buf,
        )
        if t <= 1.0:
            still = True
    if still:
        pos[i, 0] = s0x
        pos[i, 1] = s0y
        theta[i] = start_theta[i]
    return moved, buf


@njit(cache=True, parallel=True)
def _resolve_kernel(
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    segments: np.ndarray,
    cell_start: np.ndarray,
    cell_walls: np.ndarray,
    wall_cx0: np.ndarray,
    wall_cy0: np.ndarray,
    origin_x: float,
    origin_y: float,
    cell: float,
    nx: int,
    ny: int,
    margin: float,
    offset: np.ndarray,
    theta: np.ndarray,
    start_pos: np.ndarray,
    start_theta: np.ndarray,
    sweep: bool,
    corrected: np.ndarray,
) -> None:
    for i in prange(pos.shape[0]):
        corrected[i] = False
        threshold = radius[i] + margin
        buf = np.empty(_QUERY_CAPACITY, dtype=np.int64)
        if sweep:
            moved, buf = _keep_side(i, pos, vel, theta, offset, start_pos, start_theta, segments, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, buf)
            if moved:
                corrected[i] = True
        if offset[i] > 0.0:
            for _ in range(3):
                ox = offset[i] * math.cos(theta[i])
                oy = offset[i] * math.sin(theta[i])
                x = pos[i, 0]
                y = pos[i, 1]
                vx = vel[i, 0]
                vy = vel[i, 1]
                hit = False
                cx = 0.0
                cy = 0.0
                prev_ids = np.empty(0, dtype=np.int64)
                prev_depths = np.empty(0)
                for end in range(2):
                    sign = 2.0 * end - 1.0
                    px = x + sign * ox
                    py = y + sign * oy
                    k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, px, py, threshold, buf)
                    if k > buf.shape[0]:
                        buf = np.empty(k, dtype=np.int64)
                        k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, px, py, threshold, buf)
                    ids = np.empty(k, dtype=np.int64)
                    depths = np.empty(k)
                    n = 0
                    for c in range(k):
                        dist, ux, uy = _contact(segments, buf[c], px, py)
                        if dist < threshold - 1e-9:
                            hit = True
                            depth = threshold - dist
                            push = depth
                            for j in range(prev_ids.shape[0]):
                                if prev_ids[j] == buf[c]:
                                    push = max(depth - prev_depths[j], 0.0)
                                    break
                            cx += ux * push
                            cy += uy * push
                            ids[n] = buf[c]
                            depths[n] = depth
                            n += 1
                            proj = vx * ux + vy * uy
                            if proj < 0:
                                vx -= proj * ux
                                vy -= proj * uy
                    prev_ids = ids[:n]
                    prev_depths = depths[:n]
                if not hit:
                    break
                corrected[i] = True
                pos[i, 0] = x + cx
                pos[i, 1] = y + cy
                vel[i, 0] = vx
                vel[i, 1] = vy
                if sweep:
                    _, buf = _keep_side(i, pos, vel, theta, offset, start_pos, start_theta, segments, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, buf)
            continue
        for _ in range(3):
            x = pos[i, 0]
            y = pos[i, 1]
            k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, threshold, buf)
            if k > buf.shape[0]:
                buf = np.empty(k, dtype=np.int64)
                k = query_walls(cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, x, y, threshold, buf)

            hit = False
            cx = 0.0
            cy = 0.0
            for c in range(k):
                dist, ux, uy = _contact(segments, buf[c], x, y)
                if dist < threshold - 1e-9:
                    hit = True
                    cx += ux * (threshold - dist)
                    cy += uy * (threshold - dist)
            if not hit:
                break

            corrected[i] = True
            pos[i, 0] = x + cx
            pos[i, 1] = y + cy

            vx = vel[i, 0]
            vy = vel[i, 1]
            for c in range(k):
                dist, ux, uy = _contact(segments, buf[c], x, y)
                if dist < threshold - 1e-9:
                    proj = vx * ux + vy * uy
                    if proj < 0:
                        vx -= proj * ux
                        vy -= proj * uy
            vel[i, 0] = vx
            vel[i, 1] = vy
            if sweep:
                _, buf = _keep_side(i, pos, vel, theta, offset, start_pos, start_theta, segments, cell_start, cell_walls, wall_cx0, wall_cy0, origin_x, origin_y, cell, nx, ny, buf)


@njit(cache=True)
def _nearest_ends(pos: np.ndarray, offset: np.ndarray, theta: np.ndarray, i: int, j: int) -> tuple[float, float]:
    oix = offset[i] * math.cos(theta[i])
    oiy = offset[i] * math.sin(theta[i])
    ojx = offset[j] * math.cos(theta[j])
    ojy = offset[j] * math.sin(theta[j])
    best = math.inf
    bx = 0.0
    by = 0.0
    for a in range(2):
        sa = 2.0 * a - 1.0
        for b in range(2):
            sb = 2.0 * b - 1.0
            dx = pos[i, 0] + sa * oix - pos[j, 0] - sb * ojx
            dy = pos[i, 1] + sa * oiy - pos[j, 1] - sb * ojy
            d_sq = dx * dx + dy * dy
            if d_sq < best:
                best = d_sq
                bx = dx
                by = dy
    return bx, by


@njit(cache=True)
def _separate_kernel(pos: np.ndarray, vel: np.ndarray, radius: np.ndarray, movable: np.ndarray, pairs: np.ndarray, offset: np.ndarray, theta: np.ndarray) -> None:
    for _ in range(_CONTACT_PASSES):
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]
            mi = 1.0 if movable[i] else 0.0
            mj = 1.0 if movable[j] else 0.0
            total = mi + mj
            if total == 0.0:
                continue
            if offset[i] > 0.0 or offset[j] > 0.0:
                dx, dy = _nearest_ends(pos, offset, theta, i, j)
            else:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            pen = radius[i] + radius[j] - d
            if pen <= 0.0:
                continue
            if d > 1e-9:
                nx = dx / d
                ny = dy / d
            else:
                nx = 1.0
                ny = 0.0
            if total == 1.0:
                m = i if mi == 1.0 else j
                s = 1.0 if mi == 1.0 else -1.0
                vn = (vel[m, 0] * nx + vel[m, 1] * ny) * s
                if vn < 0.0:
                    vel[m, 0] -= vn * s * nx
                    vel[m, 1] -= vn * s * ny
                continue
            pos[i, 0] += 0.5 * pen * nx
            pos[i, 1] += 0.5 * pen * ny
            pos[j, 0] -= 0.5 * pen * nx
            pos[j, 1] -= 0.5 * pen * ny
            rv = (vel[i, 0] - vel[j, 0]) * nx + (vel[i, 1] - vel[j, 1]) * ny
            if rv < 0.0:
                vel[i, 0] -= 0.5 * rv * nx
                vel[i, 1] -= 0.5 * rv * ny
                vel[j, 0] += 0.5 * rv * nx
                vel[j, 1] += 0.5 * rv * ny


def _resolve_grid(
    grid: WallGrid,
    pos: np.ndarray,
    vel: np.ndarray,
    radius: np.ndarray,
    margin: float,
    offset: np.ndarray,
    theta: np.ndarray,
    start_pos: np.ndarray | None = None,
    start_theta: np.ndarray | None = None,
) -> np.ndarray:
    """Wall contact of every row; with a start-of-step pose per row (`start_pos`, `start_theta`) no row ends its step across a wall."""
    corrected = np.empty(pos.shape[0], dtype=np.bool_)
    sweep = start_pos is not None and start_theta is not None
    if not sweep:
        start_pos = pos
        start_theta = theta
    _resolve_kernel(pos, vel, radius, grid.segments, grid.cell_start, grid.cell_walls, grid.wall_cx0, grid.wall_cy0, grid.origin_x, grid.origin_y, grid.cell, grid.nx, grid.ny, margin, offset, theta, start_pos, start_theta, sweep, corrected)
    return corrected


class WallProjectionResolver(CollisionResolver):
    def __init__(self, margin: float = 0.01):
        self._margin = margin
        self._grid = WallGrid([])
        self._warmup()

    def _warmup(self) -> None:
        grid = WallGrid([((0.0, 0.0), (1.0, 0.0))])
        pairs = np.array([[0, 1]], dtype=np.int64)
        for offset in (0.0, 0.2):
            pos = np.array([[0.1, 0.1]], dtype=np.float64)
            vel = np.array([[-1.0, -1.0]], dtype=np.float64)
            radius = np.array([0.3], dtype=np.float64)
            _resolve_grid(grid, pos, vel, radius, self._margin, np.full(1, offset), np.zeros(1))
            _resolve_grid(grid, pos, vel, radius, self._margin, np.full(1, offset), np.zeros(1), np.array([[0.1, 0.5]]), np.zeros(1))
            _separate_kernel(np.zeros((2, 2)), np.zeros((2, 2)), np.full(2, 0.3), np.ones(2, dtype=np.bool_), pairs, np.full(2, offset), np.zeros(2))

    def set_walls(self, segments: Segments) -> None:
        self._grid = WallGrid(segments)

    def resolve(self, pool: AgentPool) -> set[int]:
        self._separate(pool)
        return self._project(pool)

    def _separate(self, pool: AgentPool) -> None:
        n = pool.n
        if n < 2:
            return
        pos = pool.pos[:n]
        radius = pool.agent_radius[:n]
        offset = pool.axial_offset[:n]
        pairs = cKDTree(pos).query_pairs(2.0 * float(radius.max()) + 2.0 * float(offset.max()), output_type="ndarray")
        if len(pairs) == 0:
            return
        i, j = pairs[:, 0], pairs[:, 1]
        pairs = pairs[np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) < radius[i] + radius[j] + offset[i] + offset[j]]
        if len(pool.ignored_pairs):
            n_rows = pool.n
            key = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64) * n_rows + np.maximum(pairs[:, 0], pairs[:, 1])
            skip = np.minimum(pool.ignored_pairs[:, 0], pool.ignored_pairs[:, 1]) * n_rows + np.maximum(pool.ignored_pairs[:, 0], pool.ignored_pairs[:, 1])
            pairs = pairs[~np.isin(key, skip)]
            if len(pairs) == 0:
                return
        ids = pool.agent_ids[:n][pairs]
        pairs = pairs[np.lexsort((ids.max(axis=1), ids.min(axis=1)))]
        _separate_kernel(pos, pool.vel[:n], radius, pool.policy_idx[:n] != -1, pairs.astype(np.int64), offset, pool.theta[:n])

    def _project(self, pool: AgentPool) -> set[int]:
        n = pool.n
        if n == 0 or self._grid.segments.shape[0] == 0:
            return set()

        start_pos, start_theta = (pool.step_start_pos[:n], pool.step_start_theta[:n]) if pool.has_step_start else (None, None)
        corrected = _resolve_grid(self._grid, pool.pos[:n], pool.vel[:n], pool.agent_radius[:n], self._margin, pool.axial_offset[:n], pool.theta[:n], start_pos, start_theta)

        if not corrected.any():
            return set()
        return {int(aid) for aid in pool.agent_ids[:n][corrected]}
