"""Pushed wheelchairs: the pusher steers, the chair (its rider's agent) rides rigidly at her hands.

Each tick the manager hands `Carrier` the pushes of the interaction manager (`InteractionManager.carries`) and calls it
around the stages of the tick:

- `exclude` after perception: a pusher and her chair stop seeing each other (neighbour lists, and through
  `AgentPool.ignored_pairs` the contact separation) from when she reaches for the handles until she lets go. After
  that she sees the chair again and steers around it; only the chair keeps ignoring her and the contact separation
  leaves the pair alone until their footprints are apart, so stepping back from the handles does not shove it.
- `hold_still` before local planning: a held rider plans from rest and with no way of its own, so its planner only
  reacts to what is around the chair.
- `transmit` after local planning: that reaction (the rider's planned velocity) is added to the pusher's, so walls and
  people ahead of the chair slow and steer her.
- `steer` after the kinematic constraints: the pusher moves along her heading and turns like a pushed chair
  (`PIVOT_RATE`, `TURN_RADIUS`), the rider moves with her.
- `place` after collision: what collision did to either of them moves both, then the chair is put at her hands. A
  freshly taken chair rolls there over the formation's `grab_s` from where she found it. A pusher who let go is kept
  from getting any deeper into the chair's footprint than she was when she let go (only she moves).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arena_humansim.core.formation.push import PushFormation
    from arena_humansim.core.pool import AgentPool

PIVOT_RATE = 0.8  # rad/s, a pushed chair turned on the spot
TURN_RADIUS = 0.5  # m
MAX_TRANSMIT = 0.5  # m/s, the most the chair's planner hands the pusher per tick
REACH_RADIUS = 0.6  # m from the handles where the pusher and the chair stop avoiding each other
APART_MARGIN = 0.05  # m between their footprints that counts as apart after she lets go


def _smooth(w: float) -> float:
    w = min(max(w, 0.0), 1.0)
    return w * w * (3.0 - 2.0 * w)


class _Hold:
    __slots__ = ("formation", "grab", "rel", "t")

    def __init__(self, formation: PushFormation, rel: tuple[float, float, float]) -> None:
        self.formation = formation
        self.rel = rel  # the rider in the pusher's frame when she took the handles
        self.t = 0.0
        self.grab = float(formation.grab_s)


class Carrier:
    def __init__(self) -> None:
        self._pushes: list[tuple[int, int, PushFormation]] = []
        self._holds: dict[tuple[int, int], _Hold] = {}
        self._released: dict[tuple[int, int], float] = {}  # let go, not apart yet: the deepest overlap she may still have (m, <= 0)
        self._snap: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self._pushes = []
        self._holds.clear()
        self._released.clear()
        self._snap.clear()

    def held(self) -> set[int]:
        """Riders being pushed now."""
        return {r for _, r in self._holds}

    def update(self, pool: AgentPool, pushes: list[tuple[int, int, PushFormation]], dt: float) -> None:
        """Take this tick's pushes: start holds whose pusher just took the handles, end the ones let go."""
        self._pushes = [(p, r, f) for p, r, f in pushes if p in pool._id_to_idx and r in pool._id_to_idx]
        live: set[tuple[int, int]] = set()
        for p, r, formation in self._pushes:
            if not formation.attached:
                continue
            key = (p, r)
            live.add(key)
            hold = self._holds.get(key)
            if hold is None:
                i, j = pool.idx(p), pool.idx(r)
                self._holds[key] = _Hold(formation, self._relative(pool.pos[i], pool.theta[i], pool.pos[j], pool.theta[j]))
                self._released.pop(key, None)
            else:
                hold.t += dt
        for key in [k for k in self._holds if k not in live]:
            del self._holds[key]
            if key[0] in pool._id_to_idx and key[1] in pool._id_to_idx:
                self._released[key] = min(self._gap(pool, *key)[0], 0.0)
            if key[1] in pool._id_to_idx:  # let go, the chair stops where she left it
                j = pool.idx(key[1])
                pool.vel[j] = 0.0
                pool.prev_vel[j] = 0.0
        for key in list(self._released):
            if key in live or key[0] not in pool._id_to_idx or key[1] not in pool._id_to_idx or self._gap(pool, *key)[0] > APART_MARGIN:
                del self._released[key]

    @staticmethod
    def _relative(pp: np.ndarray, pt: float, rp: np.ndarray, rt: float) -> tuple[float, float, float]:
        c, s = math.cos(pt), math.sin(pt)
        dx, dy = float(rp[0] - pp[0]), float(rp[1] - pp[1])
        return c * dx + s * dy, -s * dx + c * dy, math.remainder(rt - pt, math.tau)

    def _target(self, hold: _Hold) -> tuple[float, float, float]:
        """The rider in the pusher's frame now: from where she found it to straight ahead at the handle offset."""
        w = _smooth(hold.t / hold.grab) if hold.grab > 0.0 else 1.0
        x0, y0, a0 = hold.rel
        x1 = float(hold.formation.handle_offset)
        return x0 + w * (x1 - x0), (1.0 - w) * y0, (1.0 - w) * a0

    @staticmethod
    def _gap(pool: AgentPool, p: int, r: int) -> tuple[float, np.ndarray]:
        """Clearance between the pusher's and the chair's footprints (negative when they overlap) and the unit normal
        from the chair to her. A footprint is the disks at its two ends, as the contact separation sees it."""
        i, j = pool.idx(p), pool.idx(r)
        best, normal = math.inf, np.array([1.0, 0.0])
        ei = [pool.pos[i] + s * pool.axial_offset[i] * np.array([math.cos(pool.theta[i]), math.sin(pool.theta[i])]) for s in (1.0, -1.0)]
        ej = [pool.pos[j] + s * pool.axial_offset[j] * np.array([math.cos(pool.theta[j]), math.sin(pool.theta[j])]) for s in (1.0, -1.0)]
        for ci in ei:
            for cj in ej:
                d = ci - cj
                n = float(np.hypot(*d))
                if n < best:
                    best, normal = n, d / n if n > 1e-9 else np.array([1.0, 0.0])
        return best - float(pool.agent_radius[i] + pool.agent_radius[j]), normal

    def _reaching(self, pool: AgentPool, p: int, formation: PushFormation) -> bool:
        if formation.attached:
            return True
        i = pool.idx(p)
        slot = formation.handles()
        return math.hypot(pool.pos[i, 0] - slot.x, pool.pos[i, 1] - slot.y) <= REACH_RADIUS

    def exclude(self, pool: AgentPool) -> None:
        """Drop each pusher-chair pair from the neighbour lists and from contact separation."""
        keys = {(p, r) for p, r, f in self._pushes if self._reaching(pool, p, f)}
        pairs = [(pool.idx(p), pool.idx(r)) for p, r in keys if p in pool._id_to_idx and r in pool._id_to_idx]
        let_go = [(pool.idx(p), pool.idx(r)) for p, r in self._released if (p, r) not in keys and p in pool._id_to_idx and r in pool._id_to_idx]
        pool.ignored_pairs = np.array(pairs + let_go, dtype=np.int64).reshape(-1, 2)
        if not (pairs or let_go) or pool.n == 0:
            return
        n = pool.n
        indptr, indices = pool.neighbor_indptr, pool.neighbor_indices
        if len(indices) == 0 or len(indptr) != n + 1:
            return
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        # row a lists what a sees: both ways for a held pair, only the chair's row once she let go
        ignored = np.array([a * n + b for a, b in pairs] + [b * n + a for a, b in pairs + let_go], dtype=np.int64)
        keep = ~np.isin(rows * n + indices.astype(np.int64), ignored)
        if keep.all():
            return
        counts = np.bincount(rows[keep], minlength=n)
        new_indptr = np.zeros(n + 1, dtype=indptr.dtype)
        np.cumsum(counts, out=new_indptr[1:])
        pool.set_neighbor_csr(new_indptr, indices[keep])

    def hold_still(self, pool: AgentPool) -> None:
        for _, r in self._holds:
            j = pool.idx(r)
            pool.has_goal[j] = False
            pool.vel[j] = 0.0
            pool.prev_vel[j] = 0.0

    def transmit(self, pool: AgentPool) -> None:
        for p, r in self._holds:
            i, j = pool.idx(p), pool.idx(r)
            dv = pool.vel[j].copy()
            norm = float(np.hypot(*dv))
            if norm > MAX_TRANSMIT:
                dv = dv * (MAX_TRANSMIT / norm)
            pool.vel[i] += dv

    def steer(self, pool: AgentPool, dt: float) -> None:
        for (p, r), hold in self._holds.items():
            i, j = pool.idx(p), pool.idx(r)
            v = pool.vel[i]
            speed = float(np.hypot(*v))
            theta = float(pool.theta[i])
            if speed > 1e-6:
                delta = math.remainder(math.atan2(v[1], v[0]) - theta, math.tau)
                limit = max(PIVOT_RATE, speed / TURN_RADIUS) * dt
                theta += min(max(delta, -limit), limit)
                pool.theta[i] = theta
            f = np.array([math.cos(theta), math.sin(theta)])
            pool.vel[i] = max(float(v @ f), 0.0) * f
            pool.has_goal_theta[i] = False
            x, y, a = self._target(hold)
            pool.pos[j] = pool.pos[i] + np.array([f[0] * x - f[1] * y, f[1] * x + f[0] * y])
            pool.theta[j] = theta + a
            pool.vel[j] = pool.vel[i]
            pool.has_goal_theta[j] = False

    def before_collision(self, pool: AgentPool) -> None:
        self._snap = {(p, r): (pool.pos[pool.idx(p)].copy(), pool.pos[pool.idx(r)].copy()) for p, r in self._holds}

    def place(self, pool: AgentPool) -> None:
        for (p, r), hold in self._holds.items():
            i, j = pool.idx(p), pool.idx(r)
            snap = self._snap.get((p, r))
            if snap is not None:
                pool.pos[i] = snap[0] + (pool.pos[i] - snap[0]) + (pool.pos[j] - snap[1])
            theta = float(pool.theta[i])
            c, s = math.cos(theta), math.sin(theta)
            x, y, a = self._target(hold)
            pool.pos[j] = pool.pos[i] + np.array([c * x - s * y, s * x + c * y])
            pool.theta[j] = theta + a
            pool.vel[j] = pool.vel[i]
            pool.prev_vel[j] = pool.vel[i]
        self._snap = {}
        for (p, r), deepest in self._released.items():
            gap, normal = self._gap(pool, p, r)
            if gap >= deepest:
                self._released[(p, r)] = min(gap, 0.0)  # she only ever gets out of the chair's way
                continue
            i = pool.idx(p)
            pool.pos[i] += (deepest - gap) * normal
            into = float(pool.vel[i] @ normal)
            if into < 0.0:
                pool.vel[i] -= into * normal
