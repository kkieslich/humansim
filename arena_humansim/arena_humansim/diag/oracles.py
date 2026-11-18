"""Counterfactual behaviours for diagnosis (test-only, never for study runs).

`YieldOracle`: walkers that are not wheelchair users give way, ideally, to a wheelchair user whose path they stand in.
A walker yields when a wheelchair user within `radius` (from the chair's front end) heads at it: the walker lies ahead of
the chair's centre and within the chair's swept half-width (0.325 m) plus its own radius plus `margin` of the chair's
axis, and does not itself walk the chair's way (its subgoal direction makes a cosine below `same_way` with the chair's
heading). It then yields for `hold_s` seconds from the last tick it was in the way:
  - mode "passive": its desired speed is 0 for the local planner (no goal force; the chair's and others' pushes and
    its own braking act);
  - mode "active": its subgoal is replaced by a point `step` m to the side of the chair's axis it already stands on
    (perpendicular to the chair's current heading), at its own desired speed: it steps out of the way.
The pool's desired speeds and subgoals are restored after each local-planner call, so nothing else sees the change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arena_humansim.core.agent_manager import AgentManager

CHAIR_HALF_WIDTH = 0.325
CHAIR_HALF_LENGTH = 0.55


class YieldOracle:
    def __init__(self, mgr: AgentManager, mode: str = "passive", radius: float = 2.0, hold_s: float = 2.0, margin: float = 0.25, step: float = 1.0, same_way: float = 0.3, chair_prefix: str = "wheelchair") -> None:
        assert mode in ("passive", "active"), mode
        self.mgr = mgr
        self.mode = mode
        self.radius = radius
        self.hold_s = hold_s
        self.margin = margin
        self.step = step
        self.same_way = same_way
        self.chair_prefix = chair_prefix
        self._wrapped: set[int] = set()
        self._until: dict[int, tuple[float, int, float]] = {}  # yielder id -> (until t, chair id, side)
        self._tick_done = -1
        self._is_chair: dict[int, bool] = {}
        self.yield_ticks = 0  # walker-ticks spent yielding
        self.events = 0  # new yield episodes

    def _chair(self, aid: int) -> bool:
        v = self._is_chair.get(aid)
        if v is None:
            ag = self.mgr._agents.get(aid)
            v = ag is not None and self.chair_prefix in ag.params.name
            self._is_chair[aid] = v
        return v

    def begin_tick(self) -> None:
        for p in self.mgr._policies:
            if id(p) in self._wrapped or not hasattr(p, "compute_pool"):
                continue
            self._wrapped.add(id(p))
            p.compute_pool = self._wrap(p.compute_pool)

    def _update(self, pool) -> None:
        tick = self.mgr._tick_count
        if tick == self._tick_done:
            return
        self._tick_done = tick
        now = tick * self.mgr._dt
        n = pool.n
        ids = pool.agent_ids[:n]
        chair = np.array([self._chair(int(a)) for a in ids], np.bool_)
        nav = pool.has_terminal[:n]
        crows = np.flatnonzero(chair & nav)
        wrows = np.flatnonzero(~chair & nav)
        if crows.size and wrows.size:
            pos = pool.pos[:n]
            h = np.column_stack([np.cos(pool.theta[:n][crows]), np.sin(pool.theta[:n][crows])])
            rel = pos[wrows][None, :, :] - pos[crows][:, None, :]  # chair x walker
            ahead = rel[..., 0] * h[:, None, 0] + rel[..., 1] * h[:, None, 1]
            lat = h[:, None, 0] * rel[..., 1] - h[:, None, 1] * rel[..., 0]
            rw = pool.agent_radius[:n][wrows]
            dg = pool.goal_pos[:n][wrows] - pos[wrows]
            eg = dg / np.maximum(np.hypot(dg[:, 0], dg[:, 1]), 1e-6)[:, None]
            cos_way = eg[None, :, 0] * h[:, None, 0] + eg[None, :, 1] * h[:, None, 1]
            inway = (ahead > 0.0) & (ahead < CHAIR_HALF_LENGTH + self.radius) & (np.abs(lat) < CHAIR_HALF_WIDTH + rw[None, :] + self.margin) & (cos_way < self.same_way)
            ci, wi = np.nonzero(inway)
            for c, w in zip(ci, wi):
                wid = int(ids[wrows[w]])
                side = 1.0 if lat[c, w] >= 0.0 else -1.0
                if wid not in self._until or self._until[wid][0] < now:
                    self.events += 1
                self._until[wid] = (now + self.hold_s, int(ids[crows[c]]), side)
        for wid in [k for k, v in self._until.items() if v[0] < now or k not in pool._id_to_idx]:
            del self._until[wid]

    def _wrap(self, orig):
        oracle = self

        def compute_pool(pool, store_forces=False, dt=1.0):
            oracle._update(pool)
            if not oracle._until:
                return orig(pool, store_forces=store_forces, dt=dt)
            rows, goals = [], []
            for wid, (_, cid, side) in oracle._until.items():
                i = pool._id_to_idx.get(wid)
                if i is None:
                    continue
                rows.append(i)
                if oracle.mode == "active":
                    j = pool._id_to_idx.get(cid)
                    th = pool.theta[j] if j is not None else 0.0
                    nl = np.array([-np.sin(th), np.cos(th)])
                    goals.append(pool.pos[i] + side * oracle.step * nl)
            rows = np.array(rows, np.int64)
            saved_v = pool.desired_vel[rows].copy()
            saved_g = pool.goal_pos[rows].copy()
            if oracle.mode == "passive":
                pool.desired_vel[rows] = 0.0
            else:
                pool.goal_pos[rows] = np.array(goals)
            try:
                return orig(pool, store_forces=store_forces, dt=dt)
            finally:
                pool.desired_vel[rows] = saved_v
                pool.goal_pos[rows] = saved_g
                oracle.yield_ticks += int((pool.has_goal[rows]).sum())

        return compute_pool


class GoalSpread:
    """Capacity ablation (not a behaviour proposal): every walker aims at its own point across the passage instead of the
    shared route corner. Each walker draws a fixed u in [-1, 1]; its subgoal moves perpendicular to the line from the
    walker to the subgoal by u * s, s = the free half-width at the subgoal on that side (ray cast to the walls) minus the
    walker's radius and 0.15 m, capped at `cap`. Applied to `pool.set_goals` after the global planner, so the planner's
    routes, its replanning and every other module see their own subgoals unchanged; only the local planner's target
    moves. `types`: the walker types it applies to (empty = all)."""

    def __init__(self, mgr: AgentManager, cap: float = 1.5, types: set[str] | None = None, seed: int = 7) -> None:
        from arena_humansim.utils.wall_grid import WallGrid

        self.mgr = mgr
        self.cap = cap
        self.types = types or set()
        self.rng = np.random.default_rng(seed)
        self.u: dict[int, float] = {}
        gp = mgr._global_planner
        segs = np.asarray(getattr(gp, "_wall_segments", []), dtype=np.float64).reshape(-1, 4)
        self.grid = WallGrid(segs) if len(segs) else None
        pool = mgr._pool
        orig = pool.set_goals
        spread = self

        def set_goals(goals):
            orig(goals)
            spread._shift(pool, goals)

        pool.set_goals = set_goals

    def _shift(self, pool, goals) -> None:
        from arena_humansim.utils.wall_grid import cast_rays

        if self.grid is None:
            gp = self.mgr._global_planner
            segs = np.asarray(getattr(gp, "_wall_segments", []), dtype=np.float64).reshape(-1, 4)
            if not len(segs):
                return
            from arena_humansim.utils.wall_grid import WallGrid

            self.grid = WallGrid(segs)
        rows, aids = [], []
        for aid in goals:
            i = pool._id_to_idx.get(aid)
            if i is None:
                continue
            if self.types:
                ag = self.mgr._agents.get(aid)
                if ag is None or ag.params.name not in self.types:
                    continue
            rows.append(i)
            aids.append(aid)
        if not rows:
            return
        rows = np.array(rows)
        g = pool.goal_pos[rows].copy()
        d = g - pool.pos[rows]
        dn = np.hypot(d[:, 0], d[:, 1])
        ok = dn > 0.5
        perp = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(dn, 1e-9)[:, None]
        u = np.array([self.u.setdefault(a, float(self.rng.uniform(-1.0, 1.0))) for a in aids])
        side = np.where(u >= 0, 1.0, -1.0)[:, None] * perp
        free = cast_rays(self.grid, g, side, self.cap + 1.0)
        s = np.clip(free - pool.agent_radius[rows] - 0.15, 0.0, self.cap)
        pool.goal_pos[rows] = np.where(ok[:, None], g + (np.abs(u) * s)[:, None] * side, g)
