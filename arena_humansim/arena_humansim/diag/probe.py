"""Force and velocity probe for chosen walker types (test-only diagnostics).

Attach to a running `AgentManager` with `ForceProbe(mgr, types)`; call `begin_tick()` before each `mgr.tick()` and
`flush(t)` every `window` ticks. Each flushed row holds, for one tracked walker, the means over the window of:
goal force (`f_att`), neighbour push (`f_rep`), wall push (`f_wall`) as the local planner computed them, the velocity
after the planner (`v_plan`), after the kinematic limits (`v_kin`), after the locomotion constraints (`v_con`: forward
only for along-heading bodies, stall recovery), the displacement of the integration step and of the contact separation,
the fraction of ticks in stall recovery (rotate, reverse) and held at a passage, the unit direction to its subgoal, and
the neighbour push split by the neighbour's type, with the K strongest neighbours by id.
The per-neighbour push replicates `SFMPlanner._compute_forces_pool` pair by pair; `check` holds the largest difference
between its sum and the planner's `f_rep` (it must stay at rounding level).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from arena_humansim.core.pool import KIND_ROBOT
from arena_humansim.local_planner.sfm import SFMPlanner, _closest_disk_diff

if TYPE_CHECKING:
    from arena_humansim.core.agent_manager import AgentManager
    from arena_humansim.core.pool import AgentPool

_EPS = 1e-6
VEC_FIELDS = ("f_att", "f_rep", "f_wall", "v_plan", "v_kin", "v_con", "dp_int", "dp_sep", "e_goal")
TYPE_GROUPS = ("adult", "wheelchair", "cane", "other")


def type_group(name: str) -> int:
    if name.startswith("wheelchair") or "wheelchair" in name:
        return 1
    if "cane" in name or name.startswith("blind"):
        return 2
    if name == "adult" or name.startswith("adult") or name.startswith("cd_adult") or "adult" in name:
        return 0
    return 3


def pair_forces(planner: SFMPlanner, pool: AgentPool, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each neighbour's push on each of `rows`, exactly as the planner sums it: (observer rows, neighbour rows, forces)."""
    n = pool.n
    indptr = pool.neighbor_indptr
    indices = pool.neighbor_indices
    if rows.size == 0 or len(indices) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2))
    counts = indptr[rows + 1] - indptr[rows]
    obs = np.repeat(rows, counts).astype(np.int64)
    if obs.size == 0:
        return obs, obs.copy(), np.zeros((0, 2))
    nbr = np.concatenate([indices[indptr[r] : indptr[r + 1]] for r in rows]).astype(np.int64)
    pos = pool.pos[:n]
    d_goal = pool.goal_pos[:n][obs] - pos[obs]
    e_goal = d_goal / np.maximum(np.hypot(d_goal[:, 0], d_goal[:, 1]), _EPS)[:, None]
    offsets = pool.axial_offset[:n]
    if np.any(offsets > 0.0):
        diff = _closest_disk_diff(pos, pool.theta[:n], offsets, obs, nbr)
    else:
        diff = pos[obs] - pos[nbr]
    dists = np.maximum(np.hypot(diff[:, 0], diff[:, 1]), _EPS)
    normals = diff / dists[:, None]
    cls = pool.interaction_class[:n]
    planner._ensure_gain_classes(int(cls.max()) + 1)
    n_cls = planner._gain_strength_scale.shape[0]
    oc = np.clip(cls[obs].astype(np.int64), 0, n_cls - 1)
    nc = np.clip(cls[nbr].astype(np.int64), 0, n_cls - 1)
    radii = pool.agent_radius[:n]
    r_ij = radii[obs] + radii[nbr]
    eff_s = planner._repulsion_strength[:n][obs] * planner._gain_strength_scale[oc, nc]
    eff_r = planner._repulsion_range[:n][obs] * planner._gain_range_scale[oc, nc]
    robot = pool.kind[:n][nbr] == KIND_ROBOT
    if np.any(robot):
        eff_s = eff_s * np.where(robot, pool.robot_repulsion_gain[:n][obs], 1.0)
        eff_r = eff_r * np.where(robot, pool.robot_reaction_gain[:n][obs], 1.0)
    eff_r = np.maximum(eff_r, 1e-4)
    mag = eff_s * np.exp((r_ij - dists) / eff_r)
    lam = planner._anisotropy[:n][obs]
    cos_phi = np.sum(-normals * e_goal, axis=1)
    mag = mag * (lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi))
    f = mag[:, None] * normals
    bias = planner._side_bias[:n][obs] * planner.side
    if planner.side != 0.0 and np.any(bias != 0.0):
        oncoming = (cos_phi > 0.0) & (np.sum(pool.vel[:n][nbr] * e_goal, axis=1) < 0.0) & pool.has_goal[:n][obs]
        side = np.column_stack([e_goal[:, 1], -e_goal[:, 0]])
        f = f + (mag * bias * oncoming)[:, None] * side
    return obs, nbr, f


class ForceProbe:
    def __init__(self, mgr: AgentManager, types: set[str] | list[str], top_k: int = 4) -> None:
        self.mgr = mgr
        self.types = set(types)
        self.top_k = top_k
        self._wrapped: set[int] = set()
        self._type_of: dict[int, str] = {}
        self._acc: dict[int, dict[str, np.ndarray]] = {}
        self._cnt: dict[int, int] = defaultdict(int)
        self._mode: dict[int, np.ndarray] = {}
        self._pairs: dict[int, dict[int, np.ndarray]] = {}
        self._grp: dict[int, np.ndarray] = {}
        self._pos_before: np.ndarray | None = None
        self.rows: list[tuple] = []
        self.check = 0.0
        self._install()

    # ---- bookkeeping -------------------------------------------------------------------------------------------
    def _name(self, aid: int) -> str:
        name = self._type_of.get(aid)
        if name is None:
            ag = self.mgr._agents.get(aid)
            name = ag.params.name if ag is not None else ""
            self._type_of[aid] = name
        return name

    def tracked_rows(self, pool: AgentPool) -> np.ndarray:
        n = pool.n
        return np.array([i for i in range(n) if self._name(int(pool.agent_ids[i])) in self.types], np.int64)

    def _slot(self, aid: int) -> dict[str, np.ndarray]:
        acc = self._acc.get(aid)
        if acc is None:
            acc = {k: np.zeros(2) for k in VEC_FIELDS}
            self._acc[aid] = acc
            self._mode[aid] = np.zeros(4)  # rotate, reverse, hold, steer_to_goal
            self._pairs[aid] = {}
            self._grp[aid] = np.zeros((len(TYPE_GROUPS), 2))
        return acc

    # ---- hooks -------------------------------------------------------------------------------------------------
    def _install(self) -> None:
        mgr = self.mgr
        kin = mgr._apply_kinematic_constraints_vectorized
        con = mgr._locomotion.constrain
        integ = mgr._integrate_state_vectorized
        res = mgr._collision.resolve
        probe = self

        def kin_w(pool):
            kin(pool)
            probe._take(pool, "v_kin", pool.vel)

        def con_w(pool, dt, provides_heading):
            con(pool, dt, provides_heading)
            probe._take(pool, "v_con", pool.vel)
            loc = mgr._locomotion
            for i in probe._rows_now:
                m = probe._mode.get(int(pool.agent_ids[i]))
                if m is None:
                    continue
                m[0] += loc.mode[i] == 1
                m[1] += loc.mode[i] == 2
                m[2] += bool(loc.hold[i])
                m[3] += bool(pool.steer_to_goal[i])

        def integ_w(pool):
            p0 = pool.pos[: pool.n].copy()
            integ(pool)
            probe._take(pool, "dp_int", pool.pos[: pool.n] - p0)

        def res_w(pool):
            p0 = pool.pos[: pool.n].copy()
            out = res(pool)
            probe._take(pool, "dp_sep", pool.pos[: pool.n] - p0)
            return out

        mgr._apply_kinematic_constraints_vectorized = kin_w
        mgr._locomotion.constrain = con_w
        mgr._integrate_state_vectorized = integ_w
        mgr._collision.resolve = res_w
        self._rows_now = np.zeros(0, np.int64)

    def _wrap_planner(self, planner) -> None:
        if id(planner) in self._wrapped or not hasattr(planner, "compute_pool"):
            return
        self._wrapped.add(id(planner))
        orig = planner.compute_pool
        probe = self

        def compute_pool(pool, store_forces=False, dt=1.0):
            n = pool.n
            pidx = next((k for k, p in enumerate(probe.mgr._policies) if p is planner), -1)
            rows = probe._rows_now
            own = rows[(pool.policy_idx[rows] == pidx) & pool.has_goal[rows]] if rows.size else rows
            pf = pair_forces(planner, pool, own) if isinstance(planner, SFMPlanner) and own.size else None
            orig(pool, store_forces=True, dt=dt)
            arrs = planner._last_force_arrays
            if own.size and arrs is not None:
                _, _, f_att, f_rep, f_wall = arrs
                d_goal = pool.goal_pos[:n][own] - pool.pos[:n][own]
                e_goal = d_goal / np.maximum(np.hypot(d_goal[:, 0], d_goal[:, 1]), _EPS)[:, None]
                for k, i in enumerate(own):
                    aid = int(pool.agent_ids[i])
                    acc = probe._slot(aid)
                    acc["f_att"] += f_att[i]
                    acc["f_rep"] += f_rep[i]
                    acc["f_wall"] += f_wall[i]
                    acc["v_plan"] += pool.vel[i]
                    acc["e_goal"] += e_goal[k]
                    probe._cnt[aid] += 1
                if pf is not None:
                    obs, nbr, f = pf
                    total = np.zeros((n, 2))
                    np.add.at(total, obs, f)
                    probe.check = max(probe.check, float(np.max(np.abs(total[own] - f_rep[own]))) if own.size else 0.0)
                    for o, j, fv in zip(obs, nbr, f):
                        aid = int(pool.agent_ids[o])
                        nid = int(pool.agent_ids[j])
                        d = probe._pairs[aid]
                        if nid in d:
                            d[nid] += fv
                        else:
                            d[nid] = fv.copy()
                        probe._grp[aid][type_group(probe._name(nid))] += fv
            if not store_forces:
                planner._last_force_arrays = None

        planner.compute_pool = compute_pool

    def _take(self, pool: AgentPool, key: str, arr: np.ndarray) -> None:
        for i in self._rows_now:
            aid = int(pool.agent_ids[i])
            if aid in self._acc:
                self._acc[aid][key] += arr[i]

    def begin_tick(self) -> None:
        """Before `mgr.tick()`: wrap planners that appeared since and fix the tracked rows. Rows can change within a
        tick only by spawning and despawning, which happen before the local planner runs, so the rows are re-read
        there too."""
        for p in self.mgr._policies:
            self._wrap_planner(p)
        mgr = self.mgr
        orig = mgr._locomotion.restore
        if not getattr(orig, "_probe", False):
            probe = self

            def restore(pool):
                probe._rows_now = probe.tracked_rows(pool)
                orig(pool)

            restore._probe = True  # type: ignore[attr-defined]
            mgr._locomotion.restore = restore

    def flush(self, t: float, window_ticks: int) -> None:
        for aid, acc in self._acc.items():
            c = max(self._cnt.get(aid, 0), 1)
            pairs = self._pairs[aid]
            top = sorted(pairs.items(), key=lambda kv: -float(np.hypot(*kv[1])))[: self.top_k]
            top_ids = [k for k, _ in top] + [-1] * (self.top_k - len(top))
            top_f = [v / c for _, v in top] + [np.zeros(2)] * (self.top_k - len(top))
            self.rows.append(
                (
                    t,
                    aid,
                    self._cnt.get(aid, 0),
                    *(acc[k] / c for k in VEC_FIELDS[:6]),
                    acc["dp_int"],
                    acc["dp_sep"],
                    acc["e_goal"] / c,
                    self._mode[aid] / max(window_ticks, 1),
                    self._grp[aid] / c,
                    np.array(top_ids),
                    np.array(top_f),
                )
            )
        self._acc.clear()
        self._cnt.clear()
        self._mode.clear()
        self._pairs.clear()
        self._grp.clear()

    def save(self, path: str) -> None:
        if not self.rows:
            np.savez_compressed(path, t=np.zeros(0))
            return
        cols = list(zip(*self.rows))
        out = {"t": np.array(cols[0]), "id": np.array(cols[1], np.int64), "n": np.array(cols[2], np.int64)}
        for k, j in zip(VEC_FIELDS, range(3, 12)):
            out[k] = np.array(cols[j])
        out["mode"] = np.array(cols[12])
        out["grp"] = np.array(cols[13])
        out["top_id"] = np.array(cols[14], np.int64)
        out["top_f"] = np.array(cols[15])
        out["check"] = np.array(self.check)
        np.savez_compressed(path, **out)
