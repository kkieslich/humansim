from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np

from arena_humansim.utils.mesh import TOL, Mesh
from arena_humansim.utils.router import VERIFY_EPS, Router
from arena_humansim.utils.types import Pose2D, Segments

from . import LANE_SIDES, GlobalPlanner, PlanRequest

SNAP_CACHE = 4096
FRAME_INSET = 0.01
#: Planning clearances round up to this step, so agents of similar size share a router
CLEARANCE_STEP = 0.05
#: A lane's corners keep walkers' bodies this far from the walls
LANE_WALL_GAP = 0.05
#: Weidmann's speed-density relation, v = v0 (1 - exp(-GAMMA (1/rho - 1/RHO_MAX))) (Weidmann 1993): its constants
WEIDMANN_GAMMA = 1.913
WEIDMANN_RHO_MAX = 5.4
#: The density around a mesh edge counts the agents within this distance of its middle [m]
CONGESTION_RADIUS = 1.5
#: The most a step through a jam costs, as a multiple of its length
CONGESTION_CAP = 20.0
#: The smallest body radius a lane assumes
MIN_BODY_RADIUS = 0.15
#: A walker pushed off its route's floor gets back along the route its body fits within this distance [m], checked every
#: `ESCAPE_STEP` [m]
ESCAPE_M, ESCAPE_STEP = 3.0, 0.1


def _keys(segs: np.ndarray) -> set[tuple[float, ...]]:
    flip = (segs[:, 0] > segs[:, 2]) | ((segs[:, 0] == segs[:, 2]) & (segs[:, 1] > segs[:, 3]))
    return set(map(tuple, np.where(flip[:, None], segs[:, [2, 3, 0, 1]], segs).tolist()))


class NavMeshPlanner(GlobalPlanner):
    def __init__(
        self,
        replan_distance: float = 1.0,
        inflation_radius: float = 0.38,
        comfort_radius: float = 0.6,
        clearance_margin: float = 0.0,
        lane_side: str = "none",
        lane_offset: float = 0.0,
        congestion_interval: float = 0.0,
    ) -> None:
        super().__init__(replan_distance)
        self._congestion_interval = congestion_interval
        self._slowdown: np.ndarray | None = None
        self._congestion_tick = 0
        self._inflation_radius = inflation_radius
        self._comfort_radius = comfort_radius
        self._margin = clearance_margin
        self._lane_side = self._side(lane_side)
        self._lane_offset = lane_offset

        self._mesh: Mesh | None = None
        self._router: Router | None = None
        self._routers: dict[float, Router] = {}
        self._body_routers: dict[float, Router] = {}
        self._routed: dict[int, tuple[float, float]] = {}
        self._blocked_edges = np.zeros(0, np.int64)
        self._meshed: set[tuple[float, ...]] = set()
        self._snapped: dict[tuple[float, float], tuple[float, float] | None] = {}

    @staticmethod
    def _side(lane_side: str) -> str:
        if lane_side not in LANE_SIDES:
            raise ValueError(f"lane_side must be one of {', '.join(LANE_SIDES)}, not {lane_side!r}")
        return lane_side

    def configure(
        self,
        *,
        inflation_radius: float,
        resolution: float,
        comfort_radius: float,
        clearance_margin: float | None = None,
        lane_side: str | None = None,
        lane_offset: float | None = None,
        congestion_interval: float | None = None,
    ) -> None:
        if congestion_interval is not None and congestion_interval != self._congestion_interval:
            self._congestion_interval = congestion_interval
            self._set_slowdown(None)
        margin = self._margin if clearance_margin is None else clearance_margin
        side = self._lane_side if lane_side is None else self._side(lane_side)
        offset = self._lane_offset if lane_offset is None else lane_offset
        if (margin, side, offset) != (self._margin, self._lane_side, self._lane_offset):
            self._margin, self._lane_side, self._lane_offset = margin, side, offset
            self._routers.clear()
            self._body_routers.clear()
            self._routed.clear()
            self._forget_paths()
        if (inflation_radius, comfort_radius) == (self._inflation_radius, self._comfort_radius):
            return
        self._inflation_radius = inflation_radius
        self._comfort_radius = comfort_radius
        self._router = None
        self.set_walls(self._wall_segments)

    def set_walls(self, segments: Segments) -> None:
        self._forget_paths()
        self._snapped.clear()
        self._wall_segments = list(segments)
        segs = np.array(segments, dtype=np.float64).reshape(-1, 4)
        segs = segs[np.hypot(*(segs[:, 2:] - segs[:, :2]).T) > TOL]
        keys = _keys(segs)
        if self._block(keys):
            return
        self._mesh = None
        self._router = None
        self._routers = {}
        self._body_routers = {}
        self._routed = {}
        self._blocked_edges = np.zeros(0, np.int64)
        self._slowdown = None
        if not len(segs):
            return

        began = time.perf_counter()
        reach = 2 * max(self._comfort_radius, self._inflation_radius * 1.05)
        self._mesh = Mesh.build(segs, reach)
        self._router = Router(self._mesh, self._inflation_radius, self._comfort_radius)
        self._meshed = keys
        build_ms = (time.perf_counter() - began) * 1e3
        self._logger.info(f"Walls meshed: {len(self._mesh.V)} triangles, {len(self._mesh.P)} vertices, {len(segments)} segment(s), inflation={self._inflation_radius}m, comfort={self._comfort_radius}m, {build_ms:.0f} ms")

    def _block(self, keys: set[tuple[float, ...]]) -> bool:
        """Block the added walls on the mesh. False when the change needs a new mesh."""
        if self._mesh is None or self._router is None or not self._meshed <= keys:
            return False
        began = time.perf_counter()
        added = np.array(sorted(keys - self._meshed), dtype=np.float64).reshape(-1, 4)
        edges = self._mesh.free_edges(added[:, :2], added[:, 2:])
        if (edges < 0).any():
            return False
        self._router.set_blocked(edges)
        for router in (*self._routers.values(), *self._body_routers.values()):
            router.set_blocked(edges)
        self._blocked_edges = edges
        self._logger.info(f"Walls blocked on the mesh: {len(edges)} edge(s), {(time.perf_counter() - began) * 1e3:.0f} ms")
        return True

    def clearance_for(self, agent_radius: float) -> float:
        """The clearance an agent of `agent_radius` plans with: its radius plus the clearance margin, at least the
        inflation radius, rounded up to `CLEARANCE_STEP` and at most half the mesh's width cap."""
        wanted = agent_radius + self._margin
        if self._margin <= 0.0 or wanted <= self._inflation_radius + 1e-9:  # margin 0: every agent at the inflation radius, as before
            return self._inflation_radius
        r = math.ceil(wanted / CLEARANCE_STEP - 1e-9) * CLEARANCE_STEP
        if self._mesh is not None:
            r = min(r, self._mesh.reach / 2.0)
        return round(max(r, self._inflation_radius), 6)

    def _router_for(self, clearance: float) -> Router:
        """The router that keeps `clearance` from the walls, built on first use."""
        assert self._router is not None and self._mesh is not None
        if clearance <= self._inflation_radius + 1e-9:
            return self._router
        if clearance not in self._routers:
            router = Router(self._mesh, clearance, self._comfort_radius)
            if len(self._blocked_edges):
                router.set_blocked(self._blocked_edges)
            if self._slowdown is not None:
                router.set_congestion(self._slowdown)
            self._routers[clearance] = router
        return self._routers[clearance]

    def body_clearance(self, agent_radius: float) -> float | None:
        """The clearance of an agent's last resort route: its body radius (at least `MIN_BODY_RADIUS`) rounded up to
        `CLEARANCE_STEP`; None where that is no narrower than the inflation radius, which the agent tries before."""
        r = round(math.ceil(max(agent_radius, MIN_BODY_RADIUS) / CLEARANCE_STEP - 1e-9) * CLEARANCE_STEP, 6)
        return r if r < self._inflation_radius - 1e-9 else None

    def _body_router(self, clearance: float) -> Router:
        """The router that keeps only `clearance`, below the inflation radius, from the walls, built on first use. Its
        routes take plain lengths: they only get a walker back to where its own route starts (`_escape`)."""
        assert self._mesh is not None
        if clearance not in self._body_routers:
            router = Router(self._mesh, clearance, self._comfort_radius)
            if len(self._blocked_edges):
                router.set_blocked(self._blocked_edges)
            self._body_routers[clearance] = router
        return self._body_routers[clearance]

    def compute(self, agents, high_level_commands):  # noqa: ANN001, ANN201 - as GlobalPlanner.compute
        agents = list(agents)
        if self._routed:
            present = {a.state.agent_id for a in agents}
            self._routed = {aid: key for aid, key in self._routed.items() if aid in present}
        if self._router is not None and self._congestion_interval > 0.0:
            self._congestion_tick += 1
            if self._congestion_tick >= max(1, round(self._congestion_interval / self._dt)):
                self._congestion_tick = 0
                self._set_slowdown(self._slowdown_at(np.array([(a.state.pose.x, a.state.pose.y) for a in agents], dtype=np.float64).reshape(-1, 2)))
        return super().compute(agents, high_level_commands)

    def _slowdown_at(self, positions: np.ndarray) -> np.ndarray | None:
        """How much slower than at free flow walkers move through each mesh edge, from the density of `positions` within
        `CONGESTION_RADIUS` of its middle by Weidmann's relation (capped at `CONGESTION_CAP`); None without anyone."""
        assert self._mesh is not None
        if not len(positions):
            return None
        from scipy.spatial import cKDTree

        counts = cKDTree(positions).query_ball_point(self._mesh.e_mid, CONGESTION_RADIUS, return_length=True)
        rho = np.asarray(counts, dtype=np.float64) / (math.pi * CONGESTION_RADIUS**2)
        slow = np.ones(len(rho))
        busy = rho > 0.0
        inv = 1.0 / rho[busy] - 1.0 / WEIDMANN_RHO_MAX
        share = np.where(inv > 0.0, 1.0 - np.exp(-WEIDMANN_GAMMA * np.maximum(inv, 0.0)), 0.0)
        slow[busy] = np.minimum(1.0 / np.maximum(share, 1.0 / CONGESTION_CAP), CONGESTION_CAP)
        return slow

    def _set_slowdown(self, slowdown: np.ndarray | None) -> None:
        self._slowdown = slowdown
        for router in (self._router, *self._routers.values()):
            if router is not None:
                router.set_congestion(slowdown)

    def _lanes(self, router: Router, polyline: np.ndarray) -> np.ndarray:
        """`polyline` with its inner corners moved `lane_offset` to the lane side, so walkers in opposite directions keep
        to opposite sides of a passage instead of meeting head-on on one line. A corner moves as far as its walkers'
        bodies keep `LANE_WALL_GAP` from the walls (their clearance without the margin): the full offset, half of it,
        or not at all. The ends stay. A body-radius route (`_escape`) keeps no lane."""
        if self._lane_side == "none" or self._lane_offset <= 0.0 or len(polyline) < 3 or router.r < self._inflation_radius - 1e-9:
            return polyline
        body = max(router.r - self._margin, MIN_BODY_RADIUS) + LANE_WALL_GAP
        sign = 1.0 if self._lane_side == "right" else -1.0
        d = np.diff(polyline, axis=0)
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
        normals = sign * np.column_stack([d[:, 1], -d[:, 0]])
        n = normals[:-1] + normals[1:]
        length = np.hypot(n[:, 0], n[:, 1])
        n = np.where(length[:, None] > 1e-6, n / np.maximum(length, 1e-9)[:, None], normals[:-1])
        inner = polyline[1:-1]
        moved = inner.copy()
        settled = np.zeros(len(inner), dtype=bool)
        for share in (1.0, 0.5):
            cand = inner + n * (self._lane_offset * share)
            fits = ~settled & (router.clearance(cand) >= body)
            moved[fits] = cand[fits]
            settled |= fits
        if not settled.any():
            return polyline
        out = polyline.copy()
        out[1:-1] = moved
        if router._clear_many([out])[0] >= body - VERIFY_EPS:
            return out
        out[1:-1] = inner + (moved - inner) * 0.5
        return out if router._clear_many([out])[0] >= body - VERIFY_EPS else polyline

    def _has_map(self) -> bool:
        return self._router is not None

    def snap_terminal(self, pose: Pose2D) -> Pose2D:
        if self._router is None:
            return pose
        key = (pose.x, pose.y)
        if key not in self._snapped:
            if len(self._snapped) >= SNAP_CACHE:
                self._snapped.clear()
            p = np.array(key, dtype=np.float64)
            q = self._router.snap(p)
            self._snapped[key] = None if q is None or q is p else (float(q[0]), float(q[1]))
        hit = self._snapped[key]
        return pose if hit is None else Pose2D(x=hit[0], y=hit[1], theta=pose.theta)

    def snap_terminals(self, poses: Sequence[Pose2D]) -> list[Pose2D]:
        if self._router is not None:
            fresh = [key for key in {(p.x, p.y) for p in poses} if key not in self._snapped]
            if fresh:
                if len(self._snapped) + len(fresh) >= SNAP_CACHE:
                    self._snapped.clear()
                clear = self._router.clearance(np.array(fresh, dtype=np.float64)) >= self._router.r
                self._snapped.update((key, None) for key, ok in zip(fresh, clear.tolist(), strict=True) if ok)
        return [self.snap_terminal(p) for p in poses]

    def _sees(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        assert self._router is not None
        return self._router.line_of_sight(starts, ends)

    def _nearest_reachable(self, start: Pose2D, target: Pose2D) -> Pose2D | None:
        assert self._router is not None
        s = self._router.snap(np.array([start.x, start.y], dtype=np.float64))
        if s is None:
            return None
        q = self._router.nearest_reachable(s, np.array([target.x, target.y], dtype=np.float64))
        if q is None:
            return None
        return Pose2D(x=float(q[0]), y=float(q[1]), theta=target.theta)

    def components(self, poses: Sequence[Pose2D]) -> list[frozenset[int]]:
        if self._router is None or not poses:
            return super().components(poses)
        return self._router.components(self._inside(np.array([(p.x, p.y) for p in poses], dtype=np.float64)))

    def route_costs(self, starts: Sequence[Pose2D], goal: Pose2D) -> np.ndarray | None:
        if self._router is None:
            return None
        if not starts:
            return np.zeros(0)
        pts = self._inside(np.array([(p.x, p.y) for p in starts], dtype=np.float64))
        return self._router.costs(pts, self._inside(np.array([goal.x, goal.y], dtype=np.float64)))

    def _inside(self, p: np.ndarray) -> np.ndarray:
        """p moved onto the meshed frame when it lies beyond it."""
        assert self._mesh is not None
        frame = self._mesh.P[-self._mesh.n_frame :]
        return np.clip(p, frame.min(0) + FRAME_INSET, frame.max(0) - FRAME_INSET)

    def _rejoin(self, router: Router, start: np.ndarray, goal: np.ndarray, polyline: np.ndarray) -> np.ndarray:
        """Route between the framed points extended to the real end points."""
        if (polyline[0] != start).any():
            seen = router.line_of_sight(start[None], polyline[1][None])[0]
            polyline = np.vstack([start, polyline[1 if seen else 0 :]])
        if (polyline[-1] != goal).any():
            seen = router.line_of_sight(polyline[-2][None], goal[None])[0]
            polyline = np.vstack([polyline[: -1 if seen else len(polyline)], goal])
        return polyline

    def _plan(self, requests: list[PlanRequest]) -> dict[int, list[Pose2D] | None]:
        assert self._router is not None
        plans: dict[int, list[Pose2D] | None] = {agent_id: None for agent_id, _, _ in requests}
        by_goal: dict[tuple[float, float, float], list[tuple[int, Pose2D, Pose2D]]] = {}
        for agent_id, agent_pos, target in requests:
            clearance = self.clearance_for(self._radius_of(agent_id))
            by_goal.setdefault((target.x, target.y, clearance), []).append((agent_id, agent_pos, target))
        for (_, _, clearance), group in by_goal.items():
            router = self._router_for(clearance)
            found = self._plan_group(router, group)
            if router is not self._router:
                # no route keeps the margin (a door narrower than the agent's comfort): the bare inflation radius
                missing = [entry for entry in group if found.get(entry[0]) is None]
                if missing:
                    found.update(self._plan_group(self._router, missing))
            for agent_id, _, target in group:
                if found.get(agent_id) is not None:
                    self._routed[agent_id] = (round(target.x, 3), round(target.y, 3))
            for agent_id, agent_pos, target in group:
                if found.get(agent_id) is None and (way := self._escape(agent_id, agent_pos, target, clearance)) is not None:
                    found[agent_id] = way
            plans.update(found)
        return plans

    def _escape(self, agent_id: int, agent_pos: Pose2D, target: Pose2D, clearance: float) -> list[Pose2D] | None:
        """The way back for an agent the crowd pushed off the floor its route keeps, into a gap narrower than the
        inflation radius (a niche in a wall, between a pillar and a wall), where no route at its `clearance` or the
        inflation radius starts: along the route its body fits (`body_clearance`) to the first point within `ESCAPE_M`
        that keeps the inflation radius and reaches the target from there, then on by its own route. Without it, the
        agent walks straight at its target into the wall, or stands in the niche, for good. Only for an agent that had a
        route to this target before, so a target behind a door narrower than the inflation radius stays out of reach.
        None where there is no such way."""
        assert self._router is not None
        body = self.body_clearance(self._radius_of(agent_id))
        if body is None or self._routed.get(agent_id) != (round(target.x, 3), round(target.y, 3)):
            return None
        snapped = self.snap_terminal(target)
        goal = np.array([snapped.x, snapped.y], dtype=np.float64)
        reach = self._router.components(goal[None])[0]
        router = self._body_router(body)
        start = router.snap(np.array([agent_pos.x, agent_pos.y], dtype=np.float64))
        end = router.snap(goal)
        route = None if start is None or end is None or not reach else router.plan(start, end, try_line_of_sight=False)
        if route is None:
            return None
        line = route[0]
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
        at = np.arange(ESCAPE_STEP, min(ESCAPE_M, along[-1]) + 1e-9, ESCAPE_STEP)
        pts = np.column_stack([np.interp(at, along, line[:, 0]), np.interp(at, along, line[:, 1])])
        if not len(pts):
            return None
        keeps = self._router.clearance(pts) >= self._router.r - VERIFY_EPS
        back = next((i for i, comp in enumerate(self._router.components(pts)) if keeps[i] and comp & reach), None)
        if back is None:
            return None
        out = Pose2D(x=float(pts[back, 0]), y=float(pts[back, 1]))
        entry = [(agent_id, out, target)]
        rest = self._plan_group(self._router_for(clearance), entry).get(agent_id) or self._plan_group(self._router, entry).get(agent_id)
        if rest is None:
            return None
        head = [Pose2D(x=float(x), y=float(y)) for x, y in line[1 : int(np.searchsorted(along, at[back]))]]
        return [Pose2D(x=agent_pos.x, y=agent_pos.y), *head, *rest]

    def _plan_group(self, router: Router, group: list[tuple[int, Pose2D, Pose2D]]) -> dict[int, list[Pose2D] | None]:
        """Routes for agents with one target, by `router`'s clearance, their inner corners on the lane side."""
        plans: dict[int, list[Pose2D] | None] = {}
        entries = []
        for agent_id, agent_pos, target in group:
            start = router.snap(np.array([agent_pos.x, agent_pos.y], dtype=np.float64))
            if start is not None:
                entries.append((agent_id, agent_pos, target, start))
        if not entries:
            return plans
        snapped = self.snap_terminal(entries[0][2])
        goal = np.array([snapped.x, snapped.y])
        if router is not self._router:
            # a target closer to a wall than this clearance: route to the nearest point that keeps it, then step on
            near = router.snap(goal)
            goal = goal if near is None else near
        starts = np.array([start for _, _, _, start in entries])
        framed = self._inside(starts)
        beyond = (framed != starts).any(axis=1) | (self._inside(goal) != goal).any()
        direct = beyond & router.line_of_sight(starts, np.repeat(goal[None], len(starts), 0))
        routes = router.plan_many(framed[~direct], self._inside(goal))
        polylines: list[np.ndarray | None] = [None] * len(entries)
        for i, route in zip(np.nonzero(~direct)[0], routes, strict=True):
            if route is not None:
                polylines[i] = self._lanes(router, self._rejoin(router, starts[i], goal, route[0]) if beyond[i] else route[0])
        for i in np.nonzero(direct)[0]:
            polylines[i] = np.array([starts[i], goal])

        for (agent_id, agent_pos, target, _), polyline in zip(entries, polylines, strict=True):
            if polyline is None:
                continue
            waypoints = [Pose2D(x=float(x), y=float(y)) for x, y in polyline]
            if polyline[0, 0] == agent_pos.x and polyline[0, 1] == agent_pos.y:
                waypoints.pop(0)
            waypoints.insert(0, Pose2D(x=agent_pos.x, y=agent_pos.y))
            waypoints[-1] = Pose2D(x=snapped.x, y=snapped.y, theta=target.theta)
            plans[agent_id] = waypoints
        return plans
