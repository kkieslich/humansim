from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any, TypedDict

import attrs
import numpy as np

from arena_humansim.core.agents.types import Passage, WallTrailing
from arena_humansim.utils.mesh import TOL, Mesh
from arena_humansim.utils.router import VERIFY_EPS, Router
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segments
from arena_humansim.utils.wall_grid import WallGrid, cast_rays

from . import LANE_SIDES, STALL_M, GlobalPlanner, PassageCommand, PlanRequest
from .wall_following import PASSING_GAP, STEP, Shaped, approach, follow_walls, leave, sample, thin

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
#: `path_cost` weighs a path by the congestion at points this far apart along it [m]
PATH_COST_STEP = 0.5
#: The smallest body radius a lane assumes
MIN_BODY_RADIUS = 0.15
#: A walker pushed off its route's floor gets back along the route its body fits within this distance [m], checked every
#: `ESCAPE_STEP` [m]
ESCAPE_M, ESCAPE_STEP = 3.0, 0.1
#: A walker trailing a wall nears and leaves it by this much sideways per metre along its route (45 degrees)
TRAIL_RAMP = 1.0
#: A walker with a lane steers at the point furthest ahead on its route, up to `LOOK_AHEAD` [m], that it can head for
#: without leaving the route by more than `LOOK_CUT` [m], and at one at least `LOOK_MIN` [m] away from itself
LOOK_AHEAD, LOOK_STEP, LOOK_CUT, LOOK_MIN = 1.5, 0.25, 0.1, 0.5
#: A lane lies between walls that both run along the route within this much sideways per metre (20 degrees)
LANE_ALONG = 0.36
#: A walker with a lane or turns to take is in line with the corridor over this length before it enters and after it
#: leaves [m], so that the two directions' ways in and out run side by side; within it it claims a narrow stretch
LINE_UP = 4.5
#: A walker that takes turns is past its hold line, and goes on, once it is this far beyond it [m]
HOLD_SLACK = 0.5
#: The narrow stretch a route starts in is measured back from the start this far at most [m]
BACK_REACH = 30.0
#: A walker that queues looks this far ahead along its route for another in its way [m], at points this far apart
QUEUE_LOOK, QUEUE_STEP = 6.0, 0.25
#: Another is in a queueing walker's way where its body comes within this of the walker's body as it passes [m]
QUEUE_MARGIN = 0.1
#: Two walkers in each other's way are level where neither is this much further ahead of the other [m]
QUEUE_LEVEL = 0.1
#: Every walker without a `passage` block takes a stretch of its route too narrow to pass an oncoming walker by turns:
#: two pass each other where the stretch is at least their two body widths and `PASS_GAP` wider [m]. The value is what
#: the social-force walkers need: adults (0.5 m bodies) jam head-on in a 1.0 m passage and pass in 1.1 m and 1.15 m ones
#: (verify/passage minimal world); no source beyond that.
PASS_GAP = 0.1
#: A route is measured for stretches where its walker may have to wait for an oncoming walker as wide as this [m]: the
#: widest body the shipped types allow (adult radius clip_high 0.40 m; a manual wheelchair is 0.65 m wide)
WIDEST_BODY = 0.8
#: A walker waiting for its turn stops with its body middle this far before the narrow stretch [m]
TURN_HOLD = 1.0
#: Routes are measured for narrow stretches at points this far apart [m]; a stretch's width at a point is its narrowest
#: chord through the point within `TURN_TILT` of square to the route, measured up to `TURN_FAR` from it
TURN_STEP, TURN_FAR = 0.25, 2.0
TURN_TILT = (0.0, math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3)
#: A walker that has moved less than `STALL_M` in this long while not held for its turn no longer keeps those at the
#: other end of its narrow stretch waiting (one that stands in or before the stretch for another reason) [s]
TURN_STUCK_S = 20.0


class _Fit(TypedDict):
    """How a walker's body fits a shaped route (`wall_following`'s `body`, `axial` and `ends`)."""

    body: float
    axial: float
    ends: float


@attrs.frozen
class Narrow:
    """A stretch of a route narrower than its walker's two-way width: its start and end along the route [m], where the
    walker holds before it (`line`, along the route), and points every `STEP` along it with the route's direction at
    each."""

    start: float
    end: float
    line: float
    points: np.ndarray
    tangents: np.ndarray


@attrs.define
class Way:
    """What a walker with a lane or turns to take needs of its route: the cached waypoint list it belongs to, its points,
    their distance along it, where it keeps a lane, and its narrow stretches in order."""

    waypoints: list[Pose2D]
    points: np.ndarray
    at: np.ndarray
    lane: np.ndarray
    narrows: list[Narrow]

    def point(self, s: float) -> tuple[float, float]:
        """The route's point `s` metres along it, on its first leg's line before the start."""
        if s < 0.0:
            u = self.direction(0.0)
            return float(self.points[0, 0] + s * u[0]), float(self.points[0, 1] + s * u[1])
        return float(np.interp(s, self.at, self.points[:, 0])), float(np.interp(s, self.at, self.points[:, 1]))

    def look(self, s: float, x: float, y: float) -> tuple[float, float]:
        """The point to steer at from `s` metres along the route, standing at (x, y): the furthest one up to `LOOK_AHEAD`
        on whose straight line from the route's point at `s` stays within `LOOK_CUT` of the route (so corners are not
        cut), but one at least `LOOK_MIN` away where the route goes on that far."""
        end = float(self.at[-1])
        sx, sy = self.point(s)
        best = None
        for k in range(1, int(round(LOOK_AHEAD / LOOK_STEP)) + 1):
            e = min(s + k * LOOK_STEP, end)
            ex, ey = self.point(e)
            inner = self.points[(self.at > s) & (self.at < e)]
            if len(inner):
                chord = math.hypot(ex - sx, ey - sy)
                off = np.abs((inner[:, 0] - sx) * (ey - sy) - (inner[:, 1] - sy) * (ex - sx)) / max(chord, 1e-9)
                if float(off.max()) > LOOK_CUT and best is not None and math.hypot(best[0] - x, best[1] - y) >= LOOK_MIN:
                    break
            best = (ex, ey)
            if e >= end:
                break
        assert best is not None
        return best

    def direction(self, s: float) -> tuple[float, float]:
        """The route's unit direction `s` metres along it."""
        i = int(np.clip(np.searchsorted(self.at, s, side="right") - 1, 0, len(self.points) - 2))
        while i < len(self.points) - 2 and self.at[i + 1] - self.at[i] <= 1e-9:
            i += 1
        leg = self.points[i + 1] - self.points[i]
        length = max(float(np.hypot(leg[0], leg[1])), 1e-9)
        return float(leg[0] / length), float(leg[1] / length)


@attrs.define
class TurnWay:
    """A route measured for stretches narrower than its walker needs to pass an oncoming walker (`_turn_way`): the
    cached waypoint list it belongs to, its points, their distance along it, and each narrow stretch with its least
    width."""

    waypoints: list[Pose2D]
    points: np.ndarray
    at: np.ndarray
    narrows: list[tuple[Narrow, float]]


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
        #: set while `paths` plans legs for the stair ride timing: their request ids are leg numbers, not agents
        self._pricing = False
        #: the walkers' positions at the last congestion update, a KD-tree
        self._crowd = None
        #: per agent of the last `compute`: its wall trailing and passage settings, and half its body length
        self._manners: dict[int, tuple[WallTrailing, Passage, float]] = {}
        self._ways: dict[int, Way] = {}
        self._shapes: dict[int, tuple[int, Shaped]] = {}
        self._claims: dict[int, tuple[float, float, int]] = {}
        self._commands: dict[int, PassageCommand] = {}
        self._gives_way: set[tuple[int, int]] = set()
        self._ray_grid: tuple[list, WallGrid] | None = None
        #: narrow-passage turns of walkers without a `passage` block: measured routes, claims (mouth x, y, tick), and
        #: where each walker last moved (x, y, tick) and whether the last `compute` held it
        self._turn_ways: dict[int, TurnWay] = {}
        self._turn_claims: dict[int, tuple[float, float, int]] = {}
        self._turn_moved: dict[int, tuple[float, float, int]] = {}
        self._turn_held: set[int] = set()

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
        **_: Any,  # noqa: ANN401 - as GlobalPlanner.configure
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
        self._manners = {}
        for agent in agents:
            params = getattr(agent, "params", None)
            if params is None:
                continue
            trailing: WallTrailing | None = getattr(params, "wall_trailing", None)
            passage: Passage | None = getattr(params, "passage", None)
            if (trailing is not None and trailing.active) or (passage is not None and passage.active):
                length = float(getattr(getattr(params, "locomotion", None), "footprint_length", 0.0) or 0.0)
                self._manners[agent.state.agent_id] = (trailing if trailing is not None else WallTrailing(), passage if passage is not None else Passage(), max(length / 2.0, float(params.agent_radius)))
        goals = super().compute(agents, high_level_commands)
        self._commands = {}
        if self._manners or self._ways:
            self._drive(agents, goals)
        self._take_turns(agents, high_level_commands, goals)
        return goals

    def _takes_straight_legs(self, agent_id: int) -> bool:
        manner = self._manners.get(agent_id)
        return manner is None or not manner[1].active

    def passage_commands(self) -> dict[int, PassageCommand]:
        """What the last `compute` asks of the locomotion of walkers that keep a lane or take turns in narrow passages."""
        return self._commands

    def _cast(self, origins: np.ndarray, directions: np.ndarray, far: float) -> np.ndarray:
        """The distance to the first wall along each ray, `far` where there is none within it."""
        if self._ray_grid is None or self._ray_grid[0] is not self._wall_segments:
            self._ray_grid = (self._wall_segments, WallGrid(self._wall_segments))
        return cast_rays(self._ray_grid[1], origins, directions, far)

    def _shape(self, agent_id: int, router: Router, polyline: np.ndarray) -> np.ndarray:
        """The route of an agent that trails walls (`wall_trailing`) or keeps a lane in corridors (`passage`): laid
        `offset` from the wall it follows, on the lane side where it has the choice. A lane is entered and left in line
        with it (`approach`, `leave`), and a stretch narrower than the agent's `two_way_width` gets an approach beside
        its mouth where the agent waits. Any other agent's route as it is."""
        self._shapes.pop(agent_id, None)
        manner = self._manners.get(agent_id)
        if manner is None or len(polyline) < 2 or router.r < self._inflation_radius - 1e-9:
            return polyline
        trailing, passage, half = manner
        radius = self._radius_of(agent_id)
        body = max(radius, MIN_BODY_RADIUS) + LANE_WALL_GAP
        prefer = {"right": -1.0, "left": 1.0}.get(self._lane_side, 0.0)
        if trailing.active:
            shaped = follow_walls(polyline, self._cast, router.clearance, body=body, offset=trailing.offset, reach=trailing.reach, prefer=prefer, either=True, ramp=TRAIL_RAMP, bridge=trailing.reach)
            narrow = 0.0
        else:
            lanes = passage.lanes and prefer != 0.0
            fit: _Fit = {"body": body, "axial": max(half - radius, 0.0), "ends": radius}
            lane = passage.lane_width / 2.0 if lanes else 0.0
            shaped = follow_walls(polyline, self._cast, router.clearance, offset=lane, reach=passage.corridor_width, prefer=prefer, either=False, corridor=passage.corridor_width, span=passage.two_way_width, early=False, along=LANE_ALONG, bridge=passage.corridor_width, **fit)
            narrow = passage.two_way_width if passage.takes_turns else 0.0
            if lanes:
                shaped = leave(shaped, shaped.follows, router.clearance, length=LINE_UP, **fit)
                wide = shaped.follows.copy()
                edges = np.flatnonzero(np.diff(np.concatenate([[False], wide, [False]]).astype(np.int8)))
                for a, b in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
                    if (shaped.width[a : min(a + 2, b)] < narrow).any():
                        wide[a:b] = False
                shaped = approach(shaped, wide, router.clearance, length=LINE_UP, prefer=prefer, **fit)
            if passage.takes_turns:
                beside = lane + radius + PASSING_GAP if lanes else 0.0
                shaped = approach(shaped, shaped.width < narrow, router.clearance, length=LINE_UP, lateral=beside, prefer=prefer, waits=True, **fit)
        shaped = thin(shaped, narrow)
        self._shapes[agent_id] = (len(shaped.points), shaped)
        return shaped.points

    def _from_here(self, router: Router, agent_id: int, agent_pos: Pose2D, polyline: np.ndarray) -> np.ndarray:
        """`polyline` starting where the agent stands instead of where the router moved its start off a wall: a walker in
        a lane stands closer to the wall than its route's clearance, and its new route must not begin with a step
        sideways. Kept as it is where the straight leg from the agent to the route's second point is not clear."""
        here = np.array([agent_pos.x, agent_pos.y], dtype=np.float64)
        if len(polyline) < 2 or (polyline[0] == here).all():
            return polyline
        body = max(self._radius_of(agent_id), MIN_BODY_RADIUS)
        direct = np.vstack([here, polyline[1:]])
        return direct if router._clear_many([direct[:2]])[0] >= body - VERIFY_EPS else polyline

    def _keep_way(self, agent_id: int, waypoints: list[Pose2D], extra: int) -> None:
        """Note the lane stretches and narrow stretches of a route just planned for an agent that takes passages by rule;
        `extra` is how many waypoints the plan put before the shaped route's own."""
        self._ways.pop(agent_id, None)
        manner = self._manners.get(agent_id)
        hit = self._shapes.pop(agent_id, None)
        if manner is None or hit is None or not manner[1].active:
            return
        passage, shaped = manner[1], hit[1]
        if hit[0] + extra != len(waypoints):
            return
        points = np.array([(w.x, w.y) for w in waypoints], dtype=np.float64)
        lane = np.concatenate([np.zeros(extra, dtype=bool), shaped.follows])
        width = np.concatenate([np.full(extra, np.inf), shaped.width])
        waits = np.concatenate([np.zeros(extra, dtype=bool), shaped.waits])
        at = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
        narrows: list[Narrow] = []
        if passage.takes_turns:
            tight = width < passage.two_way_width
            edges = np.flatnonzero(np.diff(np.concatenate([[False], tight, [False]]).astype(np.int8)))
            done = 0
            for a, b in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
                start = float(at[a]) if a else -self._narrow_behind(points, passage.two_way_width)
                line = start - passage.hold_back
                stand = np.flatnonzero(waits[done:a])
                if len(stand):
                    line = min(line, float(at[done + stand[-1]]))
                done = b
                if b - a < 2:
                    pts, tangents = points[a : a + 1], np.array([[1.0, 0.0]])
                    if a + 1 < len(points):
                        leg = points[a + 1] - points[a]
                        tangents = (leg / max(float(np.hypot(*leg)), 1e-9))[None]
                else:
                    pts, tangents = sample(points[a:b])
                narrows.append(Narrow(start, float(at[b - 1]), line, pts, tangents))
        self._ways[agent_id] = Way(waypoints, points, at, lane, narrows)

    def _narrow_behind(self, points: np.ndarray, narrow: float) -> float:
        """How far the stretch narrower than `narrow` that a route starts in reaches back behind the route's start,
        along its first leg (`BACK_REACH` at most): where a walker that has to roll back out of it finds its mouth."""
        leg = points[1] - points[0]
        length = float(np.hypot(leg[0], leg[1]))
        if length <= 1e-9:
            return 0.0
        u = leg / length
        left = np.array([-u[1], u[0]])
        xs = np.arange(STEP, BACK_REACH + 1e-9, STEP)
        spots = points[0] - xs[:, None] * u
        width = self._cast(spots, np.repeat(left[None], len(xs), 0), narrow) + self._cast(spots, np.repeat(-left[None], len(xs), 0), narrow)
        blocked = self._cast(points[:1], -u[None], BACK_REACH)[0]
        wide = np.flatnonzero((width >= narrow) | (xs >= blocked))
        return float(xs[wide[0]] - STEP) if len(wide) else BACK_REACH

    def _drive(self, agents: list, goals: dict[int, Pose2D]) -> None:
        """Steer the walkers that keep lanes or take turns (`passage`) along their routes, and say which of them wait.

        Each steers at a point up to `LOOK_AHEAD` on along its route (`Way.look`). Where its route keeps a lane, its heading follows that point (`PassageCommand.lane`). Before a
        stretch narrower than its `two_way_width` it claims the stretch once within `LINE_UP` of it,
        and holds `hold_back` before it (beside the mouth where its approach leads there) while a walker of this kind is in the stretch the other way, or one waiting at the other
        end claimed it before any of those waiting at its own end. Two that met inside: the one that came less far rolls
        back behind its hold line. Behind another of its kind it queues (`_queue`)."""
        present = {a.state.agent_id: a for a in agents}
        seen: dict[int, tuple[Way, float, Narrow | None, str, float]] = {}
        for aid in list(self._ways):
            way, manner, entry = self._ways[aid], self._manners.get(aid), self._path_cache.get(aid)
            if aid not in present or manner is None or entry is None or entry[1] is not way.waypoints:
                del self._ways[aid]
                continue
            passage, half = manner[1], manner[2]
            idx = min(entry[3], len(way.points) - 2)
            pose = present[aid].state.pose
            leg = way.points[idx + 1] - way.points[idx]
            length = float(np.hypot(leg[0], leg[1]))
            along = 0.0 if length <= 1e-9 else min(max(((pose.x - way.points[idx, 0]) * leg[0] + (pose.y - way.points[idx, 1]) * leg[1]) / length, 0.0), length)
            s = float(way.at[idx]) + along
            if aid in goals:
                x, y = way.look(s, pose.x, pose.y)
                goals[aid] = Pose2D(x=x, y=y, theta=goals[aid].theta)
            narrow = next((n for n in way.narrows if n.end + half > s), None) if passage.takes_turns else None
            phase, line = "far", 0.0
            if narrow is not None:
                line = narrow.line
                phase = "inside" if s > line + HOLD_SLACK else "near" if s >= narrow.start - LINE_UP else "far"
            if phase == "far":
                self._claims.pop(aid, None)
            else:
                assert narrow is not None
                ex, ey = way.point(narrow.start)
                claim = self._claims.get(aid)
                if claim is None or math.hypot(claim[0] - ex, claim[1] - ey) > 1.0:
                    self._claims[aid] = (ex, ey, self._tick)
            seen[aid] = (way, s, narrow, phase, line)
            self._commands[aid] = PassageCommand(lane=bool(way.lane[idx] and way.lane[idx + 1]))
        self._claims = {aid: claim for aid, claim in self._claims.items() if aid in seen}

        busy = [aid for aid, (_, _, _, phase, _) in seen.items() if phase != "far"]
        against: dict[int, list[int]] = {aid: [] for aid in busy}
        beside: dict[int, list[int]] = {aid: [aid] for aid in busy}
        for i, a in enumerate(busy):
            for b in busy[i + 1 :]:
                reach = max(self._manners[a][1].two_way_width, self._manners[b][1].two_way_width)
                met = self._meet(seen[a][2], seen[b][2], reach)
                for one, other in ((a, b), (b, a)):
                    if met < 0:
                        against[one].append(other)
                    elif met > 0:
                        beside[one].append(other)
        back = set()
        for aid in busy:
            _, s, _, phase, line = seen[aid]
            if phase != "inside":
                continue
            for other in against[aid]:
                if seen[other][3] == "inside" and (s - line, -aid) < (seen[other][1] - seen[other][4], -other):
                    back.add(aid)
        limits: dict[int, float] = {}
        for aid in busy:
            _, _, _, phase, line = seen[aid]
            hold = aid in back
            if phase == "near":
                ours = min((self._claims[o][2], o) for o in beside[aid] if seen[o][3] == "near")
                hold = any((seen[o][3] == "inside" and o not in back) or (seen[o][3] == "near" and (self._claims[o][2], o) < ours) for o in against[aid])
            if hold:
                limits[aid] = line
        for aid, arc in self._queue(present, seen).items():
            limits[aid] = min(limits.get(aid, math.inf), arc)
        for aid, arc in limits.items():
            way = seen[aid][0]
            x, y = way.point(arc)
            ux, uy = way.direction(max(arc - 1e-6, 0.0))
            self._commands[aid] = PassageCommand(lane=self._commands[aid].lane, hold=(x, y, ux, uy), back=aid in back)

    def _turn_way(self, waypoints: list[Pose2D], limit: float) -> TurnWay:
        """`waypoints` measured for stretches narrower than `limit`: at points `TURN_STEP` apart, the narrowest chord
        through each within `TURN_TILT` of square to the route (a door crossed at a slant is no wider than its width)."""
        points = np.array([(w.x, w.y) for w in waypoints], dtype=np.float64).reshape(-1, 2)
        at = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
        if len(points) < 2 or at[-1] <= 1e-9:
            return TurnWay(waypoints, points, at, [])
        spots, tangents = sample(points, TURN_STEP)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(spots, axis=0).T))])
        normal = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        width = self._cast(spots, normal, TURN_FAR) + self._cast(spots, -normal, TURN_FAR)
        slant = np.flatnonzero(width < 2.0 * limit)
        for tilt in TURN_TILT[1:]:
            if not len(slant):
                break
            c, si = math.cos(tilt), math.sin(tilt)
            d = np.column_stack([c * normal[slant, 0] - si * normal[slant, 1], si * normal[slant, 0] + c * normal[slant, 1]])
            width[slant] = np.minimum(width[slant], self._cast(spots[slant], d, TURN_FAR) + self._cast(spots[slant], -d, TURN_FAR))
        tight = width < limit
        edges = np.flatnonzero(np.diff(np.concatenate([[False], tight, [False]]).astype(np.int8)))
        narrows = []
        for a, b in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
            start = float(s[a]) if a else -self._narrow_behind(points, limit)
            narrows.append((Narrow(start, float(s[b - 1]), start - TURN_HOLD, spots[a:b], tangents[a:b]), float(width[a:b].min())))
        return TurnWay(waypoints, points, at, narrows)

    @staticmethod
    def _same_tail(old: list[Pose2D], new: list[Pose2D]) -> bool:
        """Whether `new` runs on along `old` from its second point to the end, as the straight legs a walker takes to its
        target are rebuilt from where it stands at every `compute`."""
        k = len(new) - 1
        if k < 1 or len(old) < k:
            return False
        return all(a.x == b.x and a.y == b.y for a, b in zip(old[len(old) - k :], new[1:], strict=True))

    @staticmethod
    def _along_way(way: TurnWay, x: float, y: float) -> float:
        """How far along `way` the point closest to (x, y) lies [m]."""
        a, b = way.points[:-1], way.points[1:]
        leg = b - a
        length2 = np.maximum((leg**2).sum(axis=1), 1e-12)
        u = np.clip(((x - a[:, 0]) * leg[:, 0] + (y - a[:, 1]) * leg[:, 1]) / length2, 0.0, 1.0)
        d = np.hypot(a[:, 0] + u * leg[:, 0] - x, a[:, 1] + u * leg[:, 1] - y)
        k = int(np.argmin(d))
        return float(way.at[k] + u[k] * math.sqrt(length2[k]))

    def _take_turns(self, agents: list, high_level_commands: dict, goals: dict[int, Pose2D]) -> None:
        """Narrow passages one direction at a time for every navigating walker without a `passage` block.

        A stretch of a walker's route narrower than its body, `WIDEST_BODY` and `PASS_GAP` is one it may have to take by
        turns. Two walkers whose such stretches are one passage taken in opposite directions (`_meet`), and
        narrower there than their two bodies and `PASS_GAP`, do not both go in: within `LINE_UP` of the stretch a walker
        holds `TURN_HOLD` before it while one coming the other way is inside, or while one waiting at the other end
        claimed the stretch before every walker waiting at its own end. Two that met inside: the one that came less far
        rolls back behind its hold line. A walker that stood `TURN_STUCK_S` without being held keeps no one waiting.
        Without this rule two streams that meet in a passage narrower than two bodies press against each other, and the
        walkers behind keep pressing in from both ends, so neither front can give way."""
        navigating = {aid for aid, cmd in high_level_commands.items() if isinstance(cmd, HighLevelCommand) and cmd.type == CommandType.NAVIGATE}
        present = {a.state.agent_id: a for a in agents}
        seen: dict[int, tuple[float, float, Narrow, float, str]] = {}
        for aid, agent in present.items():
            manner = self._manners.get(aid)
            entry = self._path_cache.get(aid)
            if aid not in navigating or entry is None or (manner is not None and manner[1] is not None and manner[1].takes_turns):
                self._turn_ways.pop(aid, None)
                continue
            radius = max(self._radius_of(aid), MIN_BODY_RADIUS)
            way = self._turn_ways.get(aid)
            if way is None or (way.waypoints is not entry[1] and not self._same_tail(way.waypoints, entry[1])):
                way = self._turn_way(entry[1], 2.0 * radius + WIDEST_BODY + PASS_GAP)
            self._turn_ways[aid] = way
            if not way.narrows:
                continue
            pose = agent.state.pose
            s = self._along_way(way, pose.x, pose.y)
            hit = next((n for n in way.narrows if n[0].end + radius > s), None)
            if hit is None:
                continue
            narrow, width = hit
            if s > narrow.line + HOLD_SLACK:
                phase = "inside"
            elif s >= narrow.start - LINE_UP:
                phase = "near"
            else:
                continue
            seen[aid] = (s, radius, narrow, width, phase)
        tick = self._tick
        for aid in [a for a in self._turn_moved if a not in present]:
            del self._turn_moved[aid]
        stuck: set[int] = set()
        for aid in seen:
            pose = present[aid].state.pose
            moved = self._turn_moved.get(aid)
            if moved is None or math.hypot(pose.x - moved[0], pose.y - moved[1]) > STALL_M or aid in self._turn_held:
                self._turn_moved[aid] = (pose.x, pose.y, tick)
            elif (tick - moved[2]) * self._dt >= TURN_STUCK_S:
                stuck.add(aid)
        claims: dict[int, tuple[float, float, int]] = {}
        for aid, (_, _, narrow, _, phase) in seen.items():
            mouth = narrow.points[0]
            claim = self._turn_claims.get(aid)
            if claim is None or math.hypot(claim[0] - mouth[0], claim[1] - mouth[1]) > 1.0:
                claim = (float(mouth[0]), float(mouth[1]), tick)
            claims[aid] = claim
        self._turn_claims = claims
        busy = list(seen)
        against: dict[int, list[int]] = {aid: [] for aid in busy}
        beside: dict[int, list[int]] = {aid: [aid] for aid in busy}
        for i, a in enumerate(busy):
            for b in busy[i + 1 :]:
                sa, sb = seen[a], seen[b]
                width = min(sa[3], sb[3])
                met = self._meet(sa[2], sb[2], max(width, 2.0 * (sa[1] + sb[1])))
                if met < 0 and width < 2.0 * (sa[1] + sb[1]) + PASS_GAP:
                    against[a].append(b)
                    against[b].append(a)
                elif met > 0:
                    beside[a].append(b)
                    beside[b].append(a)
        back: set[int] = set()
        for aid in busy:
            s, _, narrow, _, phase = seen[aid]
            if phase != "inside":
                continue
            for other in against[aid]:
                so = seen[other]
                if so[4] == "inside" and other not in stuck and (s - narrow.line, -aid) < (so[0] - so[2].line, -other):
                    back.add(aid)
        held: set[int] = set()
        for aid in busy:
            s, _, narrow, _, phase = seen[aid]
            hold = aid in back
            if phase == "near" and against[aid]:
                ours = min((claims[o][2], o) for o in beside[aid] if seen[o][4] == "near")
                hold = any(o not in stuck and ((seen[o][4] == "inside" and o not in back) or (seen[o][4] == "near" and (claims[o][2], o) < ours)) for o in against[aid])
            if hold:
                held.add(aid)
        # the waiting walkers of one end stand in line beside the mouth, nearest first, so that those coming out pass
        order = sorted((a for a in held if a not in back), key=lambda a: (round(claims[a][0], 0), round(claims[a][1], 0), seen[a][2].start - seen[a][0]))
        place: dict[int, int] = {}
        for a in order:
            mouth = (round(claims[a][0], 0), round(claims[a][1], 0))
            place[a] = sum(1 for b in place if (round(claims[b][0], 0), round(claims[b][1], 0)) == mouth)
        for aid in held:
            s, radius, narrow, width, _ = seen[aid]
            way = self._turn_ways[aid]
            if aid in back:
                x, y, ux, uy = self._way_at(way, narrow.line)
            else:
                x, y, ux, uy = self._wait_spot(way, narrow, width, radius, place[aid])
                if aid in goals:
                    goals[aid] = Pose2D(x=x, y=y, theta=goals[aid].theta)
            self._commands[aid] = PassageCommand(hold=(x, y, ux, uy), back=aid in back)
        self._turn_held = held

    @staticmethod
    def _way_at(way: TurnWay, arc: float) -> tuple[float, float, float, float]:
        """The point `arc` metres along `way` (on its first leg's line behind the start where `arc` < 0) and the route's
        unit direction there."""
        k = int(np.clip(np.searchsorted(way.at, max(arc, 0.0), side="right") - 1, 0, len(way.points) - 2))
        leg = way.points[k + 1] - way.points[k]
        length = max(float(np.hypot(leg[0], leg[1])), 1e-9)
        ux, uy = float(leg[0] / length), float(leg[1] / length)
        if arc < 0.0:
            return float(way.points[0, 0] + arc * ux), float(way.points[0, 1] + arc * uy), ux, uy
        return float(np.interp(arc, way.at, way.points[:, 0])), float(np.interp(arc, way.at, way.points[:, 1])), ux, uy

    def _wait_spot(self, way: TurnWay, narrow: Narrow, width: float, radius: float, place: int) -> tuple[float, float, float, float]:
        """Where a walker waits for its turn, `place` walkers from the mouth: `TURN_HOLD` and one body plus `PASS_GAP`
        per walker ahead of it before the stretch, beside the line through the mouth on the lane side (the other side, or
        on the line, where the floor does not reach), and the route's direction there."""
        assert self._router is not None
        x, y, ux, uy = self._way_at(way, narrow.start - TURN_HOLD - place * (2.0 * radius + PASS_GAP))
        side = 1.0 if self._lane_side == "left" else -1.0
        lateral = width / 2.0 + radius + PASSING_GAP
        here = np.array([x, y])
        spots = np.array([(x - side * lateral * uy, y + side * lateral * ux), (x + side * lateral * uy, y - side * lateral * ux)])
        fits = (self._router.clearance(spots) >= radius + LANE_WALL_GAP) & self._router.line_of_sight(np.repeat(here[None], 2, 0), spots)
        for spot, ok in zip(spots, fits.tolist(), strict=True):
            if ok:
                return float(spot[0]), float(spot[1]), ux, uy
        return x, y, ux, uy

    def _queue(self, present: dict, seen: dict[int, tuple[Way, float, Narrow | None, str, float]]) -> dict[int, float]:
        """How far along its route each walker with a `queue_space` may go behind another of this kind in its way.

        One is in a walker's way where its body lies on the walker's route within `QUEUE_LOOK` ahead and it faces along
        that route rather than against it (one coming the other way is steered round, or met by turns in a narrow
        stretch). The walker stays behind it by its `queue_space` (measured from the middle of each body, less for a body
        that stands across). Two in each other's way, as where two routes join: the one that has the other further ahead
        gives way (the higher id where they are level) and keeps giving way until they are apart, so they go one after
        the other instead of side by side."""
        ids = [aid for aid, manner in self._manners.items() if manner[1].active and aid in present]
        if len(ids) < 2:
            return {}
        pos = np.array([(present[aid].state.pose.x, present[aid].state.pose.y) for aid in ids], dtype=np.float64)
        theta = np.array([present[aid].state.pose.theta for aid in ids], dtype=np.float64)
        radius = np.array([self._radius_of(aid) for aid in ids], dtype=np.float64)
        half = np.array([self._manners[aid][2] for aid in ids], dtype=np.float64)
        row = {aid: i for i, aid in enumerate(ids)}
        steps = np.arange(0.0, QUEUE_LOOK + 1e-9, QUEUE_STEP)
        ahead: dict[tuple[int, int], tuple[float, float]] = {}
        for aid, (way, s, _, _, _) in seen.items():
            space = self._manners[aid][1].queue_space
            if space <= 0.0 or aid not in row:
                continue
            i = row[aid]
            at = np.minimum(s + steps, float(way.at[-1]))
            line = np.column_stack([np.interp(at, way.at, way.points[:, 0]), np.interp(at, way.at, way.points[:, 1])])
            near = np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1]) <= QUEUE_LOOK + 2.0
            near[i] = False
            if not near.any():
                continue
            others = np.flatnonzero(near)
            d = np.hypot(line[:, None, 0] - pos[None, others, 0], line[:, None, 1] - pos[None, others, 1])
            k = np.argmin(d, axis=0)
            across = d[k, np.arange(len(others))]
            tangent = np.array([way.direction(float(a)) for a in at[k]])
            angle = theta[others] - np.arctan2(tangent[:, 1], tangent[:, 0])
            turned = np.abs(np.sin(angle))
            wide = radius[others] + (half[others] - radius[others]) * turned
            long = half[others] - (half[others] - radius[others]) * turned
            front = (((pos[others] - pos[i]) * np.array(way.direction(s))).sum(axis=1) > 0.0) & (np.cos(angle) > 0.0)
            for j in np.flatnonzero(front & (across <= radius[i] + wide + QUEUE_MARGIN)):
                ahead[(aid, ids[others[j]])] = (float(at[k[j]] - s), max(space - half[i], half[i]) + float(long[j]))
        limits: dict[int, float] = {}
        gives_way: set[tuple[int, int]] = set()
        for (aid, other), (gap, room) in ahead.items():
            back = ahead.get((other, aid))
            if back is not None:
                if (other, aid) in self._gives_way:
                    continue
                if (aid, other) not in self._gives_way and (back[0] > gap + QUEUE_LEVEL or (abs(back[0] - gap) <= QUEUE_LEVEL and other > aid)):
                    continue
                gives_way.add((aid, other))
            limits[aid] = min(limits.get(aid, math.inf), seen[aid][1] + gap - room)
        self._gives_way = gives_way
        return limits

    def _meet(self, a: Narrow | None, b: Narrow | None, reach: float) -> int:
        """How two routes' narrow stretches meet: -1 where they are one passage taken in opposite directions, 1 in the
        same direction, 0 where they are not one passage. One passage: they come within `reach` of each other with no
        wall between."""
        if a is None or b is None:
            return 0
        d = np.hypot(a.points[:, None, 0] - b.points[None, :, 0], a.points[:, None, 1] - b.points[None, :, 1])
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        gap = float(d[i, j])
        if gap > reach:
            return 0
        if gap > 1e-6 and self._cast(a.points[i][None], ((b.points[j] - a.points[i]) / gap)[None], gap)[0] < gap - 1e-6:
            return 0
        return -1 if float(a.tangents[i] @ b.tangents[j]) < 0.0 else 1

    def _slowdown_at(self, positions: np.ndarray) -> np.ndarray | None:
        """How much slower than at free flow walkers move through each mesh edge, from the density of `positions` within
        `CONGESTION_RADIUS` of its middle by Weidmann's relation (capped at `CONGESTION_CAP`); None without anyone."""
        assert self._mesh is not None
        if not len(positions):
            return None
        from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]  # scipy exports it from a compiled module the checker cannot see

        self._crowd = cKDTree(positions)
        return self._weidmann(self._crowd.query_ball_point(self._mesh.e_mid, CONGESTION_RADIUS, return_length=True))

    @staticmethod
    def _weidmann(counts: np.ndarray) -> np.ndarray:
        """The slowdown at points with `counts` walkers within `CONGESTION_RADIUS` (Weidmann, capped at `CONGESTION_CAP`)."""
        rho = np.asarray(counts, dtype=np.float64) / (math.pi * CONGESTION_RADIUS**2)
        slow = np.ones(len(rho))
        busy = rho > 0.0
        inv = 1.0 / rho[busy] - 1.0 / WEIDMANN_RHO_MAX
        share = np.where(inv > 0.0, 1.0 - np.exp(-WEIDMANN_GAMMA * np.maximum(inv, 0.0)), 0.0)
        slow[busy] = np.minimum(1.0 / np.maximum(share, 1.0 / CONGESTION_CAP), CONGESTION_CAP)
        return slow

    def path_cost(self, path: Sequence[Pose2D]) -> float:
        """The path's length weighted by the mean congestion slowdown at points every `PATH_COST_STEP` along it, from the
        crowd the last congestion update saw (`congestion_interval`); its plain length while congestion is off."""
        length = super().path_cost(path)
        if self._slowdown is None or self._crowd is None or len(path) < 2 or length <= 0.0:
            return length
        pts = np.array([(p.x, p.y) for p in path], dtype=np.float64)
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        at = np.arange(PATH_COST_STEP / 2.0, along[-1], PATH_COST_STEP)
        if not len(at):
            at = np.array([along[-1] / 2.0])
        samples = np.column_stack([np.interp(at, along, pts[:, 0]), np.interp(at, along, pts[:, 1])])
        return float(length * np.mean(self._weidmann(self._crowd.query_ball_point(samples, CONGESTION_RADIUS, return_length=True))))

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

    def paths(self, legs: Sequence[tuple[Pose2D, Pose2D]]) -> list[list[Pose2D] | None]:
        """As `GlobalPlanner.paths`, planned at the inflation radius and without any agent's state: the legs are numbered
        from 0, so their ids would read and overwrite the radius, route memory and escape of the agents with those ids."""
        self._pricing = True
        try:
            return super().paths(legs)
        finally:
            self._pricing = False

    def _plan(self, requests: list[PlanRequest]) -> dict[int, list[Pose2D] | None]:
        assert self._router is not None
        plans: dict[int, list[Pose2D] | None] = {agent_id: None for agent_id, _, _ in requests}
        if self._pricing:
            by_target: dict[tuple[float, float], list[tuple[int, Pose2D, Pose2D]]] = {}
            for request in requests:
                by_target.setdefault((request[2].x, request[2].y), []).append(request)
            for group in by_target.values():
                plans.update(self._plan_group(self._router, group))
            return plans
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
            if agent_id in self._manners and not self._pricing:
                polyline = self._shape(agent_id, router, self._from_here(router, agent_id, agent_pos, polyline))
            waypoints = [Pose2D(x=float(x), y=float(y)) for x, y in polyline]
            same = polyline[0, 0] == agent_pos.x and polyline[0, 1] == agent_pos.y
            if same:
                waypoints.pop(0)
            waypoints.insert(0, Pose2D(x=agent_pos.x, y=agent_pos.y))
            waypoints[-1] = Pose2D(x=snapped.x, y=snapped.y, theta=target.theta)
            plans[agent_id] = waypoints
            if self._manners and not self._pricing:
                self._keep_way(agent_id, waypoints, 0 if same else 1)
        return plans
