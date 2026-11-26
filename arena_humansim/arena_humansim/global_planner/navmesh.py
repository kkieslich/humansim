from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any, TypedDict

import attrs
import numpy as np

from arena_humansim.core.agents.types import Passage, WallTrailing
from arena_humansim.utils.mesh import TOL, Mesh, _pt_seg
from arena_humansim.utils.router import VERIFY_EPS, Router
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segments
from arena_humansim.utils.wall_grid import WallGrid, cast_rays

from . import BLOCK_NEAR, LANE_SIDES, STALL_M, GlobalPlanner, PassageCommand, PlanRequest, Robots
from ._grid import next_waypoint
from .wall_following import PASSING_GAP, STEP, Shaped, approach, follow_walls, leave, sample, thin

SNAP_CACHE = 4096
#: Priced legs kept between changes of the map, blocked edges, congestion or lanes (`NavMeshPlanner.paths`)
PRICED_LEGS = 20000
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
#: The end of a narrow passage that has it keeps it while those at the other end have waited less than this [s] (counted
#: from when it got the passage): two crowds take it in platoons, and no end waits much longer than this plus the time
#: the walkers inside need to come out. No source, our choice
TURN_YIELD_S = 10.0
#: A walker keeps its place in the order of a passage (its claim) while its route leads through the stretch whose mouth
#: lies within this of the one it claimed [m], also where the crowd pushed it back beyond `LINE_UP`; a passage is the
#: same between two calls where its walkers' stretches lie within this of where it was
TURN_SAME = 1.5
#: A walker blocked by a robot standing still goes round it this far from the robot's body [m], on points of a circle
#: round the robot at most `ROUND_STEP` [rad] apart, its route checked against the robot every `ROUND_SAMPLE` [m]
ROUND_GAP, ROUND_STEP, ROUND_SAMPLE = 0.2, math.pi / 6.0, 0.1
#: A route runs into a robot where it comes within the two bodies plus this of the robot's middle [m]
ROUND_HIT = 0.05
#: Robot-aware routers kept (keyed by clearance, walker radius and the robots standing still they keep off), and how much
#: a step across a mesh edge that passes through a robot's body (plus the walker's) costs on them, per metre
ROUND_ROUTERS, ROUND_PENALTY = 8, 1000.0


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


def _inside_polygon(points: tuple[tuple[float, float], ...], x: float, y: float) -> bool:
    """Whether (x, y) lies inside the polygon `points` (even-odd rule)."""
    inside = False
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


@attrs.define
class NoGo:
    """The walls an agent class plans with where stair or escalator halves it does not climb are walled off
    (`NavMeshPlanner.set_areas`): their mesh, its inflation router, the walls it was meshed on and the mesh edges of the
    walls added since (blocked, as on the main mesh), the routers that keep wider clearances and the body routers."""

    mesh: Mesh
    router: Router
    meshed: set[tuple[float, ...]] = attrs.field(factory=set)
    blocked: np.ndarray = attrs.field(factory=lambda: np.zeros(0, np.int64))
    routers: dict[float, Router] = attrs.field(factory=dict)
    body: dict[float, Router] = attrs.field(factory=dict)


@attrs.define
class TurnPassage:
    """One narrow passage that walkers take by turns (`_take_turns`): where it is, a direction along it that says which
    end is which (+1 along, -1 against), the end that has it (0 none) and since when [tick], the last call that saw it
    [tick], and the walker that came first at that end when it got the passage (-1 none): the end keeps the passage
    until this walker went in."""

    x: float
    y: float
    ux: float
    uy: float
    serving: int = 0
    since: int = 0
    seen: int = 0
    first: int = -1


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
        #: priced legs (start x, y, goal x, y: path) and the planner state they were priced on (`paths`)
        self._priced: dict[tuple[float, float, float, float], list[Pose2D] | None] = {}
        self._priced_state: tuple | None = None
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
        #: the wall ends no other wall meets (`_free_ends`), for the walls they were found on
        self._free: tuple[list, np.ndarray] | None = None
        #: narrow-passage turns of walkers without a `passage` block: measured routes, claims (mouth x, y, tick), and
        #: where each walker last moved (x, y, tick) and whether the last `compute` held it
        self._turn_ways: dict[int, TurnWay] = {}
        self._turn_claims: dict[int, tuple[float, float, int]] = {}
        self._turn_moved: dict[int, tuple[float, float, int]] = {}
        self._turn_held: set[int] = set()
        self._turn_passages: list[TurnPassage] = []
        #: the stair and escalator halves (name: kind, footprint), per set of halves an agent class does not climb its
        #: walled-off mesh (`NoGo`), and per agent of the last `compute` that set (empty: it climbs every half)
        self._areas: dict[str, tuple[str, tuple[tuple[float, float], ...]]] = {}
        self._nogo: dict[frozenset[str], NoGo] = {}
        #: how long each walled-off mesh took to build [ms], for the record
        self.nogo_builds: list[float] = []
        self._class_of: dict[int, frozenset[str]] = {}
        #: per blocked walker: the route `_reroute` last saw or made for it with the robots it goes round, and the side it
        #: goes round each robot by (+1 counter-clockwise, -1 clockwise); routers that also keep off robots standing still
        self._rounded: dict[int, tuple[list[Pose2D], tuple]] = {}
        self._round_side: dict[tuple[int, int], float] = {}
        self._robot_routers: dict[tuple, tuple[Router, np.ndarray]] = {}

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
            self._robot_routers.clear()
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
            self._block_nogo(keys)
            return
        self._nogo.clear()
        self._mesh = None
        self._router = None
        self._routers = {}
        self._body_routers = {}
        self._robot_routers = {}
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
        self._robot_routers.clear()
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

    def set_areas(self, areas: dict[str, tuple[str, Sequence[tuple[float, float]]]]) -> None:
        """The stair and escalator halves (name: kind, footprint). An agent plans on walls where the halves of the kinds
        it does not climb (its `climbs`) are walled off, so no route, straight leg or nearest reachable point of a
        wheelchair user leads into a stair or escalator half. Without them it walked into a half where its goal lay
        beyond it (the nearest point it could reach was up to the seam) or its route cut a half's corner."""
        areas = {name: (kind, tuple((float(x), float(y)) for x, y in fp)) for name, (kind, fp) in areas.items() if kind and len(fp) >= 3}
        if areas != self._areas:
            self._areas = areas
            self._nogo.clear()
            self._forget_paths()

    def _block_nogo(self, keys: set[tuple[float, ...]]) -> None:
        """The walled-off meshes through a change of walls the main mesh took by blocking (doors and other mechanisms
        that add walls and take them away again): the walls added since each was meshed are blocked on it; one that
        lost a wall it was meshed on, or cannot block one, is dropped and built again on first use."""
        for closed, nogo in list(self._nogo.items()):
            if not nogo.meshed <= keys:
                del self._nogo[closed]
                continue
            added = np.array(sorted(keys - nogo.meshed), dtype=np.float64).reshape(-1, 4)
            edges = nogo.mesh.free_edges(added[:, :2], added[:, 2:])
            if (edges < 0).any():
                del self._nogo[closed]
                continue
            nogo.blocked = edges
            for router in (nogo.router, *nogo.routers.values(), *nogo.body.values()):
                router.set_blocked(edges)

    def _closed_to(self, agent: object) -> frozenset[str]:
        """The halves an agent does not climb: those whose kind is not among its `climbs` (none without the attribute).
        One that stands on such a half (the crowd pushed it there) plans on the open walls, its way off; so a class has
        one walled-off mesh, not one per half its agents stand on."""
        climbs = getattr(getattr(agent, "params", None), "climbs", None)
        if climbs is None or not self._areas:
            return frozenset()
        pose = agent.state.pose  # pyright: ignore[reportAttributeAccessIssue]  # every planned agent has a state
        closed = frozenset(name for name, (kind, _) in self._areas.items() if kind not in climbs)
        if any(_inside_polygon(self._areas[name][1], pose.x, pose.y) for name in closed):
            return frozenset()
        return closed

    def _nogo_for(self, closed: frozenset[str]) -> NoGo:
        """The walled-off mesh for the halves `closed`, built on first use: the walls and the halves' outlines."""
        hit = self._nogo.get(closed)
        if hit is None:
            began = time.perf_counter()
            walls = np.array(self._wall_segments, dtype=np.float64).reshape(-1, 4)
            walls = walls[np.hypot(*(walls[:, 2:] - walls[:, :2]).T) > TOL]
            outlines = [(*a, *b) for name in sorted(closed) for fp in [self._areas[name][1]] for a, b in zip(fp, fp[1:] + fp[:1], strict=True)]
            segs = np.vstack([walls, np.array(outlines, dtype=np.float64).reshape(-1, 4)])
            segs = segs[np.hypot(*(segs[:, 2:] - segs[:, :2]).T) > TOL]
            mesh = Mesh.build(segs, 2 * max(self._comfort_radius, self._inflation_radius * 1.05))
            hit = self._nogo[closed] = NoGo(mesh, Router(mesh, self._inflation_radius, self._comfort_radius), meshed=_keys(walls))
            self.nogo_builds.append((time.perf_counter() - began) * 1e3)
            self._logger.info(f"Walls meshed with {len(closed)} stair or escalator half(s) walled off: {len(mesh.V)} triangles, {self.nogo_builds[-1]:.0f} ms")
            if self._crowd is not None:
                self._nogo_congestion(hit)
        return hit

    def _nogo_congestion(self, nogo: NoGo) -> None:
        """Weigh a walled-off mesh's routers by the crowd of the last congestion update (`_slowdown_at`)."""
        slowdown = None if self._crowd is None else self._weidmann(self._crowd.query_ball_point(nogo.mesh.e_mid, CONGESTION_RADIUS, return_length=True))
        for router in (nogo.router, *nogo.routers.values()):
            router.set_congestion(slowdown)

    def _base(self, closed: frozenset[str]) -> Router:
        """The inflation router of agents that do not climb the halves `closed`."""
        assert self._router is not None
        return self._router if not closed else self._nogo_for(closed).router

    def _router_for(self, clearance: float, closed: frozenset[str] = frozenset()) -> Router:
        """The router that keeps `clearance` from the walls, built on first use."""
        assert self._router is not None and self._mesh is not None
        if closed:
            nogo = self._nogo_for(closed)
            if clearance <= self._inflation_radius + 1e-9:
                return nogo.router
            if clearance not in nogo.routers:
                nogo.routers[clearance] = Router(nogo.mesh, clearance, self._comfort_radius)
                if len(nogo.blocked):
                    nogo.routers[clearance].set_blocked(nogo.blocked)
                if self._crowd is not None:
                    self._nogo_congestion(nogo)
            return nogo.routers[clearance]
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

    def _body_router(self, clearance: float, closed: frozenset[str] = frozenset()) -> Router:
        """The router that keeps only `clearance`, below the inflation radius, from the walls, built on first use. Its
        routes take plain lengths: they only get a walker back to where its own route starts (`_escape`)."""
        assert self._mesh is not None
        if closed:
            nogo = self._nogo_for(closed)
            if clearance not in nogo.body:
                nogo.body[clearance] = Router(nogo.mesh, clearance, self._comfort_radius)
                if len(nogo.blocked):
                    nogo.body[clearance].set_blocked(nogo.blocked)
            return nogo.body[clearance]
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
                for nogo in self._nogo.values():
                    self._nogo_congestion(nogo)
        self._class_of = {}
        if self._areas:
            for agent in agents:
                closed = self._closed_to(agent)
                if closed:
                    self._class_of[agent.state.agent_id] = closed
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
        if self._blockers or self._rounded:
            self._reroute(agents, goals, high_level_commands)
        self._commands = {}
        if self._manners or self._ways:
            self._drive(agents, goals)
        self._take_turns(agents, high_level_commands, goals)
        return goals

    def _takes_straight_legs(self, agent_id: int) -> bool:
        if self._blockers.get(agent_id):
            return False
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
        ends = self._free_ends()
        if len(ends):
            width = np.minimum(width, self._jamb_width(spots, normal, ends, limit))
        tight = width < limit
        edges = np.flatnonzero(np.diff(np.concatenate([[False], tight, [False]]).astype(np.int8)))
        narrows = []
        for a, b in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
            start = float(s[a]) if a else -self._narrow_behind(points, limit)
            narrows.append((Narrow(start, float(s[b - 1]), start - TURN_HOLD, spots[a:b], tangents[a:b]), float(width[a:b].min())))
        return TurnWay(waypoints, points, at, narrows)

    def _free_ends(self) -> np.ndarray:
        """The wall ends no other wall meets, at least `2 * MIN_BODY_RADIUS` from any other wall (the jambs of doors and
        openings in walls without thickness; a break in a wall too small for anyone to pass is none), (n, 2)."""
        if self._free is not None and self._free[0] is self._wall_segments:
            return self._free[1]
        segs = np.array(self._wall_segments, dtype=np.float64).reshape(-1, 4)
        segs = segs[np.hypot(*(segs[:, 2:] - segs[:, :2]).T) > TOL]
        ends = np.concatenate([segs[:, :2], segs[:, 2:]])
        owner = np.concatenate([np.arange(len(segs)), np.arange(len(segs))])
        _, at, count = np.unique(np.round(ends / TOL).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
        free = []
        for k in np.flatnonzero(count[at.reshape(-1)] == 1).tolist():
            d, _ = _pt_seg(ends[k][None], segs[:, :2], segs[:, 2:])
            d[owner[k]] = np.inf
            if not (d < 2.0 * MIN_BODY_RADIUS).any():
                free.append(ends[k])
        self._free = (self._wall_segments, np.array(free, dtype=np.float64).reshape(-1, 2))
        return self._free[1]

    def _jamb_width(self, spots: np.ndarray, normal: np.ndarray, ends: np.ndarray, limit: float) -> np.ndarray:
        """How narrow the route is at each of `spots` beside a free wall end within `limit` of it: on each side of the
        route, the way to the nearest such end in sight within the widest `TURN_TILT` of the side's normal, or on a side
        without one to the nearest wall along the `TURN_TILT` rays; inf where neither side has such an end.

        A door in a wall without thickness (the arena-world walls are lines) has no chord through a point beside it that
        ends on walls at both ends, so the rays of `_turn_way` find it no narrower than the rooms it joins."""
        width = np.full(len(spots), np.inf)
        reach = np.hypot(spots[:, None, 0] - ends[None, :, 0], spots[:, None, 1] - ends[None, :, 1])
        si, ei = np.nonzero(reach < limit)
        if not len(si):
            return width
        dist = reach[si, ei]
        u = (ends[ei] - spots[si]) / dist[:, None]
        seen = self._cast(spots[si], u, dist - 1e-3) >= dist - 1e-3 - 1e-9
        along = (u * normal[si]).sum(axis=1)
        cone = math.cos(max(abs(t) for t in TURN_TILT)) - 1e-9
        side = np.full((len(spots), 2), np.inf)
        for k, sign in enumerate((1.0, -1.0)):
            mine = seen & (sign * along >= cone)
            np.minimum.at(side[:, k], si[mine], dist[mine])
        has = np.isfinite(side)
        rows, k = np.nonzero(has.any(axis=1)[:, None] & ~has)
        if len(rows):
            n = normal[rows] * np.where(k == 0, 1.0, -1.0)[:, None]
            c, s = np.cos(TURN_TILT), np.sin(TURN_TILT)
            dirs = np.stack([c[:, None] * n[None, :, 0] - s[:, None] * n[None, :, 1], s[:, None] * n[None, :, 0] + c[:, None] * n[None, :, 1]], axis=-1)
            hits = self._cast(np.tile(spots[rows], (len(TURN_TILT), 1)), dirs.reshape(-1, 2), TURN_FAR).reshape(len(TURN_TILT), -1)
            side[rows, k] = hits.min(axis=0)
        one = has.any(axis=1)
        width[one] = side[one].sum(axis=1)
        return width

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
        narrower there than their two bodies and `PASS_GAP`, do not both go in. The walkers whose stretches meet form one
        passage (`_turn_passage`), and one end of it has it at a time (`_serve`): within `LINE_UP` of the stretch a walker
        of the other end, or one while someone coming the other way is inside, holds `TURN_HOLD` before it, in line
        beside the mouth with the others of its end in the order they came. A walker's place in that order (its claim)
        dates from when it first came within `LINE_UP` and lasts while its route leads through the stretch, also where the
        crowd pushed it back. Two that met inside: the one that came less far rolls back behind its hold line. A walker
        that stood `TURN_STUCK_S` without being held keeps no one waiting.
        Without this rule two streams that meet in a passage narrower than two bodies press against each other, and the
        walkers behind keep pressing in from both ends, so neither front can give way."""
        navigating = {aid for aid, cmd in high_level_commands.items() if isinstance(cmd, HighLevelCommand) and cmd.type == CommandType.NAVIGATE}
        present = {a.state.agent_id: a for a in agents}
        for aid in [a for a in self._turn_ways if a not in present]:
            del self._turn_ways[aid]
        seen: dict[int, tuple[float, float, Narrow, float, str]] = {}
        ahead: dict[int, np.ndarray] = {}
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
            ahead[aid] = narrow.points[0]
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
        # a walker's place in the order of its passage: from when it first came within `LINE_UP`, for as long as its route
        # leads through the same stretch (a crowd that pushes it back does not cost it its place)
        claims: dict[int, tuple[float, float, int]] = {}
        for aid, mouth in ahead.items():
            claim = self._turn_claims.get(aid)
            if claim is not None and math.hypot(claim[0] - mouth[0], claim[1] - mouth[1]) <= TURN_SAME:
                claims[aid] = claim
            elif aid in seen:
                claims[aid] = (float(mouth[0]), float(mouth[1]), tick)
        self._turn_claims = claims
        busy = list(seen)
        against: dict[int, list[int]] = {aid: [] for aid in busy}
        group = {aid: aid for aid in busy}

        def root(a: int) -> int:
            while group[a] != a:
                group[a] = group[group[a]]
                a = group[a]
            return a

        for i, a in enumerate(busy):
            for b in busy[i + 1 :]:
                sa, sb = seen[a], seen[b]
                width = min(sa[3], sb[3])
                met = self._meet(sa[2], sb[2], max(width, 2.0 * (sa[1] + sb[1])))
                if met != 0:
                    group[root(a)] = root(b)
                if met < 0 and width < 2.0 * (sa[1] + sb[1]) + PASS_GAP:
                    against[a].append(b)
                    against[b].append(a)
        back: set[int] = set()
        for aid in busy:
            s, _, narrow, _, phase = seen[aid]
            if phase != "inside":
                continue
            for other in against[aid]:
                so = seen[other]
                if so[4] == "inside" and other not in stuck and (s - narrow.line, -aid) < (so[0] - so[2].line, -other):
                    back.add(aid)
        passages: dict[int, list[int]] = {}
        for aid in busy:
            passages.setdefault(root(aid), []).append(aid)
        way_of: dict[int, int] = {}
        serving: dict[int, int] = {}
        for members in passages.values():
            passage = self._turn_passage(members, seen, tick)
            ref = np.array([passage.ux, passage.uy])
            for aid in members:
                narrow = seen[aid][2]
                way_of[aid] = 1 if float(narrow.tangents[len(narrow.tangents) // 2] @ ref) >= 0.0 else -1
            self._serve(passage, [a for a in members if against[a]], seen, claims, way_of, back, stuck, tick)
            for aid in members:
                serving[aid] = passage.serving
        # one of the end that does not have the passage whom the crowd behind pushed past its hold line, but not yet into
        # the stretch, rolls back behind the line: it neither keeps the other end waiting nor stands in its way out
        for aid in busy:
            s, _, narrow, _, phase = seen[aid]
            if phase == "inside" and s < narrow.start and serving[aid] not in (0, way_of[aid]) and any(o not in stuck for o in against[aid]):
                back.add(aid)
        held: set[int] = set()
        for aid in busy:
            phase = seen[aid][4]
            hold = aid in back
            if phase == "near" and against[aid]:
                # the end that does not have the passage waits, as does one while someone of the other end is inside
                hold = way_of[aid] != serving[aid] or any(o not in stuck and seen[o][4] == "inside" and o not in back for o in against[aid])
            if hold:
                held.add(aid)
        # the waiting walkers of one end stand in line beside the mouth in the order they came, so that those coming out pass
        order = sorted((a for a in held if a not in back), key=lambda a: (round(claims[a][0], 0), round(claims[a][1], 0), claims[a][2], a))
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

    def _turn_passage(self, members: list[int], seen: dict, tick: int) -> TurnPassage:
        """The passage the narrow stretches of `members` belong to: the one of the last call within `TURN_SAME` of them, or
        a new one where none is. Passages no call saw for a second are forgotten."""
        self._turn_passages = [p for p in self._turn_passages if (tick - p.seen) * self._dt <= 1.0]
        mids = np.array([seen[a][2].points[len(seen[a][2].points) // 2] for a in members])
        best, gap = None, TURN_SAME
        for passage in self._turn_passages:
            d = float(np.hypot(mids[:, 0] - passage.x, mids[:, 1] - passage.y).min())
            if d <= gap:
                best, gap = passage, d
        if best is None:
            narrow = seen[members[0]][2]
            t = narrow.tangents[len(narrow.tangents) // 2]
            best = TurnPassage(float(mids[0, 0]), float(mids[0, 1]), float(t[0]), float(t[1]), seen=tick)
            self._turn_passages.append(best)
        best.seen = tick
        return best

    def _serve(self, passage: TurnPassage, contested: list[int], seen: dict, claims: dict, way_of: dict[int, int], back: set[int], stuck: set[int], tick: int) -> None:
        """Which end of `passage` has it, from the walkers that cannot pass someone coming the other way (`contested`).

        The end whose first walker came first gets it, and keeps it while its walkers come; the other end gets it once
        this end has none left before or in the passage, or once its first walker has waited `TURN_YIELD_S` (counted from
        when this end got the passage) and the walker this end got it for went in (or stood `TURN_STUCK_S`). Those of
        this end already inside come out first. Without the last condition a slow walker (a wheelchair user waiting
        beside the mouth while faster walkers of its end go first) can miss every turn of its end."""
        first: dict[int, tuple[int, int]] = {}
        inside: set[int] = set()
        for aid in contested:
            if aid in stuck or aid in back:
                continue
            if seen[aid][4] == "inside" and seen[aid][0] >= seen[aid][2].start:
                inside.add(way_of[aid])
            else:
                first[way_of[aid]] = min(first.get(way_of[aid], (tick, aid)), (claims[aid][2], aid))
        if passage.serving not in first and passage.serving not in inside:
            passage.serving = next(iter(inside)) if len(inside) == 1 else min(first, key=lambda d: (first[d], d)) if first else 0
            passage.since = tick
            passage.first = first[passage.serving][1] if passage.serving in first else -1
        elif (
            -passage.serving in first
            and (tick - max(first[-passage.serving][0], passage.since)) * self._dt >= TURN_YIELD_S
            and not any(aid == passage.first and seen[aid][4] == "near" for aid in contested if aid not in stuck and aid not in back)
        ):
            passage.serving = -passage.serving
            passage.since = tick
            passage.first = first[passage.serving][1]

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

    def path_costs(self, paths: Sequence[Sequence[Pose2D]]) -> list[float]:
        """`path_cost` of each path, the crowd around the sample points of all of them counted in one query."""
        lengths = [GlobalPlanner.path_cost(self, path) for path in paths]
        if self._slowdown is None or self._crowd is None:
            return lengths
        weighed, chunks = [], []
        for k, (path, length) in enumerate(zip(paths, lengths, strict=True)):
            if len(path) < 2 or length <= 0.0:
                continue
            pts = np.array([(p.x, p.y) for p in path], dtype=np.float64)
            along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
            at = np.arange(PATH_COST_STEP / 2.0, along[-1], PATH_COST_STEP)
            if not len(at):
                at = np.array([along[-1] / 2.0])
            weighed.append(k)
            chunks.append(np.column_stack([np.interp(at, along, pts[:, 0]), np.interp(at, along, pts[:, 1])]))
        if not chunks:
            return lengths
        slow = self._weidmann(self._crowd.query_ball_point(np.vstack(chunks), CONGESTION_RADIUS, return_length=True))
        bounds = np.cumsum([0] + [len(chunk) for chunk in chunks])
        out = list(lengths)
        for k, a, b in zip(weighed, bounds[:-1].tolist(), bounds[1:].tolist(), strict=True):
            out[k] = float(lengths[k] * np.mean(slow[a:b]))
        return out

    def leg_floors(self, legs: Sequence[tuple[Pose2D, Pose2D]]) -> list[float]:
        """The straight distance between the two points each leg's priced path (`paths`) runs between: its first point
        is the snapped start, its last the target snapped once more (`_plan_group`). No path is shorter, and the
        congestion weights of `path_cost` are at least 1."""
        if not self._has_map():
            return [math.hypot(b.x - a.x, b.y - a.y) for a, b in legs]
        ends = self.snap_terminals([pose for leg in legs for pose in leg])
        out = []
        for k in range(len(legs)):
            a, b = ends[2 * k], self.snap_terminal(ends[2 * k + 1])
            out.append(math.hypot(b.x - a.x, b.y - a.y))
        return out

    def _set_slowdown(self, slowdown: np.ndarray | None) -> None:
        self._slowdown = slowdown
        for router in (self._router, *self._routers.values()):
            if router is not None:
                router.set_congestion(slowdown)
        for router, penalty in self._robot_routers.values():
            router.set_congestion(penalty if slowdown is None else slowdown * penalty)

    def _lanes(self, router: Router, polyline: np.ndarray) -> np.ndarray:
        """`polyline` with its inner corners moved `lane_offset` to the lane side, so walkers in opposite directions keep
        to opposite sides of a passage instead of meeting head-on on one line. A corner moves as far as its walkers'
        bodies keep `LANE_WALL_GAP` from the walls (their clearance without the margin): the full offset, half of it,
        or not at all. The ends stay. A body-radius route (`_escape`) keeps no lane."""
        return self._lanes_many(router, [polyline])[0]

    def _lanes_many(self, router: Router, polylines: Sequence[np.ndarray]) -> list[np.ndarray]:
        """`_lanes` of each polyline, the walls looked up for the corners and the shifted routes of all of them at once."""
        out = list(polylines)
        if self._lane_side == "none" or self._lane_offset <= 0.0 or router.r < self._inflation_radius - 1e-9:
            return out
        todo = [k for k, polyline in enumerate(polylines) if len(polyline) >= 3]
        if not todo:
            return out
        body = max(router.r - self._margin, MIN_BODY_RADIUS) + LANE_WALL_GAP
        sign = 1.0 if self._lane_side == "right" else -1.0
        shifts = []
        for k in todo:
            d = np.diff(polylines[k], axis=0)
            d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
            normals = sign * np.column_stack([d[:, 1], -d[:, 0]])
            n = normals[:-1] + normals[1:]
            length = np.hypot(n[:, 0], n[:, 1])
            shifts.append(np.where(length[:, None] > 1e-6, n / np.maximum(length, 1e-9)[:, None], normals[:-1]))
        inner = np.vstack([polylines[k][1:-1] for k in todo])
        n = np.vstack(shifts)
        moved = inner.copy()
        settled = np.zeros(len(inner), dtype=bool)
        for share in (1.0, 0.5):
            cand = inner + n * (self._lane_offset * share)
            fits = ~settled & (router.clearance(cand) >= body)
            moved[fits] = cand[fits]
            settled |= fits
        bounds = np.cumsum([0] + [len(polylines[k]) - 2 for k in todo]).tolist()
        shifted = []
        for k, a, b in zip(todo, bounds[:-1], bounds[1:], strict=True):
            if settled[a:b].any():
                full = polylines[k].copy()
                full[1:-1] = moved[a:b]
                shifted.append((k, full, a, b))
        halves = []
        for (k, full, a, b), clear in zip(shifted, router._clear_many([full for _, full, _, _ in shifted]) if shifted else [], strict=True):
            if clear >= body - VERIFY_EPS:
                out[k] = full
                continue
            half = polylines[k].copy()
            half[1:-1] = inner[a:b] + (moved[a:b] - inner[a:b]) * 0.5
            halves.append((k, half))
        for (k, half), clear in zip(halves, router._clear_many([half for _, half in halves]) if halves else [], strict=True):
            if clear >= body - VERIFY_EPS:
                out[k] = half
        return out

    def forget_agents(self) -> None:
        super().forget_agents()
        self._rounded.clear()
        self._round_side.clear()

    def _reroute(self, agents: list, goals: dict[int, Pose2D], high_level_commands: dict[int, HighLevelCommand]) -> None:
        """Routes of walkers blocked by a robot standing still (`_note_blockers`) go round it instead of into it.

        Where a walker's route ahead runs into such a robot, the stretch near the robot is replaced by an arc round it,
        `ROUND_GAP` clear of both bodies, on the side its body fits between the robot and the walls (and other robots
        standing still), the shorter one where both do; the route goes on as before behind the robot. A walker trailing a
        wall so steps off the wall line, passes the robot and comes back onto the line. Where neither side fits (the
        robot closes a passage), the walker plans anew on the map with that robot on it (`_plan_round`), and waits as
        before only where there is no way at all. A route once checked is not checked again until it or the robots
        change. Only robots that blocked this walker count: robots that move, and robots standing still elsewhere, are
        left to the local forces."""
        blocked = self.blockers()
        for aid in [a for a in self._rounded if a not in blocked]:
            del self._rounded[aid]
        self._round_side = {k: v for k, v in self._round_side.items() if k[0] in blocked and k[1] in blocked[k[0]]}
        if not blocked or self._router is None:
            return
        still = self.still_robots()
        standing = {(rid, round(x, 2), round(y, 2), round(r, 2)) for rid, (x, y, r) in still.items()}
        for key in [k for k in self._robot_routers if not set(k[2]) <= standing]:
            del self._robot_routers[key]
        present = {a.state.agent_id: a.state.pose for a in agents}
        for aid, robots in blocked.items():
            entry, pos, cmd = self._path_cache.get(aid), present.get(aid), high_level_commands.get(aid)
            if entry is None or pos is None or aid not in goals or not isinstance(cmd, HighLevelCommand) or cmd.type != CommandType.NAVIGATE:
                continue
            key = tuple(sorted(robots.items()))
            seen = self._rounded.get(aid)
            if seen is not None and seen[0] is entry[1] and seen[1] == key:
                continue
            goal_key, waypoints, _, idx = entry
            hit, route = self._round(aid, pos, waypoints, idx, robots, still)
            if hit and route is None:
                route = self._plan_round(aid, pos, cmd.target_pose, robots)
                if route is not None:
                    self._logger.debug(f"agent {aid}: robot(s) {sorted(robots)} close its way, planned anew round them")
            if route is None:
                self._rounded[aid] = (waypoints, key)
                continue
            idx = self.advance_along_path(pos, route, 0)
            self._path_cache[aid] = (goal_key, route, np.array([(w.x, w.y) for w in route], dtype=np.float64).reshape(-1, 2), idx)
            goals[aid] = next_waypoint(route, idx)
            self._rounded[aid] = (route, key)

    def _round(self, agent_id: int, pos: Pose2D, waypoints: list[Pose2D], idx: int, robots: Robots, still: Robots) -> tuple[bool, list[Pose2D] | None]:
        """Whether the route ahead of the walker runs into one of `robots`, and the route going round them by arcs
        (`_round_one`); None where it runs into none, or where one of them it cannot go round."""
        radius = self._radius_of(agent_id)
        tail = np.array([(w.x, w.y) for w in waypoints[idx + 1 :]], dtype=np.float64).reshape(-1, 2)
        if not len(tail):
            return False, None
        poly = np.vstack([[pos.x, pos.y], tail])
        done: set[int] = set()
        while True:
            hit = self._hit(poly, {rid: rob for rid, rob in robots.items() if rid not in done}, radius)
            if hit is None:
                break
            rid, at, pts, d, h = hit
            done.add(rid)
            poly = self._round_one(agent_id, rid, robots[rid], radius, poly, at, pts, d, h, still)
            if poly is None:
                return True, None
        if not done:
            return False, None
        route = [Pose2D(x=pos.x, y=pos.y), *(Pose2D(x=float(x), y=float(y)) for x, y in poly[1:-1]), waypoints[-1]]
        return True, route

    @staticmethod
    def _hit(poly: np.ndarray, robots: Robots, radius: float) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int] | None:
        """The first of `robots` that the route `poly` runs into (comes within both bodies plus `ROUND_HIT`): its id, the
        route sampled every `ROUND_SAMPLE` (distances along it and points), each sample's distance from the robot's
        middle, and the first sample that runs into it. None where it runs into none, or only into robots it ends next to
        (its target lies by the robot: going round does not get it there)."""
        if not robots or len(poly) < 2:
            return None
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])
        at = np.unique(np.concatenate([np.arange(0.0, along[-1], ROUND_SAMPLE), along]))
        pts = np.column_stack([np.interp(at, along, poly[:, 0]), np.interp(at, along, poly[:, 1])])
        best = None
        for rid, (x, y, r) in robots.items():
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            hits = np.flatnonzero(d < r + radius + ROUND_HIT)
            if not len(hits) or not (d[hits[0] :] >= r + radius + ROUND_GAP).any():
                continue
            if best is None or at[hits[0]] < at[best[2]]:
                best = (rid, d, int(hits[0]))
        if best is None:
            return None
        return best[0], at, pts, best[1], best[2]

    def _round_one(self, agent_id: int, rid: int, robot: tuple[float, float, float], radius: float, poly: np.ndarray, at: np.ndarray, pts: np.ndarray, d: np.ndarray, h: int, still: Robots) -> np.ndarray | None:
        """`poly` with its stretch near the robot (round sample `h`, see `_hit`) replaced by an arc round it, on the side
        the walker took round this robot before where that still fits, else on the shorter side that fits. A walker
        standing by the robot (blocked, within `BLOCK_NEAR` of its body) goes round from where it stands, at its own
        distance from the robot (the local forces hold it there), or at both bodies plus `ROUND_GAP` where only that
        fits; the arc of one further away starts where its route comes within `ROUND_GAP`. A side fits where the walker's
        body keeps clear of the walls and of the other robots standing still. None where neither fits."""
        x, y, r = robot
        touch = r + radius
        near = float(d[0]) <= touch + BLOCK_NEAR + ROUND_HIT
        rings = dict.fromkeys((min(max(float(d[0]), touch + ROUND_GAP), touch + BLOCK_NEAR), touch + ROUND_GAP)) if near else {touch + ROUND_GAP: None}
        body = max(radius, MIN_BODY_RADIUS)
        walls = self._body_router(round(math.ceil(body / CLEARANCE_STEP - 1e-9) * CLEARANCE_STEP, 6))
        others = [(ox, oy, orad) for oid, (ox, oy, orad) in still.items() if oid != rid]
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])
        for ring in rings:
            after = np.flatnonzero(d[h:] >= ring)
            if not len(after):
                continue
            b = h + int(after[0])
            if near:
                a, p_in = 0, pts[0]
            else:
                before = np.flatnonzero(d[:h] >= ring)
                a = int(before[-1]) + 1 if len(before) else 0
                p_in = pts[a]
            p_out = pts[b]
            head = np.vstack([poly[along < at[a] - 1e-9], p_in[None]])
            rest = np.vstack([p_out[None], poly[along > at[b] + 1e-9]])
            t_in = math.atan2(p_in[1] - y, p_in[0] - x)
            t_out = math.atan2(p_out[1] - y, p_out[0] - x)
            ccw = (t_out - t_in) % (2.0 * math.pi)
            fits: dict[float, tuple[float, np.ndarray]] = {}
            for side, sweep in ((1.0, ccw), (-1.0, ccw - 2.0 * math.pi)):
                n = max(1, math.ceil(abs(sweep) / ROUND_STEP - 1e-9))
                angles = t_in + sweep * np.arange(1, n) / n
                arc = np.column_stack([x + ring * np.cos(angles), y + ring * np.sin(angles)])
                middle = np.vstack([p_in[None], arc, p_out[None]])
                if walls._clear_many([middle])[0] < body - VERIFY_EPS:
                    continue
                if others:
                    legs = np.hypot(*np.diff(middle, axis=0).T)
                    m_at = np.concatenate([[0.0], np.cumsum(legs)])
                    s = np.arange(0.0, m_at[-1] + 1e-9, ROUND_SAMPLE)
                    spots = np.column_stack([np.interp(s, m_at, middle[:, 0]), np.interp(s, m_at, middle[:, 1])])
                    if any(float(np.hypot(spots[:, 0] - ox, spots[:, 1] - oy).min()) < orad + radius for ox, oy, orad in others):
                        continue
                fits[side] = (float(np.hypot(*np.diff(middle, axis=0).T).sum()), middle)
            if not fits:
                continue
            side = self._round_side.get((agent_id, rid))
            if side not in fits:
                side = min(fits, key=lambda k: fits[k][0])
            self._round_side[(agent_id, rid)] = side
            return np.vstack([head[:-1], fits[side][1], rest[1:]])
        return None

    def _plan_round(self, agent_id: int, pos: Pose2D, target: Pose2D, robots: Robots) -> list[Pose2D] | None:
        """A route to `target` that keeps off `robots` (`_robot_router`), at the walker's clearance or else the inflation
        radius; None where every way runs into them."""
        radius = self._radius_of(agent_id)
        for clearance in dict.fromkeys((self.clearance_for(radius), self._inflation_radius)):
            route = self._plan_group(self._robot_router(clearance, robots, radius), [(agent_id, pos, target)]).get(agent_id)
            if route is not None and self._hit(np.array([(w.x, w.y) for w in route], dtype=np.float64), robots, radius) is None:
                return route
        return None

    def _robot_router(self, clearance: float, robots: Robots, radius: float) -> Router:
        """A router keeping `clearance` on which every step across a mesh edge that passes within `radius` of the body of
        one of `robots` costs `ROUND_PENALTY` times its length (on top of the congestion weights), built on first use. A
        route avoids those edges wherever any other way exists; a step through them stays possible, so a walker is never
        walled in by the coarse mesh (a route that still runs into a robot is dropped by `_plan_round`)."""
        assert self._mesh is not None
        key = (clearance, round(radius, 2), tuple(sorted((rid, round(x, 2), round(y, 2), round(r, 2)) for rid, (x, y, r) in robots.items())))
        hit = self._robot_routers.get(key)
        if hit is None:
            mesh = self._mesh
            near = np.zeros(len(mesh.ea), dtype=bool)
            pa, pb = mesh.P[mesh.ea], mesh.P[mesh.eb]
            for x, y, r in robots.values():
                near |= _pt_seg(np.array([x, y], dtype=np.float64), pa, pb)[0] <= r + radius
            penalty = np.where(near, ROUND_PENALTY, 1.0)
            router = Router(mesh, clearance, self._comfort_radius)
            if len(self._blocked_edges):
                router.set_blocked(self._blocked_edges)
            router.set_congestion(penalty if self._slowdown is None else self._slowdown * penalty)
            if len(self._robot_routers) >= ROUND_ROUTERS:
                self._robot_routers.pop(next(iter(self._robot_routers)))
            hit = self._robot_routers[key] = (router, penalty)
        return hit[0]

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

    def _sees_legs(self, agent_ids: list[int], starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """`_sees`, and for an agent that does not climb some halves also on its walled-off walls."""
        clear = self._sees(starts, ends)
        if self._class_of:
            by: dict[frozenset[str], list[int]] = {}
            for k, aid in enumerate(agent_ids):
                if aid in self._class_of:
                    by.setdefault(self._class_of[aid], []).append(k)
            for closed, rows in by.items():
                clear[rows] &= self._base(closed).line_of_sight(starts[rows], ends[rows])
        return clear

    def _nearest_reachable(self, start: Pose2D, target: Pose2D, agent_id: int | None = None) -> Pose2D | None:
        assert self._router is not None
        router = self._router if agent_id is None else self._base(self._class_of.get(agent_id, frozenset()))
        s = router.snap(np.array([start.x, start.y], dtype=np.float64))
        if s is None:
            return None
        q = router.nearest_reachable(s, np.array([target.x, target.y], dtype=np.float64))
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
        from 0, so their ids would read and overwrite the radius, route memory and escape of the agents with those ids.

        A leg's path depends only on its ends, the map, the blocked edges, the congestion weights and the lane settings:
        legs asked again while none of these changed (the route choice of many walkers from a landing to one goal, until
        the next congestion update) are answered from `_priced` (at most `PRICED_LEGS`)."""
        state = (self._router, self._blocked_edges, self._slowdown, self._lane_side, self._lane_offset, self._margin)
        if self._priced_state is None or len(self._priced) >= PRICED_LEGS or any(a is not b and a != b if isinstance(a, (str, float, int)) else a is not b for a, b in zip(state, self._priced_state, strict=True)):
            self._priced, self._priced_state = {}, state
        keys = [(a.x, a.y, b.x, b.y) for a, b in legs]
        todo = {key: leg for key, leg in zip(keys, legs, strict=True) if key not in self._priced}
        if todo:
            self._pricing = True
            try:
                self._priced.update(zip(todo, super().paths(list(todo.values())), strict=True))
            finally:
                self._pricing = False
        return [self._priced[key] for key in keys]

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
        by_goal: dict[tuple[float, float, float, frozenset[str]], list[tuple[int, Pose2D, Pose2D]]] = {}
        for agent_id, agent_pos, target in requests:
            clearance = self.clearance_for(self._radius_of(agent_id))
            by_goal.setdefault((target.x, target.y, clearance, self._class_of.get(agent_id, frozenset())), []).append((agent_id, agent_pos, target))
        for (_, _, clearance, closed), group in by_goal.items():
            base = self._base(closed)
            router = self._router_for(clearance, closed)
            found = self._plan_group(router, group, base)
            if router is not base:
                # no route keeps the margin (a door narrower than the agent's comfort): the bare inflation radius
                missing = [entry for entry in group if found.get(entry[0]) is None]
                if missing:
                    found.update(self._plan_group(base, missing, base))
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
        closed = self._class_of.get(agent_id, frozenset())
        base = self._base(closed)
        snapped = self.snap_terminal(target)
        goal = np.array([snapped.x, snapped.y], dtype=np.float64)
        reach = base.components(goal[None])[0]
        router = self._body_router(body, closed)
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
        keeps = base.clearance(pts) >= base.r - VERIFY_EPS
        back = next((i for i, comp in enumerate(base.components(pts)) if keeps[i] and comp & reach), None)
        if back is None:
            return None
        out = Pose2D(x=float(pts[back, 0]), y=float(pts[back, 1]))
        entry = [(agent_id, out, target)]
        rest = self._plan_group(self._router_for(clearance, closed), entry, base).get(agent_id) or self._plan_group(base, entry, base).get(agent_id)
        if rest is None:
            return None
        head = [Pose2D(x=float(x), y=float(y)) for x, y in line[1 : int(np.searchsorted(along, at[back]))]]
        return [Pose2D(x=agent_pos.x, y=agent_pos.y), *head, *rest]

    def _fits(self, router: Router, agent_id: int, polyline: np.ndarray) -> bool:
        """Whether an agent's body (its radius, at least `MIN_BODY_RADIUS`) keeps `LANE_WALL_GAP` from every wall along
        `polyline`: so a gap it is routed through is at least its body width plus that, on both sides. A wheelchair
        (0.325 m) needs 0.75 m, an adult of 0.25 m 0.6 m; the inflation radius asks 0.76 m of everyone."""
        need = max(self._radius_of(agent_id), MIN_BODY_RADIUS) + LANE_WALL_GAP
        return bool(router._clear_many([polyline])[0] >= need - 1e-9)

    def _plan_group(self, router: Router, group: list[tuple[int, Pose2D, Pose2D]], base: Router | None = None) -> dict[int, list[Pose2D] | None]:
        """Routes for agents with one target, by `router`'s clearance, their inner corners on the lane side; `base` is
        the inflation router of their walls (`self._router` where they climb every half)."""
        base = self._router if base is None else base
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
        if router is not base:
            # a target closer to a wall than this clearance: route to the nearest point that keeps it, then step on
            near = router.snap(goal)
            goal = goal if near is None else near
        starts = np.array([start for _, _, _, start in entries])
        framed = self._inside(starts)
        beyond = (framed != starts).any(axis=1) | (self._inside(goal) != goal).any()
        # the sight line is looked for only where it is used: a start or the goal outside the mesh frame
        direct = np.zeros(len(starts), dtype=bool)
        if beyond.any():
            direct[beyond] = router.line_of_sight(starts[beyond], np.repeat(goal[None], int(beyond.sum()), 0))
        # priced routes (`paths`) only need their length: their funnels are not checked against the walls
        routes = router.plan_many(framed[~direct], self._inside(goal), verify=not self._pricing)
        if not self._pricing and any(route is None for route in routes):
            # a channel at the radius exists, but its funnel clips a wall corner where the gap is barely wider than two
            # radii (approached at a slant) and no repair or other channel keeps the radius: the agent walks that
            # funnel where its own body keeps `LANE_WALL_GAP` from every wall along it (`_fits`), else it has no route
            lost = [k for k, route in enumerate(routes) if route is None]
            ids = [entries[i][0] for i in np.nonzero(~direct)[0]]
            for k, route in zip(lost, router.plan_many(framed[~direct][lost], self._inside(goal), verify=False), strict=True):
                if route is not None and self._fits(router, ids[k], route[0]):
                    routes[k] = route
        polylines: list[np.ndarray | None] = [None] * len(entries)
        found = [(int(i), self._rejoin(router, starts[i], goal, route[0]) if beyond[i] else route[0]) for i, route in zip(np.nonzero(~direct)[0], routes, strict=True) if route is not None]
        for (i, _), polyline in zip(found, self._lanes_many(router, [polyline for _, polyline in found]), strict=True):
            polylines[i] = polyline
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
