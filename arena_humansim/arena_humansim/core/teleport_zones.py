from __future__ import annotations

import enum
import math
from collections.abc import Sequence

import attrs
import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.pool import KIND_ROBOT, AgentPool
from arena_humansim.global_planner import GlobalPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

WAIT_RADIUS = 1.0
HOLD_RADIUS = 0.3
SLOT_SPACING = 0.7
SLOT_MARGIN = 0.35
STAIR_SPEED = 0.7
LANDING_BACKOFF = 0.1
#: Route choice (`route_choice`): a walker re-decides between walking and a stair or escalator at most this often [s]
CHOICE_INTERVAL = 5.0
#: ... and leaves its choice only for an option cheaper by this share of its cost, at least `SWITCH_M` metres
SWITCH_SHARE = 0.1
SWITCH_M = 3.0
#: A stair half's hold keeps at least this from every wall, or no walker reaches it and no one ever crosses [m]
HOLD_CLEARANCE = 0.25
#: Route choice re-decisions are spread over this many slots of a second `CHOICE_INTERVAL` by agent id (the next decision
#: comes 5-10 s later), so a crowd decided together does not re-decide (and recompute its costs) in one tick
CHOICE_STAGGER = 20
#: A pedestrian on a stair flight stays this far short of its landing until it lands, so waiting fliers never stand on landings [m]
FLIGHT_WAIT_GAP = 1.0
#: Landing spots a flight tries when its own is taken: steps across the partner half, within its footprint [m]
LANDING_STEP = 0.6


def _inside(pts: tuple[tuple[float, float], ...], x: float, y: float) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(pts, pts[-1:] + pts[:-1], strict=True):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def _slots(zone: Zone) -> tuple[Pose2D, ...]:
    """Spots on a grid inside the footprint, the one farthest from the entry first, the hold pose when none fits."""
    if not zone.footprint:
        return (zone.hold,)
    xs, ys = [x for x, _ in zone.footprint], [y for _, y in zone.footprint]
    axes = []
    for lo, hi in ((min(xs) + SLOT_MARGIN, max(xs) - SLOT_MARGIN), (min(ys) + SLOT_MARGIN, max(ys) - SLOT_MARGIN)):
        n = max(1, math.floor(max(hi - lo, 0.0) / SLOT_SPACING) + 1)
        mid = (lo + hi) / 2.0
        axes.append([mid + (i - (n - 1) / 2.0) * SLOT_SPACING for i in range(n)])
    spots = [Pose2D(x=x, y=y, theta=zone.hold.theta) for x in axes[0] for y in axes[1] if _inside(zone.footprint, x, y)]
    spots.sort(key=lambda p: -math.hypot(p.x - zone.entry.x, p.y - zone.entry.y))
    return tuple(spots) or (zone.hold,)


def _center(zone: Zone) -> tuple[float, float]:
    """Footprint area centroid, the vertex mean for a degenerate footprint, the hold pose without one."""
    pts = zone.footprint
    if not pts:
        return zone.hold.x, zone.hold.y
    area = cx = cy = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1], strict=True):
        cross = ax * by - bx * ay
        area += cross
        cx += (ax + bx) * cross
        cy += (ay + by) * cross
    if abs(area) < 1e-9:
        return sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    return cx / (3.0 * area), cy / (3.0 * area)


@attrs.frozen
class Zone:
    name: str
    destinations: tuple[str, ...]
    entry: Pose2D
    wait: Pose2D
    hold: Pose2D
    footprint: tuple[tuple[float, float], ...] = ()
    delay: float = 0.0
    kind: str = ""
    belt: float = 0.0
    slots: tuple[Pose2D, ...] = attrs.field(init=False, default=attrs.Factory(_slots, takes_self=True))
    center: tuple[float, float] = attrs.field(init=False, default=attrs.Factory(_center, takes_self=True))

    @property
    def axis(self) -> tuple[float, float]:
        return math.cos(self.entry.theta), math.sin(self.entry.theta)

    def along(self, x: float, y: float) -> float:
        """Distance of x, y from the entry along the zone's facing."""
        ax, ay = self.axis
        return (x - self.entry.x) * ax + (y - self.entry.y) * ay

    def contains(self, x: float, y: float) -> bool:
        """Whether a pedestrian at x, y stands inside the zone."""
        if self.kind:
            return _inside(self.footprint, x, y)
        if math.hypot(x - self.hold.x, y - self.hold.y) <= HOLD_RADIUS:
            return True
        return _inside(self.footprint, x, y)

    def past_seam(self, x: float, y: float) -> bool:
        """Whether a pedestrian at x, y stands on this stair half at or beyond its seam trigger line."""
        return self.contains(x, y) and self.along(x, y) >= self.along(self.hold.x, self.hold.y)


class Phase(enum.IntEnum):
    WALK = 0
    WAIT = 1
    BOARD = 2
    HOLD = 3


@attrs.define
class Transfer:
    goal: tuple[float, float]
    zone: str
    destination: str
    phase: Phase = Phase.WALK
    since: int = 0
    slot: int = -1
    held: int = 0
    yielded: int = 0
    #: picked by route choice over a walkable route, not because walking cannot reach the goal
    chosen: bool = False


@attrs.define
class _Flight:
    """A pedestrian on a stair flight: from where it crossed the seam to its landing on the partner half `dest`, `ticks`
    long. It moves from `start` to `wait`, `FLIGHT_WAIT_GAP` short of the landing, and waits there until it lands."""

    start: tuple[float, float]
    end: tuple[float, float, float]
    t0: int
    ticks: int
    dest: str = ""
    wait: tuple[float, float] = (0.0, 0.0)


@attrs.frozen
class _Choice:
    goal: tuple[float, float]
    edge: tuple[str, str] | None
    at: float


@attrs.frozen
class _Graph:
    area: dict[int, int]
    edges: dict[int, list[tuple[str, str, int]]]


class TeleportZones:
    """Teleport zones, their access and the pedestrians transferring through them."""

    def __init__(self) -> None:
        self.zones: dict[str, Zone] = {}
        self.open: dict[str, bool] = {}
        self.transfers: dict[int, Transfer] = {}
        self.riders: set[int] = set()
        self._banned: dict[int, tuple[tuple[float, float], list[str]]] = {}
        self._held: set[int] = set()
        #: walkers that can climb a stair or escalator take it where that is cheaper than walking (walk to it, climb at
        #: stair or belt speed, walk on), by the global planner's congestion-weighted route costs; off: zones only join
        #: floor that walking cannot reach
        self.route_choice = False
        self._choices: dict[int, _Choice] = {}
        self._flights: dict[int, _Flight] = {}
        #: the stair kinds each pedestrian's type climbs, as last steered: a type that does not climb a half is never carried over its seam
        self._climbs: dict[int, tuple[str, ...]] = {}
        #: the half each pedestrian landed on, while it stands on it: it is not carried back over that half's seam unless
        #: its route takes the half. Kept through `forget`, which ends the transfer at the landing.
        self._landed: dict[int, str] = {}

    def set_zones(self, zones: Sequence[Zone], is_open: Sequence[bool]) -> None:
        """Replace every zone, keeping the transfers through zones that come back unchanged or stairs that keep their destination."""
        old = self.zones
        self.zones = {z.name: z for z in zones}
        self.open = {z.name: bool(o) for z, o in zip(zones, is_open, strict=True)}
        kept = {name for name, z in self.zones.items() if old.get(name) == z}
        for aid, transfer in list(self.transfers.items()):
            zone = self.zones.get(transfer.zone)
            stair = zone is not None and bool(zone.kind) and transfer.zone in old and bool(old[transfer.zone].kind) and transfer.destination in zone.destinations
            if transfer.zone not in kept and not stair:
                del self.transfers[aid]
        for aid, (goal, names) in list(self._banned.items()):
            names = [name for name in names if name in kept]
            if names:
                self._banned[aid] = (goal, names)
            else:
                del self._banned[aid]
        self.riders.clear()
        self._held.clear()
        self._choices.clear()
        if not self.zones:
            self._flights.clear()
            self._climbs.clear()
            self._landed.clear()

    def clear_transfers(self) -> None:
        self.transfers.clear()
        self._banned.clear()
        self.riders.clear()
        self._held.clear()
        self._choices.clear()
        self._flights.clear()
        self._climbs.clear()
        self._landed.clear()

    def forget(self, aid: int) -> None:
        self.transfers.pop(aid, None)
        self._banned.pop(aid, None)
        self._held.discard(aid)
        self._choices.pop(aid, None)
        self._flights.pop(aid, None)
        self._climbs.pop(aid, None)

    @property
    def flying(self) -> frozenset[int]:
        """Pedestrians on a stair flight: their per-tick moves along it are not the end of their transfer."""
        return frozenset(self._flights)

    def steer(self, agents: Sequence[BaseAgent], cmds: dict[int, HighLevelCommand], now: float | None = None) -> dict[int, HighLevelCommand]:
        """Commands with each transferring pedestrian's target swapped for its cabin's wait pose or spot inside, or its stair's seam trigger point.

        With `route_choice`, `now` [s] paces each walker's choice between walking and a stair (every call without it)."""
        if not self.zones:
            return cmds
        out = dict(cmds)
        groups: dict[GlobalPlanner, list[tuple[BaseAgent, HighLevelCommand]]] = {}
        for agent in agents:
            aid = agent.state.agent_id
            if agent.state.kind != KIND_ROBOT:
                self._climbs[aid] = tuple(agent.params.climbs)
            cmd = cmds.get(aid)
            flight = self._flights.get(aid)
            if flight is not None and cmd is not None:
                out[aid] = attrs.evolve(cmd, target_pose=Pose2D(x=flight.end[0], y=flight.end[1], theta=flight.end[2]))
                continue
            if cmd is not None and cmd.type == CommandType.NAVIGATE and agent.state.kind != KIND_ROBOT:
                groups.setdefault(agent.global_planner, []).append((agent, cmd))
        blocked: dict[tuple[str, ...], set[str]] = {}
        choose = self.route_choice and any(z.kind and z.destinations for z in self.zones.values())
        for planner, group in groups.items():
            graph = self._graph(planner)
            comps = planner.components([agent.state.pose for agent, _ in group] + [cmd.target_pose for _, cmd in group])
            if choose:
                self._decide(planner, graph, group, comps, now)
            for k, (agent, cmd) in enumerate(group):
                aid = agent.state.agent_id
                near = (agent.state.pose.x, agent.state.pose.y)
                goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
                transfer = self.transfers.get(aid)
                if transfer is not None and transfer.goal != goal:
                    transfer = None
                    del self.transfers[aid]
                banned = self._banned.get(aid)
                if banned is not None and banned[0] != goal:
                    banned = None
                    del self._banned[aid]
                climbs = agent.params.climbs
                if climbs not in blocked:
                    blocked[climbs] = self._blocked(climbs)
                if transfer is None or (transfer.phase == Phase.WALK and not self._aboard(transfer, *near)):
                    start, end = comps[k], comps[len(group) + k]
                    if choose and self._walks_to(graph, start, end):
                        choice = self._choices.get(aid)
                        edge = choice.edge if choice is not None and choice.goal == goal else None
                        if edge is None:
                            self.transfers.pop(aid, None)
                            continue
                        if transfer is None or (transfer.zone, transfer.destination) != edge:
                            transfer = self.transfers[aid] = Transfer(goal=goal, zone=edge[0], destination=edge[1], chosen=True)
                        out[aid] = attrs.evolve(cmd, target_pose=self.zones[edge[0]].hold)
                        continue
                    keep = None if transfer is None else (transfer.zone, transfer.destination)
                    edge = self._route(graph, start, end, blocked[climbs] | set(() if banned is None else banned[1]), near, keep)
                    if edge is None and banned is not None and len(banned[1]) > 1:
                        self._banned[aid] = (goal, banned[1][-1:])
                        edge = self._route(graph, start, end, blocked[climbs] | set(banned[1][-1:]), near, keep)
                    if edge is None and banned is not None:
                        del self._banned[aid]
                        edge = self._route(graph, start, end, blocked[climbs], near, keep)
                    if edge is None:
                        self.transfers.pop(aid, None)
                        continue
                    if transfer is None or (transfer.zone, transfer.destination) != edge:
                        transfer = self.transfers[aid] = Transfer(goal=goal, zone=edge[0], destination=edge[1])
                z = self.zones[transfer.zone]
                if z.kind:
                    out[aid] = attrs.evolve(cmd, target_pose=z.hold)
                    continue
                out[aid] = attrs.evolve(cmd, target_pose=z.wait if transfer.phase in (Phase.WALK, Phase.WAIT) else z.slots[transfer.slot])
        return out

    @staticmethod
    def _walks_to(graph: _Graph, start: frozenset[int], goal: frozenset[int]) -> bool:
        """Whether walking reaches the goal: start and goal share an area (as `_route` sees them)."""
        starts = {graph.area.get(c, c) for c in start}
        goals = {graph.area.get(c, c) for c in goal}
        return bool(starts and goals and starts & goals)

    def _climb(self, zone: Zone, speed: float) -> tuple[float, float]:
        """Length of a stair half from its entry to its seam [m], and how much longer than walking it takes for a walker
        of `speed`: a still stair caps it at `STAIR_SPEED`, a running belt carries it at belt speed."""
        length = max(0.0, zone.along(zone.hold.x, zone.hold.y))
        on = abs(zone.belt) if zone.belt else min(speed, STAIR_SPEED)
        return length, (speed / max(on, 1e-3) if speed > 0.0 else 1.0)

    def _decide(self, planner: GlobalPlanner, graph: _Graph, group: list[tuple[BaseAgent, HighLevelCommand]], comps: list[frozenset[int]], now: float | None) -> None:
        """Route choice for the walkers of `group` whose goal walking reaches and whose choice is due: walk, or a stair or
        escalator they climb, whichever costs less: the planner's walking cost to the half's seam, the half at stair or
        belt speed, and from the far half's landing on to the goal. A walker keeps its choice unless another option is
        cheaper by `SWITCH_SHARE` of it (at least `SWITCH_M`); once on a stair it is not asked again."""
        due: list[int] = []
        for k, (agent, cmd) in enumerate(group):
            aid = agent.state.agent_id
            goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
            x, y = agent.state.pose.x, agent.state.pose.y
            transfer = self.transfers.get(aid)
            if transfer is not None and transfer.goal == goal and (transfer.phase != Phase.WALK or self._aboard(transfer, x, y)):
                continue
            if not self._walks_to(graph, comps[k], comps[len(group) + k]):
                continue
            choice = self._choices.get(aid)
            if choice is not None and choice.goal == goal and now is not None and now - choice.at < CHOICE_INTERVAL:
                banned = self._banned.get(aid)
                closed = self._blocked(agent.params.climbs) | set(() if banned is None or banned[0] != goal else banned[1])
                if choice.edge is None or not set(choice.edge) & closed:
                    continue
            due.append(k)
        if not due:
            return
        stairs = sorted(name for name, z in self.zones.items() if z.kind and z.destinations and self.open[name])
        walk: dict[int, float] = {}
        by_goal: dict[tuple[float, float], list[int]] = {}
        for k in due:
            cmd = group[k][1]
            by_goal.setdefault((round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3)), []).append(k)
        for ks in by_goal.values():
            costs = planner.route_costs([group[k][0].state.pose for k in ks], group[ks[0]][1].target_pose)
            if costs is None:
                return
            walk.update(zip(ks, costs.tolist(), strict=True))
        # the stair options each walker may take and could win by a straight-line bound, and the one it holds
        options: dict[int, list[tuple[str, str]]] = {}
        held: dict[int, tuple[bool, tuple[str, str] | None]] = {}
        for k in due:
            agent, cmd = group[k]
            aid = agent.state.agent_id
            goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
            choice, transfer = self._choices.get(aid), self.transfers.get(aid)
            if choice is not None and choice.goal == goal:
                held[k] = (True, choice.edge)
            elif transfer is not None and transfer.goal == goal and transfer.chosen:
                held[k] = (True, (transfer.zone, transfer.destination))
            else:
                held[k] = (False, None)
            banned = self._banned.get(aid)
            closed = self._blocked(agent.params.climbs) | set(() if banned is None or banned[0] != goal else banned[1])
            speed = float(agent.state.desired_velocity)
            p, g = agent.state.pose, cmd.target_pose
            for name in stairs:
                if name in closed:
                    continue
                z = self.zones[name]
                if z.belt < 0.0:  # a belt running back to the entry: no one standing on it reaches the seam
                    continue
                lz, fz = self._climb(z, speed)
                for dest in z.destinations:
                    d = self.zones.get(dest)
                    if d is None or dest in closed:
                        continue
                    ld, fd = self._climb(d, speed)
                    # to and from the mouths: a lower bound of walking there
                    to_mouth = max(0.0, math.hypot(p.x - z.entry.x, p.y - z.entry.y) - self._mouth(z))
                    from_mouth = max(0.0, math.hypot(g.x - d.entry.x, g.y - d.entry.y) - self._mouth(d))
                    bound = to_mouth + lz * fz + max(0.0, z.delay) * speed + ld * fd + from_mouth
                    if bound < walk[k] or held[k][1] == (name, dest):
                        options.setdefault(k, []).append((name, dest))
        into: dict[str, dict[int, float]] = {}
        onward: dict[tuple[str, tuple[float, float]], float] = {}
        for name in stairs:
            ks = [k for k in due if any(e[0] == name for e in options.get(k, ()))]
            if ks:
                costs = planner.route_costs([group[k][0].state.pose for k in ks], self.zones[name].hold)
                into[name] = dict(zip(ks, (costs.tolist() if costs is not None else [math.inf] * len(ks)), strict=True))
        # onward from each landing to every goal that needs it: one tree, at the landing (walking costs are symmetric)
        wanted: dict[str, dict[tuple[float, float], Pose2D]] = {}
        for k, edges in options.items():
            cmd = group[k][1]
            goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
            for _, dest in edges:
                wanted.setdefault(dest, {})[goal] = cmd.target_pose
        for dest in sorted(wanted):
            d = self.zones[dest]
            landing = Pose2D(x=d.hold.x - d.axis[0] * LANDING_BACKOFF, y=d.hold.y - d.axis[1] * LANDING_BACKOFF)
            goals = sorted(wanted[dest])
            costs = planner.route_costs([wanted[dest][g] for g in goals], landing)
            for g, c in zip(goals, (costs.tolist() if costs is not None else [math.inf] * len(goals)), strict=True):
                onward[(dest, g)] = float(c)
        for k in due:
            agent, cmd = group[k]
            aid = agent.state.agent_id
            goal = (round(cmd.target_pose.x, 3), round(cmd.target_pose.y, 3))
            speed = float(agent.state.desired_velocity)
            priced: dict[tuple[str, str] | None, float] = {None: walk[k]}
            for name, dest in options.get(k, ()):
                lz, fz = self._climb(self.zones[name], speed)
                ld, fd = self._climb(self.zones[dest], speed)
                priced[(name, dest)] = into[name][k] + lz * (fz - 1.0) + max(0.0, self.zones[name].delay) * speed + onward[(dest, goal)] + ld * (fd - 1.0)
            best = min(priced, key=lambda e: (priced[e], () if e is None else e))
            known, prev = held[k]
            if known and best != prev:
                kept = priced.get(prev, math.inf)
                if math.isfinite(kept) and priced[best] + max(SWITCH_M, SWITCH_SHARE * kept) >= kept:
                    best = prev
            if not math.isfinite(priced[best]):
                best = None
            # the next decision falls into this walker's slot of the interval
            stamp = 0.0 if now is None else now + (aid % CHOICE_STAGGER) * CHOICE_INTERVAL / CHOICE_STAGGER
            self._choices[aid] = _Choice(goal=goal, edge=best, at=stamp)

    @staticmethod
    def _mouth(zone: Zone) -> float:
        """Half the width of a half's mouth: the farthest its footprint reaches across its axis."""
        ax, ay = zone.axis
        return max((abs(-(x - zone.entry.x) * ay + (y - zone.entry.y) * ax) for x, y in zone.footprint), default=0.0)

    def _blocked(self, climbs: tuple[str, ...]) -> set[str]:
        """Stairs halves a pedestrian climbing these kinds is not routed into: other kinds and closed halves."""
        return {name for name, z in self.zones.items() if z.kind and (z.kind not in climbs or not self.open[name])}

    def _aboard(self, transfer: Transfer, x: float, y: float) -> bool:
        zone = self.zones[transfer.zone]
        return bool(zone.kind) and zone.contains(x, y)

    def carry(self, pool: AgentPool) -> None:
        """Stand the pedestrians on a running belt and carry them at belt speed, cap walkers on a still stair at stair speed and keep the ones held at a seam still."""
        self.riders = set()
        for i, zone in self._on_stairs(pool):
            aid = int(pool.agent_ids[i])
            climbs = self._climbs.get(aid)
            if climbs is not None and zone.kind not in climbs:  # pushed onto a half it does not climb: it walks off
                continue
            if zone.belt:
                self.riders.add(aid)
                pool.desired_vel[i] = 0.0
            if aid in self._held:
                pool.vel[i] = 0.0
            elif zone.belt:
                ax, ay = zone.axis
                pool.vel[i, 0] = zone.belt * ax
                pool.vel[i, 1] = zone.belt * ay
            else:
                speed = math.hypot(float(pool.vel[i, 0]), float(pool.vel[i, 1]))
                if speed > STAIR_SPEED:
                    pool.vel[i] *= STAIR_SPEED / speed
            pool.prev_vel[i] = pool.vel[i]

    def standing(self, pool: AgentPool) -> np.ndarray:
        """Mask over the pool's rows of the pedestrians standing on a running belt."""
        mask = np.zeros(pool.n, dtype=np.bool_)
        for aid in self.riders:
            idx = pool._id_to_idx.get(aid)
            if idx is not None:
                mask[idx] = True
        return mask

    def _on_stairs(self, pool: AgentPool) -> list[tuple[int, Zone]]:
        """Pool rows of the self-driven pedestrians standing on a stair half, with that half."""
        stairs = [z for z in self.zones.values() if z.kind]
        if not stairs:
            return []
        n = pool.n
        pos = pool.pos[:n]
        free = (pool.kind[:n] != KIND_ROBOT) & (pool.policy_idx[:n] >= 0)
        found: dict[int, Zone] = {}
        for zone in stairs:
            xs, ys = [x for x, _ in zone.footprint], [y for _, y in zone.footprint]
            near = free & (pos[:, 0] >= min(xs)) & (pos[:, 0] <= max(xs)) & (pos[:, 1] >= min(ys)) & (pos[:, 1] <= max(ys))
            for i in np.flatnonzero(near).tolist():
                if i not in found and zone.contains(float(pos[i, 0]), float(pos[i, 1])):
                    found[i] = zone
        return sorted(found.items(), key=lambda item: item[0])

    @staticmethod
    def _landing(zone: Zone, dest: Zone, x: float, y: float, theta: float) -> tuple[float, float, float]:
        """Pose past the destination's seam matching a crossing of the zone's seam at x, y, the side offset kept relative to the heading, on the destination's center line when the offset leaves its footprint."""
        ax, ay = zone.axis
        bx, by = dest.axis
        across = -(x - zone.hold.x) * ay + (y - zone.hold.y) * ax
        lx, ly = dest.hold.x - bx * LANDING_BACKOFF, dest.hold.y - by * LANDING_BACKOFF
        if dest.contains(lx + by * across, ly - bx * across):
            lx, ly = lx + by * across, ly - bx * across
        return lx, ly, math.remainder(theta + dest.entry.theta + math.pi - zone.entry.theta, math.tau)

    @staticmethod
    def _occupied(pool: AgentPool, i: int, x: float, y: float, landed: Sequence[tuple[float, float, float]], skip: Sequence[int] = ()) -> bool:
        """Whether row i standing at x, y would overlap another agent (but the rows in `skip`) or a landing already taken this tick."""
        n, r = pool.n, float(pool.agent_radius[i])
        overlap = np.hypot(pool.pos[:n, 0] - x, pool.pos[:n, 1] - y) < pool.agent_radius[:n] + r
        overlap[i] = False
        if len(skip):
            overlap[list(skip)] = False
        return bool(overlap.any()) or any(math.hypot(lx - x, ly - y) < lr + r for lx, ly, lr in landed)

    def _cross(self, pool: AgentPool, tick: int = 0, dt: float = 0.05) -> list[tuple[int, float, float, float]]:
        """Teleports of the pedestrians past a stair seam whose landing is clear, the others held still at the seam. On a
        half with a `delay` (the stair flight the halves leave out) the pedestrian boards the flight instead (`_fly`). A
        pedestrian whose type does not climb the half's kind is never carried over. Nor is one standing on the half it
        landed on, unless its route takes that half: one that drifts back past the seam after landing walks off."""
        self._held = set()
        teleports: list[tuple[int, float, float, float]] = []
        landed: list[tuple[float, float, float]] = []
        on = self._on_stairs(pool)
        here = {int(pool.agent_ids[i]): zone.name for i, zone in on}
        self._landed = {aid: name for aid, name in self._landed.items() if here.get(aid) == name}
        for i, zone in on:
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            if not zone.destinations or not zone.past_seam(x, y):
                continue
            aid = int(pool.agent_ids[i])
            if aid in self._flights:
                continue
            climbs = self._climbs.get(aid)
            if climbs is not None and zone.kind not in climbs:
                continue
            transfer = self.transfers.get(aid)
            if self._landed.get(aid) == zone.name and (transfer is None or transfer.zone != zone.name):
                continue
            dest = transfer.destination if transfer is not None and transfer.zone == zone.name else zone.destinations[0]
            lx, ly, theta = self._landing(zone, self.zones[dest], x, y, float(pool.theta[i]))
            if zone.delay > 0.0:
                gap = min(FLIGHT_WAIT_GAP, math.hypot(lx - x, ly - y))
                ux, uy = (lx - x, ly - y)
                norm = max(math.hypot(ux, uy), 1e-9)
                wait = (lx - ux / norm * gap, ly - uy / norm * gap)
                self._flights[aid] = _Flight(start=(x, y), end=(lx, ly, theta), t0=tick, ticks=max(1, round(zone.delay / dt)), dest=dest, wait=wait)
                continue
            if self._occupied(pool, i, lx, ly, landed):
                self._held.add(aid)
                ax, ay = zone.axis
                excess = zone.along(x, y) - zone.along(zone.hold.x, zone.hold.y)
                pool.pos[i, 0] = x - excess * ax
                pool.pos[i, 1] = y - excess * ay
                pool.vel[i] = 0.0
                pool.prev_vel[i] = 0.0
                pool.desired_vel[i] = 0.0
                continue
            landed.append((lx, ly, float(pool.agent_radius[i])))
            teleports.append((aid, lx, ly, theta))
            self._landed[aid] = dest
        return teleports

    def hold(self, pool: AgentPool, tick: int, timeout_ticks: int, dt: float, let_out_ticks: int) -> list[tuple[int, float, float, float]]:
        """Advance every cabin rider's phase, slow the waiting ones to desired speed 0, stand the holding ones still and return the due self-teleports, stair crossings included, as id, x, y, heading."""
        robots = self._zone_robots(pool)
        alighting = self._alighting(pool) if let_out_ticks > 0 else set()
        teleports: list[tuple[int, float, float, float]] = []
        for aid in list(self.transfers):
            transfer = self.transfers[aid]
            idx = pool._id_to_idx.get(aid)
            if idx is None:
                self.forget(aid)
                continue
            zone = self.zones[transfer.zone]
            if zone.kind:
                continue
            is_open = self.open[transfer.zone]
            x, y = float(pool.pos[idx, 0]), float(pool.pos[idx, 1])
            covered = self._covered(zone, robots.get(zone.name, ()), float(pool.agent_radius[idx]))
            if transfer.phase == Phase.WALK and math.hypot(x - zone.wait.x, y - zone.wait.y) <= WAIT_RADIUS:
                transfer.phase, transfer.since = Phase.WAIT, tick
            if transfer.phase == Phase.BOARD and transfer.slot in covered:
                slot = self._free_slot(zone, covered)
                if slot is None:
                    transfer.phase, transfer.slot = Phase.WAIT, -1
                else:
                    transfer.slot = slot
            if transfer.phase == Phase.WAIT and not is_open:
                transfer.yielded = 0
            if transfer.phase == Phase.WAIT and is_open:
                if zone.name in alighting and transfer.yielded < let_out_ticks:
                    transfer.yielded += 1
                else:
                    slot = self._free_slot(zone, covered)
                    if slot is not None:
                        transfer.phase, transfer.slot, transfer.since = Phase.BOARD, slot, tick
            elif transfer.phase == Phase.BOARD and self._at_slot(zone, transfer, x, y):
                transfer.phase, transfer.held = Phase.HOLD, tick
            elif transfer.phase == Phase.BOARD and not is_open and not zone.contains(x, y):
                transfer.phase, transfer.slot = Phase.WAIT, -1
            elif transfer.phase == Phase.HOLD and is_open and not zone.contains(x, y):
                transfer.phase = Phase.BOARD
            if transfer.phase != Phase.WALK and tick - transfer.since >= timeout_ticks:
                del self.transfers[aid]
                names = self._banned.setdefault(aid, (transfer.goal, []))[1]
                if transfer.zone in names:
                    names.remove(transfer.zone)
                names.append(transfer.zone)
                continue
            if transfer.phase in (Phase.WAIT, Phase.HOLD):
                pool.desired_vel[idx] = 0.0
            if transfer.phase == Phase.HOLD:
                pool.vel[idx] = 0.0
                pool.prev_vel[idx] = 0.0
            if transfer.phase == Phase.HOLD and zone.delay >= 0 and tick - transfer.held >= round(zone.delay / dt):
                cx, cy = zone.center
                dx, dy = self.zones[transfer.destination].center
                teleports.append((aid, x - cx + dx, y - cy + dy, float(pool.theta[idx])))
        teleports.extend(self._cross(pool, tick, dt))
        teleports.extend(self._fly(pool, tick, timeout_ticks))
        return teleports

    def _fly(self, pool: AgentPool, tick: int, timeout_ticks: int = 0) -> list[tuple[int, float, float, float]]:
        """Move each pedestrian on a stair flight along it, standing: a straight line from its seam crossing towards its
        landing, stopping `FLIGHT_WAIT_GAP` short of it (across the stair's footprint between the halves and through the
        seam walls). Once its time is over it lands, in the order the flights end: on its landing, or on the nearest free
        spot across the partner half (`LANDING_STEP` apart, within the footprint). Waiting fliers never count as standing on
        a landing. A flier that finds no free spot for `timeout_ticks` after its time lands on its own spot anyway."""
        moves: list[tuple[int, float, float, float]] = []
        landed: list[tuple[float, float, float]] = []
        rows = [pool._id_to_idx[a] for a in self._flights if a in pool._id_to_idx]
        for aid in sorted(self._flights, key=lambda a: (self._flights[a].t0 + self._flights[a].ticks, a)):
            flight = self._flights[aid]
            idx = pool._id_to_idx.get(aid)
            if idx is None:
                del self._flights[aid]
                continue
            pool.vel[idx] = 0.0
            pool.prev_vel[idx] = 0.0
            pool.desired_vel[idx] = 0.0
            (sx, sy), (wx, wy), (ex, ey, et) = flight.start, flight.wait, flight.end
            heading = math.atan2(ey - sy, ex - sx)
            share = (tick - flight.t0) / flight.ticks
            if share < 1.0:
                moves.append((aid, sx + (wx - sx) * share, sy + (wy - sy) * share, heading))
                continue
            spot = self._landing_spot(pool, idx, flight, landed, rows)
            if spot is None and tick - (flight.t0 + flight.ticks) >= timeout_ticks > 0:
                spot = (ex, ey)
            if spot is None:
                moves.append((aid, wx, wy, heading))
                continue
            landed.append((spot[0], spot[1], float(pool.agent_radius[idx])))
            moves.append((aid, spot[0], spot[1], et))
            self._landed[aid] = flight.dest
            del self._flights[aid]
        return moves

    def _landing_spot(self, pool: AgentPool, idx: int, flight: _Flight, landed: Sequence[tuple[float, float, float]], fliers: Sequence[int]) -> tuple[float, float] | None:
        """The flight's landing if free, else the nearest free spot across the partner half within its footprint, None
        when every one is taken (by agents that are not flying, or by landings of this tick)."""
        ex, ey, _ = flight.end
        dest = self.zones.get(flight.dest)
        spots = [(ex, ey)]
        if dest is not None and dest.footprint:
            bx, by = dest.axis
            reach = 2.0 * self._mouth(dest)
            steps = int(reach / LANDING_STEP) + 1
            for k in range(1, steps + 1):
                for sign in (1.0, -1.0):
                    x, y = ex + by * sign * k * LANDING_STEP, ey - bx * sign * k * LANDING_STEP
                    if _inside(dest.footprint, x, y):
                        spots.append((x, y))
        for x, y in spots:
            if not self._occupied(pool, idx, x, y, landed, fliers):
                return (x, y)
        return None

    def unreachable(self, zones: Sequence[Zone], walls: Sequence[tuple[tuple[float, float], tuple[float, float]]]) -> list[str]:
        """The stair halves among `zones` no walker can cross: the hold outside the footprint, or closer than
        `HOLD_CLEARANCE` to a wall (the seam trigger lies on the seam wall)."""
        out = []
        segs = np.array([(a[0], a[1], b[0], b[1]) for a, b in walls], dtype=np.float64).reshape(-1, 4)
        for zone in zones:
            if not zone.kind:
                continue
            h = zone.hold
            if not _inside(zone.footprint, h.x, h.y):
                out.append(zone.name)
                continue
            if len(segs):
                d = segs[:, 2:] - segs[:, :2]
                t = np.clip(((h.x - segs[:, 0]) * d[:, 0] + (h.y - segs[:, 1]) * d[:, 1]) / np.maximum((d * d).sum(1), 1e-12), 0.0, 1.0)
                gap = np.hypot(segs[:, 0] + t * d[:, 0] - h.x, segs[:, 1] + t * d[:, 1] - h.y).min()
                if gap < HOLD_CLEARANCE:
                    out.append(zone.name)
        return out

    def _free_slot(self, zone: Zone, covered: set[int]) -> int | None:
        taken = {t.slot for t in self.transfers.values() if t.zone == zone.name and t.phase in (Phase.BOARD, Phase.HOLD)} | covered
        return next((i for i in range(len(zone.slots)) if i not in taken), None)

    def _alighting(self, pool: AgentPool) -> set[str]:
        """Zones someone stands in who is not waiting for, boarding or holding in that zone."""
        out: set[str] = set()
        for i in range(pool.n):
            transfer = self.transfers.get(int(pool.agent_ids[i]))
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            for zone in self.zones.values():
                if zone.name in out or (transfer is not None and transfer.zone == zone.name and transfer.phase != Phase.WALK):
                    continue
                if zone.contains(x, y):
                    out.add(zone.name)
        return out

    def _zone_robots(self, pool: AgentPool) -> dict[str, list[tuple[float, float, float]]]:
        """Robots standing in each cabin as x, y, radius."""
        out: dict[str, list[tuple[float, float, float]]] = {}
        cabins = [z for z in self.zones.values() if not z.kind]
        for i in range(pool.n):
            if pool.kind[i] != KIND_ROBOT:
                continue
            x, y = float(pool.pos[i, 0]), float(pool.pos[i, 1])
            for zone in cabins:
                if zone.contains(x, y):
                    out.setdefault(zone.name, []).append((x, y, float(pool.agent_radius[i])))
        return out

    @staticmethod
    def _covered(zone: Zone, robots: Sequence[tuple[float, float, float]], radius: float) -> set[int]:
        """Spots a pedestrian of this radius cannot stand on because a robot in the zone overlaps them."""
        return {i for i, spot in enumerate(zone.slots) for x, y, r in robots if math.hypot(spot.x - x, spot.y - y) < r + radius}

    @staticmethod
    def _at_slot(zone: Zone, transfer: Transfer, x: float, y: float) -> bool:
        spot = zone.slots[transfer.slot]
        return math.hypot(x - spot.x, y - spot.y) <= HOLD_RADIUS

    def _graph(self, planner: GlobalPlanner) -> _Graph:
        """Teleport edges between areas, an area being the components that one zone's entry and hold pose join."""
        names = sorted(self.zones)
        comps = planner.components([p for name in names for p in (self.zones[name].entry, self.zones[name].hold)])
        parent: dict[int, int] = {}

        def find(c: int) -> int:
            while parent.setdefault(c, c) != c:
                c = parent[c]
            return c

        sides = {name: comps[2 * i] | comps[2 * i + 1] for i, name in enumerate(names)}
        for side in sides.values():
            roots = sorted({find(c) for c in side})
            for r in roots[1:]:
                parent[r] = roots[0]
        area = {c: find(c) for c in parent}
        edges: dict[int, list[tuple[str, str, int]]] = {}
        for name in names:
            if not sides[name]:
                continue
            a = area[min(sides[name])]
            for dest in self.zones[name].destinations:
                if dest not in sides or not sides[dest]:
                    continue
                b = area[min(sides[dest])]
                if a != b:
                    edges.setdefault(a, []).append((name, dest, b))
        return _Graph(area=area, edges=edges)

    def _route(
        self,
        graph: _Graph,
        start: frozenset[int],
        goal: frozenset[int],
        banned: set[str],
        near: tuple[float, float],
        keep: tuple[str, str] | None,
    ) -> tuple[str, str] | None:
        """Zone and destination of the first edge of a fewest-teleports route from start to goal, `keep` when it is one, else the one whose entry is nearest to `near`, None when start reaches goal on foot or no route exists."""
        starts = {graph.area.get(c, c) for c in start}
        goals = {graph.area.get(c, c) for c in goal}
        if not starts or not goals or starts & goals:
            return None
        frontier: dict[int, set[tuple[str, str]]] = {c: set() for c in starts}
        seen = set(starts)
        while frontier:
            reached: dict[int, set[tuple[str, str]]] = {}
            for c, firsts in frontier.items():
                for name, dest, to in graph.edges.get(c, ()):
                    if name in banned or to in seen:
                        continue
                    reached.setdefault(to, set()).update(firsts or {(name, dest)})
            done = set().union(*(firsts for to, firsts in reached.items() if to in goals))
            if done:
                if keep in done:
                    return keep
                return min(done, key=lambda edge: (math.hypot(self.zones[edge[0]].entry.x - near[0], self.zones[edge[0]].entry.y - near[1]), edge))
            seen.update(reached)
            frontier = reached
        return None
