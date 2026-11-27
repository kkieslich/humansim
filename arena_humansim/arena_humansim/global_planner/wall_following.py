from __future__ import annotations

import math
from collections.abc import Callable

import attrs
import numpy as np

#: Spacing of the samples a route is measured and moved at [m]
STEP = 0.5
#: A route nears or leaves a wall by at most this much per metre along it (about 27 degrees)
RAMP = 0.5
#: A wall runs along the route where its distance from the route changes by less than this per metre (45 degrees)
ALONG = 1.0
#: A wall is followed where it runs along the route for longer than the way to it and back times this
WORTH = 1.0
#: The other side than the preferred one costs this much per metre followed
OTHER_SIDE = 0.25
#: A stretch of wall shorter than this is no wall to follow [m]
MIN_RUN = 1.5
#: A corner counts from this turn of the route on [rad]
CORNER = 0.15
#: A sideways move smaller than this is none [m]
SMALL = 0.02
#: A walker waiting before a narrow stretch keeps its body this far to the side of the stretch's mouth [m]
PASSING_GAP = 0.15
#: The approach comes in from beside the mouth to the lane by this much sideways per metre (35 degrees)
ENTRY_RAMP = 0.7
#: The leg onto an approach line leaves the route at most this far before the line [m]
JOIN_REACH = 10.0
#: A corner point added where the route turns away from its wall keeps this much more than the body's clearance from
#: the walls [m]: where walls meet, the social-force wall push holds a body's middle up to about 0.14 m further out than
#: its radius (two walls, strength 3.0, range 0.1, pull 1.5 m/s2), and the body clearance already holds 0.05 of that. A
#: corner point closer than that is one the walker cannot reach. No source beyond that.
CORNER_MARGIN = 0.10
#: A route point where the route turns back on itself by more than this [rad] (120 degrees) is a hairpin: a spike into a
#: corner that the walker would have to turn round on the spot
HAIRPIN = 2.0 * math.pi / 3.0

Cast = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
Clearance = Callable[[np.ndarray], np.ndarray]


@attrs.frozen
class Shaped:
    """A route laid along walls: its points, where it follows a wall, the side (+1 left, -1 right, 0 none), the free
    width across it (inf where a side has no wall within the rays' range), where a walker waiting for the narrow
    stretch ahead stands (`approach`), and for each point where it would lie at the full offset from its wall (`line`:
    the point itself where it follows none or is there already; `approach` and `leave` put the route's ends on it)."""

    points: np.ndarray
    follows: np.ndarray
    side: np.ndarray
    width: np.ndarray
    waits: np.ndarray
    line: np.ndarray

    def keep(self, mask: np.ndarray) -> Shaped:
        return Shaped(self.points[mask], self.follows[mask], self.side[mask], self.width[mask], self.waits[mask], self.line[mask])


def sample(polyline: np.ndarray, step: float = STEP) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` or less along `polyline`, its vertices among them, and the unit direction of the leg each
    lies on (a vertex: the leg it starts; the last point: the leg it ends)."""
    legs = np.diff(polyline, axis=0)
    length = np.hypot(legs[:, 0], legs[:, 1])
    keep = length > 1e-9
    starts, legs, length = polyline[:-1][keep], legs[keep], length[keep]
    if not len(legs):
        return polyline[:1].copy(), np.array([[1.0, 0.0]])
    pieces = np.maximum(np.ceil(length / step - 1e-9).astype(np.int64), 1)
    leg = np.repeat(np.arange(len(legs)), pieces)
    first = np.concatenate([[0], np.cumsum(pieces)[:-1]])
    share = (np.arange(len(leg)) - first[leg]) / pieces[leg]
    points = starts[leg] + share[:, None] * legs[leg]
    tangents = (legs / length[:, None])[leg]
    return np.vstack([points, polyline[-1]]), np.vstack([tangents, tangents[-1]])


def _along(points: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])


def _edges(mask: np.ndarray) -> list[tuple[int, int]]:
    """The runs of True in `mask` as (first index, index after the last)."""
    cut = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8)))
    return list(zip(cut[::2].tolist(), cut[1::2].tolist(), strict=True))


def _runs(mask: np.ndarray, at: np.ndarray, min_run: float) -> np.ndarray:
    """`mask` without its runs shorter than `min_run` along the arc lengths `at`."""
    out = mask.copy()
    for a, b in _edges(mask):
        if at[b - 1] - at[a] < min_run:
            out[a:b] = False
    return out


def _bridge(mask: np.ndarray, d: np.ndarray, known: np.ndarray, at: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """`mask` with its gaps of up to `gap` metres closed, and the wall distance `d` over each closed gap: the measured
    one where `known`, else a straight line between the gap's two ends. So a wall is followed straight across a door or
    a side passage."""
    out, filled = mask.copy(), d.copy()
    held = np.flatnonzero(mask)
    for i, j in zip(held[:-1].tolist(), held[1:].tolist(), strict=True):
        if j - i > 1 and at[j] - at[i] <= gap:
            out[i + 1 : j] = True
            across = np.interp(at[i + 1 : j], [at[i], at[j]], [d[i], d[j]])
            filled[i + 1 : j] = np.where(known[i + 1 : j], d[i + 1 : j], across)
    return out, filled


def _ramp(shift: np.ndarray, at: np.ndarray, slope: float) -> np.ndarray:
    """`shift` lowered so that it starts and ends at 0 and changes by at most `slope` per metre along `at`."""
    out = shift.copy()
    out[0] = out[-1] = 0.0
    gap = np.diff(at) * slope
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1] + gap[i - 1])
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] + gap[i])
    return out


def _spread(shift: np.ndarray, at: np.ndarray, slope: float) -> np.ndarray:
    """`shift` raised so that it falls off by `slope` per metre beyond each stretch: the route starts towards a wall
    before the wall's stretch begins and is at the wall where it does."""
    out = shift.copy()
    gap = np.diff(at) * slope
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] - gap[i - 1])
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1] - gap[i])
    return out


def _choose(cost: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """The cheapest sequence of states (0 none, 1 left, 2 right) along the samples: `cost` per sample and state plus the
    sideways way between neighbouring samples' `shift`s."""
    n = len(cost)
    total = cost[0].copy()
    back = np.zeros((n, 3), dtype=np.int64)
    for i in range(1, n):
        step = total[:, None] + np.abs(shift[i][None, :] - shift[i - 1][:, None])
        back[i] = np.argmin(step, axis=0)
        total = step[back[i], np.arange(3)] + cost[i]
    states = np.zeros(n, dtype=np.int64)
    states[-1] = int(np.argmin(total))
    for i in range(n - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states


def _fits(points: np.ndarray, clearance: Clearance, body: float, axial: float, ends: float, facing: np.ndarray | None = None) -> np.ndarray:
    """Per point of a polyline whether a body keeps its distance from the walls there: `body` at its middle and, for a
    long body, `ends` at the middles of its two end discs, `axial` before and behind along both legs that meet at the
    point (or along `facing`, a unit direction per point)."""
    ok = clearance(points) >= body
    if axial <= 0.0 or len(points) < 2:
        return ok
    if facing is None:
        legs = np.diff(points, axis=0)
        legs = legs / np.maximum(np.hypot(legs[:, 0], legs[:, 1]), 1e-9)[:, None]
        ways = (np.vstack([legs[:1], legs]), np.vstack([legs, legs[-1:]]))
    else:
        ways = (facing,)
    for along in ways:
        ok &= clearance(points + axial * along) >= ends
        ok &= clearance(points - axial * along) >= ends
    return ok


def _legs(points: np.ndarray, clearance: Clearance, body: float, axial: float = 0.0, ends: float = 0.0) -> np.ndarray:
    """Per leg of the polyline whether a body keeps its distance from the walls along it, checked every 0.1 m: `body`
    at its middle and, for a long body (`axial`), `ends` at its two end discs along the leg."""
    legs = np.diff(points, axis=0)
    length = np.hypot(legs[:, 0], legs[:, 1])
    pieces = np.maximum(np.ceil(length / 0.1).astype(np.int64), 1)
    leg = np.concatenate([np.arange(len(legs)), np.repeat(np.arange(len(legs)), pieces)])
    first = np.concatenate([[0], np.cumsum(pieces)[:-1]])
    inner = np.repeat(np.arange(len(legs)), pieces)
    share = (np.arange(len(inner)) - first[inner] + 1) / pieces[inner]
    dense = np.vstack([points[:-1], points[:-1][inner] + share[:, None] * legs[inner]])
    ok = clearance(dense) >= body - 5e-3
    if axial > 0.0:
        along = (legs / np.maximum(length, 1e-9)[:, None])[leg]
        ok &= (clearance(dense + axial * along) >= ends - 5e-3) & (clearance(dense - axial * along) >= ends - 5e-3)
    out = np.ones(len(legs), dtype=bool)
    np.logical_and.at(out, leg, ok)
    return out


def _clear(points: np.ndarray, clearance: Clearance, body: float, axial: float = 0.0, ends: float = 0.0, moved: np.ndarray | None = None) -> bool:
    """Whether a body keeps its distance from the walls along the whole polyline (`_legs`). With `moved` (per point)
    only the legs that touch a moved point count."""
    ok = _legs(points, clearance, body, axial, ends)
    return bool(ok.all() if moved is None else ok[moved[:-1] | moved[1:]].all())


def follow_walls(
    polyline: np.ndarray,
    cast: Cast,
    clearance: Clearance,
    *,
    body: float,
    offset: float,
    reach: float,
    prefer: float,
    either: bool,
    corridor: float = 0.0,
    span: float = 0.0,
    ramp: float = RAMP,
    early: bool = True,
    along: float = ALONG,
    bridge: float = 0.0,
    axial: float = 0.0,
    ends: float = 0.0,
) -> Shaped:
    """`polyline` moved sideways so that it keeps `offset` from a wall wherever one runs along it.

    A wall counts on a side where a ray across the route meets it within `reach`, its distance changing by less than
    `along` per metre over `MIN_RUN` or more, and, with `corridor` set, only between two such walls less than `corridor`
    apart; a gap of up to `bridge` metres in it (a door, a side passage) is crossed straight. `prefer` is the side taken
    where both have a wall (+1 left, -1 right, 0 the nearer); `either` False follows the preferred side only. With
    `either`, a wall is followed where that is worth the way to it and back (`WORTH`), and the route crosses to the
    other side only where that is the shorter way. The route nears and leaves a wall by `ramp` sideways per metre, with
    `early` starting before the wall's stretch where the floor allows, goes round outer corners along the wall, keeps
    its ends, and never comes closer to a wall than `body`; where it would, it moves less. A long body (`axial`: from
    its middle to the middle of each end disc) also keeps `ends` at both end discs, along the route's direction.
    `cast(origins, directions, range)` gives the distance to the first wall along each ray, `clearance(points)` the
    distance to the nearest wall. Rays reach `span` at least, for the widths the result carries. With `offset` 0 the
    route is only sampled and measured."""
    points, tangents = sample(polyline)
    n = len(points)
    left = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    far = max(reach, corridor, span) + offset
    d = np.column_stack([cast(points, left, far), cast(points, -left, far)])
    hit = d < far - 1e-9
    width = np.where(hit.all(axis=1), d.sum(axis=1), np.inf)
    at = _along(points)
    none = Shaped(points, np.zeros(n, dtype=bool), np.zeros(n), width, np.zeros(n, dtype=bool), points)
    if n < 3 or offset <= 0.0:
        return none

    gap = np.maximum(np.diff(at), 1e-9)
    change = np.abs(np.diff(d, axis=0)) / gap[:, None]
    steady = np.ones((n, 2), dtype=bool)
    steady[1:] &= change <= along
    steady[:-1] |= change <= along
    steady[0] = change[0] <= along
    steady &= hit
    allowed = (d <= reach) & steady
    if corridor > 0.0:
        allowed &= (steady.all(axis=1) & (width < corridor))[:, None]
    sides = (1.0, -1.0)
    for k in range(2):
        if not either and sides[k] != prefer:
            allowed[:, k] = False
        if bridge > 0.0:
            allowed[:, k], d[:, k] = _bridge(allowed[:, k], d[:, k], hit[:, k] & (d[:, k] <= reach), at, bridge)
        allowed[:, k] = _runs(allowed[:, k], at, MIN_RUN)
    if not allowed.any():
        return none

    weight = np.concatenate([[gap[0] / 2], (gap[:-1] + gap[1:]) / 2, [gap[-1] / 2]])
    want = np.column_stack([np.zeros(n), np.maximum(d[:, 0] - offset, 0.0), -np.maximum(d[:, 1] - offset, 0.0)])
    cost = np.full((n, 3), np.inf)
    cost[:, 0] = np.where(allowed.any(axis=1), weight * (WORTH if either else 1e6), 0.0)
    for k in range(2):
        other = OTHER_SIDE if prefer not in (0.0, sides[k]) else 0.0
        cost[:, k + 1] = np.where(allowed[:, k], weight * other, np.inf)
    cost[0, 1:] = cost[-1, 1:] = np.inf
    cost[0, 0] = cost[-1, 0] = 0.0
    states = _choose(cost, want)
    full = want[np.arange(n), states]
    shift = full.copy()
    if early:
        shift = _spread(np.maximum(shift, 0.0), at, ramp) - _spread(np.maximum(-shift, 0.0), at, ramp)
    for _ in range(12):
        shift = _ramp(np.maximum(shift, 0.0), at, ramp) - _ramp(np.maximum(-shift, 0.0), at, ramp)
        shift = np.where(np.abs(shift) < SMALL, 0.0, shift)
        there = points + shift[:, None] * left
        cut = ~_legs(there, clearance, body, axial, ends)
        tight = (~_fits(there, clearance, body, axial, ends) | np.concatenate([[False], cut]) | np.concatenate([cut, [False]])) & (shift != 0.0)
        if not tight.any():
            break
        shift = np.where(tight, 0.5 * shift, shift)
    else:
        return none
    moved = points + shift[:, None] * left
    lined = points + full[:, None] * left
    line = np.where((_fits(lined, clearance, body, axial, ends, tangents) & (states != 0))[:, None], lined, moved)
    for corners in (True, False):
        out, follows, side, wide, changed, on = _corners(moved, line, tangents, states, shift, width, clearance, body, far, corners)
        if _clear(out, clearance, body, axial, ends, changed):
            return Shaped(out, follows, side, wide, np.zeros(len(out), dtype=bool), on)
    return none


def _corners(moved: np.ndarray, line: np.ndarray, tangents: np.ndarray, states: np.ndarray, shift: np.ndarray, width: np.ndarray, clearance: Clearance, body: float, far: float, add: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The moved route with its sides, widths, which points moved and each point's place at the full offset, and with
    `add` the corner of the wall's offset line added where the route turns away from the wall it follows, so that it
    goes round the outer corner along the wall instead of cutting it."""
    side = np.where(states == 1, 1.0, np.where(states == 2, -1.0, 0.0))
    pts: list[np.ndarray] = []
    on: list[np.ndarray] = []
    follows: list[bool] = []
    sides: list[float] = []
    wide: list[float] = []
    changed: list[bool] = []
    n = len(moved)
    for i in range(n):
        pts.append(moved[i])
        on.append(line[i])
        follows.append(bool(states[i] != 0))
        sides.append(float(side[i]))
        wide.append(float(width[i]))
        changed.append(bool(shift[i] != 0.0))
        if not add or i == n - 1 or states[i] == 0 or states[i] != states[i + 1] or shift[i] == 0.0 or shift[i + 1] == 0.0:
            continue
        a, b = tangents[i], tangents[i + 1]
        turn = a[0] * b[1] - a[1] * b[0]
        if abs(turn) < math.sin(CORNER) or turn * side[i] > 0.0:
            continue
        gap = moved[i + 1] - moved[i]
        u = (gap[0] * b[1] - gap[1] * b[0]) / turn
        v = (a[0] * gap[1] - a[1] * gap[0]) / turn
        if not (0.0 < u <= 2.0 * far and 0.0 < v <= 2.0 * far):
            continue
        corner = moved[i] + u * a
        if clearance(corner[None])[0] >= body + CORNER_MARGIN:
            pts.append(corner)
            on.append(corner)
            follows.append(True)
            sides.append(float(side[i]))
            wide.append(float(width[i]))
            changed.append(True)
    return np.array(pts), np.array(follows, dtype=bool), np.array(sides), np.array(wide), np.array(changed, dtype=bool), np.array(on)


def _join(points: np.ndarray, at: np.ndarray, spot: np.ndarray, axis: np.ndarray, candidates: range, clearance: Clearance, body: float, axial: float, ends: float, slope: float) -> int | None:
    """The route point among `candidates` (nearest first) from which a straight leg to `spot` is clear and makes
    `slope` or less sideways per metre against `axis`; where none is that flat, the flattest clear one; None where no
    leg is clear within `JOIN_REACH` along the route."""
    best, flattest = None, np.inf
    for j in candidates:
        if abs(at[j] - at[candidates[0]]) > JOIN_REACH:
            break
        leg = spot - points[j]
        along, across = abs(float(leg @ axis)), abs(float(leg[0] * axis[1] - leg[1] * axis[0]))
        steep = across / along if along > 1e-9 else np.inf
        if steep >= flattest or not _clear(np.vstack([points[j], spot]), clearance, body, axial, ends):
            continue
        best, flattest = j, steep
        if steep <= slope:
            break
    return best


def _splice(shaped: Shaped, stop: int, block: np.ndarray, go: int, side: float, waits: np.ndarray, stretch: np.ndarray) -> tuple[Shaped, np.ndarray]:
    """`shaped` with its points from `stop` up to `go` replaced by `block` (followed, on `side`, on their line), and the
    stretch mask with it."""
    count = len(block)
    return (
        Shaped(
            np.vstack([shaped.points[:stop], block, shaped.points[go:]]),
            np.concatenate([shaped.follows[:stop], np.ones(count, dtype=bool), shaped.follows[go:]]),
            np.concatenate([shaped.side[:stop], np.full(count, side), shaped.side[go:]]),
            np.concatenate([shaped.width[:stop], np.full(count, np.inf), shaped.width[go:]]),
            np.concatenate([shaped.waits[:stop], waits, shaped.waits[go:]]),
            np.vstack([shaped.line[:stop], block, shaped.line[go:]]),
        ),
        np.concatenate([stretch[:stop], np.zeros(count, dtype=bool), stretch[go:]]),
    )


def approach(
    shaped: Shaped,
    stretch: np.ndarray,
    clearance: Clearance,
    *,
    body: float,
    length: float,
    lateral: float = 0.0,
    prefer: float = 0.0,
    waits: bool = False,
    ramp: float = ENTRY_RAMP,
    ramp_in: float = 1.0,
    axial: float = 0.0,
    ends: float = 0.0,
) -> Shaped:
    """`shaped` with a straight approach before each stretch (`stretch`: per point whether it belongs to one): for
    `length` before the stretch the route runs along the stretch's own direction and enters it on its line
    (`Shaped.line`), so that a walker that cannot sidestep is lined up before it enters, on its own side of the mouth,
    and those waiting for it stand in a line. With `lateral` the approach lies that far out on the `prefer` side,
    beside the mouth rather than in front of it, where the floor allows, and comes in to the stretch's line by `ramp`
    at the entry; with `waits` its points before it comes in are marked as where a walker waits (`Shaped.waits`). The
    route leaves its old line early enough to reach the approach at `ramp_in` sideways per metre or less where a clear
    leg allows (within `JOIN_REACH`), else by the flattest clear leg. The approach is cut where the floor ends, and left
    out where no clear leg leads onto it; the route then comes in as `follow_walls` laid it. Its points count as
    followed."""
    for a, b in _edges(stretch)[::-1]:
        points, line = shaped.points, shaped.line
        if a == 0 or a + 1 >= len(points):
            continue
        at = _along(points)
        head = a
        while head < b and np.hypot(*(line[head] - points[head])) > SMALL:
            head += 1
        ahead = max(min(int(np.searchsorted(at, at[a] + STEP)), len(points) - 1), a + 1)
        axis = line[ahead] - line[a]
        span = float(np.hypot(axis[0], axis[1]))
        if span <= 1e-9:
            continue
        axis = axis / span
        out = prefer * np.array([-axis[1], axis[0]])
        before = int(np.flatnonzero(stretch[:a])[-1]) if stretch[:a].any() else 0
        room = min(length, float(at[a] - at[before]) - (STEP if before else 0.0))
        if float(np.hypot(*(line[a] - points[0]))) <= length + STEP:
            room = min(room, float((line[a] - points[0]) @ axis) - STEP / 2.0)
        xs = np.arange(STEP, room + 1e-9, STEP)
        if not len(xs):
            continue
        shift = np.minimum(lateral if prefer != 0.0 else 0.0, xs * ramp)
        ok = np.ones(len(xs), dtype=bool)
        for k in range(len(xs)):
            if k:
                shift[k] = min(shift[k], shift[k - 1] + ramp * STEP)
            for share in (1.0, 0.5, 0.0):
                spot = line[a] - xs[k] * axis + shift[k] * share * out
                if clearance(spot[None])[0] >= body and clearance(np.array([spot + axial * axis, spot - axial * axis])).min() >= ends:
                    shift[k] *= share
                    break
            else:
                ok[k] = False
        way_in = line[a] - xs[:, None] * axis + shift[:, None] * out
        count = int(np.argmin(ok)) if not ok.all() else len(xs)
        tail = np.vstack([line[a:head], points[head : head + 1]]) if head < len(points) else line[a:head]
        while count and not _clear(np.vstack([way_in[:count][::-1], tail]), clearance, body, axial, ends):
            count -= 1
        if count == 0:
            continue
        block = way_in[:count][::-1]
        first = max(int(np.searchsorted(at, at[a] - xs[count - 1] - STEP / 2.0, side="right")) - 1, before)
        old = points[first + 1 : a]
        if head == a and len(old) == count and np.hypot(*(old - block).T).max() < SMALL:
            continue
        join = _join(points, at, block[0], axis, range(first, before - 1, -1), clearance, body, axial, ends, ramp_in)
        if join is None:
            continue
        stands = (shift[:count] >= shift[:count].max() - 1e-9)[::-1] if waits else np.zeros(count, dtype=bool)
        if head > a:
            shaped = attrs.evolve(shaped, points=np.vstack([points[:a], line[a:head], points[head:]]))
        shaped, stretch = _splice(shaped, join + 1, block, a, prefer, stands, stretch)
    return shaped


def leave(shaped: Shaped, stretch: np.ndarray, clearance: Clearance, *, body: float, length: float, ramp_out: float = 1.0, axial: float = 0.0, ends: float = 0.0) -> Shaped:
    """`shaped` with a straight way out after each stretch (`stretch`: per point whether it belongs to one): the route
    stays on the stretch's line (`Shaped.line`) to its end and runs on along its direction for `length` beyond, as far
    as the floor allows, so that a walker that cannot sidestep leaves on its own side of the mouth with its body in
    line, and only then turns for its old route, by `ramp_out` sideways per metre or less where a clear leg allows,
    else by the flattest clear leg. Left out where no clear leg leads back; the route then leaves as `follow_walls` laid
    it. Its points count as followed."""
    for a, b in _edges(stretch)[::-1]:
        points, line = shaped.points, shaped.line
        last = b - 1
        if last <= 0 or b >= len(points) - 1:
            continue
        at = _along(points)
        tail = b
        while tail > a and np.hypot(*(line[tail - 1] - points[tail - 1])) > SMALL:
            tail -= 1
        behind = min(max(int(np.searchsorted(at, at[last] - STEP, side="right")) - 1, 0), last - 1)
        axis = line[last] - line[behind]
        span = float(np.hypot(axis[0], axis[1]))
        if span <= 1e-9:
            continue
        axis = axis / span
        after = b + int(np.flatnonzero(stretch[b:])[0]) if stretch[b:].any() else len(points) - 1
        room = min(length, float(at[after] - at[last]) - STEP)
        if float(np.hypot(*(points[-1] - line[last]))) <= length + STEP:
            room = min(room, float((points[-1] - line[last]) @ axis) - STEP / 2.0)
        xs = np.arange(STEP, room + 1e-9, STEP)
        if not len(xs):
            continue
        way_out = line[last] + xs[:, None] * axis
        ok = (clearance(way_out) >= body) & (clearance(way_out + axial * axis) >= ends) & (clearance(way_out - axial * axis) >= ends)
        count = int(np.argmin(ok)) if not ok.all() else len(xs)
        lead = np.vstack([points[tail - 1 : tail], line[tail:b]]) if tail > 0 else line[tail:b]
        while count and not _clear(np.vstack([lead, way_out[:count]]), clearance, body, axial, ends):
            count -= 1
        if count == 0:
            continue
        block = way_out[:count]
        first = min(int(np.searchsorted(at, at[last] + xs[count - 1] + STEP / 2.0)), after)
        old = points[b:first]
        if tail == b and len(old) == count and np.hypot(*(old - block).T).max() < SMALL:
            continue
        join = _join(points, at, block[-1], axis, range(first, after + 1), clearance, body, axial, ends, ramp_out)
        if join is None:
            continue
        if tail < b:
            shaped = attrs.evolve(shaped, points=np.vstack([points[:tail], line[tail:b], points[b:]]))
        shaped, stretch = _splice(shaped, b, block, join, float(shaped.side[last]), np.zeros(count, dtype=bool), stretch)
    return shaped


def unspike(shaped: Shaped, clearance: Clearance, *, body: float, axial: float = 0.0, ends: float = 0.0) -> Shaped:
    """`shaped` without its hairpins: a point where the route turns back on itself by more than `HAIRPIN` is dropped
    where the straight leg between its neighbours keeps the body clear of the walls (`_legs`), so the route turns the
    corner instead of running into it and back. Its ends stay."""
    p = shaped.points
    keep = np.ones(len(p), dtype=bool)
    if len(p) < 3:
        return shaped
    changed = True
    while changed:
        changed = False
        idx = np.flatnonzero(keep)
        for k in range(1, len(idx) - 1):
            i, j, m = idx[k - 1], idx[k], idx[k + 1]
            a, b = p[j] - p[i], p[m] - p[j]
            na, nb = float(np.hypot(a[0], a[1])), float(np.hypot(b[0], b[1]))
            if na < 1e-9 or nb < 1e-9 or float(a @ b) / (na * nb) >= math.cos(HAIRPIN):
                continue
            if _legs(np.vstack([p[i], p[m]]), clearance, body, axial, ends).all():
                keep[j] = False
                changed = True
                break
    return shaped if keep.all() else shaped.keep(keep)


def thin(shaped: Shaped, narrow: float = 0.0) -> Shaped:
    """`shaped` without the points that lie on the straight line between their neighbours and change nothing: same side
    as both, on the same side of the `narrow` width, and waiting or not as both."""
    p = shaped.points
    if len(p) < 3:
        return shaped
    tight = shaped.width < narrow
    a, b = p[1:-1] - p[:-2], p[2:] - p[1:-1]
    straight = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) <= 1e-6 * np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-12)
    straight &= (a * b).sum(axis=1) > 0.0
    same = np.ones(len(p) - 2, dtype=bool)
    for flag in (shaped.side, tight, shaped.waits, shaped.follows):
        same &= (flag[1:-1] == flag[:-2]) & (flag[1:-1] == flag[2:])
    return shaped.keep(np.concatenate([[True], ~(straight & same), [True]]))
