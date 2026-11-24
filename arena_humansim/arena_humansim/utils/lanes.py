"""Pedestrian lanes: free-space bands per station, per-walker tracks and their checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import attrs
import numpy as np
from numpy.typing import NDArray

from arena_humansim.utils.geometry import point_in_polygon
from arena_humansim.utils.types import (
    Lane,
    LaneConfig,
    LaneSettings,
    LaneSide,
    Pose2D,
    Segments,
    Shape,
    ShapeType,
    SinkConfig,
    SourceConfig,
    Track,
)

MAX_TRACK_SPACING = 3.0
JOIN_TOLERANCE = 0.05
STATION_DIVISIONS = 4
GRID_EPS = 1e-9
VERTEX_EPS = 1e-9
WALL_DISTANCE_CHUNK = 4096
RAY_CHUNK = 1_000_000
BAND_SAMPLES = 41
DEFAULT_AGENT_RADIUS = 0.35
MIN_ADJACENT_RUN_WIDTH = 2.0

Run = tuple[float, float]


def resolve_clearance(inflation: float, resolution: float) -> float:
    cells = math.ceil(inflation / resolution - GRID_EPS)
    return cells * resolution


def truncated_normal(sigma: float, rng: np.random.Generator) -> float:
    while True:
        value = float(rng.normal(0.0, sigma))
        if -1.0 <= value <= 1.0:
            return value


def truncated_sd_factor(sigma_q: float) -> float:
    a = 1.0 / sigma_q
    pdf = math.exp(-a * a / 2.0) / math.sqrt(2.0 * math.pi)
    return math.sqrt(1.0 - 2.0 * a * pdf / math.erf(a / math.sqrt(2.0)))


def segments_array(walls: Segments) -> NDArray[np.float64]:
    return np.asarray(walls, dtype=np.float64).reshape(-1, 2, 2)


def wall_distance(points: NDArray[np.float64], segments: NDArray[np.float64]) -> NDArray[np.float64]:
    a = segments[:, 0, :]
    ab = segments[:, 1, :] - a
    length_sq = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=2) / length_sq[None], 0.0, 1.0)
    closest = a[None] + t[:, :, None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def derive_track_spacing(vertices: NDArray[np.float64], chord_tolerance: float) -> float:
    spacing = MAX_TRACK_SPACING
    for a, b, c in zip(vertices[:-2], vertices[1:-1], vertices[2:], strict=True):
        cross = abs((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])
        if cross < 1e-12:
            continue
        radius = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(a - c) / (2.0 * cross)
        spacing = min(spacing, math.sqrt(8.0 * chord_tolerance * radius))
    return spacing


def sigma_meters(lane: Lane) -> float:
    sigma = lane.settings.sigma_q
    return sigma * truncated_sd_factor(sigma) * float(np.median(lane.half_width))


def build_lane(
    config: LaneConfig,
    walls: Segments,
    *,
    flow: str,
    agent_radius: float,
    clearance: float,
    source_extent: float,
) -> Lane:
    _check_inputs(config, walls, flow, agent_radius, clearance)
    vertices, junctions = _polyline(config, flow)
    seg_len = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    vertex_s = np.concatenate([[0.0], np.cumsum(seg_len)])
    length = float(vertex_s[-1])
    _check_turns(vertices, seg_len, flow)

    spacing = _track_spacing(config, vertices, flow)
    arrival = config.arrival_radius if config.arrival_radius is not None else spacing / 3.0
    if not 0.0 < arrival < spacing / 2.0:
        raise ValueError(f"flow {flow!r}: arrival radius must lie below half the track spacing")
    separation = agent_radius if config.two_way else 0.0
    if config.two_way and config.offset is not None and abs(config.offset) < separation:
        raise ValueError(f"flow {flow!r}: a two-way offset must lie outside the separation zone")
    spawn_length = config.spawn_length if config.spawn_length is not None else max(0.0, source_extent - arrival)
    if spawn_length >= length:
        raise ValueError(f"flow {flow!r}: spawn length {spawn_length:.2f} m must be shorter than the lane ({length:.2f} m)")
    sweep = max(config.sweep_length if config.sweep_length is not None else spacing, spacing)

    station_step = min(spacing / STATION_DIVISIONS, config.chord_tolerance)
    stations = np.unique(np.concatenate([np.arange(0.0, length, station_step), vertex_s, [length]]))
    x, y, nx, ny = _frame(vertex_s, vertices, stations)
    segments = segments_array(walls)
    chord = config.chord_tolerance
    wall_radius = clearance + chord
    offsets = _offset_grid(x, y, nx, ny, segments, wall_radius, separation, chord)
    grid_reach = float(offsets[-1])
    wall_blocked = _wall_blocked_grid(x, y, nx, ny, offsets, segments, wall_radius)
    swept_free = ~_sweep(wall_blocked, stations, sweep / 2.0)
    right, left = _hallway_edges(swept_free, offsets, stations, flow)
    zone = separation + chord if config.two_way else 0.0
    lows, highs = _station_runs(swept_free, offsets, config, right, left, zone, stations, flow)
    center = (lows + highs) / 2.0
    if config.offset is not None:
        inside = (lows - GRID_EPS <= config.offset) & (config.offset <= highs + GRID_EPS)
        center = np.where(inside, config.offset, center)
    half = np.minimum(center - lows, highs - center)
    center = _window_mean(center, stations, sweep / 2.0)
    half = _window_mean(half, stations, sweep / 2.0)
    center, half = _clip(center, half, lows, highs)

    center, half, merge_lengths = _blend_junctions(config, center, half, stations, vertex_s, seg_len, junctions, flow)
    center, half = _clip(center, half, lows, highs)

    settings = LaneSettings(
        two_way=config.two_way,
        side=config.side,
        sigma_q=config.sigma_q,
        chord_tolerance=chord,
        track_spacing=spacing,
        arrival_radius=arrival,
        merge_length=config.merge_length if config.merge_length is not None else max(merge_lengths, default=0.0),
        spawn_length=spawn_length,
        sweep_length=sweep,
        separation=separation,
        clearance=clearance,
    )
    lane = Lane(
        flow=flow,
        settings=settings,
        s=stations,
        x=x,
        y=y,
        nx=nx,
        ny=ny,
        center=center,
        half_width=half,
        vertex_s=vertex_s,
        vertex_x=vertices[:, 0],
        vertex_y=vertices[:, 1],
        length=length,
        goal_s=np.empty(0),
        grid_reach=grid_reach,
    )
    lane = attrs.evolve(lane, goal_s=goal_corners(lane))
    check_band(lane, walls)
    return lane


def goal_corners(lane: Lane, others: Sequence[Lane] | None = None) -> NDArray[np.float64]:
    """Arc lengths of the corners where a track keeps its goal: outside corners of two-way lanes, with a lane inside the turn."""
    if not lane.settings.two_way:
        return np.empty(0)
    found: list[float] = []
    for k in range(1, len(lane.vertex_x) - 1):
        cross = _vertex_cross(lane, k)
        if abs(cross) <= GRID_EPS:
            continue
        turn = math.copysign(1.0, cross)
        s_vertex = float(lane.vertex_s[k])
        center = float(np.interp(s_vertex, lane.s, lane.center))
        if center * turn >= 0.0:
            continue
        if others is None or any(_band_enters_turn(lane, k, other) for other in others if other is not lane):
            found.append(s_vertex)
    return np.array(found, dtype=np.float64)


def with_goal_corners(lane: Lane, others: Sequence[Lane]) -> Lane:
    """The lane with its goal corners limited to the corners where another lane is inside the turn."""
    goals = goal_corners(lane, others)
    return lane if np.array_equal(goals, lane.goal_s) else attrs.evolve(lane, goal_s=goals)


def lane_affected(lane: Lane, changed: NDArray[np.float64]) -> bool:
    """Whether a changed segment lies within the lane's grid reach of its stations."""
    if len(changed) == 0:
        return False
    near = lane.grid_reach + lane.settings.clearance + lane.settings.chord_tolerance
    return bool((_min_wall_distance(np.column_stack([lane.x, lane.y]), changed) <= near).any())


def _vertex_cross(lane: Lane, k: int) -> float:
    """Cross product of the legs entering and leaving vertex k, positive for a left turn."""
    t_in_x = float(lane.vertex_x[k] - lane.vertex_x[k - 1])
    t_in_y = float(lane.vertex_y[k] - lane.vertex_y[k - 1])
    t_out_x = float(lane.vertex_x[k + 1] - lane.vertex_x[k])
    t_out_y = float(lane.vertex_y[k + 1] - lane.vertex_y[k])
    return t_in_x * t_out_y - t_in_y * t_out_x


def _corner_reach(lane: Lane, k: int) -> float:
    """Distance from vertex k within which the inside of a corner can be walked: far band edge times miter length, plus arrival radius."""
    t_in = _unit_vector(float(lane.vertex_x[k] - lane.vertex_x[k - 1]), float(lane.vertex_y[k] - lane.vertex_y[k - 1]))
    t_out = _unit_vector(float(lane.vertex_x[k + 1] - lane.vertex_x[k]), float(lane.vertex_y[k + 1] - lane.vertex_y[k]))
    n_in = (-t_in[1], t_in[0])
    n_out = (-t_out[1], t_out[0])
    denominator = 1.0 + n_in[0] * n_out[0] + n_in[1] * n_out[1]
    miter = math.hypot((n_in[0] + n_out[0]) / denominator, (n_in[1] + n_out[1]) / denominator)
    s_vertex = float(lane.vertex_s[k])
    center = float(np.interp(s_vertex, lane.s, lane.center))
    half = float(np.interp(s_vertex, lane.s, lane.half_width))
    return max(abs(center - half), abs(center + half)) * miter + lane.settings.arrival_radius


def _band_enters_turn(lane: Lane, k: int, other: Lane) -> bool:
    """Whether a point of the other lane's band lies inside the turn at vertex k, within reach of that corner."""
    t_in = _unit_vector(float(lane.vertex_x[k] - lane.vertex_x[k - 1]), float(lane.vertex_y[k] - lane.vertex_y[k - 1]))
    t_out = _unit_vector(float(lane.vertex_x[k + 1] - lane.vertex_x[k]), float(lane.vertex_y[k + 1] - lane.vertex_y[k]))
    turn = math.copysign(1.0, _vertex_cross(lane, k))
    lateral = np.linspace(-1.0, 1.0, BAND_SAMPLES)
    offsets = other.center[:, None] + other.half_width[:, None] * lateral[None, :]
    px = (other.x[:, None] + offsets * other.nx[:, None]).ravel() - lane.vertex_x[k]
    py = (other.y[:, None] + offsets * other.ny[:, None]).ravel() - lane.vertex_y[k]
    near = np.hypot(px, py) <= _corner_reach(lane, k)
    inside = (turn * (t_in[0] * py - t_in[1] * px) > 0.0) & (turn * (t_out[0] * py - t_out[1] * px) > 0.0)
    return bool(np.any(near & inside))


def build_track(lane: Lane, q: float, s0: float) -> Track:
    spacing = lane.settings.track_spacing
    corners = lane.goal_s
    vertices = corners[corners > s0 + VERTEX_EPS]
    grid = np.arange(s0 + spacing, lane.length, spacing)
    grid = grid[~_within_any(grid, _corner_folds(lane, q))]
    protected = np.unique(np.concatenate([[s0], vertices, [lane.length]]))
    stops = np.unique(np.concatenate([protected, grid]))
    center = np.interp(stops, lane.s, lane.center)
    half = np.interp(stops, lane.s, lane.half_width)
    x, y, nx, ny = _frame(lane.vertex_s, np.column_stack([lane.vertex_x, lane.vertex_y]), stops)
    lateral = center + q * half
    xy = np.column_stack([x + lateral * nx, y + lateral * ny])
    grid_stop = ~np.isin(stops, protected)
    anchors = xy[~grid_stop]
    near_anchor = np.min(np.linalg.norm(xy[:, None, :] - anchors[None, :, :], axis=2), axis=1) < lane.settings.arrival_radius
    xy = xy[~(grid_stop & near_anchor)]
    kept = _spaced_points(xy, lane.settings.arrival_radius)
    points = [Pose2D(x=float(px), y=float(py)) for px, py in xy[kept]]
    return Track(points=points, radii=[lane.settings.arrival_radius] * len(points))


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def _corner_folds(lane: Lane, q: float) -> list[tuple[float, float]]:
    """Arc intervals beside each corner where the offset track folds back on itself."""
    folds: list[tuple[float, float]] = []
    vx, vy = lane.vertex_x, lane.vertex_y
    for k in range(1, len(vx) - 1):
        t_in = _unit_vector(float(vx[k] - vx[k - 1]), float(vy[k] - vy[k - 1]))
        t_out = _unit_vector(float(vx[k + 1] - vx[k]), float(vy[k + 1] - vy[k]))
        n_in = (-t_in[1], t_in[0])
        n_out = (-t_out[1], t_out[0])
        s_vertex = float(lane.vertex_s[k])
        lateral = float(np.interp(s_vertex, lane.s, lane.center) + q * np.interp(s_vertex, lane.s, lane.half_width))
        denominator = max(1.0 + n_in[0] * n_out[0] + n_in[1] * n_out[1], GRID_EPS)
        along_in = lateral * (n_out[0] * t_in[0] + n_out[1] * t_in[1]) / denominator
        along_out = lateral * (n_in[0] * t_out[0] + n_in[1] * t_out[1]) / denominator
        if along_in < 0.0:
            folds.append((s_vertex + along_in, s_vertex))
        if along_out > 0.0:
            folds.append((s_vertex, s_vertex + along_out))
    return folds


def _within_any(values: NDArray[np.float64], intervals: list[tuple[float, float]]) -> NDArray[np.bool_]:
    inside = np.zeros(len(values), dtype=np.bool_)
    for low, high in intervals:
        inside |= (values > low + GRID_EPS) & (values < high - GRID_EPS)
    return inside


def _spaced_points(xy: NDArray[np.float64], minimum_gap: float) -> NDArray[np.intp]:
    """Indices of the kept track points, the first and last always kept and no two closer than minimum_gap."""
    kept = [0]
    for index in range(1, len(xy) - 1):
        if float(np.hypot(*(xy[index] - xy[kept[-1]]))) >= minimum_gap:
            kept.append(index)
    last = len(xy) - 1
    while len(kept) > 1 and float(np.hypot(*(xy[last] - xy[kept[-1]]))) < minimum_gap:
        kept.pop()
    kept.append(last)
    return np.array(kept, dtype=np.intp)


def check_band(lane: Lane, walls: Segments) -> None:
    bounds = np.stack([lane.center - lane.half_width, lane.center, lane.center + lane.half_width])
    px = lane.x[None, :] + bounds * lane.nx[None, :]
    py = lane.y[None, :] + bounds * lane.ny[None, :]
    distance = _min_wall_distance(np.column_stack([px.ravel(), py.ravel()]), segments_array(walls)).reshape(3, -1)
    bad = np.nonzero((distance <= lane.settings.clearance).any(axis=0))[0]
    if bad.size:
        raise ValueError(f"flow {lane.flow!r}: band reaches the clearance of a wall at station {lane.s[bad[0]]:.2f} m")


def check_endpoints(lane: Lane, source: SourceConfig, sink: SinkConfig) -> None:
    for index in np.nonzero(lane.s <= lane.settings.spawn_length + GRID_EPS)[0]:
        for x, y in _band_edges(lane, int(index)):
            if not _inside_source(source, x, y):
                raise ValueError(f"flow {lane.flow!r}: spawn band leaves source {source.name!r} at station {lane.s[index]:.2f} m")
    for x, y in _band_edges(lane, len(lane.s) - 1):
        if not _inside_sink(sink, x, y):
            raise ValueError(f"flow {lane.flow!r}: end band leaves sink {sink.name!r}")


def resolve_lane(
    source: SourceConfig,
    walls: Segments,
    sinks: Mapping[str, SinkConfig],
    clearance: float,
) -> Lane:
    config = source.lane_config
    if config is None:
        raise ValueError(f"flow {source.name!r}: the source has no lane")
    if len(source.agent.sink_affinity) != 1:
        raise ValueError(f"flow {source.name!r}: a lane flow needs exactly one sink affinity")
    sink_name = source.agent.sink_affinity[0].sink_name
    if sink_name not in sinks:
        raise ValueError(f"flow {source.name!r}: sink {sink_name!r} is not declared")
    tangent = _first_tangent(config, source.name)
    agent_radius = source.agent.agent_radius if source.agent.agent_radius > 0.0 else DEFAULT_AGENT_RADIUS
    lane = build_lane(
        config,
        walls,
        flow=source.name,
        agent_radius=agent_radius,
        clearance=clearance,
        source_extent=_source_extent(source.shape, tangent),
    )
    check_endpoints(lane, source, sinks[sink_name])
    return lane


def _check_inputs(config: LaneConfig, walls: Segments, flow: str, agent_radius: float, clearance: float) -> None:
    if not walls:
        raise ValueError(f"flow {flow!r}: the lane needs walls to keep it in")
    if config.sigma_q <= 0.0:
        raise ValueError(f"flow {flow!r}: sigma_q must be positive")
    if config.chord_tolerance <= 0.0:
        raise ValueError(f"flow {flow!r}: chord_tolerance must be positive")
    if agent_radius <= 0.0 or clearance <= 0.0:
        raise ValueError(f"flow {flow!r}: agent radius and clearance must be positive")
    overrides = {
        "track_spacing": config.track_spacing,
        "arrival_radius": config.arrival_radius,
        "merge_length": config.merge_length,
        "spawn_length": config.spawn_length,
        "sweep_length": config.sweep_length,
    }
    for name, value in overrides.items():
        if value is not None and value <= 0.0:
            raise ValueError(f"flow {flow!r}: {name} must be positive when given")


def _polyline(config: LaneConfig, flow: str) -> tuple[NDArray[np.float64], list[int]]:
    if not config.centerlines:
        raise ValueError(f"flow {flow!r}: the lane has no segments")
    vertices: list[tuple[float, float]] = []
    junctions: list[int] = []
    for index, centerline in enumerate(config.centerlines):
        if len(centerline) < 2:
            raise ValueError(f"flow {flow!r}: segment {index} needs at least two points")
        points = [(point.x, point.y) for point in centerline]
        if vertices:
            gap = math.dist(vertices[-1], points[0])
            if gap > JOIN_TOLERANCE:
                raise ValueError(f"flow {flow!r}: segment {index} does not connect to the previous one (gap {gap:.3f} m)")
            junctions.append(len(vertices) - 1)
            points = points[1:]
        for point in points:
            if not vertices or math.dist(vertices[-1], point) > GRID_EPS:
                vertices.append(point)
    if len(vertices) < 2:
        raise ValueError(f"flow {flow!r}: the lane has no length")
    if any(junction >= len(vertices) - 1 for junction in junctions):
        raise ValueError(f"flow {flow!r}: a segment has no length")
    return np.array(vertices, dtype=np.float64), junctions


def _check_turns(vertices: NDArray[np.float64], seg_len: NDArray[np.float64], flow: str) -> None:
    if len(vertices) < 3:
        return
    tangents = np.diff(vertices, axis=0) / seg_len[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    if bool(np.any(np.linalg.norm(normals[:-1] + normals[1:], axis=1) < 1e-3)):
        raise ValueError(f"flow {flow!r}: the lane turns back on itself")


def _track_spacing(config: LaneConfig, vertices: NDArray[np.float64], flow: str) -> float:
    if config.track_spacing is not None:
        return config.track_spacing
    return derive_track_spacing(vertices, config.chord_tolerance)


def _frame(vertex_s: NDArray[np.float64], vertices: NDArray[np.float64], s: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    count = len(vertex_s) - 1
    k = np.clip(np.searchsorted(vertex_s, s, side="right") - 1, 0, count - 1)
    seg_len = vertex_s[k + 1] - vertex_s[k]
    delta = vertices[k + 1] - vertices[k]
    frac = (s - vertex_s[k]) / seg_len
    xy = vertices[k] + frac[:, None] * delta
    tangent = delta / seg_len[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    prev = np.maximum(k - 1, 0)
    prev_len = np.maximum(vertex_s[k] - vertex_s[prev], GRID_EPS)
    prev_tangent = (vertices[k] - vertices[prev]) / prev_len[:, None]
    prev_normal = np.column_stack([-prev_tangent[:, 1], prev_tangent[:, 0]])
    summed = normal + prev_normal
    cosine = np.einsum("ij,ij->i", normal, prev_normal)
    miter = summed / np.maximum(1.0 + cosine, GRID_EPS)[:, None]
    at_vertex = (k > 0) & (np.abs(s - vertex_s[k]) < VERTEX_EPS)
    normal = np.where(at_vertex[:, None], miter, normal)
    return xy[:, 0], xy[:, 1], normal[:, 0], normal[:, 1]


def _offset_grid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
    segments: NDArray[np.float64],
    wall_radius: float,
    separation: float,
    chord: float,
) -> NDArray[np.float64]:
    reach = max(_wall_reach(x, y, nx, ny, segments) + wall_radius, separation + chord)
    count = math.ceil(reach / chord)
    return np.arange(-count, count + 1, dtype=np.float64) * chord


def _wall_reach(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
    segments: NDArray[np.float64],
) -> float:
    """Farthest nearest-wall hit of the normal rays, or the farthest wall endpoint when a ray hits nothing."""
    origin = np.column_stack([x, y])
    normal = np.column_stack([nx, ny])
    rows = max(1, RAY_CHUNK // len(segments))
    reach = 0.0
    open_side = False
    for start in range(0, len(origin), rows):
        for sign in (1.0, -1.0):
            hit = _ray_distance(origin[start : start + rows], sign * normal[start : start + rows], segments)
            if np.isinf(hit).any():
                open_side = True
            else:
                reach = max(reach, float(hit.max()))
    if not open_side:
        return reach
    ends = segments.reshape(-1, 2)
    lateral = np.abs((ends[None, :, 0] - x[:, None]) * nx[:, None] + (ends[None, :, 1] - y[:, None]) * ny[:, None])
    return float(lateral.max())


def _ray_distance(origin: NDArray[np.float64], direction: NDArray[np.float64], segments: NDArray[np.float64]) -> NDArray[np.float64]:
    a = segments[:, 0, :]
    edge = segments[:, 1, :] - a
    w = a[None, :, :] - origin[:, None, :]
    d = direction[:, None, :]
    denom = d[..., 0] * edge[None, :, 1] - d[..., 1] * edge[None, :, 0]
    safe = np.where(np.abs(denom) > GRID_EPS, denom, 1.0)
    t = (w[..., 0] * edge[None, :, 1] - w[..., 1] * edge[None, :, 0]) / safe
    u = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / safe
    hit = (np.abs(denom) > GRID_EPS) & (t >= -GRID_EPS) & (u >= -GRID_EPS) & (u <= 1.0 + GRID_EPS)
    return np.where(hit, np.maximum(t, 0.0), np.inf).min(axis=1)


def _wall_blocked_grid(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    nx: NDArray[np.float64],
    ny: NDArray[np.float64],
    offsets: NDArray[np.float64],
    segments: NDArray[np.float64],
    wall_radius: float,
) -> NDArray[np.bool_]:
    px = x[:, None] + offsets[None, :] * nx[:, None]
    py = y[:, None] + offsets[None, :] * ny[:, None]
    distance = _min_wall_distance(np.column_stack([px.ravel(), py.ravel()]), segments).reshape(px.shape)
    return distance < wall_radius - GRID_EPS


def _min_wall_distance(points: NDArray[np.float64], segments: NDArray[np.float64]) -> NDArray[np.float64]:
    chunks = [wall_distance(points[start : start + WALL_DISTANCE_CHUNK], segments) for start in range(0, len(points), WALL_DISTANCE_CHUNK)]
    return np.concatenate(chunks) if chunks else np.empty(0)


def _window_bounds(stations: NDArray[np.float64], half_window: float) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    lo = np.searchsorted(stations, stations - half_window - GRID_EPS, side="left")
    hi = np.searchsorted(stations, stations + half_window + GRID_EPS, side="right")
    return lo, hi


def _sweep(blocked: NDArray[np.bool_], stations: NDArray[np.float64], half_window: float) -> NDArray[np.bool_]:
    prefix = np.vstack([np.zeros((1, blocked.shape[1]), dtype=np.int64), np.cumsum(blocked, axis=0, dtype=np.int64)])
    lo, hi = _window_bounds(stations, half_window)
    return prefix[hi] - prefix[lo] > 0


def _window_mean(values: NDArray[np.float64], stations: NDArray[np.float64], half_window: float) -> NDArray[np.float64]:
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    lo, hi = _window_bounds(stations, half_window)
    return (prefix[hi] - prefix[lo]) / (hi - lo)


def _hallway_edges(
    free: NDArray[np.bool_],
    offsets: NDArray[np.float64],
    stations: NDArray[np.float64],
    flow: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Right and left walls of the hallway at each station: its free space within the widest hallway the centerline runs through."""
    widest = _widest_hallway(free, offsets)
    if widest is None:
        raise ValueError(f"flow {flow!r}: no walkable space on the centerline")
    inside = np.abs(offsets) <= widest + GRID_EPS
    low, high, any_free = _free_edges(free & inside[None, :], offsets)
    if not bool(np.all(any_free)):
        raise ValueError(f"flow {flow!r}: no walkable space at station {stations[int(np.argmin(any_free))]:.2f} m")
    return low, high


def _widest_hallway(free: NDArray[np.bool_], offsets: NDArray[np.float64]) -> float | None:
    """Half-width of the widest free run that holds the centerline, over the stations where the centerline is free."""
    halves = []
    for row in free:
        run = _run_containing(_free_runs(row, offsets), 0.0)
        if run is not None:
            halves.append(max(-run[0], run[1]))
    return max(halves) if halves else None


def _free_edges(free: NDArray[np.bool_], offsets: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    low = np.empty(free.shape[0])
    high = np.empty(free.shape[0])
    any_free = np.asarray(free.any(axis=1), dtype=np.bool_)
    for index, row in enumerate(free):
        runs = _free_runs(row, offsets)
        if not runs:
            continue
        main = _run_containing(runs, 0.0)
        if main is None:
            low[index], high[index] = runs[0][0], runs[-1][1]
            continue
        start = runs.index(main)
        lo_idx = 0
        while lo_idx < start and runs[lo_idx][1] - runs[lo_idx][0] < MIN_ADJACENT_RUN_WIDTH:
            lo_idx += 1
        hi_idx = len(runs) - 1
        while hi_idx > start and runs[hi_idx][1] - runs[hi_idx][0] < MIN_ADJACENT_RUN_WIDTH:
            hi_idx -= 1
        low[index], high[index] = runs[lo_idx][0], runs[hi_idx][1]
    return low, high, any_free


def _station_runs(
    free: NDArray[np.bool_],
    offsets: NDArray[np.float64],
    config: LaneConfig,
    right: NDArray[np.float64],
    left: NDArray[np.float64],
    zone: float,
    stations: NDArray[np.float64],
    flow: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lows = np.empty(len(stations))
    highs = np.empty(len(stations))
    edges: Run = (float(offsets[0]), float(offsets[-1]))
    for index, station in enumerate(stations):
        center = float((right[index] + left[index]) / 2.0)
        row = free[index]
        if zone > 0.0:
            row = row & ~(np.abs(offsets - center) < zone - GRID_EPS)
        runs = _clipped_runs(_free_runs(row, offsets), float(right[index]), float(left[index]))
        lows[index], highs[index] = _choose_run(runs, config, center, edges, float(station), flow)
    return lows, highs


def _free_runs(free_row: NDArray[np.bool_], offsets: NDArray[np.float64]) -> list[Run]:
    padded = np.concatenate([[False], free_row, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0] - 1
    return [(float(offsets[a]), float(offsets[b])) for a, b in zip(starts, ends, strict=True)]


def _choose_run(runs: list[Run], config: LaneConfig, center: float, edges: Run, station: float, flow: str) -> Run:
    chosen = _primary_run(runs, config, center)
    if chosen is None:
        raise ValueError(f"flow {flow!r}: no walkable space at station {station:.2f} m")
    if chosen[0] <= edges[0] + GRID_EPS or chosen[1] >= edges[1] - GRID_EPS:
        raise ValueError(f"flow {flow!r}: no wall on the open side at station {station:.2f} m")
    return chosen


def _primary_run(runs: list[Run], config: LaneConfig, center: float) -> Run | None:
    if config.offset is not None:
        inside = [run for run in runs if _contains(run, config.offset)]
        if inside:
            return inside[0]
        if config.offset > 0.0:
            return _nearest_within(runs, 0.0, math.inf, config.offset)
        return _nearest_within(runs, -math.inf, 0.0, config.offset)
    if config.side is LaneSide.RIGHT:
        return _nearest_within(runs, -math.inf, center, center)
    if config.side is LaneSide.LEFT:
        return _nearest_within(runs, center, math.inf, center)
    return _nearest_within(runs, -math.inf, math.inf, center)


def _nearest_within(runs: list[Run], low: float, high: float, anchor: float) -> Run | None:
    clipped = _clipped_runs(runs, low, high)
    return min(clipped, key=lambda run: _distance(run, anchor)) if clipped else None


def _clipped_runs(runs: list[Run], low: float, high: float) -> list[Run]:
    return [clipped for clipped in (_clip_run(run, low, high) for run in runs) if clipped is not None]


def _clip_run(run: Run, low: float, high: float) -> Run | None:
    lo, hi = max(run[0], low), min(run[1], high)
    return (lo, hi) if hi >= lo - GRID_EPS else None


def _run_containing(runs: list[Run], value: float) -> Run | None:
    return next((run for run in runs if _contains(run, value)), None)


def _contains(run: Run, value: float) -> bool:
    return run[0] - GRID_EPS <= value <= run[1] + GRID_EPS


def _distance(run: Run, anchor: float) -> float:
    if _contains(run, anchor):
        return 0.0
    return min(abs(anchor - run[0]), abs(anchor - run[1]))


def _clip(
    center: NDArray[np.float64],
    half: NDArray[np.float64],
    lows: NDArray[np.float64],
    highs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lo = np.maximum(center - half, lows)
    hi = np.minimum(center + half, highs)
    empty = hi < lo
    lo = np.where(empty, lows, lo)
    hi = np.where(empty, highs, hi)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _blend_junctions(
    config: LaneConfig,
    center: NDArray[np.float64],
    half: NDArray[np.float64],
    stations: NDArray[np.float64],
    vertex_s: NDArray[np.float64],
    seg_len: NDArray[np.float64],
    junctions: list[int],
    flow: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float]]:
    derived: list[float] = []
    for vertex in junctions:
        junction_s = float(vertex_s[vertex])
        station = int(np.argmin(np.abs(stations - junction_s)))
        before = max(station - 1, 0)
        after = min(station + 1, len(stations) - 1)
        fits = float(min(seg_len[vertex - 1], seg_len[vertex]))
        if config.merge_length is None:
            length = min(fits, 2.0 * float(half[before]), 2.0 * float(half[after]))
        else:
            length = config.merge_length
            if length > fits + GRID_EPS:
                raise ValueError(f"flow {flow!r}: merge length {length:.2f} m does not fit the segments at station {junction_s:.2f} m")
        derived.append(length)
        if length <= GRID_EPS:
            continue
        start = junction_s - length
        window = (stations >= start - GRID_EPS) & (stations <= junction_s + GRID_EPS)
        frac = np.clip((stations[window] - start) / length, 0.0, 1.0)
        c_start = float(np.interp(start, stations, center))
        h_start = float(np.interp(start, stations, half))
        center[window] = c_start + (center[station] - c_start) * frac
        half[window] = h_start + (half[station] - h_start) * frac
    return center, half, derived


def _band_edges(lane: Lane, index: int) -> list[tuple[float, float]]:
    offsets = (lane.center[index] - lane.half_width[index], lane.center[index], lane.center[index] + lane.half_width[index])
    return [(float(lane.x[index] + o * lane.nx[index]), float(lane.y[index] + o * lane.ny[index])) for o in offsets]


def _first_tangent(config: LaneConfig, flow: str) -> tuple[float, float]:
    if not config.centerlines or len(config.centerlines[0]) < 2:
        raise ValueError(f"flow {flow!r}: the first segment needs two points")
    first, second = config.centerlines[0][0], config.centerlines[0][1]
    dx, dy = second.x - first.x, second.y - first.y
    length = math.hypot(dx, dy)
    if length <= GRID_EPS:
        raise ValueError(f"flow {flow!r}: the first segment has no length")
    return dx / length, dy / length


def _source_extent(shape: Shape, tangent: tuple[float, float]) -> float:
    if shape.type == ShapeType.CIRCLE and shape.radius > 0.0:
        return 2.0 * shape.radius
    if not shape.vertices:
        return 0.0
    along = [vertex.x * tangent[0] + vertex.y * tangent[1] for vertex in shape.vertices]
    return max(along) - min(along)


def _inside_source(source: SourceConfig, x: float, y: float) -> bool:
    shape = source.shape
    if shape.type == ShapeType.CIRCLE and shape.radius > 0.0:
        return math.hypot(x - source.pose.x, y - source.pose.y) <= shape.radius
    return point_in_polygon(x - source.pose.x, y - source.pose.y, shape.vertices)


def _inside_sink(sink: SinkConfig, x: float, y: float) -> bool:
    if sink.shape.type == ShapeType.POLYGON and sink.shape.vertices:
        return point_in_polygon(x - sink.pose.x, y - sink.pose.y, sink.shape.vertices)
    return math.hypot(x - sink.pose.x, y - sink.pose.y) < sink.absorption_radius
