import math
from collections.abc import Callable, Sequence

import numpy as np

from arena_humansim.utils.lanes import build_track, truncated_normal
from arena_humansim.utils.loggable import Loggable
from arena_humansim.utils.rng import RNG, derive_seed
from arena_humansim.utils.types import (
    AgentLifetime,
    Lane,
    Pose2D,
    RateKeyframe,
    Shape,
    ShapeType,
    SinkConfig,
    SourceConfig,
    SourceType,
    SpawnRequest,
    WaypointMode,
)


class SpawnScheduler(Loggable):
    def __init__(self, rng: RNG | np.random.Generator, allocate_id: Callable[[], int] | None = None):
        self._sources: dict[str, SourceConfig] = {}
        self._sinks: dict[str, SinkConfig] = {}
        self._alive_count: dict[str, int] = {}
        self._total_count: dict[str, int] = {}
        self._agent_source: dict[int, str] = {}
        self._rng = rng
        self._base_seed = int(rng.integers(0, 2**62)) if isinstance(rng, np.random.Generator) else -1
        self._streams: dict[str, np.random.Generator] = {}
        self._local_next_id = 1
        self._allocate_id = allocate_id if allocate_id is not None else self._allocate_local_id

    def _allocate_local_id(self) -> int:
        aid = self._local_next_id
        self._local_next_id += 1
        return aid

    def _stream(self, name: str) -> np.random.Generator:
        if isinstance(self._rng, RNG):
            return self._rng.get_substream(f"spawn_scheduler/{name}")
        gen = self._streams.get(name)
        if gen is None:
            gen = np.random.default_rng(self._seed(name))
            self._streams[name] = gen
        return gen

    def _seed(self, name: str) -> int:
        base = self._rng.seed if isinstance(self._rng, RNG) else self._base_seed
        return derive_seed(base, f"spawn_scheduler/{name}")

    def _event_stream(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self._seed(name))

    def tick(self, tick_count: int, dt: float) -> list[SpawnRequest]:
        sim_time_s = tick_count * dt
        requests = []
        for src in self._sources.values():
            if src.lane_config is not None and src.lane is None:
                continue
            alive = self._alive_count.get(src.name, 0)
            total = self._total_count.get(src.name, 0)
            if src.type is SourceType.MAX:
                n = src.max_concurrent - alive
            else:
                rate = self._interpolate_rate(src.rate_profile, sim_time_s)
                if rate <= 0.0:
                    continue
                n = int(self._stream(f"{src.name}/poisson").poisson(rate * dt))
                if src.max_concurrent > 0:
                    n = min(n, src.max_concurrent - alive)
            if src.max_total > 0:
                n = min(n, src.max_total - total)
            for _ in range(max(0, n)):
                req = self._sample_spawn_request(src, tick_count, total)
                req.agent_id = self._allocate_id()
                requests.append(req)
                alive += 1
                total += 1
            self._alive_count[src.name] = alive
            self._total_count[src.name] = total

        if requests:
            self._logger.debug(f"Tick {tick_count}: {len(requests)} spawn request(s)")
        return requests

    def register_agent(self, agent_id: int, source_name: str) -> None:
        self._agent_source[agent_id] = source_name

    def notify_despawn(self, agent_id: int) -> None:
        source_name = self._agent_source.pop(agent_id, None)
        if source_name is not None and source_name in self._alive_count:
            self._alive_count[source_name] = max(0, self._alive_count[source_name] - 1)

    @staticmethod
    def validate(config: SourceConfig) -> None:
        if config.type is SourceType.MAX:
            if config.max_concurrent <= 0:
                raise ValueError(f"source {config.name!r}: type max needs max_concurrent > 0")
            if config.rate_profile:
                raise ValueError(f"source {config.name!r}: type max takes no rate_profile")
        if config.lane_config is not None:
            if len(config.agent.sink_affinity) != 1:
                raise ValueError(f"source {config.name!r}: a lane flow needs exactly one sink affinity")
            if config.lane is None:
                raise ValueError(f"source {config.name!r}: the lane is not resolved")

    def add_source(self, config: SourceConfig) -> str:
        self.validate(config)
        self._sources[config.name] = config
        self._alive_count.setdefault(config.name, 0)
        self._total_count.setdefault(config.name, 0)
        self._logger.info(f"Source added: {config.name}")
        return config.name

    def remove_source(self, name: str) -> None:
        self._sources.pop(name, None)
        self._logger.info(f"Source removed: {name}")

    def clear_sources(self) -> None:
        self._sources.clear()

    def set_sinks(self, sinks: dict[str, SinkConfig]) -> None:
        self._sinks = sinks

    def reset_counts(self) -> None:
        self._alive_count.clear()
        self._total_count.clear()
        self._agent_source.clear()
        self._streams.clear()
        self._local_next_id = 1

    @staticmethod
    def _interpolate_rate(profile: Sequence[RateKeyframe], t: float) -> float:
        if not profile:
            return 0.0
        if len(profile) == 1:
            return profile[0].rate
        if t <= profile[0].t:
            return profile[0].rate
        if t >= profile[-1].t:
            return profile[-1].rate

        for i in range(len(profile) - 1):
            if profile[i].t <= t <= profile[i + 1].t:
                dt = profile[i + 1].t - profile[i].t
                if dt <= 0:
                    return profile[i + 1].rate
                alpha = (t - profile[i].t) / dt
                return profile[i].rate + alpha * (profile[i + 1].rate - profile[i].rate)

        return profile[-1].rate

    def _sample_lane_spawn_request(self, src: SourceConfig, lane: Lane, tick_count: int, seq: int) -> SpawnRequest:
        tmpl = src.agent
        rng = self._event_stream(f"{src.name}/event/{seq}")
        desired_velocity = rng.uniform(tmpl.desired_velocity_min, tmpl.desired_velocity_max)
        q = truncated_normal(lane.settings.sigma_q, rng)
        s0 = float(rng.uniform(0.0, lane.settings.spawn_length))
        track = build_track(lane, q, s0)
        first, second = track.points[0], track.points[1]
        pose = Pose2D(x=first.x, y=first.y, theta=math.atan2(second.y - first.y, second.x - first.x))
        lifetime = AgentLifetime(
            source_name=src.name,
            source_seq=seq,
            spawn_tick=tick_count,
            target_sink_name=tmpl.sink_affinity[0].sink_name,
        )
        robot_awareness = tmpl.robot_awareness if tmpl.robot_awareness is not None else tmpl.robot_aware
        return SpawnRequest(
            pose=pose,
            desired_velocity=desired_velocity,
            agent_radius=tmpl.agent_radius,
            agent_type=tmpl.agent_type,
            waypoints=track.points,
            lifetime=lifetime,
            robot_awareness=robot_awareness,
            waypoint_radii=track.radii,
            waypoint_mode=WaypointMode.ONCE,
        )

    def _sample_spawn_request(self, src: SourceConfig, tick_count: int, seq: int) -> SpawnRequest:
        if src.lane is not None:
            return self._sample_lane_spawn_request(src, src.lane, tick_count, seq)
        tmpl = src.agent
        rng = self._event_stream(f"{src.name}/event/{seq}")

        pose = self._sample_pose_in_shape(src.pose, src.shape, rng)

        desired_velocity = rng.uniform(tmpl.desired_velocity_min, tmpl.desired_velocity_max)

        target_sink_name = ""
        waypoints: list[Pose2D] = []
        if tmpl.sink_affinity:
            weights = np.array([sa.weight for sa in tmpl.sink_affinity])
            weights = weights / weights.sum()
            chosen = int(rng.choice(len(tmpl.sink_affinity), p=weights))
            target_sink_name = tmpl.sink_affinity[chosen].sink_name

            if target_sink_name and target_sink_name in self._sinks:
                sink_pose = self._sinks[target_sink_name].pose
                waypoints = [Pose2D(x=sink_pose.x, y=sink_pose.y, theta=sink_pose.theta)]
                pose.theta = math.atan2(sink_pose.y - pose.y, sink_pose.x - pose.x)

        lifetime = AgentLifetime(
            source_name=src.name,
            source_seq=seq,
            spawn_tick=tick_count,
            target_sink_name=target_sink_name,
        )

        robot_awareness = tmpl.robot_awareness if tmpl.robot_awareness is not None else tmpl.robot_aware
        return SpawnRequest(
            pose=pose,
            desired_velocity=desired_velocity,
            agent_radius=tmpl.agent_radius,
            agent_type=tmpl.agent_type,
            waypoints=waypoints,
            lifetime=lifetime,
            robot_awareness=robot_awareness,
        )

    def _sample_pose_in_shape(self, center: Pose2D, shape: Shape, rng: np.random.Generator | None = None) -> Pose2D:
        if rng is None:
            rng = self._stream("pose")
        if shape.type == ShapeType.CIRCLE and shape.radius > 0:
            r = shape.radius * math.sqrt(float(rng.random()))
            angle = float(rng.uniform(0, 2 * math.pi))
            return Pose2D(
                x=center.x + r * math.cos(angle),
                y=center.y + r * math.sin(angle),
                theta=0.0,
            )
        if shape.vertices:
            return self._sample_pose_in_polygon(shape.vertices, center, rng)
        return Pose2D(x=center.x, y=center.y, theta=0.0)

    def _sample_pose_in_polygon(self, vertices: Sequence[Pose2D], center: Pose2D, rng: np.random.Generator | None = None) -> Pose2D:
        if rng is None:
            rng = self._stream("pose")
        n = len(vertices)
        if n < 3:
            if vertices:
                return Pose2D(x=center.x + vertices[0].x, y=center.y + vertices[0].y)
            return Pose2D(x=center.x, y=center.y)

        v0 = vertices[0]
        triangles = _triangulate(vertices)
        areas = [abs((v1.x - v0_.x) * (v2.y - v0_.y) - (v2.x - v0_.x) * (v1.y - v0_.y)) / 2.0 for v0_, v1, v2 in triangles]

        total_area = sum(areas)
        if total_area <= 0:
            return Pose2D(x=center.x + v0.x, y=center.y + v0.y)

        r = float(rng.random()) * total_area
        cumulative = 0.0
        tri = triangles[0]
        for tri, area in zip(triangles, areas, strict=True):  # noqa: B007
            cumulative += area
            if cumulative >= r:
                break

        u = float(rng.random())
        v = float(rng.random())
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        w = 1.0 - u - v

        a, b, c = tri
        x = center.x + w * a.x + u * b.x + v * c.x
        y = center.y + w * a.y + u * b.y + v * c.y
        return Pose2D(x=x, y=y, theta=0.0)


def _cross(o: Pose2D, a: Pose2D, b: Pose2D) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def _triangulate(vertices: Sequence[Pose2D]) -> list[tuple[Pose2D, Pose2D, Pose2D]]:
    """Triangles that cover a simple polygon exactly: the fan from the first vertex for a convex polygon (as before), ear
    clipping for a concave one, whose fan would reach outside it and count some of its area twice. Turns within float32
    noise of straight count as convex. A polygon ear clipping cannot reduce keeps the fan; a self-intersecting polygon
    is not covered exactly either way."""
    n = len(vertices)
    fan = [(vertices[0], vertices[i], vertices[i + 1]) for i in range(1, n - 1)]
    turns = [_cross(vertices[i - 1], vertices[i], vertices[(i + 1) % n]) for i in range(n)]
    extent = max(max(v.x for v in vertices) - min(v.x for v in vertices), max(v.y for v in vertices) - min(v.y for v in vertices))
    tol = 1e-9 * extent * extent
    if all(t >= -tol for t in turns) or all(t <= tol for t in turns):
        return fan
    signed = sum(vertices[i].x * vertices[(i + 1) % n].y - vertices[(i + 1) % n].x * vertices[i].y for i in range(n))
    idx = list(range(n)) if signed > 0.0 else list(range(n - 1, -1, -1))
    eps = 1e-12
    triangles: list[tuple[Pose2D, Pose2D, Pose2D]] = []
    while len(idx) > 3:
        for k in range(len(idx)):
            a, b, c = vertices[idx[k - 1]], vertices[idx[k]], vertices[idx[(k + 1) % len(idx)]]
            turn = _cross(a, b, c)
            if abs(turn) <= eps:
                idx.pop(k)  # a collinear vertex adds no area
                break
            if turn < 0.0:
                continue  # reflex
            corners = {(a.x, a.y), (b.x, b.y), (c.x, c.y)}
            # a vertex at the same point as a corner (a self-touching ring) does not block the ear
            others = (vertices[j] for j in idx if j not in (idx[k - 1], idx[k], idx[(k + 1) % len(idx)]) and (vertices[j].x, vertices[j].y) not in corners)
            if any(_cross(a, b, q) >= -eps and _cross(b, c, q) >= -eps and _cross(c, a, q) >= -eps for q in others):
                continue
            triangles.append((a, b, c))
            idx.pop(k)
            break
        else:
            return fan
    a, b, c = (vertices[j] for j in idx)
    if abs(_cross(a, b, c)) > eps:
        triangles.append((a, b, c))
    return triangles or fan
