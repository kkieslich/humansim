from __future__ import annotations

import math

import numpy as np
import pytest
from arena_humansim.utils.lanes import (
    build_lane,
    build_track,
    derive_track_spacing,
    goal_corners,
    resolve_clearance,
    truncated_normal,
)
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import Lane, LaneConfig, LaneSide, Pose2D, Segments, SourceConfig, Track
from numpy.typing import NDArray

from tests._lane_layouts import corridor, straight

CLEARANCE = resolve_clearance(0.38, 0.2)


def lane_of(config: LaneConfig, walls: Segments, extent: float = 1.5, flow: str = "test") -> Lane:
    return build_lane(config, walls, flow=flow, agent_radius=0.25, clearance=CLEARANCE, source_extent=extent)


def test_lane_side_accepts_only_right_left_and_center() -> None:
    assert LaneSide("right") is LaneSide.RIGHT
    with pytest.raises(ValueError):
        LaneSide("up")


def test_source_without_a_lane_config_has_no_lane() -> None:
    source = SourceConfig()
    assert source.lane_config is None
    assert source.lane is None


@pytest.mark.parametrize(("inflation", "expected"), [(0.38, 0.4), (0.4, 0.4), (0.41, 0.6)])
def test_resolve_clearance_rounds_the_inflation_up_to_whole_cells(inflation: float, expected: float) -> None:
    assert resolve_clearance(inflation, 0.2) == pytest.approx(expected)


def test_one_way_center_lane_fills_the_centerd_corridor() -> None:
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.CENTER), corridor(4.0, 20.0))
    assert abs(float(np.median(lane.center))) < 0.05
    assert float(np.median(lane.half_width)) > 1.45


def test_right_lane_in_an_open_corridor_keeps_to_the_right_half() -> None:
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), corridor(4.0, 20.0))
    assert float(np.max(lane.center + lane.half_width)) <= 1e-9
    assert float(np.median(lane.center)) < -0.7


def test_offset_centers_the_band_and_bounds_its_half_width_by_the_walls() -> None:
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, offset=-0.5)
    lane = lane_of(config, corridor(4.0, 20.0))
    assert abs(float(np.median(lane.center)) + 0.5) < 0.05
    assert abs(float(np.median(lane.half_width)) - 1.05) < 0.05


def test_offset_next_to_a_wall_narrows_the_band_instead_of_leaving_the_corridor() -> None:
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, offset=-1.3)
    lane = lane_of(config, corridor(4.0, 20.0))
    assert float(np.min(lane.center - lane.half_width)) >= -1.55 - 1e-9
    assert float(np.max(lane.center + lane.half_width)) <= -1.05 + 1e-9


def test_two_way_band_stays_right_of_the_separation_zone() -> None:
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, two_way=True)
    lane = lane_of(config, corridor(4.0, 20.0))
    assert float(np.max(lane.center + lane.half_width)) <= -0.30 + 1e-9


def test_two_way_offset_inside_the_separation_zone_is_an_error() -> None:
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, two_way=True, offset=0.1)
    with pytest.raises(ValueError, match="flow 'test'"):
        lane_of(config, corridor(4.0, 20.0))


def test_five_meter_corridor_widens_the_band_by_the_corridor_ratio() -> None:
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, two_way=True)
    four = lane_of(config, corridor(4.0, 20.0))
    five = lane_of(config, corridor(5.0, 20.0))
    ratio = float(np.median(five.half_width)) / float(np.median(four.half_width))
    assert abs(ratio - 1.4) < 0.05


def test_pillar_on_the_centerline_moves_the_band_to_the_free_side() -> None:
    walls = corridor(4.0, 20.0) + [((9.7, -0.3), (9.7, 0.3))]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), walls)
    near = np.abs(lane.s - 9.7) < 0.5
    assert float(np.max(lane.center[near] + lane.half_width[near])) <= -0.75 + 1e-9


def test_narrow_gap_between_obstacles_keeps_the_band_on_one_side() -> None:
    walls = corridor(4.0, 20.0) + [((8.0, -0.6), (8.0, -0.2)), ((9.0, -0.6), (9.0, -0.2))]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), walls)
    gap = (lane.s > 8.0) & (lane.s < 9.0)
    assert float(np.max(lane.center[gap] + lane.half_width[gap])) <= -1.05 + 1e-9
    assert float(np.max(lane.center + lane.half_width)) <= 1e-9


def test_side_colonnade_moves_the_hallway_center_and_the_right_lane_keeps_its_side() -> None:
    walls = corridor(4.0, 20.0) + [((10.0, -1.7), (10.0, -1.3))]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), walls)
    at_column = int(np.argmin(np.abs(lane.s - 10.0)))
    open_station = int(np.argmin(np.abs(lane.s - 1.0)))
    assert float(lane.center[at_column] + lane.half_width[at_column]) == pytest.approx(0.35, abs=0.06)
    assert float(lane.center[at_column] - lane.half_width[at_column]) == pytest.approx(-0.85, abs=0.06)
    assert float(lane.center[open_station] + lane.half_width[open_station]) == pytest.approx(0.0, abs=0.06)


def test_colonnade_on_the_lane_side_with_no_room_is_generated_with_the_center_moved() -> None:
    columns = [((x, -1.2), (x, -0.8)) for x in (4.0, 8.0, 12.0, 16.0)]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, sweep_length=4.5), corridor(4.0, 20.0) + columns)
    at_column = int(np.argmin(np.abs(lane.s - 8.0)))
    assert float(lane.center[at_column] + lane.half_width[at_column]) == pytest.approx(0.60, abs=0.06)
    assert float(lane.half_width[at_column]) > 0.0


def test_colonnade_on_the_lane_side_narrows_the_band_and_moves_the_center_with_it() -> None:
    columns = [((x, -1.7), (x, -1.3)) for x in (4.0, 8.0, 12.0, 16.0)]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), corridor(4.0, 20.0) + columns)
    at_column = int(np.argmin(np.abs(lane.s - 8.0)))
    open_station = int(np.argmin(np.abs(lane.s - 0.5)))
    assert float(lane.half_width[at_column]) < float(lane.half_width[open_station])
    assert float(lane.center[at_column] + lane.half_width[at_column]) == pytest.approx(0.35, abs=0.06)


def test_two_way_left_lane_with_a_column_on_its_side_keeps_its_side_of_the_moved_center() -> None:
    columns = [((x, 1.4), (x, 1.8)) for x in (4.0, 8.0, 12.0, 16.0)]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.LEFT, two_way=True, sweep_length=4.5), corridor(4.0, 20.0) + columns)
    at_column = int(np.argmin(np.abs(lane.s - 8.0)))
    assert float(lane.center[at_column] - lane.half_width[at_column]) == pytest.approx(0.0, abs=0.06)
    assert float(lane.center[at_column] + lane.half_width[at_column]) == pytest.approx(0.95, abs=0.06)


def test_two_way_lanes_split_the_space_about_the_moved_center() -> None:
    walls = corridor(5.0, 20.0) + [((10.0, 1.4), (10.0, 1.8))]
    right = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, two_way=True), walls)
    left = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.LEFT, two_way=True), walls)
    at_column = int(np.argmin(np.abs(right.s - 10.0)))
    assert float(right.center[at_column] + right.half_width[at_column]) == pytest.approx(-0.85, abs=0.06)
    assert float(left.center[at_column] - left.half_width[at_column]) == pytest.approx(-0.25, abs=0.06)


def test_two_way_lanes_split_the_space_when_a_pillar_leaves_room_behind_it() -> None:
    walls = corridor(6.0, 20.0) + [((10.0, 1.4), (10.0, 1.6))]
    right = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, two_way=True), walls)
    left = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.LEFT, two_way=True), walls)
    at_column = int(np.argmin(np.abs(right.s - 10.0)))
    assert float(right.half_width[at_column]) == pytest.approx(float(left.half_width[at_column]), abs=0.06)
    assert float(right.center[at_column] + right.half_width[at_column]) < 0.0


def test_centerd_colonnade_leaves_each_two_way_lane_its_side() -> None:
    walls = corridor(5.0, 20.0) + [((10.0, -0.3), (10.0, 0.3))]
    right = build_lane(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, two_way=True), walls, flow="east", agent_radius=0.35, clearance=CLEARANCE, source_extent=1.5)
    left = build_lane(LaneConfig(centerlines=straight(20.0), side=LaneSide.LEFT, two_way=True), walls, flow="west", agent_radius=0.35, clearance=CLEARANCE, source_extent=1.5)
    at_column = int(np.argmin(np.abs(right.s - 10.0)))
    assert float(right.center[at_column] + right.half_width[at_column]) <= -0.60
    assert float(left.center[at_column] - left.half_width[at_column]) >= 0.60


def test_opening_in_one_wall_does_not_move_the_hallway_center() -> None:
    walls = [
        ((0.0, -2.0), (20.0, -2.0)),
        ((0.0, 2.0), (8.0, 2.0)),
        ((11.0, 2.0), (20.0, 2.0)),
        ((8.0, 2.0), (8.0, 8.0)),
        ((11.0, 2.0), (11.0, 8.0)),
        ((8.0, 8.0), (11.0, 8.0)),
    ]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), walls)
    at_opening = int(np.argmin(np.abs(lane.s - 9.5)))
    assert float(lane.center[at_opening] + lane.half_width[at_opening]) == pytest.approx(0.0, abs=0.06)


@pytest.mark.parametrize("side", [LaneSide.RIGHT, LaneSide.LEFT])
def test_room_behind_a_solid_wall_does_not_widen_the_hallway(side: LaneSide) -> None:
    walls = [
        ((0.0, -2.0), (20.0, -2.0)),
        ((0.0, 2.0), (20.0, 2.0)),
        ((0.0, 8.3), (20.0, 8.3)),
        ((0.0, 2.0), (0.0, 8.3)),
        ((20.0, 2.0), (20.0, 8.3)),
    ]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=side), walls)
    assert float(np.max(lane.center + lane.half_width)) <= 2.0
    assert float(np.min(lane.center - lane.half_width)) >= -2.0


@pytest.mark.parametrize("side", [LaneSide.RIGHT, LaneSide.LEFT])
def test_door_in_the_wall_to_a_room_keeps_the_band_in_the_corridor_away_from_the_door(side: LaneSide) -> None:
    walls = [
        ((0.0, -2.0), (20.0, -2.0)),
        ((0.0, 2.0), (1.0, 2.0)),
        ((2.0, 2.0), (20.0, 2.0)),
        ((0.0, 8.3), (20.0, 8.3)),
        ((0.0, 2.0), (0.0, 8.3)),
        ((20.0, 2.0), (20.0, 8.3)),
    ]
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=side), walls)
    far = lane.s > 12.0
    assert float(np.max((lane.center + lane.half_width)[far])) <= 2.0
    assert float(np.min((lane.center - lane.half_width)[far])) >= -2.0


def test_disconnected_segments_raise_naming_the_flow() -> None:
    config = LaneConfig(
        centerlines=[
            [Pose2D(x=0.0, y=0.0), Pose2D(x=10.0, y=0.0)],
            [Pose2D(x=12.0, y=0.0), Pose2D(x=20.0, y=0.0)],
        ],
        side=LaneSide.RIGHT,
    )
    with pytest.raises(ValueError, match="flow 'test'"):
        lane_of(config, corridor(4.0, 20.0))


def test_two_segment_lane_is_one_continuous_lane() -> None:
    config = LaneConfig(
        centerlines=[
            [Pose2D(x=0.0, y=0.0), Pose2D(x=10.0, y=0.0)],
            [Pose2D(x=10.0, y=0.0), Pose2D(x=20.0, y=0.0)],
        ],
        side=LaneSide.RIGHT,
    )
    lane = lane_of(config, corridor(4.0, 20.0))
    assert lane.length == pytest.approx(20.0)
    assert bool(np.all(np.isfinite(lane.center)))


def test_corridor_narrower_than_a_body_is_generated_with_a_zero_width_lane() -> None:
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), corridor(0.9, 20.0))
    assert float(np.max(lane.half_width)) <= 1e-9


def test_corridor_with_no_walkable_center_is_an_error() -> None:
    with pytest.raises(ValueError, match="no walkable space"):
        lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT), corridor(0.6, 20.0))


def test_track_stop_next_to_a_corner_is_dropped_so_no_two_goals_sit_close_together() -> None:
    config = LaneConfig(
        centerlines=[[Pose2D(x=0.0, y=0.0), Pose2D(x=9.04, y=0.0)], [Pose2D(x=9.04, y=0.0), Pose2D(x=9.04, y=6.0)]],
        side=LaneSide.CENTER,
        offset=0.0,
        track_spacing=1.5,
    )
    walls = [((-10.0, -10.0), (30.0, -10.0)), ((30.0, -10.0), (30.0, 30.0)), ((30.0, 30.0), (-10.0, 30.0)), ((-10.0, 30.0), (-10.0, -10.0))]
    lane = lane_of(config, walls)
    track = build_track(lane, q=0.0, s0=0.0)
    xy = np.array([[p.x, p.y] for p in track.points])
    gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert float(gaps.min()) >= lane.settings.arrival_radius - 1e-9


def test_two_corner_vertices_closer_than_an_arrival_radius_leave_one_goal() -> None:
    config = LaneConfig(
        centerlines=[[Pose2D(x=0.0, y=0.0), Pose2D(x=0.11, y=0.0)], [Pose2D(x=0.11, y=0.0), Pose2D(x=8.0, y=0.0)]],
        side=LaneSide.CENTER,
        offset=0.0,
        track_spacing=1.5,
    )
    walls = [((-2.0, -2.0), (10.0, -2.0)), ((10.0, -2.0), (10.0, 2.0)), ((10.0, 2.0), (-2.0, 2.0)), ((-2.0, 2.0), (-2.0, -2.0))]
    lane = lane_of(config, walls)
    track = build_track(lane, q=0.0, s0=0.0)
    xy = np.array([[p.x, p.y] for p in track.points])
    gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert float(gaps.min()) >= lane.settings.arrival_radius - 1e-9


def test_offset_corner_whose_goals_fold_onto_each_other_keeps_one_goal() -> None:
    config = LaneConfig(
        centerlines=[[Pose2D(x=0.0, y=0.0), Pose2D(x=9.04, y=0.0)], [Pose2D(x=9.04, y=0.0), Pose2D(x=9.04, y=6.0)]],
        side=LaneSide.CENTER,
        offset=0.8,
        track_spacing=1.65,
    )
    walls = [((-2.0, -4.0), (12.0, -4.0)), ((12.0, -4.0), (12.0, 8.0)), ((12.0, 8.0), (-2.0, 8.0)), ((-2.0, 8.0), (-2.0, -4.0))]
    lane = lane_of(config, walls)
    track = build_track(lane, q=0.0, s0=0.0)
    xy = np.array([[p.x, p.y] for p in track.points])
    gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    assert float(gaps.min()) >= lane.settings.arrival_radius - 1e-9


def _l_corridor() -> Segments:
    return [
        ((-1.0, -2.0), (11.04, -2.0)),
        ((11.04, -2.0), (11.04, 8.0)),
        ((11.04, 8.0), (7.04, 8.0)),
        ((7.04, 8.0), (7.04, 2.0)),
        ((7.04, 2.0), (-1.0, 2.0)),
        ((-1.0, 2.0), (-1.0, -2.0)),
    ]


def _left_corner(offset: float) -> Lane:
    config = LaneConfig(
        centerlines=[[Pose2D(x=0.0, y=0.0), Pose2D(x=9.04, y=0.0)], [Pose2D(x=9.04, y=0.0), Pose2D(x=9.04, y=6.0)]],
        side=LaneSide.CENTER,
        offset=offset,
        track_spacing=1.65,
    )
    return lane_of(config, _l_corridor())


def _reversals(track: Track) -> list[tuple[float, float]]:
    """Track points where the path turns back on itself by more than 120 degrees."""
    xy = np.array([[p.x, p.y] for p in track.points])
    found = []
    for i in range(1, len(xy) - 1):
        a = xy[i] - xy[i - 1]
        b = xy[i + 1] - xy[i]
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            continue
        if float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) < math.cos(math.radians(120.0)):
            found.append((float(xy[i][0]), float(xy[i][1])))
    return found


@pytest.mark.parametrize("q", [-1.0, 0.0, 1.0])
def test_track_does_not_fold_back_at_an_inside_corner(q: float) -> None:
    track = build_track(_left_corner(0.8), q=q, s0=0.0)
    assert _reversals(track) == []


def _l_corner_lane(offset: float, mirror: bool) -> Lane:
    """L-turn lane at one vertex; mirror flips the layout in y and the sign of the offset."""
    sign = -1.0 if mirror else 1.0
    config = LaneConfig(
        centerlines=[
            [Pose2D(x=0.0, y=0.0), Pose2D(x=9.04, y=0.0)],
            [Pose2D(x=9.04, y=0.0), Pose2D(x=9.04, y=sign * 6.0)],
        ],
        side=LaneSide.CENTER,
        offset=sign * offset,
        track_spacing=1.65,
    )
    walls = [((x0, sign * y0), (x1, sign * y1)) for (x0, y0), (x1, y1) in _l_corridor()]
    return lane_of(config, walls)


def _offset_corner(lane: Lane, index: int, lateral: float) -> NDArray[np.float64]:
    """Point where the offset lines of the two legs at a vertex meet, at the given lateral distance."""
    legs: list[list[float]] = []
    for start, end in ((index - 1, index), (index, index + 1)):
        leg = np.array([lane.vertex_x[end] - lane.vertex_x[start], lane.vertex_y[end] - lane.vertex_y[start]])
        leg = leg / np.linalg.norm(leg)
        legs.append([-leg[1], leg[0]])
    miter = np.linalg.solve(np.array(legs), np.ones(2))
    return np.array([lane.vertex_x[index], lane.vertex_y[index]]) + lateral * miter


def _l_corner_pair(mirror: bool) -> tuple[Lane, Lane]:
    """Two-way pair at one L-turn: a right-hand lane on the centerline and one on its reverse."""
    sign = -1.0 if mirror else 1.0
    forward = [
        [Pose2D(x=0.0, y=0.0), Pose2D(x=9.04, y=0.0)],
        [Pose2D(x=9.04, y=0.0), Pose2D(x=9.04, y=sign * 6.0)],
    ]
    backward = [[Pose2D(x=point.x, y=point.y) for point in reversed(segment)] for segment in reversed(forward)]
    walls = [((x0, sign * y0), (x1, sign * y1)) for (x0, y0), (x1, y1) in _l_corridor()]
    return (
        lane_of(LaneConfig(centerlines=forward, side=LaneSide.RIGHT, two_way=True, track_spacing=1.65), walls),
        lane_of(LaneConfig(centerlines=backward, side=LaneSide.RIGHT, two_way=True, track_spacing=1.65), walls),
    )


def _turn_sign(lane: Lane) -> float:
    """Sign of the turn at the lane's corner, positive for a left turn."""
    t_in = np.array([lane.vertex_x[1] - lane.vertex_x[0], lane.vertex_y[1] - lane.vertex_y[0]])
    t_out = np.array([lane.vertex_x[2] - lane.vertex_x[1], lane.vertex_y[2] - lane.vertex_y[1]])
    return math.copysign(1.0, float(t_in[0] * t_out[1] - t_in[1] * t_out[0]))


def _assert_track_passes_corner(lane: Lane, q: float) -> None:
    """The track at lateral q passes the offset corner of the lane's band."""
    vertex = lane.vertex_s[1]
    lateral = float(np.interp(vertex, lane.s, lane.center) + q * np.interp(vertex, lane.s, lane.half_width))
    corner = _offset_corner(lane, 1, lateral)
    track = build_track(lane, q=q, s0=0.0)
    xy = np.array([[p.x, p.y] for p in track.points])
    assert float(np.min(np.linalg.norm(xy - corner, axis=1))) < 1e-6


@pytest.mark.parametrize("mirror", [False, True])
def test_a_one_way_lane_keeps_no_corner_goal(mirror: bool) -> None:
    """Without an opposing lane no corner goal is kept, so a track may cut the inside of the turn."""
    assert goal_corners(_l_corner_lane(-0.8, mirror)).size == 0


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("q", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_outer_lane_of_a_two_way_pair_keeps_its_outside_corner_point(mirror: bool, q: float) -> None:
    """With an inner lane in the corridor, the outer lane of a two-way pair still passes every corner goal."""
    forward, backward = _l_corner_pair(mirror)
    outer = forward if _turn_sign(forward) > 0.0 else backward
    _assert_track_passes_corner(outer, q)


@pytest.mark.parametrize("mirror", [False, True])
def test_only_the_outer_lane_of_a_two_way_pair_keeps_a_corner_goal(mirror: bool) -> None:
    """The inner lane of a two-way pair may cut the corner; the outer lane keeps its goal."""
    forward, backward = _l_corner_pair(mirror)
    outer = forward if _turn_sign(forward) > 0.0 else backward
    inner = backward if outer is forward else forward
    assert goal_corners(outer).shape == (1,)
    assert abs(float(goal_corners(outer)[0]) - float(outer.vertex_s[1])) < 1e-9
    assert goal_corners(inner).size == 0


def _u_corridor_walls() -> Segments:
    return [
        ((-1.0, -2.0), (11.04, -2.0)),
        ((11.04, -2.0), (11.04, 8.0)),
        ((11.04, 8.0), (-1.0, 8.0)),
        ((-1.0, 8.0), (-1.0, -2.0)),
        ((-1.0, 2.0), (7.04, 2.0)),
        ((7.04, 2.0), (7.04, 4.0)),
        ((7.04, 4.0), (-1.0, 4.0)),
        ((7.04, 2.0), (9.04, 2.0)),
    ]


def test_an_outer_corner_keeps_its_goal_only_where_the_other_lane_is_inside_the_turn() -> None:
    walls = _u_corridor_walls()
    outer = lane_of(
        LaneConfig(
            centerlines=[
                [Pose2D(x=0.0, y=0.0), Pose2D(x=9.04, y=0.0)],
                [Pose2D(x=9.04, y=0.0), Pose2D(x=9.04, y=6.0)],
                [Pose2D(x=9.04, y=6.0), Pose2D(x=0.0, y=6.0)],
            ],
            side=LaneSide.RIGHT,
            two_way=True,
            track_spacing=1.65,
        ),
        walls,
    )
    partner = lane_of(
        LaneConfig(centerlines=[[Pose2D(x=9.04, y=0.0), Pose2D(x=0.0, y=0.0)]], side=LaneSide.RIGHT, two_way=True, track_spacing=1.65),
        walls,
    )
    assert goal_corners(outer, [partner]).tolist() == pytest.approx([float(outer.vertex_s[1])])
    assert goal_corners(outer, []).size == 0
    assert goal_corners(outer).size == 2


def test_an_outer_lane_keeps_no_goal_when_the_other_lane_is_far_away() -> None:
    far_walls = [((-2.0, 28.0), (22.0, 28.0)), ((-2.0, 32.0), (22.0, 32.0)), ((-2.0, 28.0), (-2.0, 32.0)), ((22.0, 28.0), (22.0, 32.0))]
    walls = _l_corridor() + far_walls
    outer = lane_of(
        LaneConfig(
            centerlines=[[Pose2D(x=0.0, y=0.0), Pose2D(x=9.04, y=0.0)], [Pose2D(x=9.04, y=0.0), Pose2D(x=9.04, y=6.0)]],
            side=LaneSide.RIGHT,
            two_way=True,
            track_spacing=1.65,
        ),
        walls,
    )
    far = lane_of(
        LaneConfig(centerlines=[[Pose2D(x=0.0, y=30.0), Pose2D(x=20.0, y=30.0)]], side=LaneSide.RIGHT, two_way=True, track_spacing=1.65),
        walls,
    )
    assert goal_corners(outer, [far]).size == 0
    assert goal_corners(outer).size == 1


def test_track_spacing_is_capped_on_a_straight_lane() -> None:
    vertices = np.array([[0.0, 0.0], [20.0, 0.0]])
    assert derive_track_spacing(vertices, 0.05) == pytest.approx(3.0)


def test_track_spacing_follows_the_bend_radius() -> None:
    vertices = np.array([[2.0, 0.0], [1.0, math.sqrt(3.0)], [-1.0, math.sqrt(3.0)]])
    assert derive_track_spacing(vertices, 0.05) == pytest.approx(math.sqrt(0.8))


def test_track_follows_the_centerline_in_order_with_arrival_radii() -> None:
    lane = lane_of(LaneConfig(centerlines=straight(20.0), side=LaneSide.CENTER), corridor(4.0, 20.0))
    track = build_track(lane, q=0.0, s0=0.0)
    xs = [point.x for point in track.points]
    assert xs == sorted(xs)
    assert xs[-1] == pytest.approx(20.0)
    assert track.radii == [lane.settings.arrival_radius] * len(track.points)


def test_truncated_normal_stays_inside_bounds_and_matches_closed_form_sd() -> None:
    sigma = 0.5
    generator = RNG(7).get_substream("q")
    draws = np.array([truncated_normal(sigma, generator) for _ in range(10_000)])
    a = 1.0 / sigma
    phi = math.exp(-a * a / 2.0) / math.sqrt(2.0 * math.pi)
    big_phi = 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))
    expected_sd = sigma * math.sqrt(1.0 - 2.0 * a * phi / (2.0 * big_phi - 1.0))
    assert float(draws.min()) >= -1.0
    assert float(draws.max()) <= 1.0
    assert abs(float(draws.std()) - expected_sd) < 0.01


def far_walls() -> Segments:
    return [((-40.0, 60.0), (60.0, 60.0)), ((-40.0, -60.0), (60.0, -60.0)), ((-40.0, -60.0), (-40.0, 60.0)), ((60.0, -60.0), (60.0, 60.0))]


def test_corridor_inside_a_much_larger_world_builds_the_same_band() -> None:
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT)
    alone = lane_of(config, corridor(4.0, 20.0))
    enclosed = lane_of(config, corridor(4.0, 20.0) + far_walls())
    np.testing.assert_allclose(enclosed.center, alone.center)
    np.testing.assert_allclose(enclosed.half_width, alone.half_width)


def test_offset_grid_of_a_corridor_in_a_larger_world_stays_near_the_corridor_width() -> None:
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT)
    lane = lane_of(config, corridor(4.0, 20.0) + far_walls())
    assert 2.0 <= lane.grid_reach < 3.0


def _hall_then_corridor_with_room() -> Segments:
    return [
        ((-3.0, -5.0), (10.0, -5.0)),
        ((-3.0, 5.0), (10.0, 5.0)),
        ((-3.0, -5.0), (-3.0, 5.0)),
        ((10.0, -5.0), (10.0, -2.0)),
        ((10.0, 2.0), (10.0, 5.0)),
        ((10.0, -2.0), (30.0, -2.0)),
        ((10.0, 2.0), (30.0, 2.0)),
        ((10.0, 8.3), (30.0, 8.3)),
        ((10.0, 2.0), (10.0, 8.3)),
        ((30.0, -2.0), (30.0, 8.3)),
    ]


@pytest.mark.parametrize("side", [LaneSide.RIGHT, LaneSide.LEFT])
def test_room_behind_a_corridor_wall_after_a_wide_hall_leaves_both_bands_their_width(side: LaneSide) -> None:
    config = LaneConfig(centerlines=straight(27.0), side=side, two_way=True)
    lane = lane_of(config, _hall_then_corridor_with_room())
    in_corridor = lane.s > 16.0
    assert float(np.max((lane.center + lane.half_width)[in_corridor])) <= 2.0
    assert float(np.min((lane.center - lane.half_width)[in_corridor])) >= -2.0
    assert float(np.min(lane.half_width[in_corridor])) > 0.6
