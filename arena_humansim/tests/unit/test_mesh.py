from __future__ import annotations

import math

import numpy as np
import pytest

from arena_humansim.utils.mesh import MIN_LEN, SNAP, Mesh, SegIndex, _clip_dist, _expand, _pt_seg, _seg_seg, add_feet, refine_gabriel, split_segments

from ._helpers import two_rooms


def _segs(*rows: tuple[float, float, float, float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _xy(pts: np.ndarray) -> list[tuple[float, float]]:
    return sorted((round(float(x), 6) + 0.0, round(float(y), 6) + 0.0) for x, y in pts)


def _edge_keys(edges: np.ndarray, n: int) -> np.ndarray:
    return np.minimum(edges[:, 0], edges[:, 1]) * n + np.maximum(edges[:, 0], edges[:, 1])


def test_expand_concatenates_ranges() -> None:
    assert _expand(np.array([2, 5]), np.array([2, 3])).tolist() == [2, 3, 5, 6, 7]


def test_pt_seg_clamps_to_the_ends() -> None:
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    d, t = _pt_seg(np.array([[-1.0, 0.0], [0.5, 1.0], [3.0, 0.0]]), a, b)
    assert d.tolist() == [1.0, 1.0, 2.0]
    assert t.tolist() == [0.0, 0.5, 1.0]


def test_seg_seg_is_zero_on_a_crossing_and_the_gap_otherwise() -> None:
    p0, p1 = np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[2.0, 2.0], [2.0, 0.0]])
    w0, w1 = np.array([[0.0, 2.0], [0.0, 1.0]]), np.array([[2.0, 0.0], [2.0, 1.0]])
    assert _seg_seg(p0, p1, w0, w1).tolist() == [0.0, 1.0]


def test_clip_dist_measures_only_inside_the_wedge() -> None:
    corner = np.array([[0.0, 0.0], [0.0, 0.0]])
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 1.0]])
    u = np.array([[2.0, -1.0], [-1.0, -1.0]])
    w = np.array([[-1.0, 2.0], [-2.0, -2.0]])
    got = _clip_dist(corner, a, b, u, w)
    assert got[0] == pytest.approx(math.sqrt(0.5))
    assert got[1] == math.inf


@pytest.mark.parametrize(
    ("segs", "points", "n_edges"),
    [
        (_segs((0.0, 0.0, 2.0, 2.0), (0.0, 2.0, 2.0, 0.0)), [(0.0, 0.0), (0.0, 2.0), (1.0, 1.0), (2.0, 0.0), (2.0, 2.0)], 4),
        (_segs((0.0, 0.0, 2.0, 0.0), (1.0, 0.0, 1.0, 1.0)), [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 0.0)], 3),
        (_segs((0.0, 0.0, 2.0, 0.0), (1.0, 0.0, 3.0, 0.0)), [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], 3),
        (_segs((0.0, 0.0, 2.0, 0.0), (0.0, 0.0, 2.0, 0.0)), [(0.0, 0.0), (2.0, 0.0)], 1),
        (_segs((0.0, 0.0, 2.0, 0.0), (5.0, 5.0, 5.0, 5.0)), [(0.0, 0.0), (2.0, 0.0)], 1),
    ],
    ids=["crossing", "t_junction", "collinear_overlap", "duplicate", "zero_length"],
)
def test_split_segments_cuts_at_every_contact(segs: np.ndarray, points: list[tuple[float, float]], n_edges: int) -> None:
    pts, edges = split_segments(segs)
    assert _xy(pts) == points
    assert len(edges) == n_edges


def test_add_feet_splits_a_wall_under_a_near_vertex() -> None:
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0], [2.0, 3.0]])
    edges = np.array([[0, 1], [2, 3]])
    new_pts, new_edges = add_feet(pts, edges, 2.0)
    assert _xy(new_pts) == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 3.0), (4.0, 0.0)]
    assert len(new_edges) == 3


def test_add_feet_skips_a_foot_hidden_behind_another_wall() -> None:
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0], [2.0, 3.0], [1.0, 0.5], [3.0, 0.5]])
    edges = np.array([[0, 1], [2, 3], [4, 5]])
    new_pts, _ = add_feet(pts, edges, 2.0)
    assert (2.0, 0.5) in _xy(new_pts)
    assert (2.0, 0.0) not in _xy(new_pts)


def test_refine_gabriel_clears_every_diametral_circle() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
    edges = np.array([[0, 1], [2, 3]])
    new_pts, is_input, new_edges = refine_gabriel(pts, np.ones(4, bool), edges, np.empty((0, 2)))
    assert len(new_pts) > 4
    assert is_input.tolist() == [True] * 4 + [False] * (len(new_pts) - 4)
    a, b = new_pts[new_edges[:, 0]], new_pts[new_edges[:, 1]]
    for mid, rad in zip((a + b) / 2, np.hypot(*(b - a).T) / 2, strict=True):
        assert rad * 2 <= MIN_LEN or np.sum(np.hypot(*(new_pts - mid).T) <= rad * (1 + 1e-9)) == 2


def test_seg_index_finds_the_closest_wall_point() -> None:
    index = SegIndex(np.array([[0.0, 0.0], [0.0, 6.0]]), np.array([[4.0, 0.0], [4.0, 6.0]]))
    hit = index.nearest(np.array([2.0, 1.0]), 2.0)
    assert hit is not None
    assert hit[0] == pytest.approx(1.0)
    assert hit[1].tolist() == [2.0, 0.0]
    assert hit[2].tolist() == [4.0, 0.0]


def test_seg_index_sees_nothing_between_far_walls() -> None:
    index = SegIndex(np.array([[0.0, 0.0], [0.0, 6.0]]), np.array([[4.0, 0.0], [4.0, 6.0]]))
    assert index.nearest(np.array([2.0, 3.0]), 1.0) is None
    assert index.point_clearance(np.array([[2.0, 3.0]]), 1.0).tolist() == [math.inf]


def test_seg_index_keeps_the_cells_along_a_diagonal_wall() -> None:
    index = SegIndex(np.array([[0.0, 0.0]]), np.array([[10.0, 10.0]]))
    assert index.point_clearance(np.array([[5.0, 5.2]]), 0.5)[0] == pytest.approx(0.2 / math.sqrt(2.0))


def test_mesh_contains_every_wall_sub_edge() -> None:
    mesh = Mesh.build(np.array(two_rooms(2.0)).reshape(-1, 4), 1.0)
    assert np.isin(_edge_keys(mesh.con_edges, len(mesh.P)), mesh._edge_key).all()


def test_free_edges_finds_a_door_and_refuses_walls_and_non_edges() -> None:
    mesh = Mesh.build(np.array(two_rooms(2.0)).reshape(-1, 4), 1.0)
    door = mesh.free_edges(np.array([[6.0, 2.0]]), np.array([[6.0, 4.0]]))
    assert door[0] >= 0
    assert mesh.e_len[door[0]] == pytest.approx(2.0)
    a, b = mesh.P[mesh.con_edges[:1, 0]], mesh.P[mesh.con_edges[:1, 1]]
    assert mesh.free_edges(a, b).tolist() == [-1]
    assert mesh.free_edges(np.array([[3.0, 3.0]]), np.array([[9.0, 3.0]])).tolist() == [-1]
    assert len(mesh.free_edges(np.zeros((0, 2)), np.zeros((0, 2)))) == 0


def _dock(x: float, y: float, theta: float) -> list[tuple[float, float, float, float]]:
    """A 1.2 x 1.0 m dock footprint (a charger of the walkway world) at (x, y), turned by theta."""
    c, s = math.cos(theta), math.sin(theta)
    corners = [(x + c * u - s * v, y + s * u + c * v) for u, v in ((0.25, -0.5), (1.45, -0.5), (1.45, 0.5), (0.25, 0.5))]
    return [(*a, *b) for a, b in zip(corners, corners[1:] + corners[:1], strict=True)]


def _room_with_docks(gap: float, f32: bool) -> np.ndarray:
    """Two chargers side by side in a room, as on the walkway world (x 152.9, theta 3.1264, the second 15 mm along):
    `gap` is how far apart their shared edges lie; float32 as the walls come over AddWalls (Point32)."""
    room = [(145.0, 15.0, 160.0, 15.0), (160.0, 15.0, 160.0, 30.0), (160.0, 30.0, 145.0, 30.0), (145.0, 30.0, 145.0, 15.0)]
    segs = np.array(room + _dock(152.9143, 23.4839, 3.1264) + _dock(152.8991, 22.4839 + gap, 3.1264), dtype=np.float64)
    return segs.astype(np.float32).astype(np.float64) if f32 else segs


def test_footprints_that_share_an_edge_a_hair_apart_mesh_as_when_they_share_it() -> None:
    """Two docks side by side share an edge; in the walkway world their two copies of it lay 1e-4 m apart, and float32
    on the way. Kept apart, the two lines forced splits down to MIN_LEN along the whole edge: 3800 triangles more on
    the walkway world, its mesh four times slower to build. Within SNAP they are one edge, and the mesh is the one of
    the exactly shared edge."""
    exact = Mesh.build(_room_with_docks(0.0, f32=False), 1.2)
    for gap in (1e-4, -1e-4, 2e-5):
        near = Mesh.build(_room_with_docks(gap, f32=True), 1.2)
        assert len(near.P) == len(exact.P)
        assert len(near.V) == len(exact.V)
        np.testing.assert_allclose(np.sort(near.P, axis=0), np.sort(exact.P, axis=0), atol=2 * SNAP)
        np.testing.assert_allclose(np.sort(near.arc_w), np.sort(exact.arc_w), atol=2 * SNAP)


def test_footprints_further_apart_than_snap_keep_their_gap() -> None:
    """A real gap (5 mm) stays: the rule merges only what lies within SNAP."""
    mesh = Mesh.build(_room_with_docks(5e-3, f32=False), 1.2)
    exact = Mesh.build(_room_with_docks(0.0, f32=False), 1.2)
    assert len(mesh.P) > len(exact.P)


def test_a_wall_that_stops_short_of_another_by_less_than_snap_meets_it() -> None:
    touching = Mesh.build(_segs((0.0, 0.0, 4.0, 0.0), (2.0, 0.0, 2.0, 2.0), (0.0, 3.0, 4.0, 3.0)), 1.2)
    short = Mesh.build(_segs((0.0, 0.0, 4.0, 0.0), (2.0, 0.0005, 2.0, 2.0), (0.0, 3.0, 4.0, 3.0)), 1.2)
    assert len(short.P) == len(touching.P)
    assert len(short.V) == len(touching.V)
