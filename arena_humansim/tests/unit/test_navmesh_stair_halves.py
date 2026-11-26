"""An agent that does not climb a stair or escalator half (a wheelchair user) is never routed into one: its routes, straight
legs and nearest reachable point keep out of every half of a kind it does not climb (`NavMeshPlanner.set_areas`).

The walls and halves are those of the north stair of the T junction of smu_link_full_v3 (both flights, joined in place at
their seams, which the planner takes as walls). In the v3 study crowd a wheelchair user whose goal lay beyond the stair
was routed to the nearest point it could reach, up to the seam, 1.32 m into the deck-side top half (r1.0, seed 7)."""

from __future__ import annotations

from collections.abc import Callable

import attrs
import numpy as np

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.global_planner import LANE_OFFSET_M
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D

#: smu_link_full_v3, closed and outside walls touching x 116-134, y 42-51
WALLS = [
    ((116.33, 41.978), (116.33, 42.296)),
    ((116.33, 42.296), (116.339, 42.385)),
    ((116.339, 42.385), (116.85, 42.819)),
    ((116.85, 42.819), (116.77, 43.462)),
    ((116.77, 43.462), (116.674, 43.6)),
    ((116.674, 43.6), (116.73, 44.142)),
    ((116.73, 44.142), (116.687, 44.267)),
    ((116.687, 44.267), (116.561, 44.315)),
    ((116.561, 44.315), (116.725, 44.519)),
    ((116.725, 44.519), (116.808, 44.414)),
    ((116.808, 44.414), (116.89, 44.699)),
    ((116.89, 44.699), (116.77, 45.584)),
    ((116.77, 45.584), (116.81, 46.14)),
    ((116.81, 46.14), (116.743, 46.66)),
    ((116.743, 46.66), (116.93, 47.1)),
    ((116.93, 47.1), (116.674, 47.44)),
    ((116.674, 47.44), (116.807, 47.587)),
    ((116.807, 47.587), (116.648, 47.61)),
    ((116.648, 47.61), (116.661, 47.73)),
    ((116.661, 47.73), (116.405, 47.639)),
    ((116.405, 47.639), (116.59, 47.824)),
    ((116.59, 47.824), (116.59, 48.07)),
    ((116.59, 48.07), (116.73, 48.139)),
    ((116.73, 48.139), (116.66, 48.21)),
    ((116.66, 48.21), (116.655, 48.12)),
    ((116.655, 48.12), (116.55, 48.123)),
    ((116.55, 48.123), (116.55, 48.717)),
    ((116.55, 48.717), (116.655, 48.72)),
    ((116.655, 48.72), (116.662, 48.576)),
    ((116.662, 48.576), (116.761, 48.675)),
    ((116.761, 48.675), (116.67, 48.421)),
    ((116.67, 48.421), (116.742, 48.39)),
    ((116.742, 48.39), (116.89, 48.779)),
    ((116.89, 48.779), (116.89, 49.143)),
    ((116.89, 49.143), (116.65, 49.504)),
    ((116.65, 49.504), (116.65, 50.016)),
    ((116.65, 50.016), (116.81, 50.93)),
    ((116.81, 50.93), (116.926, 51.0)),
    ((116.926, 51.0), (116.85, 51.076)),
    ((131.6, 44.23), (131.6, 42.65)),
    ((131.6, 42.65), (124.2, 42.65)),
    ((124.2, 42.65), (123.1, 42.65)),
    ((123.1, 42.65), (123.1, 44.23)),
    ((123.1, 44.23), (124.2, 44.23)),
    ((124.2, 44.23), (131.6, 44.23)),
    ((128.09, 48.0), (132.6, 48.0)),
    ((132.6, 48.0), (132.6, 48.217)),
    ((132.6, 48.217), (133.2, 48.215)),
    ((133.2, 48.215), (133.2, 47.718)),
    ((133.2, 47.718), (133.4, 47.7)),
    ((133.4, 47.7), (133.4, 47.618)),
    ((133.4, 47.618), (133.6, 47.6)),
    ((133.6, 47.6), (133.618, 47.5)),
    ((133.618, 47.5), (133.882, 47.5)),
    ((133.882, 47.5), (133.9, 47.6)),
    ((133.9, 47.6), (134.0, 47.618)),
    ((134.0, 47.618), (134.0, 48.284)),
    ((134.0, 48.284), (134.304, 48.35)),
    ((134.68, 48.505), (131.65, 48.45)),
    ((131.65, 48.45), (131.6, 48.52)),
    ((131.6, 48.52), (131.592, 48.3)),
    ((131.592, 48.3), (131.56, 48.3)),
    ((131.56, 48.3), (131.337, 48.37)),
    ((131.337, 48.37), (131.343, 48.3)),
    ((131.343, 48.3), (130.994, 48.3)),
    ((130.994, 48.3), (130.91, 48.345)),
    ((130.91, 48.345), (130.91, 48.47)),
    ((130.91, 48.47), (131.19, 48.43)),
    ((131.19, 48.43), (131.185, 48.65)),
    ((131.185, 48.65), (130.296, 48.67)),
    ((130.296, 48.67), (130.19, 48.776)),
    ((130.19, 48.776), (130.241, 49.05)),
    ((130.241, 49.05), (129.559, 49.05)),
    ((129.559, 49.05), (129.547, 48.85)),
    ((129.547, 48.85), (129.724, 48.85)),
    ((129.724, 48.85), (129.536, 48.658)),
    ((129.536, 48.658), (129.462, 48.732)),
    ((129.462, 48.732), (129.202, 48.689)),
    ((129.202, 48.689), (129.139, 48.77)),
    ((129.139, 48.77), (129.092, 48.622)),
    ((129.092, 48.622), (128.97, 48.744)),
    ((128.97, 48.744), (128.97, 48.9)),
    ((128.97, 48.9), (128.87, 48.9)),
    ((128.87, 48.9), (128.79, 48.782)),
    ((128.79, 48.782), (128.832, 48.69)),
    ((128.832, 48.69), (128.024, 48.63)),
    ((128.024, 48.63), (127.919, 48.525)),
    ((127.919, 48.525), (127.957, 48.63)),
    ((127.957, 48.63), (127.884, 48.65)),
    ((127.884, 48.65), (127.424, 48.65)),
    ((127.424, 48.65), (127.3, 48.73)),
    ((127.3, 48.73), (127.262, 48.65)),
    ((127.262, 48.65), (127.104, 48.65)),
    ((127.104, 48.65), (126.33, 48.77)),
    ((126.33, 48.77), (126.263, 49.33)),
    ((126.263, 49.33), (124.659, 49.37)),
    ((124.659, 49.37), (124.0, 49.114)),
    ((124.0, 49.114), (123.992, 48.934)),
    ((123.992, 48.934), (124.078, 48.951)),
    ((124.078, 48.951), (124.256, 48.75)),
    ((124.256, 48.75), (124.016, 48.75)),
    ((124.016, 48.75), (123.876, 48.85)),
    ((123.876, 48.85), (123.17, 48.85)),
    ((123.17, 48.85), (123.17, 48.372)),
    ((123.17, 48.372), (123.578, 48.3)),
    ((123.578, 48.3), (123.0, 48.3)),
    ((123.0, 48.3), (123.0, 48.236)),
    ((123.0, 48.236), (123.4, 48.2)),
    ((123.4, 48.2), (123.4, 48.172)),
    ((123.4, 48.172), (123.61, 48.14)),
    ((123.61, 48.14), (123.65, 48.0)),
    ((123.65, 48.0), (127.688, 48.0)),
    ((127.688, 48.0), (127.71, 48.04)),
    ((127.71, 48.04), (128.01, 48.04)),
    ((128.01, 48.04), (128.09, 48.0)),
    ((131.5, 42.65), (129.75, 42.65)),
    ((128.0, 42.65), (129.75, 42.65)),
    ((126.7, 42.65), (124.95, 42.65)),
    ((123.2, 42.65), (124.95, 42.65)),
    ((131.5, 44.23), (129.75, 44.23)),
    ((131.5, 48.0), (129.75, 48.0)),
    ((128.0, 48.0), (129.75, 48.0)),
    ((128.0, 44.23), (129.75, 44.23)),
    ((126.7, 44.23), (124.95, 44.23)),
    ((126.7, 48.0), (124.95, 48.0)),
    ((123.2, 48.0), (124.95, 48.0)),
    ((123.2, 44.23), (124.95, 44.23)),
]
#: the stair seams joined in place (TeleportZones.gates), walls for the global planner
GATES = [((124.95, 48.0), (124.95, 44.23)), ((129.75, 48.0), (129.75, 44.23))]
#: a box round them, so the floor is bounded
BOX = [((115.0, 41.0), (135.0, 41.0)), ((135.0, 41.0), (135.0, 52.0)), ((135.0, 52.0), (115.0, 52.0)), ((115.0, 52.0), (115.0, 41.0))]
#: the halves of the north stair: name, kind, footprint
HALVES = {
    "stairs_t_junction_north_upper_top": ("stairs", ((123.2, 44.23), (124.95, 44.23), (124.95, 48.0), (123.2, 48.0))),
    "stairs_t_junction_north_upper_bottom": ("stairs", ((126.7, 48.0), (124.95, 48.0), (124.95, 44.23), (126.7, 44.23))),
    "stairs_t_junction_north_lower_top": ("stairs", ((128.0, 44.23), (129.75, 44.23), (129.75, 48.0), (128.0, 48.0))),
    "stairs_t_junction_north_lower_bottom": ("stairs", ((131.5, 48.0), (129.75, 48.0), (129.75, 44.23), (131.5, 44.23))),
}
START = (122.40, 47.85)
#: the landing between the two flights: no walking route from the deck, only over the seams
BEYOND = (127.35, 46.1)


def _planner() -> NavMeshPlanner:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25, lane_side="left", lane_offset=LANE_OFFSET_M)
    planner.set_walls([*WALLS, *GATES, *BOX])
    planner.set_timing(0.05, 8.0)
    planner.set_areas(HALVES)
    return planner


def _in_a_half(x: float, y: float) -> bool:
    return any(min(p[0] for p in fp) < x < max(p[0] for p in fp) and min(p[1] for p in fp) < y < max(p[1] for p in fp) for _, fp in HALVES.values())


def _go(planner: NavMeshPlanner, agent: BaseAgent, target: tuple[float, float]) -> list[Pose2D]:
    """The route the agent walks (its cached waypoints), or where it walks straight at without one."""
    cmd = {1: HighLevelCommand(agent_id=1, type=CommandType.NAVIGATE, target_pose=Pose2D(x=target[0], y=target[1]), desired_velocity=0.9)}
    goal = planner.compute([agent], cmd)[1]
    return planner.get_cached_paths().get(1) or [agent.state.pose, goal]


def test_a_wheelchair_with_its_goal_beyond_a_stair_is_not_routed_into_the_stair(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner()
    chair = agent_factory(agent_id=1, x=START[0], y=START[1])
    chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
    points = _go(planner, chair, BEYOND)
    assert not any(_in_a_half(p.x, p.y) for p in points)
    for a, b in zip(points, points[1:], strict=False):
        for t in np.linspace(0.0, 1.0, 21):
            assert not _in_a_half(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))


def test_an_adult_that_climbs_stairs_is_still_routed_onto_the_stair(agent_factory: Callable[..., BaseAgent]) -> None:
    """The walls are per class: one that climbs stairs plans on the open floor, up to the seam."""
    planner = _planner()
    adult = agent_factory(agent_id=1, x=START[0], y=START[1])
    adult.params = attrs.evolve(adult.params, agent_radius=0.25, climbs=("stairs", "escalator"))
    points = _go(planner, adult, BEYOND)
    assert any(_in_a_half(p.x, p.y) for p in points)


def test_a_wheelchair_pushed_onto_a_half_plans_its_way_off(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner()
    chair = agent_factory(agent_id=1, x=124.0, y=46.5)
    chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
    points = _go(planner, chair, (119.0, 46.0))
    assert np.hypot(points[-1].x - 119.0, points[-1].y - 46.0) < 0.5


def _free_edge_near(planner: NavMeshPlanner, x: float, y: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """A mesh edge that is no wall, near (x, y): a wall there is taken by blocking, as a door that closes."""
    mesh = planner._mesh
    assert mesh is not None
    for k in np.argsort(np.hypot(mesh.e_mid[:, 0] - x, mesh.e_mid[:, 1] - y)).tolist():
        a, b = mesh.P[mesh.ea[k]], mesh.P[mesh.eb[k]]
        if mesh.free_edges(np.array([a]), np.array([b]))[0] >= 0:
            return (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    raise AssertionError("no free edge")


def test_the_walled_off_mesh_lasts_through_walls_that_come_and_go(agent_factory: Callable[..., BaseAgent]) -> None:
    """Doors and other mechanisms add walls and take them away again; the main mesh takes that by blocking edges, and
    the walled-off mesh does the same instead of being built again."""
    planner = _planner()
    chair = agent_factory(agent_id=1, x=START[0], y=START[1])
    chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
    _go(planner, chair, BEYOND)
    assert len(planner.nogo_builds) == 1
    door = _free_edge_near(planner, 119.5, 46.5)
    walls = [*WALLS, *GATES, *BOX]
    for k in range(6):
        planner.set_walls([*walls, door] if k % 2 == 0 else walls)
        points = _go(planner, chair, BEYOND)
        assert not any(_in_a_half(p.x, p.y) for p in points)
    assert len(planner.nogo_builds) == 1


def test_wheelchair_users_on_different_halves_share_one_walled_off_mesh(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner()
    for aid, (x, y) in enumerate([START, (124.0, 46.5), (125.8, 46.0), (122.0, 45.5)], start=1):
        chair = agent_factory(agent_id=aid, x=x, y=y)
        chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
        cmd = {aid: HighLevelCommand(agent_id=aid, type=CommandType.NAVIGATE, target_pose=Pose2D(x=119.0, y=46.0), desired_velocity=0.9)}
        planner.compute([chair], cmd)
    assert len(planner.nogo_builds) == 1


def test_a_walled_off_mesh_built_while_a_door_is_closed_lasts_when_it_opens(agent_factory: Callable[..., BaseAgent]) -> None:
    """The walled-off mesh is meshed on the walls of the main mesh, the door blocked on it as on the main mesh: the door
    opening takes it away by blocking too, between the same two ticks, instead of building the mesh again (2.6 s within
    a tick on the walkway world)."""
    planner = _planner()
    door = _free_edge_near(planner, 119.5, 46.5)
    walls = [*WALLS, *GATES, *BOX]
    planner.set_walls([*walls, door])
    chair = agent_factory(agent_id=1, x=START[0], y=START[1])
    chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
    _go(planner, chair, BEYOND)
    assert len(planner.nogo_builds) == 1
    for k in range(5):
        planner.set_walls(walls if k % 2 == 0 else [*walls, door])
        points = _go(planner, chair, BEYOND)
        assert not any(_in_a_half(p.x, p.y) for p in points)
    assert len(planner.nogo_builds) == 1


def _routers(planner: NavMeshPlanner) -> int:
    return len(planner._routers) + len(planner._body_routers) + sum(1 + len(n.routers) + len(n.body) for n in planner._nogo.values())


def test_prepare_builds_the_walled_off_mesh_and_routers_before_the_first_route(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = _planner()
    planner.prepare((), [0.325])
    assert len(planner.nogo_builds) == 1
    built = _routers(planner)
    chair = agent_factory(agent_id=1, x=START[0], y=START[1])
    chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
    points = _go(planner, chair, BEYOND)
    assert not any(_in_a_half(p.x, p.y) for p in points)
    assert len(planner.nogo_builds) == 1
    assert _routers(planner) == built


def test_prepare_is_idempotent() -> None:
    """Asked again, it builds nothing; classes that do not climb the same halves (every half here is a stair) share one
    walled-off mesh."""
    planner = _planner()
    for _ in range(3):
        planner.prepare(("escalator",), [0.3, 0.325])
        planner.prepare([], [0.325])
    assert len(planner.nogo_builds) == 1
    built = _routers(planner)
    planner.prepare((), [0.325, 0.3])
    assert _routers(planner) == built
    assert len(planner.nogo_builds) == 1


def test_an_agent_class_that_climbs_every_kind_needs_no_walled_off_mesh() -> None:
    planner = _planner()
    planner.prepare(("stairs", "escalator"), [0.25])
    planner.prepare(None, [0.25])
    assert planner.nogo_builds == []


def test_prepare_before_the_walls_builds_once_the_walls_and_halves_come(agent_factory: Callable[..., BaseAgent]) -> None:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, clearance_margin=0.25, lane_side="left", lane_offset=LANE_OFFSET_M)
    planner.set_timing(0.05, 8.0)
    planner.prepare((), [0.325])
    planner.set_walls([*WALLS, *GATES, *BOX])
    assert planner.nogo_builds == []
    planner.set_areas(HALVES)
    assert len(planner.nogo_builds) == 1
    chair = agent_factory(agent_id=1, x=START[0], y=START[1])
    chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
    _go(planner, chair, BEYOND)
    assert len(planner.nogo_builds) == 1


def test_walls_meshed_again_at_a_reset_are_warmed_again_at_once(agent_factory: Callable[..., BaseAgent]) -> None:
    """A change of walls the main mesh cannot block (a wall it was meshed on goes away: the next world) builds the
    prepared walled-off mesh again within that change, not at the next route."""
    planner = _planner()
    planner.prepare((), [0.325])
    assert len(planner.nogo_builds) == 1
    planner.set_walls([*WALLS[1:], *GATES, *BOX])
    assert len(planner.nogo_builds) == 2
    planner.configure(inflation_radius=0.4, resolution=0.1, comfort_radius=0.6)
    assert len(planner.nogo_builds) == 3
    chair = agent_factory(agent_id=1, x=START[0], y=START[1])
    chair.params = attrs.evolve(chair.params, agent_radius=0.325, climbs=())
    built = _routers(planner)
    _go(planner, chair, BEYOND)
    assert len(planner.nogo_builds) == 3
    assert _routers(planner) == built


def test_forget_prepared_leaves_the_next_world_cold() -> None:
    planner = _planner()
    planner.prepare((), [0.325])
    planner.forget_prepared()
    planner.set_walls([*WALLS[1:], *GATES, *BOX])
    assert len(planner.nogo_builds) == 1
