"""Route choice over stairs: a walker whose target walking reaches also rides a stair half beside its partner where that
is quicker (zone_route_choice), timed as the stairs joined in place are, on the planner's congestion weights."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import attrs
import numpy as np
import pytest
from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.teleport_zones import CHOICE_INTERVAL, CHOICE_STAGGER, STAIR_SPEED, SWITCH_M, TeleportZones, Zone
from arena_humansim.global_planner.navmesh import NavMeshPlanner
from arena_humansim.utils.types import CommandType, HighLevelCommand, Pose2D, Segments

# two rooms joined by a 3 m wide corridor over the top (about 51 m from A's middle to B's) and by a stair whose halves are
# 4 m pockets off the facing walls (about 21 m with the climb)
WALLS: Segments = [
    ((0.0, 0.0), (10.0, 0.0)),
    ((0.0, 0.0), (0.0, 10.0)),
    ((0.0, 10.0), (3.0, 10.0)),
    ((6.0, 10.0), (10.0, 10.0)),
    ((10.0, 0.0), (10.0, 4.0)),
    ((10.0, 6.0), (10.0, 10.0)),
    ((20.0, 0.0), (30.0, 0.0)),
    ((30.0, 0.0), (30.0, 10.0)),
    ((20.0, 10.0), (23.0, 10.0)),
    ((26.0, 10.0), (30.0, 10.0)),
    ((20.0, 0.0), (20.0, 4.0)),
    ((20.0, 6.0), (20.0, 10.0)),
    ((3.0, 10.0), (3.0, 22.0)),
    ((6.0, 10.0), (6.0, 19.0)),
    ((3.0, 22.0), (26.0, 22.0)),
    ((6.0, 19.0), (23.0, 19.0)),
    ((26.0, 22.0), (26.0, 10.0)),
    ((23.0, 19.0), (23.0, 10.0)),
    ((10.0, 4.0), (14.0, 4.0)),
    ((10.0, 6.0), (14.0, 6.0)),
    ((14.0, 4.0), (14.0, 6.0)),
    ((16.0, 4.0), (20.0, 4.0)),
    ((16.0, 6.0), (20.0, 6.0)),
    ((16.0, 4.0), (16.0, 6.0)),
]
IN_B = Pose2D(x=25.0, y=5.0)
IN_A = Pose2D(x=2.0, y=8.0)
SPEED = 1.1
#: the climb of both halves at stair speed, effort 1: (13.55 - 10) + (20 - 16.45) m
RUN_S = 7.1 / STAIR_SPEED


def _box(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _stairs(kind: str = "stairs", belt: float = 0.0) -> list[Zone]:
    a_entry, b_entry = Pose2D(x=10.0, y=5.0), Pose2D(x=20.0, y=5.0, theta=math.pi)
    return [
        Zone(name="st_a", destinations=("st_b",), entry=a_entry, wait=Pose2D(x=9.2, y=5.0), hold=Pose2D(x=13.55, y=5.0), footprint=_box(10.0, 4.0, 14.0, 6.0), kind=kind, belt=belt),
        Zone(name="st_b", destinations=("st_a",), entry=b_entry, wait=Pose2D(x=20.8, y=5.0, theta=math.pi), hold=Pose2D(x=16.45, y=5.0, theta=math.pi), footprint=_box(16.0, 4.0, 20.0, 6.0), kind=kind, belt=-belt),
    ]


def _zones(zones: list[Zone], route_choice: bool = True) -> TeleportZones:
    tz = TeleportZones()
    tz.set_zones(zones, [True] * len(zones))
    tz.route_choice = route_choice
    return tz


@pytest.fixture
def planner() -> NavMeshPlanner:
    p = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6)
    p.set_walls(WALLS)
    return p


def _walker(agent_factory: Callable[..., BaseAgent], planner: object, aid: int, x: float, y: float, climbs: tuple[str, ...] = ("stairs", "escalator")) -> BaseAgent:
    agent = agent_factory(agent_id=aid, x=x, y=y)
    agent.global_planner = planner
    agent.params = attrs.evolve(agent.params, climbs=climbs, desired_velocity=SPEED, stair_effort=1.0)
    agent.state.desired_velocity = SPEED
    return agent


def _go(agents: list[BaseAgent], goal: Pose2D) -> dict[int, HighLevelCommand]:
    return {a.state.agent_id: HighLevelCommand(agent_id=a.state.agent_id, type=CommandType.NAVIGATE, target_pose=goal) for a in agents}


def test_walker_takes_the_stair_over_a_long_detour(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_B), now=0.0)
    transfer = tz.transfers[1]
    assert (transfer.zone, transfer.destination, transfer.ride) == ("st_a", "st_b", True)
    assert out[1].target_pose == tz.zones["st_a"].lane


def test_without_route_choice_a_walkable_target_is_walked_to(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs(), route_choice=False)
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


def test_without_a_time_a_walkable_target_is_walked_to(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B))
    assert 1 not in tz.transfers


def test_wheelchair_walks_the_detour(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    chair, walker = _walker(agent_factory, planner, 1, 5.0, 5.0, climbs=()), _walker(agent_factory, planner, 2, 5.0, 4.0)
    out = tz.steer([chair, walker], _go([chair, walker], IN_B), now=0.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B
    assert tz.transfers[2].zone == "st_a"


def test_type_that_climbs_only_stairs_walks_past_an_escalator(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs("escalator", 0.6))
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0, climbs=("stairs",))
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in tz.transfers


def test_a_target_close_by_is_walked_to(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    out = tz.steer([walker], _go([walker], IN_A), now=0.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_A


def test_one_way_escalator_is_only_taken_in_its_direction(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    zones = _stairs("escalator", 0.6)
    zones[1] = attrs.evolve(zones[1], destinations=())
    tz = _zones(zones)
    there, back = _walker(agent_factory, planner, 1, 5.0, 5.0), _walker(agent_factory, planner, 2, 25.0, 5.0)
    tz.steer([there], _go([there], IN_B), now=0.0)
    tz.steer([back], _go([back], Pose2D(x=5.0, y=5.0)), now=0.0)
    assert tz.transfers[1].zone == "st_a"
    assert 2 not in tz.transfers


def test_a_belt_running_back_to_its_entry_is_never_ridden(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs("escalator", -0.6))
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in tz.transfers


def test_a_crowded_stair_sends_walkers_round_once_their_choice_is_due(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    assert tz.transfers[1].zone == "st_a"
    xs, ys = np.meshgrid(np.linspace(10.3, 13.7, 10), np.linspace(4.3, 5.7, 8))
    crowd = np.column_stack([xs.ravel(), ys.ravel()])
    planner._set_slowdown(planner._slowdown_at(crowd))
    tz.steer([walker], cmds, now=1.0)
    assert tz.transfers[1].zone == "st_a"  # not due yet
    out = tz.steer([walker], cmds, now=2.0 * CHOICE_INTERVAL)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


@attrs.define(eq=False)
class _Timed:
    """A planner stand-in: one free-space component, straight paths and fixed walking costs per goal [m], per (start,
    goal) where `between` has one."""

    costs: dict[tuple[float, float], float]
    between: dict[tuple[tuple[float, float], tuple[float, float]], float] = attrs.Factory(dict)
    legs: list[tuple[Pose2D, Pose2D]] = attrs.Factory(list)
    #: tree costs per goal [m] where they differ from the path costs (`leg_costs`)
    tree: dict[tuple[float, float], float] = attrs.Factory(dict)

    def components(self, poses: Sequence[Pose2D]) -> list[frozenset[int]]:
        return [frozenset({0}) for _ in poses]

    def paths(self, legs: Sequence[tuple[Pose2D, Pose2D]]) -> list[list[Pose2D] | None]:
        self.legs.extend(legs)
        return [[start, goal] for start, goal in legs]

    def path_costs(self, paths: Sequence[Sequence[Pose2D]]) -> list[float]:
        return [self.path_cost(path) for path in paths]

    def leg_floors(self, legs: Sequence[tuple[Pose2D, Pose2D]]) -> list[float]:
        return [0.0] * len(legs)

    def leg_costs(self, legs: Sequence[tuple[Pose2D, Pose2D]]) -> list[float]:
        self.legs.extend(legs)
        return [self.tree.get((round(goal.x, 2), round(goal.y, 2)), self.path_cost([start, goal])) for start, goal in legs]

    def path_cost(self, path: Sequence[Pose2D]) -> float:
        start, goal = (round(path[0].x, 2), round(path[0].y, 2)), (round(path[-1].x, 2), round(path[-1].y, 2))
        if (start, goal) in self.between:
            return self.between[(start, goal)]
        return self.costs.get(goal, math.hypot(path[-1].x - path[0].x, path[-1].y - path[0].y))


def _priced(walk: float) -> _Timed:
    # board 9 m to st_a's mouth, on 9 m from st_b's mouth to the goal
    return _Timed({(25.0, 5.0): walk, (10.0, 5.0): 9.0, (13.55, 5.0): 3.55, (20.0, 5.0): 3.55}, {((20.0, 5.0), (25.0, 5.0)): 9.0})


def _stair_time() -> float:
    return (9.0 + 9.0) / SPEED + RUN_S


def test_a_walker_keeps_its_choice_unless_another_is_clearly_quicker(agent_factory: Callable[..., BaseAgent]) -> None:
    priced = _priced(100.0)
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    assert tz.transfers[1].zone == "st_a"
    margin = SWITCH_M / SPEED
    stair = _stair_time()
    assert margin > 0.1 * stair
    priced.costs[(25.0, 5.0)] = (stair - (margin - 0.5)) * SPEED
    tz.steer([walker], cmds, now=2.0 * CHOICE_INTERVAL)
    assert tz.transfers[1].zone == "st_a"
    priced.costs[(25.0, 5.0)] = (stair - (margin + 0.5)) * SPEED
    tz.steer([walker], cmds, now=4.0 * CHOICE_INTERVAL)
    assert 1 not in tz.transfers
    priced.costs[(25.0, 5.0)] = (stair + (margin - 0.5)) * SPEED
    tz.steer([walker], cmds, now=6.0 * CHOICE_INTERVAL)
    assert 1 not in tz.transfers


def test_a_walker_on_the_stair_is_not_asked_again(agent_factory: Callable[..., BaseAgent]) -> None:
    priced = _priced(100.0)
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    walker.state.pose = Pose2D(x=12.0, y=5.0)
    priced.costs[(25.0, 5.0)] = 1.0
    out = tz.steer([walker], cmds, now=2.0 * CHOICE_INTERVAL)
    assert tz.transfers[1].zone == "st_a"
    assert out[1].target_pose == tz.zones["st_a"].lane


def test_forget_drops_the_choice(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 in tz._next_choice
    tz.forget(1)
    assert 1 not in tz.transfers
    assert 1 not in tz._next_choice


def test_the_stair_flight_counts_in_the_ride_time(agent_factory: Callable[..., BaseAgent]) -> None:
    walk = (_stair_time() + 10.0) * SPEED
    walker = _walker(agent_factory, _priced(walk), 1, 5.0, 5.0)
    fast = _zones(_stairs())
    fast.steer([walker], _go([walker], IN_B), now=0.0)
    assert fast.transfers[1].zone == "st_a"
    zones = _stairs()
    slow = _zones([attrs.evolve(zones[0], delay=20.0), zones[1]])
    slow.steer([walker], _go([walker], IN_B), now=0.0)
    assert 1 not in slow.transfers


@pytest.mark.parametrize("closing", ["st_a", "st_b"])
def test_a_walker_whose_chosen_half_or_its_partner_closes_is_routed_again_at_once(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner, closing: str) -> None:
    tz = _zones(_stairs())
    walker = _walker(agent_factory, planner, 1, 5.0, 5.0)
    cmds = _go([walker], IN_B)
    tz.steer([walker], cmds, now=0.0)
    assert tz.transfers[1].zone == "st_a"
    tz.open[closing] = False  # set_zone_access
    out = tz.steer([walker], cmds, now=1.0)
    assert 1 not in tz.transfers
    assert out[1].target_pose == IN_B


def test_walkers_that_chose_together_choose_again_spread_over_the_interval(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> None:
    tz = _zones(_stairs())
    walkers = [_walker(agent_factory, planner, aid, 5.0, 2.0 + 0.3 * (aid % 10)) for aid in range(1, 41)]
    tz.steer(walkers, _go(walkers, IN_B), now=0.0)
    due = sorted(set(tz._next_choice.values()))
    assert len(due) == CHOICE_STAGGER and min(due) == CHOICE_INTERVAL and max(due) < 2.0 * CHOICE_INTERVAL


def test_a_walker_far_from_every_stair_plans_only_its_own_walk(agent_factory: Callable[..., BaseAgent]) -> None:
    priced = _Timed({(2.0, 8.0): 3.0})
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 2.0, 5.0)
    tz.steer([walker], _go([walker], IN_A), now=0.0)
    assert 1 not in tz.transfers
    assert len(priced.legs) == 1


def test_paths_leave_the_agents_route_memory_alone(planner: NavMeshPlanner) -> None:
    planner._routed = {0: (1.0, 1.0), 1: (2.0, 2.0)}
    paths = planner.paths([(Pose2D(x=5.0, y=5.0), IN_B), (Pose2D(x=5.0, y=5.0), Pose2D(x=8.0, y=8.0))])
    assert all(p is not None for p in paths)
    assert planner._routed == {0: (1.0, 1.0), 1: (2.0, 2.0)}


def test_a_jam_weighs_the_path_cost(planner: NavMeshPlanner) -> None:
    path = [Pose2D(x=1.0, y=5.0), Pose2D(x=9.0, y=5.0)]
    assert planner.path_cost(path) == pytest.approx(8.0)
    xs, ys = np.meshgrid(np.linspace(3.0, 7.0, 10), np.linspace(4.0, 6.0, 6))
    planner._set_slowdown(planner._slowdown_at(np.column_stack([xs.ravel(), ys.ravel()])))
    assert planner.path_cost(path) > 12.0


@pytest.mark.parametrize(("pricing", "method"), [("tree", "leg_costs"), ("paths", "paths")])
def test_one_steer_prices_every_walker_as_alone_in_four_batches(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner, monkeypatch: pytest.MonkeyPatch, pricing: str, method: str) -> None:
    spots = [(5.0, 2.0 + 0.5 * k) for k in range(12)] + [(25.0, 2.0 + 0.5 * k) for k in range(12)]
    walkers = [_walker(agent_factory, planner, aid, x, y) for aid, (x, y) in enumerate(spots, start=1)]
    cmds = {**_go(walkers[:12], IN_B), **_go(walkers[12:], IN_A)}
    alone = {}
    for walker in walkers:
        tz = _zones(_stairs())
        tz.pricing = pricing
        aid = walker.state.agent_id
        alone[aid] = (tz.steer([walker], {aid: cmds[aid]}, now=0.0)[aid].target_pose, tz.transfers.get(aid))
    calls = []
    priced = getattr(planner, method)
    monkeypatch.setattr(planner, method, lambda legs: calls.append(len(legs)) or priced(legs))
    together = _zones(_stairs())
    together.pricing = pricing
    out = together.steer(walkers, cmds, now=0.0)
    for walker in walkers:
        aid = walker.state.agent_id
        target, transfer = alone[aid]
        assert out[aid].target_pose == target
        both = together.transfers.get(aid)
        assert (transfer is None) == (both is None)
        assert transfer is None or (transfer.zone, transfer.destination, transfer.ride) == (both.zone, both.destination, both.ride)
    assert list(together.transfers) == [w.state.agent_id for w in walkers if w.state.agent_id in together.transfers]
    assert len(calls) <= 4 and sum(calls) > len(walkers)


def _crowd_steers(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner) -> tuple[list[tuple], int]:
    """Five steers of a 60-walker crowd in both rooms and the corridor, half of it heading for each room, in a jam
    round A's stair mouth: every command, transfer, walk memo and choice time after each, and the legs planned."""
    rng = np.random.default_rng(3)
    spots = [(float(rng.uniform(1.0, 9.0)), float(rng.uniform(1.0, 9.0))) for _ in range(20)]
    spots += [(float(rng.uniform(21.0, 29.0)), float(rng.uniform(1.0, 9.0))) for _ in range(20)]
    spots += [(float(rng.uniform(7.0, 22.0)), float(rng.uniform(19.8, 21.2))) for _ in range(20)]
    walkers = [_walker(agent_factory, planner, aid, x, y) for aid, (x, y) in enumerate(spots, start=1)]
    goals = [IN_B, IN_A, Pose2D(x=8.0, y=2.0), Pose2D(x=28.0, y=8.0)]
    cmds = {}
    for k, walker in enumerate(walkers):
        cmds.update(_go([walker], goals[k % len(goals)]))
    xs, ys = np.meshgrid(np.linspace(8.0, 12.0, 8), np.linspace(4.2, 5.8, 4))
    planner._set_slowdown(planner._slowdown_at(np.vstack([np.column_stack([xs.ravel(), ys.ravel()]), np.array(spots)])))
    legs = []
    paths = planner.paths
    planner.paths = lambda asked: legs.append(len(asked)) or paths(asked)  # type: ignore[method-assign]
    tz = _zones(_stairs())
    tz.pricing = "paths"  # the floors cut planned paths only
    seen = []
    for step in range(5):
        for walker in walkers:
            walker.state.pose = Pose2D(x=walker.state.pose.x + 0.3 * math.cos(walker.state.agent_id + step), y=walker.state.pose.y)
        out = tz.steer(walkers, cmds, now=step * CHOICE_INTERVAL / 2.0)
        seen.append(
            (
                {aid: (cmd.target_pose.x, cmd.target_pose.y) for aid, cmd in out.items()},
                {aid: (t.zone, t.destination, t.ride) for aid, t in tz.transfers.items()},
                list(tz.transfers),
                dict(tz._walks),
                dict(tz._next_choice),
            )
        )
    return seen, sum(legs)


def test_left_out_stairs_change_no_choice_of_a_crowd(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stair left out of a walker's choice (its shortest possible way still slower than walking) is one pricing every
    way would not choose: the same crowd steered with every way planned (no floor) takes the same choices."""
    pruned, few = _crowd_steers(agent_factory, planner)
    every = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6)
    every.set_walls(WALLS)
    monkeypatch.setattr(every, "leg_floors", lambda legs: [-math.inf] * len(legs))
    full, many = _crowd_steers(agent_factory, every)
    assert pruned == full
    assert any(choice for _, choice, *_ in pruned)
    assert few < many


def test_leg_floors_are_below_the_priced_path_costs(planner: NavMeshPlanner) -> None:
    rng = np.random.default_rng(5)
    legs = [(Pose2D(x=float(rng.uniform(1.0, 29.0)), y=float(rng.uniform(1.0, 9.0))), Pose2D(x=float(rng.uniform(1.0, 29.0)), y=float(rng.uniform(1.0, 9.0)))) for _ in range(40)]
    legs += [(Pose2D(x=0.2, y=5.0), Pose2D(x=10.0, y=5.0)), (Pose2D(x=5.0, y=9.9), Pose2D(x=20.0, y=5.0))]
    xs, ys = np.meshgrid(np.linspace(3.0, 7.0, 10), np.linspace(4.0, 6.0, 6))
    planner._set_slowdown(planner._slowdown_at(np.column_stack([xs.ravel(), ys.ravel()])))
    paths = planner.paths(legs)
    costs = planner.path_costs([path for path in paths if path])
    floors = [floor for floor, path in zip(planner.leg_floors(legs), paths, strict=True) if path]
    assert len(floors) > 15
    assert all(floor <= cost + 1e-9 for floor, cost in zip(floors, costs, strict=True))


def test_path_costs_are_the_path_costs_one_by_one(planner: NavMeshPlanner) -> None:
    paths = [[Pose2D(x=1.0, y=5.0), Pose2D(x=9.0, y=5.0)], [Pose2D(x=2.0, y=2.0)], [Pose2D(x=4.0, y=4.0), Pose2D(x=4.0, y=4.0)], [Pose2D(x=1.0, y=1.0), Pose2D(x=5.0, y=5.0), Pose2D(x=9.0, y=5.5)]]
    assert planner.path_costs(paths) == [planner.path_cost(path) for path in paths]
    xs, ys = np.meshgrid(np.linspace(3.0, 7.0, 10), np.linspace(4.0, 6.0, 6))
    planner._set_slowdown(planner._slowdown_at(np.column_stack([xs.ravel(), ys.ravel()])))
    assert planner.path_costs(paths) == [planner.path_cost(path) for path in paths]
    assert planner.path_costs(paths)[0] > 12.0


def _lanes_one_by_one(planner: NavMeshPlanner, router: object, polyline: np.ndarray) -> np.ndarray:
    """jazzy's `NavMeshPlanner._lanes` for one route (59d217a), the reference of `_lanes_many`."""
    from arena_humansim.global_planner.navmesh import LANE_WALL_GAP, MIN_BODY_RADIUS
    from arena_humansim.utils.router import VERIFY_EPS

    if planner._lane_side == "none" or planner._lane_offset <= 0.0 or len(polyline) < 3 or router.r < planner._inflation_radius - 1e-9:  # type: ignore[attr-defined]
        return polyline
    body = max(router.r - planner._margin, MIN_BODY_RADIUS) + LANE_WALL_GAP  # type: ignore[attr-defined]
    sign = 1.0 if planner._lane_side == "right" else -1.0
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
        cand = inner + n * (planner._lane_offset * share)
        fits = ~settled & (router.clearance(cand) >= body)  # type: ignore[attr-defined]
        moved[fits] = cand[fits]
        settled |= fits
    if not settled.any():
        return polyline
    out = polyline.copy()
    out[1:-1] = moved
    if router._clear_many([out])[0] >= body - VERIFY_EPS:  # type: ignore[attr-defined]
        return out
    out[1:-1] = inner + (moved - inner) * 0.5
    return out if router._clear_many([out])[0] >= body - VERIFY_EPS else polyline  # type: ignore[attr-defined]


@pytest.mark.parametrize("side", ["left", "right"])
def test_lane_shifts_of_many_routes_at_once_are_those_one_by_one(side: str) -> None:
    planner = NavMeshPlanner(inflation_radius=0.38, comfort_radius=0.6, lane_side=side, lane_offset=0.4)
    planner.set_walls(WALLS)
    router = planner._router
    assert router is not None
    rng = np.random.default_rng(7)
    starts = [Pose2D(x=float(rng.uniform(1.0, 9.0)), y=float(rng.uniform(1.0, 9.0))) for _ in range(30)]
    starts += [Pose2D(x=float(rng.uniform(21.0, 29.0)), y=float(rng.uniform(1.0, 9.0))) for _ in range(30)]
    found = [np.array([(p.x, p.y) for p in path]) for path in planner.paths([(s, IN_B if s.x < 15.0 else IN_A) for s in starts]) if path]
    # the routes over the corridor turn four corners; straight and two-point ones stay as they are
    polylines = [*found, *(np.array([(4.5, y), (4.5, 14.0), (14.0, 20.5), (24.5, 14.0), (24.5, 8.0)]) for y in (2.0, 5.0)), np.array([(1.0, 1.0), (2.0, 2.0)])]
    assert sum(len(p) >= 3 for p in polylines) > 20
    many = planner._lanes_many(router, polylines)
    for polyline, batched in zip(polylines, many, strict=True):
        alone = _lanes_one_by_one(planner, router, polyline)
        assert batched.tobytes() == alone.tobytes()
    assert any(not np.array_equal(b, p) for b, p in zip(many, polylines, strict=True))



@pytest.mark.parametrize("pricing", ["tree", "paths"])
def test_both_pricings_take_the_stair_over_a_long_detour_and_walk_to_a_target_close_by(agent_factory: Callable[..., BaseAgent], planner: NavMeshPlanner, pricing: str) -> None:
    tz = _zones(_stairs())
    tz.pricing = pricing
    far, near = _walker(agent_factory, planner, 1, 5.0, 5.0), _walker(agent_factory, planner, 2, 5.0, 5.0)
    tz.steer([far, near], {**_go([far], IN_B), **_go([near], Pose2D(x=8.0, y=8.0))}, now=0.0)
    assert (tz.transfers[1].zone, tz.transfers[1].ride) == ("st_a", True)
    assert 2 not in tz.transfers


def test_tree_costs_lie_between_the_straight_line_and_a_third_over_the_planned_path(planner: NavMeshPlanner) -> None:
    """Tree costs run through edge midpoints: never below the straight line between the ends, the straight line where
    it keeps the radius, and some 10-30 % above the funnelled path where the way bends (more in a jam, weighed at mesh edges)."""
    rng = np.random.default_rng(4)
    legs = [(Pose2D(x=float(rng.uniform(1.0, 9.0)), y=float(rng.uniform(1.0, 9.0))), IN_B if k % 2 else Pose2D(x=float(rng.uniform(1.0, 9.0)), y=float(rng.uniform(1.0, 9.0)))) for k in range(30)]
    for crowd in (False, True):
        if crowd:
            xs, ys = np.meshgrid(np.linspace(3.0, 7.0, 10), np.linspace(19.5, 21.5, 4))
            planner._set_slowdown(planner._slowdown_at(np.column_stack([xs.ravel(), ys.ravel()])))
        tree = planner.leg_costs(legs)
        paths = planner.paths(legs)
        costs = planner.path_costs([path for path in paths if path])
        assert len(costs) == len(legs)
        ratios = []
        for (a, b), t, c in zip(legs, tree, costs, strict=True):
            assert t >= math.hypot(b.x - a.x, b.y - a.y) - 1e-6
            ratios.append(t / c)
        # here 1.00 for the legs within a room (their straight line), 1.12-1.26 for those round the corridor, up to
        # ~1.5 with the jam in it: the tree weighs the crowd at mesh edges, path_cost at points along the path
        assert 0.95 <= min(ratios) and max(ratios) <= (1.6 if crowd else 1.3)
        assert sum(r <= 1.0 + 1e-9 for r in ratios) >= 10
    assert planner.leg_costs([(Pose2D(x=5.0, y=5.0), Pose2D(x=15.0, y=5.0))]) == [math.inf], "a goal inside the walls is out of reach"


def test_tree_pricing_answers_legs_from_one_tree_per_goal_kept_apart_from_the_planners(planner: NavMeshPlanner) -> None:
    router = planner._router
    assert router is not None
    planner.leg_costs([(Pose2D(x=1.0 + k, y=2.0), IN_B) for k in range(8)])
    assert len(router._cost_trees) == 1 and not router._trees
    planner._set_slowdown(planner._slowdown_at(np.array([[5.0, 5.0]])))
    assert not router._cost_trees, "a congestion update drops the trees"


def test_the_shadow_counts_every_choice_and_the_ones_the_other_pricing_makes_differently(agent_factory: Callable[..., BaseAgent]) -> None:
    """Planned paths make the stair quicker (100 m walk), the tree makes walking quicker (20 m): the crowd follows its
    pricing, and the shadow counts the choice the other pricing would have made, by walker class."""
    for pricing, chosen, other in (("tree", "walk", "st_a"), ("paths", "st_a", "walk")):
        priced = _priced(100.0)
        priced.tree[(25.0, 5.0)] = 20.0
        tz = _zones(_stairs())
        tz.pricing, tz.shadow = pricing, True
        walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
        tz.steer([walker], _go([walker], IN_B), now=0.0)
        assert (1 in tz.transfers) == (chosen == "st_a")
        cls = walker.params.name or "walker"
        assert tz.shadow_choices == {cls: 1}
        assert tz.shadow_flips == {(cls, chosen, other): 1}
    tz.clear_transfers()
    assert not tz.shadow_choices and not tz.shadow_flips


def test_without_the_shadow_nothing_is_priced_twice(agent_factory: Callable[..., BaseAgent]) -> None:
    priced = _priced(100.0)
    tz = _zones(_stairs())
    walker = _walker(agent_factory, priced, 1, 5.0, 5.0)
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    once = len(priced.legs)
    tz.shadow = True
    tz.forget(1)
    priced.legs.clear()
    tz.steer([walker], _go([walker], IN_B), now=0.0)
    assert len(priced.legs) > once
    assert not tz.shadow_flips
