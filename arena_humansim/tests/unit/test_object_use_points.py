"""An object's capacity is how many walkers use it at once: use points in the line's front row, one hold clock each."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.behavior.nodes.navigation import _approach_pose_for
from arena_humansim.core.formation.anchor import PoseAnchor
from arena_humansim.core.formation.line import LineFormation
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.core.interaction_manager import InteractionManager, LifecycleEdge
from arena_humansim.core.world_knowledge import WorldKnowledge, WorldObject
from arena_humansim.utils.rng import RNG
from arena_humansim.utils.types import AnchorKind, BehaviorTreeMovement, CommandType, FormationSpec, HighLevelCommand, InteractionOutcome, Pose2D, SeekSpec

DT = 0.05
STEP = 1.0
FRONT = 0.8


@dataclass
class _FakeParams:
    reaction_time: float = 0.4
    personal_space_min: float = 0.6


@dataclass
class _FakeState:
    agent_id: int = 0
    pose: Pose2D = field(default_factory=Pose2D)
    desired_velocity: float = 1.2
    kind: int = 0


@dataclass
class _FakeAgent:
    state: _FakeState
    params: _FakeParams = field(default_factory=_FakeParams)
    movement: BehaviorTreeMovement = field(default_factory=BehaviorTreeMovement)


def _agents(n: int) -> dict[int, _FakeAgent]:
    return {aid: _FakeAgent(_FakeState(agent_id=aid, pose=Pose2D(x=-2.0 - aid, y=0.0))) for aid in range(1, n + 1)}


def _fountain(capacity: int, seats: list[Pose2D] | None = None) -> WorldObject:
    line = FormationSpec(type="line", params={"base_step": STEP, "front_offset": FRONT}, anchor_kind=AnchorKind.OBJECT)
    return WorldObject(object_id="fountain", type="fountain", pose=Pose2D(x=0.0, y=0.0, theta=0.0), capacity=capacity, formation=line, seats=list(seats or []))


def _mgr(agents: dict[int, _FakeAgent], obj: WorldObject) -> InteractionManager:
    world = WorldKnowledge()
    world.add_object(obj)
    mgr = InteractionManager(RNG(0))
    mgr.set_context(world_knowledge=world, agent_lookup=agents.get, visibility_lookup=lambda aid: set(agents) - {aid})
    return mgr


def _seek(mgr: InteractionManager, aid: int, duration: float, max_participants: int | None = None) -> None:
    spec = SeekSpec(interaction_type=InteractionType.USE, target="fountain", duration=duration, max_participants=max_participants)
    mgr.update({aid: HighLevelCommand(agent_id=aid, type=CommandType.SEEK, spec=spec)}, dt=DT)


def _walk_to_targets(mgr: InteractionManager, agents: dict[int, _FakeAgent], ids: list[int]) -> None:
    for aid in ids:
        target = mgr.formation_target(aid)
        assert target is not None
        agents[aid].state.pose = Pose2D(x=target.x, y=target.y, theta=target.theta)


def _holds(mgr: InteractionManager) -> list[tuple[LifecycleEdge, tuple[int, ...]]]:
    return [(e.edge, e.participants) for e in mgr.drain_edges() if e.edge in (LifecycleEdge.HOLD_ONSET, LifecycleEdge.RELEASED)]


def test_use_points_follow_capacity_except_for_seats_and_single_use() -> None:
    assert _fountain(3).use_points == 3
    assert _fountain(1).use_points == 1
    assert _fountain(0).use_points == 1
    assert _fountain(-1).use_points == 1
    assert _fountain(3, seats=[Pose2D(), Pose2D(x=1.0), Pose2D(x=2.0)]).use_points == 1


def test_line_points_stand_side_by_side_and_the_wait_starts_behind_the_row() -> None:
    f = LineFormation(anchor=PoseAnchor(fixed=Pose2D(x=0.0, y=0.0, theta=0.0)), agent_lookup=lambda _aid: None, base_step=STEP, front_offset=FRONT, points=3)
    for aid in (1, 2, 3):
        f.on_join(aid, participant=True)
    f.on_join(4, participant=False)
    f.on_join(5, participant=False)
    t = f.tick(DT)
    assert [(round(t[a].x, 6), round(t[a].y, 6)) for a in (1, 2, 3)] == [(-FRONT, -STEP), (-FRONT, 0.0), (-FRONT, STEP)]
    assert (t[4].x, t[4].y) == pytest.approx((-FRONT - STEP, 0.0))
    assert (t[5].x, t[5].y) == pytest.approx((-FRONT - 2 * STEP, 0.0))


def test_a_line_point_is_kept_until_its_holder_leaves_and_the_head_takes_it() -> None:
    f = LineFormation(anchor=PoseAnchor(fixed=Pose2D()), agent_lookup=lambda _aid: None, base_step=STEP, front_offset=FRONT, points=2)
    f.on_join(1, participant=True)
    f.on_join(2, participant=True)
    f.on_join(3, participant=False)
    before = f.tick(DT)
    f.on_leave(1)
    f.on_join(3, participant=True)
    after = f.tick(DT)
    assert after[2] == before[2]
    assert after[3] == before[1]
    assert f.slot_of(3) == before[1]


def test_a_point_line_needs_a_point() -> None:
    with pytest.raises(ValueError, match="at least one use point"):
        LineFormation(anchor=PoseAnchor(fixed=Pose2D()), agent_lookup=lambda _aid: None, points=0)


def test_capacity_three_serves_three_at_once_and_queues_the_rest() -> None:
    agents = _agents(5)
    mgr = _mgr(agents, _fountain(3))
    for aid in range(1, 6):
        _seek(mgr, aid, duration=1.0)
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.contract.max_participants == 3
    assert interaction.participants == [1, 2, 3] and interaction.contract.queue == [4, 5]
    mgr.drain_edges()
    _walk_to_targets(mgr, agents, [1, 2, 3])
    mgr.update({}, dt=DT)
    assert sorted(_holds(mgr)) == [(LifecycleEdge.HOLD_ONSET, (1,)), (LifecycleEdge.HOLD_ONSET, (2,)), (LifecycleEdge.HOLD_ONSET, (3,))]
    assert mgr.is_holding(interaction)
    points = {(round(mgr.formation_target(a).x, 6), round(mgr.formation_target(a).y, 6)) for a in (1, 2, 3)}
    assert len(points) == 3


def test_each_member_leaves_on_its_own_clock_and_the_queue_head_takes_its_point() -> None:
    agents = _agents(4)
    mgr = _mgr(agents, _fountain(3))
    for aid, duration in ((1, 0.52), (2, 1.02), (3, 2.02), (4, 1.02)):
        _seek(mgr, aid, duration=duration)
    interaction = next(iter(mgr.interactions.values()))
    _walk_to_targets(mgr, agents, [1, 2, 3])
    point_of_1 = mgr.formation_target(1)
    point_of_2 = mgr.formation_target(2)
    mgr.drain_edges()
    for _ in range(11):
        mgr.update({}, dt=DT)
    assert _holds(mgr)[-1] == (LifecycleEdge.RELEASED, (1,))
    assert interaction.participants == [2, 3, 4] and not interaction.contract.queue
    assert agents[1].movement.last_outcome == InteractionOutcome.COMPLETED
    assert interaction.member_holds[2] == pytest.approx(11 * DT)
    assert interaction.member_holds[3] == pytest.approx(11 * DT)
    assert mgr.formation_target(4) == point_of_1
    assert mgr.formation_target(2) == point_of_2
    _walk_to_targets(mgr, agents, [4])
    for _ in range(10):
        mgr.update({}, dt=DT)
    assert (LifecycleEdge.RELEASED, (2,)) in _holds(mgr)
    assert interaction.participants == [3, 4]
    assert interaction.member_holds[3] == pytest.approx(21 * DT)
    assert interaction.member_holds[4] == pytest.approx(10 * DT)


def test_a_member_leaving_restarts_no_other_clock() -> None:
    agents = _agents(3)
    mgr = _mgr(agents, _fountain(3))
    for aid in (1, 2, 3):
        _seek(mgr, aid, duration=5.0)
    interaction = next(iter(mgr.interactions.values()))
    _walk_to_targets(mgr, agents, [1, 2, 3])
    for _ in range(10):
        mgr.update({}, dt=DT)
    held = dict(interaction.member_holds)
    mgr.stop(2, interaction.id)
    assert interaction.member_holds == {1: held[1], 3: held[3]}
    assert mgr.is_holding(interaction)


def test_a_hold_waits_for_its_own_member_at_its_point() -> None:
    agents = _agents(2)
    mgr = _mgr(agents, _fountain(2))
    _seek(mgr, 1, duration=1.0)
    _seek(mgr, 2, duration=1.0)
    interaction = next(iter(mgr.interactions.values()))
    _walk_to_targets(mgr, agents, [1])
    mgr.drain_edges()
    for _ in range(4):
        mgr.update({}, dt=DT)
    assert _holds(mgr) == [(LifecycleEdge.HOLD_ONSET, (1,))]
    assert set(interaction.member_holds) == {1}
    assert interaction.member_holds[1] == pytest.approx(4 * DT)


def test_the_last_member_done_with_nobody_queued_completes_the_interaction() -> None:
    agents = _agents(2)
    mgr = _mgr(agents, _fountain(2))
    _seek(mgr, 1, duration=0.5)
    _seek(mgr, 2, duration=0.5)
    interaction = next(iter(mgr.interactions.values()))
    _walk_to_targets(mgr, agents, [1, 2])
    mgr.drain_edges()
    for _ in range(round(0.5 / DT) + 1):
        mgr.update({}, dt=DT)
    edges = _holds(mgr)
    assert edges[-1][0] == LifecycleEdge.RELEASED and set(edges[-1][1]) == {1, 2}
    assert interaction.outcome == InteractionOutcome.COMPLETED
    assert not interaction.member_holds


def test_capacity_one_keeps_the_shared_single_line() -> None:
    agents = _agents(2)
    mgr = _mgr(agents, _fountain(1))
    _seek(mgr, 1, duration=1.0)
    _seek(mgr, 2, duration=1.0)
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.contract.max_participants == 1
    assert interaction.participants == [1] and interaction.contract.queue == [2]
    assert "_use_points" not in interaction.state


def test_a_steps_own_max_participants_wins_over_the_capacity() -> None:
    agents = _agents(3)
    mgr = _mgr(agents, _fountain(3))
    for aid in (1, 2, 3):
        _seek(mgr, aid, duration=1.0, max_participants=2)
    interaction = next(iter(mgr.interactions.values()))
    assert interaction.contract.max_participants == 2
    assert interaction.participants == [1, 2] and interaction.contract.queue == [3]
    assert "_use_points" not in interaction.state


def test_the_approach_pose_is_the_front_row_until_the_points_are_taken() -> None:
    world = WorldKnowledge()
    obj = _fountain(3)
    world.add_object(obj)
    for participants, queue, index in ((0, 0, 0), (2, 0, 0), (3, 0, 1), (3, 1, 2)):
        world.set_counts({"fountain": queue}, {"fountain": participants})
        pose = _approach_pose_for(obj, world)
        assert pose.x == pytest.approx(-(FRONT + index * STEP)), (participants, queue)
    one = _fountain(1)
    world.add_object(one)
    world.set_counts({"fountain": 1}, {"fountain": 1})
    assert _approach_pose_for(one, world).x == pytest.approx(-(FRONT + 2 * STEP))
