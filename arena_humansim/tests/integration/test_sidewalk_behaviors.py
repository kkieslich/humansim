from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.agents import loader
from arena_humansim.core.agents.types import AgentType, RobotStepDef
from arena_humansim.core.behavior.compiler import BehaviorTreeFactory
from arena_humansim.core.behavior.nodes import SequenceStateMachine
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import AgentConfig, ModuleConfig, ScenarioConfig, SimulationParams
from arena_humansim.utils.scenario_loader import converter
from arena_humansim.utils.types import InteractionOutcome, Pose2D
from arena_humansim_msgs.msg import AgentState as AgentStateMsg
from arena_humansim_msgs.msg import AgentStates as AgentStatesMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg

ROBOT = 1
EXTERNAL_ID = 424242
SPEED = 1.0
DT = 0.05
TYPES_DIR = Path(loader.__file__).resolve().parents[3] / "config" / "agent_types"
BEHAVIORS = ("obstructing", "conversing", "queueing", "frontal", "lateral", "overtaking", "crossing", "gesturing")

Ped = tuple[int, str, float, float, float]


def _load_types() -> dict[str, AgentType]:
    raw = loader.load_agent_types_raw_from_dir(TYPES_DIR)
    resolved = loader.resolve_extends({name: r for name, (r, _) in raw.items()}, {})
    return {name: converter.structure(r, AgentType) for name, r in resolved.items()}


def _scenario(peds: list[Ped], robot_x: float) -> ScenarioConfig:
    agents = [AgentConfig(agent_id=ROBOT, kind=1, policy="straight", agent_type="robot", spawn_pose=Pose2D(x=robot_x, y=0.0))]
    agents += [AgentConfig(agent_id=aid, agent_type=f"sidewalk_{behavior}", spawn_pose=Pose2D(x=x, y=y, theta=theta)) for aid, behavior, x, y, theta in peds]
    return ScenarioConfig(
        name="sidewalk_behaviors",
        simulation=SimulationParams(seed=5, dt=DT, bt_tick_interval=2, max_ticks=4000),
        modules=ModuleConfig(global_planner="navmesh"),
        agents=agents,
        agent_types=_load_types(),
    )


def _robot_msg(tick: int, x: float) -> AgentStatesMsg:
    msg = AgentStatesMsg()
    stamp_ms = round(tick * DT * 1000)
    msg.header.stamp.sec = stamp_ms // 1000
    msg.header.stamp.nanosec = stamp_ms % 1000 * 1_000_000
    robot = AgentStateMsg()
    robot.agent_id = EXTERNAL_ID
    robot.pose = Pose2DMsg(x=x, y=0.0, theta=0.0)
    robot.radius = 0.3
    msg.agents.append(robot)
    return msg


class Rig:
    def __init__(self, mgr: AgentManager, robot_x: float, speed: float) -> None:
        self.mgr = mgr
        self.robot_x = robot_x
        self.speed = speed
        self.tick_no = 0
        self.fired: dict[int, float] = {}

    def track(self, aid: int, n: int) -> list[Pose2D]:
        poses = []
        for _ in range(n):
            self.step()
            p = self.pose(aid)
            poses.append(Pose2D(x=p.x, y=p.y, theta=p.theta))
        return poses

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.mgr._world_state_callback(_robot_msg(self.tick_no, self.robot_x))
            self.mgr.tick()
            self.tick_no += 1
            self.robot_x += self.speed * DT
            for aid in self.mgr._agents:
                if aid != ROBOT and aid not in self.fired and self.sequence(aid) != "idle":
                    self.fired[aid] = self.robot_x

    def sequence(self, aid: int) -> str:
        machine = next(n for n in self.mgr._behavior_trees[aid].root.iterate() if isinstance(n, SequenceStateMachine))
        return machine._current_name

    def pose(self, aid: int) -> Pose2D:
        return self.mgr._agents[aid].state.pose

    def velocity(self, aid: int) -> tuple[float, float]:
        return self.mgr._agents[aid].state.velocity

    def robot(self) -> Pose2D:
        return self.pose(ROBOT)


@pytest.fixture
def rig(manager_factory: Callable[..., AgentManager]) -> Callable[[list[Ped], float, float], Rig]:
    def make(peds: list[Ped], robot_x: float = 0.0, speed: float = SPEED) -> Rig:
        mgr = manager_factory(_scenario(peds, robot_x), node_name=f"test_sidewalk_{peds[0][1]}")
        return Rig(mgr, robot_x, speed)

    return make


def _angle_to(src: Pose2D, dst: Pose2D) -> float:
    return math.atan2(dst.y - src.y, dst.x - src.x)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@pytest.fixture(scope="module")
def types() -> dict[str, AgentType]:
    return _load_types()


@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_every_sidewalk_type_extends_adult_and_waits_on_a_robot_trigger(types: dict[str, AgentType], behavior: str) -> None:
    t = types[f"sidewalk_{behavior}"]
    adult = types["adult"]
    assert t.mode == "behavior_tree"
    assert t.desired_velocity == adult.desired_velocity
    assert t.agent_radius == adult.agent_radius
    assert t.initial_sequence == "idle"
    watch = t.sequences["idle"].steps["watch"]
    assert isinstance(watch, RobotStepDef) and watch.when is not None and watch.act == "wait"


@pytest.mark.parametrize(
    ("behavior", "field", "distance"),
    [("obstructing", "ahead", 10.0), ("conversing", "ahead", 10.0), ("queueing", "ahead", 10.0), ("frontal", "ahead", 20.0), ("lateral", "ahead", 4.0), ("overtaking", "behind", 6.0), ("crossing", "within", 8.0), ("gesturing", "ahead", 10.0)],
)
def test_trigger_distances_match_the_paper_table(types: dict[str, AgentType], behavior: str, field: str, distance: float) -> None:
    watch = types[f"sidewalk_{behavior}"].sequences["idle"].steps["watch"]
    assert isinstance(watch, RobotStepDef) and watch.when is not None
    assert getattr(watch.when, field) == distance


def test_obstructer_idles_until_the_robot_is_10_m_away_then_faces_it_and_stays(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "obstructing", 12.0, 0.0, math.pi)])
    r.step(30)
    assert r.sequence(2) == "idle"
    r.step(100)
    assert r.sequence(2) == "obstruct"
    assert r.fired[2] == pytest.approx(2.0, abs=0.3)
    r.step(30)
    ped = r.pose(2)
    assert r.robot().x < 8.0
    assert math.hypot(ped.x - 12.0, ped.y) < 0.3
    assert abs(_wrap(ped.theta - _angle_to(ped, r.robot()))) < 0.3


def test_frontal_walker_starts_at_20_m_and_walks_at_the_robot(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "frontal", 25.0, 0.0, math.pi)])
    r.step(60)
    assert r.sequence(2) == "idle"
    r.step(80)
    assert r.sequence(2) == "approach"
    assert r.fired[2] == pytest.approx(5.0, abs=0.3)
    r.step(20)
    vx, vy = r.velocity(2)
    ped, robot = r.pose(2), r.robot()
    assert math.hypot(vx, vy) > 0.5
    assert abs(_wrap(math.atan2(vy, vx) - _angle_to(ped, robot))) < 0.35
    gaps = []
    for _ in range(150):
        r.step()
        gaps.append(math.hypot(r.pose(2).x - r.robot().x, r.pose(2).y - r.robot().y))
    assert min(gaps) < 2.0


def test_lateral_walker_starts_at_4_m_and_crosses_the_robots_projected_path(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "lateral", 8.0, -3.0, math.pi / 2)])
    r.step(40)
    assert r.sequence(2) == "idle"
    r.step(60)
    assert r.sequence(2) == "cross"
    assert r.fired[2] == pytest.approx(4.0, abs=0.3)
    r.step(10)
    assert r.velocity(2)[1] > 0.5
    path = r.track(2, 60)
    assert max(p.y for p in path) > 0.3
    assert all(abs(p.x - 8.0) < 0.5 for p in path)


def test_overtaker_triggers_behind_the_robot_and_ends_up_ahead_of_it(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "overtaking", -4.0, 0.8, 0.0)], robot_x=-12.0)
    r.step(100)
    assert r.sequence(2) == "idle"
    r.step(120)
    assert r.sequence(2) == "overtake"
    assert r.fired[2] == pytest.approx(-4.0, abs=1.0)
    for _ in range(300):
        if r.sequence(2) != "overtake":
            break
        r.step()
    assert r.sequence(2) == "done"
    assert r.pose(2).x - r.robot().x > 1.0


def test_conversers_idle_until_10_m_then_form_one_face_to_face_group(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "conversing", 12.0, -0.6, math.pi / 2), (3, "conversing", 12.0, 0.0, math.pi), (4, "conversing", 12.0, 0.6, -math.pi / 2)])
    r.step(30)
    assert {r.sequence(a) for a in (2, 3, 4)} == {"idle"}
    r.step(150)
    assert {r.sequence(a) for a in (2, 3, 4)} == {"converse"}
    groups = [i for i in r.mgr._interaction_manager.interactions.values() if i.type == int(InteractionType.GROUP_CONVERSATION)]
    assert len(groups) == 1 and groups[0].outcome == InteractionOutcome.ACTIVE
    assert set(groups[0].participants) == {2, 3, 4}
    poses = [r.pose(a) for a in (2, 3, 4)]
    cx, cy = sum(p.x for p in poses) / 3, sum(p.y for p in poses) / 3
    for p in poses:
        assert abs(_wrap(p.theta - _angle_to(p, Pose2D(x=cx, y=cy)))) < 1.0


def test_queue_stays_a_tight_line_and_faces_along_it(rig: Callable[..., Rig]) -> None:
    ys = (-1.5, -0.9, -0.3, 0.3)
    r = rig([(2 + i, "queueing", 12.0, y, 0.0) for i, y in enumerate(ys)])
    r.step(30)
    assert {r.sequence(a) for a in range(2, 6)} == {"idle"}
    r.step(100)
    assert {r.sequence(a) for a in range(2, 6)} == {"queue"}
    assert r.robot().x < 8.0
    poses = [r.pose(a) for a in range(2, 6)]
    assert max(p.x for p in poses) - min(p.x for p in poses) < 0.2
    gaps = [b.y - a.y for a, b in zip(poses, poses[1:], strict=False)]
    assert all(0.45 < g < 0.8 for g in gaps)
    assert all(abs(_wrap(p.theta - math.pi / 2)) < 0.3 for p in poses)


def test_crosser_in_front_fires_inside_8_m_and_walks_toward_the_robot(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "crossing", 6.0, 0.0, math.pi)], robot_x=-14.0)
    r.step(100)
    assert r.sequence(2) == "idle"
    r.step(150)
    assert r.sequence(2) == "cross"
    assert r.fired[2] == pytest.approx(-2.0, abs=0.3)
    r.step(20)
    assert r.velocity(2)[0] < -0.5
    assert min(p.x for p in r.track(2, 60)) < 4.5


def test_crosser_behind_the_robot_walks_through_to_the_far_side(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "crossing", -6.0, 1.0, 0.0)], speed=0.0)
    r.step(40)
    assert r.sequence(2) == "cross"
    r.step(300)
    assert r.pose(2).x > 1.0


def test_zone_stimulus_sends_a_crosser_across_without_the_range_trigger(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "crossing", 30.0, 0.0, math.pi)], speed=0.0)
    r.step(40)
    assert r.sequence(2) == "idle"
    r.mgr._agents[2].needs.needs["robot_in_zone"].value = 100.0
    r.step(10)
    assert r.sequence(2) == "cross"


def test_gesturer_faces_the_robot_and_holds_a_stop_gesture_up(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "gesturing", 12.0, 0.0, math.pi)])
    r.step(30)
    assert r.sequence(2) == "idle"
    assert r.mgr._agents[2].movement.gestures == ()
    r.step(100)
    assert r.sequence(2) == "gesture"
    assert r.fired[2] == pytest.approx(2.0, abs=0.3)
    gestures = r.mgr._agents[2].movement.gestures
    assert [g.slot for g in gestures] == ["halt"]
    assert gestures[0].hand in ("l", "r")
    robot = r.robot()
    assert gestures[0].x == pytest.approx(robot.x, abs=0.3)
    ped = r.pose(2)
    assert abs(_wrap(ped.theta - _angle_to(ped, robot))) < 0.3


def test_a_robot_step_cannot_compile_without_the_agent_pool(rig: Callable[..., Rig]) -> None:
    r = rig([(2, "frontal", 25.0, 0.0, math.pi)])
    mgr = r.mgr
    with pytest.raises(ValueError, match="pool"):
        BehaviorTreeFactory(mgr._agent_types["sidewalk_frontal"]).build(
            agent=mgr._agents[2], world=mgr._world_knowledge, event_bus=mgr._event_bus, rng=np.random.default_rng(0), dt=0.1
        )
