from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs
import numpy as np
import py_trees

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import RobotStepDef
from arena_humansim.core.behavior.nodes.attention import GESTURE_Z_AGENT
from arena_humansim.core.behavior.nodes.helpers import _nav_command, _sample_param_dist, _set_command
from arena_humansim.core.pool import KIND_ROBOT
from arena_humansim.utils.types import BehaviorTreeMovement, GestureIntent, Pose2D

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool

HEADING_MIN_SPEED = 0.1
ARRIVAL_M = 0.5
PASSED_M = 1.0
CROSSED_M = 0.3
_RUNNING = py_trees.common.Status.RUNNING
_SUCCESS = py_trees.common.Status.SUCCESS


@attrs.frozen
class _Robot:
    x: float
    y: float
    heading: float


@attrs.frozen
class _Rel:
    along: float
    across: float
    dist: float


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _relative(robot: _Robot, x: float, y: float) -> _Rel:
    dx, dy = x - robot.x, y - robot.y
    c, s = math.cos(robot.heading), math.sin(robot.heading)
    return _Rel(along=dx * c + dy * s, across=-dx * s + dy * c, dist=math.hypot(dx, dy))


def _world(robot: _Robot, along: float, across: float) -> tuple[float, float]:
    c, s = math.cos(robot.heading), math.sin(robot.heading)
    return robot.x + along * c - across * s, robot.y + along * s + across * c


class RobotNode(py_trees.behaviour.Behaviour):
    """Wait for the nearest robot to meet the step's trigger, then stand, approach it, cross its path, walk toward it or pass it."""

    def __init__(self, name: str, agent: BaseAgent, pool: AgentPool, step: RobotStepDef, rng: np.random.Generator, dt: float) -> None:
        super().__init__(name)
        self._agent = agent
        self._pool = pool
        self._step = step
        self._rng = rng
        self._dt = dt
        self._triggered = False
        self._elapsed = 0.0
        self._duration: float | None = None
        self._heading_goal: float | None = None
        self._target: tuple[float, float] | None = None
        self._side = 1.0
        self._path: _Robot | None = None
        self._direction = (1.0, 0.0)

    def initialise(self) -> None:
        self._triggered = False
        self._elapsed = 0.0
        self._duration = _sample_param_dist(self._step.duration, self._rng) if self._step.duration is not None else None
        self._heading_goal = None
        self._target = None
        self._path = None

    def _own(self) -> tuple[float, float]:
        i = self._pool.idx(self._agent.state.agent_id)
        return float(self._pool.pos[i, 0]), float(self._pool.pos[i, 1])

    def _robot(self) -> _Robot | None:
        pool = self._pool
        robots = np.flatnonzero(pool.kind[: pool.n] == KIND_ROBOT)
        if robots.size == 0:
            return None
        ox, oy = self._own()
        d = np.hypot(pool.pos[robots, 0] - ox, pool.pos[robots, 1] - oy)
        i = int(robots[int(np.argmin(d))])
        vx, vy = float(pool.vel[i, 0]), float(pool.vel[i, 1])
        heading = math.atan2(vy, vx) if math.hypot(vx, vy) > HEADING_MIN_SPEED else float(pool.theta[i])
        return _Robot(float(pool.pos[i, 0]), float(pool.pos[i, 1]), heading)

    def _met(self, robot: _Robot) -> bool:
        trigger = self._step.when
        if trigger is None:
            return True
        rel = _relative(robot, *self._own())
        if abs(rel.across) > trigger.lateral:
            return False
        if trigger.ahead is not None and not 0.0 < rel.along <= trigger.ahead:
            return False
        if trigger.behind is not None and not -trigger.behind <= rel.along < 0.0:
            return False
        return trigger.within is None or rel.dist <= trigger.within

    def _halt(self) -> None:
        cmd = _nav_command(self._agent, self._agent.state.pose)
        cmd.desired_velocity = 0.0
        _set_command(self._agent, cmd)

    def _go(self, x: float, y: float) -> None:
        ox, oy = self._own()
        _set_command(self._agent, _nav_command(self._agent, Pose2D(x=x, y=y, theta=math.atan2(y - oy, x - ox))))

    def _latch(self, robot: _Robot) -> None:
        step = self._step
        ox, oy = self._own()
        rel = _relative(robot, ox, oy)
        self._side = -1.0 if rel.across < 0.0 else 1.0
        self._path = robot
        norm = max(rel.dist, 1e-6)
        self._direction = ((robot.x - ox) / norm, (robot.y - oy) / norm)
        if step.act == "cross":
            self._target = _world(robot, rel.along, -self._side * max(abs(rel.across), step.clearance))
        if step.act == "traverse":
            shift = self._side * step.clearance
            self._target = (ox + self._direction[0] * step.lead - math.sin(robot.heading) * shift, oy + self._direction[1] * step.lead + math.cos(robot.heading) * shift)
        if isinstance(step.face, (int, float)) and not isinstance(step.face, bool):
            self._heading_goal = _wrap(robot.heading + math.radians(float(step.face)))

    def _face(self, robot: _Robot) -> None:
        mv = self._agent.movement
        if not isinstance(mv, BehaviorTreeMovement):
            return
        ox, oy = self._own()
        if self._step.face == "robot":
            mv.heading_goal = math.atan2(robot.y - oy, robot.x - ox)
        elif self._heading_goal is not None:
            mv.heading_goal = self._heading_goal

    def _gesture(self, robot: _Robot) -> None:
        mv = self._agent.movement
        slot = self._step.gesture
        if not slot or not isinstance(mv, BehaviorTreeMovement):
            return
        hand = self._agent.params.handedness if slot == "halt" else ""
        intent = GestureIntent(slot, robot.x, robot.y, GESTURE_Z_AGENT, hand=hand)
        mv.gestures = (*(g for g in mv.gestures if g.slot != slot), intent)

    def _move(self, robot: _Robot) -> bool:
        step = self._step
        ox, oy = self._own()
        if step.act == "approach":
            dx, dy = self._direction
            self._go(robot.x + dx * step.lead, robot.y + dy * step.lead)
            return (robot.x - ox) * dx + (robot.y - oy) * dy < -PASSED_M
        if step.act in ("cross", "traverse"):
            assert self._target is not None and self._path is not None
            self._go(*self._target)
            crossed = step.act == "cross" and -self._side * _relative(self._path, ox, oy).across >= CROSSED_M
            return crossed or math.hypot(self._target[0] - ox, self._target[1] - oy) < ARRIVAL_M
        if step.act == "pass":
            self._go(*_world(robot, step.lead, self._side * step.clearance))
            return _relative(robot, ox, oy).along > PASSED_M
        return False

    def update(self) -> py_trees.common.Status:
        robot = self._robot()
        if not self._triggered:
            self._halt()
            if robot is None or not self._met(robot):
                return _RUNNING
            self._triggered = True
            self._latch(robot)
            if self._step.act == "wait":
                return _SUCCESS
        if robot is None:
            return _SUCCESS
        if self._duration is not None and self._elapsed >= self._duration:
            return _SUCCESS
        self._elapsed += self._dt
        if self._step.act == "stand":
            self._halt()
        elif self._move(robot):
            return _SUCCESS
        self._face(robot)
        self._gesture(robot)
        return _RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        del new_status
        mv = self._agent.movement
        if not isinstance(mv, BehaviorTreeMovement):
            return
        mv.heading_goal = None
        if self._step.gesture:
            mv.gestures = tuple(g for g in mv.gestures if g.slot != self._step.gesture)
