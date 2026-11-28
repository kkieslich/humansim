from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs

from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D, pose_distance

from . import AgentLookup, Formation
from .anchor import AgentAnchor, Anchor

if TYPE_CHECKING:
    pass


@attrs.define
class _Slot:
    agent_id: int
    target: Pose2D
    prev_target: Pose2D


class LineFormation(Formation):
    """Ordered line with target-propagation shuffle.

    Invariant: slot k's target is, after a reaction delay, a delayed echo of
    slot k-1's target. Slot 0 is anchored to the formation's anchor pose.
    The line extends backward along (anchor.theta + pi).

    With ``points`` > 1 (an object several walkers use at once) the front is a row of that many use points side by side,
    ``point_step`` apart across the anchor's heading, centred on the front pose. A participant keeps its point until it
    leaves; members not served yet wait in one line behind the row's centre and the head takes the next point that
    frees. The line formation of one use point is the plain line.
    """

    def __init__(
        self,
        anchor: Anchor,
        agent_lookup: AgentLookup,
        base_step: float = 1.0,
        formation_scale: float = 1.0,
        front_offset: float = 0.0,
        points: int = 1,
        point_step: float | None = None,
    ) -> None:
        if points < 1:
            raise ValueError(f"a line needs at least one use point, got {points}")
        self.anchor = anchor
        self.agent_lookup = agent_lookup
        self.base_step = base_step
        self.formation_scale = formation_scale
        self.front_offset = front_offset
        self._slots: list[_Slot] = []
        self._points: list[int | None] = [None] * points if points > 1 else []
        self._point_step = (base_step if point_step is None else point_step) * formation_scale
        self._cached_front_key: tuple[float, float, float] | None = None
        self._cached_front_pose: Pose2D | None = None

    def _front_pose(self, anchor_pose: Pose2D) -> Pose2D:
        key = (anchor_pose.x, anchor_pose.y, anchor_pose.theta)
        if key == self._cached_front_key and self._cached_front_pose is not None:
            return self._cached_front_pose
        if self.front_offset == 0.0:
            result = Pose2D(x=anchor_pose.x, y=anchor_pose.y, theta=anchor_pose.theta)
        else:
            result = Pose2D(
                x=anchor_pose.x - self.front_offset * math.cos(anchor_pose.theta),
                y=anchor_pose.y - self.front_offset * math.sin(anchor_pose.theta),
                theta=anchor_pose.theta,
            )
        self._cached_front_key = key
        self._cached_front_pose = result
        return result

    def _spacing_for(self, agent_id: int) -> float:
        agent = self.agent_lookup(agent_id)
        if agent is None:
            floor = self.base_step
        else:
            floor = max(self.base_step, agent.params.personal_space_min)
        return floor * self.formation_scale

    def _backward_offset(self, step: float, yaw: float) -> tuple[float, float]:
        backward = yaw + math.pi
        return step * math.cos(backward), step * math.sin(backward)

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        if agent_id in self._points:
            return
        if participant and self._points and None in self._points:
            self._slots = [s for s in self._slots if s.agent_id != agent_id]
            self._points[self._free_point_for(agent_id)] = agent_id
            return
        if any(s.agent_id == agent_id for s in self._slots):
            return
        self._slots.append(_Slot(agent_id=agent_id, target=Pose2D(), prev_target=Pose2D()))

    def on_leave(self, agent_id: int) -> None:
        self._points = [None if p == agent_id else p for p in self._points]
        self._slots = [s for s in self._slots if s.agent_id != agent_id]

    def _point_poses(self) -> list[Pose2D]:
        front = self._front_pose(self.anchor.pose())
        across = front.theta + 0.5 * math.pi
        mid = 0.5 * (len(self._points) - 1)
        return [Pose2D(x=front.x + (k - mid) * self._point_step * math.cos(across), y=front.y + (k - mid) * self._point_step * math.sin(across), theta=front.theta) for k in range(len(self._points))]

    def _free_point_for(self, agent_id: int) -> int:
        """The free use point nearest the joining member, the lowest index among equally near ones."""
        free = [k for k, p in enumerate(self._points) if p is None]
        agent = self.agent_lookup(agent_id)
        if agent is None:
            return free[0]
        poses = self._point_poses()
        return min(free, key=lambda k: (pose_distance(poses[k], agent.state.pose), k))

    def _slot_poses(self) -> list[Pose2D]:
        anchor_pose = self.anchor.pose()
        front_target = self._front_pose(anchor_pose)
        yaw = anchor_pose.theta
        if self._points:
            # the waiting line starts one step behind the row of use points
            poses = []
            offset_x = offset_y = 0.0
            for slot in self._slots:
                dx, dy = self._backward_offset(self._spacing_for(slot.agent_id), yaw)
                offset_x += dx
                offset_y += dy
                poses.append(Pose2D(x=front_target.x + offset_x, y=front_target.y + offset_y, theta=yaw))
            return poses
        poses = [front_target]
        offset_x = 0.0
        offset_y = 0.0
        for slot in self._slots[1:]:
            dx, dy = self._backward_offset(self._spacing_for(slot.agent_id), yaw)
            offset_x += dx
            offset_y += dy
            poses.append(Pose2D(x=front_target.x + offset_x, y=front_target.y + offset_y, theta=yaw))
        return poses

    def _anchors_front(self) -> bool:
        return not self._points and isinstance(self.anchor, AgentAnchor) and self.anchor.agent_id == self._slots[0].agent_id

    def tick(self, dt: float) -> dict[int, Pose2D]:
        del dt
        targets: dict[int, Pose2D] = {}
        if self._points:
            targets = {p: pose for p, pose in zip(self._points, self._point_poses(), strict=True) if p is not None}
        if not self._slots:
            return targets

        for slot, pose in zip(self._slots, self._slot_poses(), strict=True):
            slot.target = pose

        if self._anchors_front():
            return {s.agent_id: s.target for s in self._slots[1:]}
        return {**targets, **{s.agent_id: s.target for s in self._slots}}

    def speeds(self) -> dict[int, float]:
        if len(self._slots) < 2 or not self._anchors_front():
            return {}
        lead = self.agent_lookup(self._slots[0].agent_id)
        followers = [agent.state.desired_velocity for s in self._slots[1:] if (agent := self.agent_lookup(s.agent_id)) is not None]
        if lead is None or not followers:
            return {}
        return {lead.state.agent_id: min(lead.state.desired_velocity, *followers)}

    def slot_of(self, agent_id: int) -> Pose2D | None:
        if agent_id in self._points:
            return self._point_poses()[self._points.index(agent_id)]
        idx = next((i for i, s in enumerate(self._slots) if s.agent_id == agent_id), None)
        if idx is None or (idx == 0 and self._anchors_front()):
            return None
        return self._slot_poses()[idx]

    def arrived(self, agent_id: int) -> bool:
        if agent_id in self._points:
            target = self._point_poses()[self._points.index(agent_id)]
        else:
            slot = next((s for s in self._slots if s.agent_id == agent_id), None)
            if slot is None:
                return True
            target = slot.target
        agent = self.agent_lookup(agent_id)
        if agent is None:
            return True
        return pose_distance(agent.state.pose, target) < DISTANCE_TOLERANCE
