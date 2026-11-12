from __future__ import annotations

import math

from arena_humansim.utils.types import Pose2D

from . import AgentLookup, Formation
from .anchor import AgentAnchor, Anchor


class PushFormation(Formation):
    """A pusher walks to the push handles of the provider's wheelchair, then steers the pair.

    The anchor is the rider. Until the pusher holds the handles its slot is `handle_offset` behind the rider along the
    chair's heading, facing the same way. Within `attach_radius` and `attach_yaw` of it she takes the handles: the
    formation stops giving targets, she navigates on her own and core/carry.py keeps the chair `handle_offset` ahead of
    her, rolling it there over `grab_s` from where she found it.
    """

    def __init__(
        self,
        anchor: Anchor,
        agent_lookup: AgentLookup,
        handle_offset: float = 0.715,
        attach_radius: float = 0.25,
        attach_yaw: float = 0.15,
        push_speed: float = 0.9,
        grab_s: float = 0.6,
        formation_scale: float = 1.0,
    ) -> None:
        del formation_scale  # the offset is the chair's and the pusher's reach, not a social distance
        if not isinstance(anchor, AgentAnchor):
            raise TypeError("push formation needs an agent anchor (the rider)")
        self.anchor = anchor
        self.agent_lookup = agent_lookup
        self.rider = anchor.agent_id
        self.handle_offset = handle_offset
        self.attach_radius = attach_radius
        self.attach_yaw = attach_yaw
        self.push_speed = push_speed
        self.grab_s = grab_s
        self.pusher: int | None = None
        self.attached = False
        self._slot: Pose2D | None = None

    def on_join(self, agent_id: int, *, participant: bool = True) -> None:
        if agent_id != self.rider and self.pusher is None and participant:
            self.pusher = agent_id

    def on_leave(self, agent_id: int) -> None:
        if agent_id == self.pusher:
            self.pusher = None
            self.attached = False
            self._slot = None

    def handles(self) -> Pose2D:
        """Where the pusher stands to hold the handles of the chair as it is now."""
        rp = self.anchor.pose()
        return Pose2D(x=rp.x - self.handle_offset * math.cos(rp.theta), y=rp.y - self.handle_offset * math.sin(rp.theta), theta=rp.theta)

    def tick(self, dt: float) -> dict[int, Pose2D]:
        del dt
        if self.pusher is None or self.attached:
            self._slot = None
            return {}
        agent = self.agent_lookup(self.pusher)
        if agent is None:
            return {}
        slot = self.handles()
        self._slot = slot
        pose = agent.state.pose
        yaw = abs(math.remainder(pose.theta - slot.theta, math.tau))
        if math.hypot(pose.x - slot.x, pose.y - slot.y) <= self.attach_radius and yaw <= self.attach_yaw:
            self.attached = True
            self._slot = None
            return {}
        return {self.pusher: slot}

    def arrived(self, agent_id: int) -> bool:
        return self.attached

    def slot_of(self, agent_id: int) -> Pose2D | None:
        return self._slot if agent_id == self.pusher else None

    def speeds(self) -> dict[int, float]:
        return {self.pusher: self.push_speed} if self.attached and self.pusher is not None else {}
