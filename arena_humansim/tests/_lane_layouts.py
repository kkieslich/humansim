from __future__ import annotations

from arena_humansim.utils.types import Pose2D, Segments


def corridor(width: float, length: float) -> Segments:
    half = width / 2.0
    return [
        ((0.0, -half), (length, -half)),
        ((0.0, half), (length, half)),
    ]


def straight(length: float) -> list[list[Pose2D]]:
    return [[Pose2D(x=0.0, y=0.0), Pose2D(x=length, y=0.0)]]
