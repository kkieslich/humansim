"""Replay of the recorded wall_projection trajectory: the corrected agents exactly, positions and velocities within the
replay tolerance (last-bit differences between CPU families pass, core/replay.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arena_humansim.collision.wall_projection import WallProjectionResolver
from arena_humansim.core.pool import AgentPool
from arena_humansim.core.replay import POSITION_TOL_M, VELOCITY_TOL_MPS
from arena_humansim.utils.types import Segments
from tests.conftest import _make_agent

GOLDEN = Path(__file__).resolve().parent / "golden" / "collision_pool.json"
STEPS = 20
DT = 0.05
WALLS: Segments = [((-5.0, -1.0), (5.0, -1.0)), ((-5.0, 1.0), (5.0, 1.0))]
AGENTS: tuple[tuple[float, float, float, float, bool], ...] = (
    (0.0, 0.0, 1.0, 0.0, True),
    (0.3, 0.1, -1.0, 0.0, True),
    (0.5, -0.2, -0.5, 0.5, True),
    (2.0, 0.9, 0.0, 1.0, True),
    (-2.0, -0.85, 0.3, -1.0, True),
    (2.3, 0.7, -0.5, 0.5, True),
    (-2.4, -0.7, 0.0, 0.0, False),
    (-4.0, 0.0, 0.0, 0.0, True),
)


def _pool() -> AgentPool:
    pool = AgentPool(capacity=8)
    for i, (x, y, vx, vy, autonomous) in enumerate(AGENTS):
        pool.add_agent(_make_agent(i + 1, x=x, y=y))
        pool.vel[i] = (vx, vy)
        pool.policy_idx[i] = 0 if autonomous else -1
    return pool


def _trajectory() -> list[dict[str, list]]:
    resolver = WallProjectionResolver(margin=0.01)
    resolver.set_walls(WALLS)
    pool = _pool()
    n = pool.n
    steps = []
    for _ in range(STEPS):
        corrected = resolver.resolve(pool)
        steps.append({"pos": pool.pos[:n].tolist(), "vel": pool.vel[:n].tolist(), "corrected": sorted(corrected)})
        pool.pos[:n] += pool.vel[:n] * DT
    return steps


def test_trajectory_matches_golden() -> None:
    with open(GOLDEN) as fh:
        expected = json.load(fh)
    got = _trajectory()
    assert len(got) == len(expected)
    for step, (g, e) in enumerate(zip(got, expected, strict=True)):
        assert g.keys() == e.keys()
        assert g["corrected"] == e["corrected"], f"step {step}: corrected agents differ"
        for key, tol in (("pos", POSITION_TOL_M), ("vel", VELOCITY_TOL_MPS)):
            g_arr, e_arr = np.asarray(g[key]), np.asarray(e[key])
            assert g_arr.shape == e_arr.shape, f"step {step}: {key} shape {g_arr.shape} != {e_arr.shape}"
            err = float(np.max(np.abs(g_arr - e_arr)))
            assert err <= tol, f"step {step}: {key} off by {err:.3g} (tolerance {tol:g})"


def test_trajectory_is_deterministic_across_resolvers() -> None:
    assert _trajectory() == _trajectory()


if __name__ == "__main__":
    with open(GOLDEN, "w") as fh:
        json.dump(_trajectory(), fh, indent=1)
        fh.write("\n")
