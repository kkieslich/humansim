from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from arena_humansim.core.agents.base import BaseAgent
from arena_humansim.core.agents.types import SampledParams, SampledPerception
from arena_humansim.core.pool import AgentPool
from arena_humansim.core.replay import HEADING_TOL_RAD, POSITION_TOL_M, VELOCITY_TOL_MPS
from arena_humansim.local_planner.hsfm import HSFMPlanner
from arena_humansim.local_planner.sfm import SFMPlanner
from arena_humansim.utils.types import AgentState, Pose2D, Segments

GOLDEN_DIR = Path(__file__).parent / "golden"
DT = 0.05
STEPS = 50
# The replay tolerances: last-bit differences between CPU families pass, a behaviour change fails (core/replay.py).
TOLERANCES = {"pos": POSITION_TOL_M, "vel": VELOCITY_TOL_MPS, "theta": HEADING_TOL_RAD}
WALLS: Segments = [((-6.0, -1.2), (6.0, -1.2)), ((-6.0, 1.2), (6.0, 1.2))]
ROWS: tuple[tuple[float, ...], ...] = (
    (-4.0, 0.3, 0.0, 0.8, 0.0, 0.25, 5.0, 0.2),
    (-3.5, -0.6, 0.1, 0.6, 0.1, 0.3, 5.0, -0.5),
    (-2.5, 0.8, -0.2, 0.9, -0.1, 0.2, 5.0, 0.6),
    (-1.0, -0.2, 0.3, 0.5, 0.0, 0.35, 5.0, 0.0),
    (4.0, -0.4, 3.1, -0.7, 0.0, 0.28, -5.0, -0.3),
    (3.0, 0.5, 3.0, -0.8, 0.05, 0.22, -5.0, 0.4),
    (2.0, -0.9, -3.0, -0.6, 0.1, 0.32, -5.0, -0.8),
    (0.5, 0.1, 2.9, -0.4, -0.05, 0.26, -5.0, 0.1),
)
PLANNERS = {"sfm": SFMPlanner, "hsfm": HSFMPlanner}


def _agent(aid: int, row: tuple[float, ...], extra_params: dict[str, float]) -> BaseAgent:
    x, y, theta, vx, vy, radius = row[:6]
    state = AgentState(agent_id=aid, pose=Pose2D(x=x, y=y, theta=theta), velocity=(vx, vy), desired_velocity=1.2)
    params = SampledParams(
        name="adult",
        desired_velocity=1.2,
        agent_radius=radius,
        max_velocity=1.6,
        max_acceleration=1.5,
        max_deceleration=2.5,
        min_turning_radius=0.3,
        pivot_angular_velocity=2.0,
        reaction_time=0.4,
        personal_space_min=0.6,
        perception=SampledPerception(vision_range=5.0, vision_fov=180.0),
        local_planner_params={
            "relaxation_time": 0.5,
            "repulsion_strength": 2.1,
            "repulsion_range": 0.3,
            "anisotropy": 0.5,
            "lateral_gain": 0.3,
            "lateral_damping": 1.5,
            "angular_gain": 4.0,
            "angular_damping": 4.0,
            **extra_params,
        },
        local_planner="sfm",
    )
    return BaseAgent(state=state, params=params, global_planner=cast(Any, None), local_planner=cast(Any, None), animation=cast(Any, None))


def build_pool(planner: SFMPlanner, extra_params: dict[str, float] | None = None) -> AgentPool:
    pool = AgentPool(capacity=8)
    planner.attach(pool)
    planner.set_walls(WALLS)
    for i, row in enumerate(ROWS):
        pool.add_agent(_agent(i + 1, row, extra_params or {}))
    pool.set_goals({i + 1: Pose2D(x=row[6], y=row[7]) for i, row in enumerate(ROWS)})
    n = len(ROWS)
    indptr = np.arange(0, n * (n - 1) + 1, n - 1, dtype=np.int32)
    indices = np.array([j for i in range(n) for j in range(n) if j != i], dtype=np.int32)
    pool.set_neighbor_csr(indptr, indices)
    return pool


def run_trajectory(planner: SFMPlanner, extra_params: dict[str, float] | None = None) -> list[dict[str, list[list[float]] | list[float]]]:
    pool = build_pool(planner, extra_params)
    n = pool.n
    out = []
    for _ in range(STEPS):
        planner.compute_pool(pool, dt=DT)
        pool.pos[:n] += pool.vel[:n] * DT
        out.append({"pos": pool.pos[:n].tolist(), "vel": pool.vel[:n].tolist(), "theta": pool.theta[:n].tolist()})
    return out


def max_abs_diff(got: list[dict[str, Any]], want: list[dict[str, Any]]) -> float:
    assert len(got) == len(want)
    return max(float(np.max(np.abs(np.asarray(g[key]) - np.asarray(w[key])))) for g, w in zip(got, want, strict=True) for key in TOLERANCES)


def assert_matches_golden(got: list[dict[str, Any]], want: list[dict[str, Any]]) -> None:
    """Step by step, so a divergence is reported at the step it starts, with its size."""
    assert len(got) == len(want)
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        for key, tol in TOLERANCES.items():
            g_arr, w_arr = np.asarray(g[key]), np.asarray(w[key])
            assert g_arr.shape == w_arr.shape, f"step {step}: {key} shape {g_arr.shape} != {w_arr.shape}"
            err = float(np.max(np.abs(g_arr - w_arr)))
            assert err <= tol, f"step {step}: {key} off by {err:.3g} (tolerance {tol:g})"


@pytest.mark.parametrize("name", sorted(PLANNERS))
def test_pool_trajectory_matches_golden(name: str) -> None:
    got = run_trajectory(PLANNERS[name]())
    want = json.loads((GOLDEN_DIR / f"{name}_pool.json").read_text())
    assert_matches_golden(got, want)


if __name__ == "__main__":
    import rclpy

    from arena_humansim.utils.loggable import Loggable

    rclpy.init()
    Loggable.init_logging(rclpy.create_node("planner_golden_record"))
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cls in PLANNERS.items():
        (out_dir / f"{name}_pool.json").write_text(json.dumps(run_trajectory(cls())))
