from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import ParamDist
from arena_humansim.utils.types import Pose2D

from .sfm import SFMPlanner, _resize_1d

if TYPE_CHECKING:
    from arena_humansim.core.pool import AgentPool

_EPS = 1e-6
_MAX_TOTAL_DEVIATION = math.pi / 2.0


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


class HSFMPlanner(SFMPlanner):
    supports_pool: bool = True
    provides_heading: bool = True

    PARAM_DEFAULTS: ClassVar[dict[str, ParamDist]] = {
        **SFMPlanner.PARAM_DEFAULTS,
        "lateral_gain": ParamDist(0.3, 0.0, clip_low=0.0, clip_high=1.0),
        "lateral_damping": ParamDist(1.5, 0.0, clip_low=0.0),
        "angular_gain": ParamDist(4.0, 0.0, clip_low=0.5),
        "angular_damping": ParamDist(4.0, 0.0, clip_low=0.1),
        "heading_source": ParamDist(0.0, 0.0, clip_low=0.0, clip_high=1.0),
    }

    def __init__(
        self,
        wall_repulsion_strength: float = 3.0,
        wall_repulsion_range: float = 0.1,
    ) -> None:
        super().__init__(wall_repulsion_strength, wall_repulsion_range)
        self._omega: dict[int, float] = {}

        self._lateral_gain = np.zeros(0, dtype=np.float64)
        self._lateral_damping = np.zeros(0, dtype=np.float64)
        self._angular_gain = np.zeros(0, dtype=np.float64)
        self._angular_damping = np.zeros(0, dtype=np.float64)
        self._heading_source = np.zeros(0, dtype=np.float64)

    def _allocate_soa(self, cap: int) -> None:
        super()._allocate_soa(cap)
        self._lateral_gain = np.zeros(cap, dtype=np.float64)
        self._lateral_damping = np.zeros(cap, dtype=np.float64)
        self._angular_gain = np.zeros(cap, dtype=np.float64)
        self._angular_damping = np.zeros(cap, dtype=np.float64)
        self._heading_source = np.zeros(cap, dtype=np.float64)

    def on_pool_grow(self, new_capacity: int, old_capacity: int) -> None:
        super().on_pool_grow(new_capacity, old_capacity)
        self._lateral_gain = _resize_1d(self._lateral_gain, new_capacity, old_capacity)
        self._lateral_damping = _resize_1d(self._lateral_damping, new_capacity, old_capacity)
        self._angular_gain = _resize_1d(self._angular_gain, new_capacity, old_capacity)
        self._angular_damping = _resize_1d(self._angular_damping, new_capacity, old_capacity)
        self._heading_source = _resize_1d(self._heading_source, new_capacity, old_capacity)

    def on_pool_add(self, idx: int, agent: BaseAgent) -> None:
        super().on_pool_add(idx, agent)
        lp = agent.params.local_planner_params
        self._lateral_gain[idx] = self._param(lp, "lateral_gain")
        self._lateral_damping[idx] = self._param(lp, "lateral_damping")
        self._angular_gain[idx] = self._param(lp, "angular_gain")
        self._angular_damping[idx] = self._param(lp, "angular_damping")
        self._heading_source[idx] = self._param(lp, "heading_source")

    def on_pool_swap(self, idx: int, last: int) -> None:
        super().on_pool_swap(idx, last)
        self._lateral_gain[idx] = self._lateral_gain[last]
        self._lateral_damping[idx] = self._lateral_damping[last]
        self._angular_gain[idx] = self._angular_gain[last]
        self._angular_damping[idx] = self._angular_damping[last]
        self._heading_source[idx] = self._heading_source[last]

    def _desired_heading(self, pool: AgentPool, f_att: np.ndarray, total_f: np.ndarray, at_goal: np.ndarray) -> np.ndarray:
        n = pool.n
        theta = pool.theta[:n]
        f_att_norm = np.hypot(f_att[:, 0], f_att[:, 1])
        has_dir = f_att_norm > _EPS
        theta_des = np.where(has_dir, np.arctan2(f_att[:, 1], f_att[:, 0]), theta)
        from_total = self._heading_source[:n] >= 0.5
        if np.any(from_total):
            from_total &= np.hypot(total_f[:, 0], total_f[:, 1]) > _EPS
            total_dir = np.arctan2(total_f[:, 1], total_f[:, 0])
            d_goal = pool.goal_pos[:n] - pool.pos[:n]
            bearing = np.arctan2(d_goal[:, 1], d_goal[:, 0])
            limited = bearing + np.clip(_wrap(total_dir - bearing), -_MAX_TOTAL_DEVIATION, _MAX_TOTAL_DEVIATION)
            theta_des = np.where(from_total, np.where(at_goal, total_dir, limited), theta_des)
        steered = pool.steer_to_goal[:n] & ~at_goal
        if np.any(steered):
            d_goal = pool.goal_pos[:n] - pool.pos[:n]
            theta_des = np.where(steered, np.arctan2(d_goal[:, 1], d_goal[:, 0]), theta_des)
        return theta_des

    def compute_pool(self, pool: AgentPool, store_forces: bool = False, dt: float = 1.0) -> None:
        n = pool.n
        if n == 0:
            self._last_forces = {}
            self._last_force_arrays = None
            self._omega.clear()
            return

        f_att, f_rep, f_wall, at_goal = self._compute_forces_pool(pool)
        total_f = f_att + f_rep + f_wall

        theta = pool.theta[:n]
        vel = pool.vel[:n]
        max_v = pool.max_velocity[:n]
        k_lat = self._lateral_gain[:n]
        k_lat_d = self._lateral_damping[:n]
        k_ang = self._angular_gain[:n]
        k_ang_d = self._angular_damping[:n]
        w_max = pool.pivot_angular_velocity[:n]

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        forward = np.stack((cos_t, sin_t), axis=1)
        perp = np.stack((-sin_t, cos_t), axis=1)

        f_fwd = np.sum(total_f * forward, axis=1)
        f_perp = np.sum(total_f * perp, axis=1)
        v_perp = np.sum(vel * perp, axis=1)
        f_lat = k_lat * f_perp - k_lat_d * v_perp

        body_a = f_fwd[:, None] * forward + f_lat[:, None] * perp
        new_vel = vel + body_a * dt

        speed = np.hypot(new_vel[:, 0], new_vel[:, 1])
        too_fast = speed > max_v
        if np.any(too_fast):
            new_vel[too_fast] *= (max_v[too_fast] / speed[too_fast])[:, None]

        new_vel[at_goal] = 0.0
        pool.vel[:n] = new_vel

        omega_arr = np.array(
            [self._omega.get(int(aid), 0.0) for aid in pool.agent_ids[:n]],
            dtype=np.float64,
        )

        err = _wrap(self._desired_heading(pool, f_att, total_f, at_goal) - theta)

        alpha = k_ang * err - k_ang_d * omega_arr
        omega_arr = omega_arr + alpha * dt
        omega_arr = np.clip(omega_arr, -w_max, w_max)
        omega_arr[at_goal] = 0.0

        new_theta = _wrap(theta + omega_arr * dt)
        pool.theta[:n] = new_theta

        self._omega = {int(aid): float(omega_arr[i]) for i, aid in enumerate(pool.agent_ids[:n])}

        if store_forces:
            self._stash_force_arrays(pool, n, f_att, f_rep, f_wall)
        else:
            self._last_force_arrays = None

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        self._last_agents = agents
        if not agents:
            self._last_forces = {}
            return {}

        velocities: dict[int, tuple[float, float]] = {}
        last_forces: dict[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {}

        for agent in agents:
            aid = agent.state.agent_id
            goal = global_goals.get(aid)
            if goal is None:
                velocities[aid] = (0.0, 0.0)
                continue

            forces = self._compute_forces_scalar(agent, goal)
            if forces is None:
                velocities[aid] = (0.0, 0.0)
                continue
            (f_att_x, f_att_y), (f_rep_x, f_rep_y), (f_wall_x, f_wall_y), _ = forces

            total_fx = f_att_x + f_rep_x + f_wall_x
            total_fy = f_att_y + f_rep_y + f_wall_y

            theta = agent.state.pose.theta
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            fx, fy = -sin_t, cos_t  # perp axis

            f_fwd = total_fx * cos_t + total_fy * sin_t
            f_perp = total_fx * fx + total_fy * fy
            cur_vx, cur_vy = agent.state.velocity
            v_perp = cur_vx * fx + cur_vy * fy

            lp = agent.params.local_planner_params
            f_lat = lp["lateral_gain"] * f_perp - lp["lateral_damping"] * v_perp

            ax = f_fwd * cos_t + f_lat * fx
            ay = f_fwd * sin_t + f_lat * fy

            new_vx = cur_vx + ax * dt
            new_vy = cur_vy + ay * dt

            max_velocity = agent.params.max_velocity
            speed = math.hypot(new_vx, new_vy)
            if speed > max_velocity:
                scale = max_velocity / speed
                new_vx *= scale
                new_vy *= scale

            velocities[aid] = (float(new_vx), float(new_vy))

            f_att_norm = math.hypot(f_att_x, f_att_y)
            from_total = self._param(lp, "heading_source") >= 0.5 and math.hypot(total_fx, total_fy) > _EPS
            if from_total or f_att_norm > _EPS:
                if from_total:
                    theta_des = math.atan2(goal.y - agent.state.pose.y, goal.x - agent.state.pose.x)
                    deviation = math.atan2(total_fy, total_fx) - theta_des
                    deviation = math.atan2(math.sin(deviation), math.cos(deviation))
                    theta_des += min(max(deviation, -_MAX_TOTAL_DEVIATION), _MAX_TOTAL_DEVIATION)
                else:
                    theta_des = math.atan2(f_att_y, f_att_x)
                err = math.atan2(math.sin(theta_des - theta), math.cos(theta_des - theta))
                omega = self._omega.get(aid, 0.0)
                alpha = lp["angular_gain"] * err - lp["angular_damping"] * omega
                omega += alpha * dt
                w_max = agent.params.pivot_angular_velocity
                if omega > w_max:
                    omega = w_max
                elif omega < -w_max:
                    omega = -w_max
                self._omega[aid] = omega
                agent.state.pose.theta = math.atan2(
                    math.sin(theta + omega * dt),
                    math.cos(theta + omega * dt),
                )

            last_forces[aid] = (
                (f_att_x, f_att_y),
                (f_rep_x, f_rep_y),
                (f_wall_x, f_wall_y),
            )

        self._last_forces = last_forces
        return velocities
