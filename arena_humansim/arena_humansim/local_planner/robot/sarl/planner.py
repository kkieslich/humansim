from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from arena_humansim.core.agents import BaseAgent
from arena_humansim.utils.types import Pose2D

from ..base import RobotPolicy, fetch_to_disk, policy_cache_dir
from .obs import build_full_state, build_observable_states, rotate

if TYPE_CHECKING:
    from .model import ValueNetwork

_CHECKPOINT_URL = "https://huggingface.co/arena-rosnav/sarl/resolve/main/rl_model.pth"
_CHECKPOINT_SHA256 = "4f1e59c9a3c77db5e75b0581274123bb4e6e053733338ee1e8d81743c47d64cb"
_CHECKPOINT_FILENAME = "rl_model.pth"

_INPUT_DIM = 13
_SELF_STATE_DIM = 6
_MLP1_DIMS = [150, 100]
_MLP2_DIMS = [100, 50]
_MLP3_DIMS = [150, 100, 100, 1]
_ATTENTION_DIMS = [100, 100, 1]
_WITH_GLOBAL_STATE = False

_DEFAULT_GAMMA = 0.9
_DEFAULT_SPEED_SAMPLES = 5
_DEFAULT_ROTATION_SAMPLES = 16
_DEFAULT_DEFAULT_RADIUS = 0.3
_DEFAULT_MAX_HUMANS = 0
# CrowdNav circle-crossing training had goal distances ~4m. Beyond ~8m the value
# net is OOD and argmax becomes noisy; clamp the planner-facing goal to a carrot
# along the agent->goal line.
_DEFAULT_MAX_LOOKAHEAD = 4.0


class SARLPlanner(RobotPolicy):
    """SARL value-iteration robot navigator (Chen et al., ICRA 2019).

    Loads the upstream LeeKeyu/sarl_star `rl_model.pth` checkpoint into a vanilla
    CrowdNav ValueNetwork and selects each step's action by enumerating a holonomic
    speed/rotation grid, propagating one time_step, and picking the argmax of the
    predicted next-state value."""

    bypasses_kinematic_constraints: bool = True
    needs_global_subgoal: bool = True
    supports_pool: bool = False

    def __init__(
        self,
        checkpoint_path: str = "",
        device: str = "cpu",
        v_pref: float = 1.0,
        time_step: float = 0.25,
        seed: int = 0,
        action_space: str = "holonomic",
        gamma: float = _DEFAULT_GAMMA,
        speed_samples: int = _DEFAULT_SPEED_SAMPLES,
        rotation_samples: int = _DEFAULT_ROTATION_SAMPLES,
        default_other_radius: float = _DEFAULT_DEFAULT_RADIUS,
        max_humans: int = _DEFAULT_MAX_HUMANS,
        max_lookahead: float = _DEFAULT_MAX_LOOKAHEAD,
    ):
        self._checkpoint_path = checkpoint_path or str(policy_cache_dir("sarl") / _CHECKPOINT_FILENAME)
        self._device_str = device
        self._v_pref = float(v_pref)
        self._time_step = float(time_step)
        self._seed = int(seed)
        self._kinematics = action_space
        self._gamma = float(gamma)
        self._speed_samples = int(speed_samples)
        self._rotation_samples = int(rotation_samples)
        self._default_other_radius = float(default_other_radius)
        self._max_humans = int(max_humans)
        self._max_lookahead = float(max_lookahead)

        self._model: ValueNetwork | None = None
        self._device = None
        self._torch = None
        self._action_space: np.ndarray | None = None

    def apply_policy_params(self, params_json: str) -> None:
        if not params_json:
            return
        try:
            blob = json.loads(params_json)
        except (ValueError, TypeError):
            return
        if not isinstance(blob, dict):
            return
        if "checkpoint_path" in blob and isinstance(blob["checkpoint_path"], str):
            self._checkpoint_path = blob["checkpoint_path"]
            self._model = None
        if "device" in blob and isinstance(blob["device"], str):
            self._device_str = blob["device"]
            self._model = None
        if "v_pref" in blob:
            self._v_pref = float(blob["v_pref"])
            self._action_space = None
        if "time_step" in blob:
            self._time_step = float(blob["time_step"])
        if "seed" in blob:
            self._seed = int(blob["seed"])
            self._model = None
        if "action_space" in blob and isinstance(blob["action_space"], str):
            self._kinematics = blob["action_space"]
            self._action_space = None
        if "gamma" in blob:
            self._gamma = float(blob["gamma"])
        if "speed_samples" in blob:
            self._speed_samples = int(blob["speed_samples"])
            self._action_space = None
        if "rotation_samples" in blob:
            self._rotation_samples = int(blob["rotation_samples"])
            self._action_space = None
        if "default_other_radius" in blob:
            self._default_other_radius = float(blob["default_other_radius"])
        if "max_humans" in blob:
            self._max_humans = int(blob["max_humans"])
        if "max_lookahead" in blob:
            self._max_lookahead = float(blob["max_lookahead"])

    def _ensure_checkpoint_on_disk(self) -> None:
        path = Path(self._checkpoint_path)
        if path.exists():
            return
        default = policy_cache_dir("sarl") / _CHECKPOINT_FILENAME
        if path != default:
            raise FileNotFoundError(f"SARL checkpoint not found at {path}")
        self._logger.info(f"Fetching SARL checkpoint from {_CHECKPOINT_URL} -> {path}")
        fetch_to_disk(_CHECKPOINT_URL, path, sha256=_CHECKPOINT_SHA256)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._seed_rngs(self._seed)
        import torch

        from .model import ValueNetwork

        self._ensure_checkpoint_on_disk()
        self._torch = torch
        self._device = torch.device(self._device_str)

        try:
            ckpt = torch.load(self._checkpoint_path, map_location=self._device, weights_only=True)
        except (TypeError, RuntimeError):
            ckpt = torch.load(self._checkpoint_path, map_location=self._device)

        state = ckpt
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]

        cleaned: dict[str, object] = {}
        for k, v in state.items():
            cleaned[k.removeprefix("module.")] = v

        model = ValueNetwork(
            input_dim=_INPUT_DIM,
            self_state_dim=_SELF_STATE_DIM,
            mlp1_dims=list(_MLP1_DIMS),
            mlp2_dims=list(_MLP2_DIMS),
            mlp3_dims=list(_MLP3_DIMS),
            attention_dims=list(_ATTENTION_DIMS),
            with_global_state=_WITH_GLOBAL_STATE,
        ).to(self._device)
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._logger.info(f"Loaded SARL checkpoint from {self._checkpoint_path} on {self._device_str}")

    def _build_action_space(self, v_pref: float) -> np.ndarray:
        # Mirrors crowd_nav.policy.cadrl.build_action_space for holonomic kinematics.
        speeds = np.array(
            [(math.exp((i + 1) / self._speed_samples) - 1.0) / (math.e - 1.0) * v_pref for i in range(self._speed_samples)],
            dtype=np.float32,
        )
        rotations = np.linspace(0.0, 2.0 * np.pi, self._rotation_samples, endpoint=False, dtype=np.float32)
        actions: list[tuple[float, float]] = [(0.0, 0.0)]
        for r in rotations:
            cr = float(np.cos(r))
            sr = float(np.sin(r))
            for s in speeds:
                actions.append((float(s) * cr, float(s) * sr))
        return np.asarray(actions, dtype=np.float32)

    def _action_space_for(self, v_pref: float) -> np.ndarray:
        if self._action_space is None:
            self._action_space = self._build_action_space(v_pref)
        return self._action_space

    def _carrot_goal(self, agent: BaseAgent, goal: Pose2D) -> Pose2D:
        if self._max_lookahead <= 0.0:
            return goal
        px = float(agent.state.pose.x)
        py = float(agent.state.pose.y)
        dx = float(goal.x) - px
        dy = float(goal.y) - py
        d = math.hypot(dx, dy)
        if d <= self._max_lookahead:
            return goal
        scale = self._max_lookahead / d
        return Pose2D(x=px + dx * scale, y=py + dy * scale, theta=goal.theta)

    def compute(
        self,
        agents: Sequence[BaseAgent],
        global_goals: dict[int, Pose2D],
        dt: float = 1.0,
    ) -> dict[int, tuple[float, float]]:
        if not agents:
            return {}

        self._ensure_model()
        torch = self._torch
        assert torch is not None and self._model is not None and self._device is not None

        out: dict[int, tuple[float, float]] = {}
        for agent in agents:
            aid = int(agent.state.agent_id)
            goal = global_goals.get(aid)
            if goal is None:
                out[aid] = (0.0, 0.0)
                continue

            carrot = self._carrot_goal(agent, goal)
            self_state = build_full_state(agent, carrot)
            v_pref = float(agent.params.desired_velocity) if agent.params.desired_velocity > 0 else self._v_pref
            actions = self._action_space_for(v_pref)
            human_states = build_observable_states(agent, self._default_other_radius, max_humans=self._max_humans)
            out[aid] = self._best_action(self_state, actions, human_states)
        return out

    def _best_action(
        self,
        self_state: tuple[float, ...],
        actions: np.ndarray,
        human_states: list[tuple[float, float, float, float, float]],
    ) -> tuple[float, float]:
        # SARL fans out 5x16+1 candidate actions per tick; the only difference between
        # candidates is the next-self row, so all 81 value-net forwards collapse into a
        # single (n_actions, n_humans, 13) batch.
        torch = self._torch
        assert torch is not None and self._model is not None and self._device is not None

        n_actions = actions.shape[0]
        self_arr = np.asarray(self_state, dtype=np.float32)
        next_self = np.broadcast_to(self_arr, (n_actions, 9)).copy()
        next_self[:, 0] = self_arr[0] + actions[:, 0] * self._time_step
        next_self[:, 1] = self_arr[1] + actions[:, 1] * self._time_step
        next_self[:, 2] = actions[:, 0]
        next_self[:, 3] = actions[:, 1]

        if human_states:
            h_arr = np.asarray(human_states, dtype=np.float32)
            next_humans = h_arr.copy()
            next_humans[:, 0] += h_arr[:, 2] * self._time_step
            next_humans[:, 1] += h_arr[:, 3] * self._time_step
        else:
            far = 1e3
            next_humans = np.array(
                [[self_arr[0] + far, self_arr[1] + far, 0.0, 0.0, 0.3]],
                dtype=np.float32,
            )

        n_humans = next_humans.shape[0]
        joint = np.empty((n_actions, n_humans, 14), dtype=np.float32)
        joint[:, :, :9] = next_self[:, None, :]
        joint[:, :, 9:14] = next_humans[None, :, :]

        rotated = rotate(joint.reshape(-1, 14)).reshape(n_actions, n_humans, 13)
        with torch.no_grad():
            values = self._model(torch.from_numpy(rotated).to(self._device))
        best_idx = int(values.view(-1).argmax().item())
        return float(actions[best_idx, 0]), float(actions[best_idx, 1])


if __name__ == "__main__":
    import types
    from collections.abc import Sequence as _Seq
    from typing import Any as _Any

    class _FakePose:
        def __init__(self, x: float, y: float, theta: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.theta = theta

    class _FakeBelief:
        def __init__(self, observed: _Seq[_Any]) -> None:
            self.observed_agents = observed

    class _FakeObserved:
        def __init__(self, aid: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
            self.agent_id = aid
            self.pose = _FakePose(x, y)
            self.velocity = (vx, vy)

    def make_agent(aid: int, x: float, y: float, observed: _Seq[_Any]) -> types.SimpleNamespace:
        a = types.SimpleNamespace()
        a.state = types.SimpleNamespace(
            agent_id=aid,
            pose=_FakePose(x, y),
            velocity=(0.0, 0.0),
        )
        a.params = types.SimpleNamespace(
            agent_radius=0.3,
            desired_velocity=1.0,
            max_velocity=1.0,
        )
        a.belief = _FakeBelief(observed)
        return a

    obs1 = [_FakeObserved(10, 1.0, 0.5), _FakeObserved(11, -1.0, 1.5)]
    agents = [make_agent(1, 0.0, 0.0, obs1)]
    goals = {1: _FakePose(5.0, 0.0)}

    p1 = SARLPlanner(seed=42)
    v1 = p1.compute(agents, goals)  # type: ignore[arg-type]

    p2 = SARLPlanner(seed=42)
    v2 = p2.compute(agents, goals)  # type: ignore[arg-type]

    a1 = np.array(list(v1.values()))
    a2 = np.array(list(v2.values()))
    assert np.array_equal(a1, a2), f"mismatch: {a1} vs {a2}"
    print("SARL determinism OK")
