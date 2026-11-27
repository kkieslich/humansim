from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.replay import ReplayManager, StateError, _compare_agent_states
from arena_humansim.utils.types import AgentState, Pose2D


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_spawn(agent_id: int, params: dict) -> dict:
    return {
        "event": "spawn",
        "agent_id": agent_id,
        "agent_type": params.get("name", "adult"),
        "params": params,
    }


def _nested_params(name: str = "adult") -> dict:
    return {
        "name": name,
        "desired_velocity": 1.1,
        "agent_radius": 0.25,
        "max_velocity": 1.5,
        "max_acceleration": 1.5,
        "max_deceleration": 2.5,
        "min_turning_radius": 0.3,
        "pivot_angular_velocity": 2.0,
        "stair_effort": 2.2,
        "perception": {"vision_range": 7.5, "vision_fov": 200.0},
        "local_planner_params": {
            "relaxation_time": 0.7,
            "repulsion_strength": 3.0,
            "repulsion_range": 0.4,
            "anisotropy": 0.6,
        },
        "perception_stack": ["default"],
        "local_planner": "sfm",
        "global_planner": "dijkstra",
        "animation": "noop",
    }


def _flat_legacy_params(name: str = "child") -> dict:
    return {
        "name": name,
        "desired_velocity": 0.9,
        "agent_radius": 0.2,
        "perception_stack": ["default"],
        "local_planner": "sfm",
        "global_planner": "dijkstra",
        "animation": "noop",
        "vision_range": 4.0,
        "vision_fov": 120.0,
        "relaxation_time": 0.6,
        "repulsion_strength": 2.5,
        "repulsion_range": 0.35,
        "anisotropy": 0.55,
    }


def _tick_record(tick: int, agents: dict | None = None, commands: dict | None = None) -> dict:
    return {
        "tick": tick,
        "timestamp": float(tick) * 0.05,
        "agents": agents or {},
        "interactions": {},
        "commands": commands or {},
    }


def _make_replay(tmp_path: Path, records: list[dict]) -> ReplayManager:
    log_path = tmp_path / "session.jsonl"
    _write_jsonl(log_path, records)
    replay = ReplayManager()
    replay.load(str(log_path))
    return replay


def test_load_separates_spawns_and_ticks(tmp_path: Path) -> None:
    records = [
        _make_spawn(1, _nested_params("adult")),
        _make_spawn(2, _nested_params("child")),
        _tick_record(0),
        _tick_record(1),
        _tick_record(2),
    ]
    replay = _make_replay(tmp_path, records)
    assert replay.tick_count == 3
    assert sorted(replay.spawned_agent_ids) == [1, 2]


def test_load_ignores_blank_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps(_make_spawn(1, _nested_params())) + "\n")
        f.write("\n")
        f.write(json.dumps(_tick_record(0)) + "\n")
        f.write("   \n")
    replay = ReplayManager()
    replay.load(str(log_path))
    assert replay.tick_count == 1
    assert replay.spawned_agent_ids == [1]


def test_get_tick_by_logical_number_not_index(tmp_path: Path) -> None:
    records = [
        _tick_record(5),
        _tick_record(10),
        _tick_record(15),
    ]
    replay = _make_replay(tmp_path, records)

    r10 = replay.get_tick(10)
    assert r10 is not None
    assert r10["tick"] == 10

    r5 = replay.get_tick(5)
    assert r5 is not None and r5["tick"] == 5

    r15 = replay.get_tick(15)
    assert r15 is not None and r15["tick"] == 15

    assert replay.get_tick(7) is None
    assert replay.get_tick(0) is None


def test_get_spawn_params_nested(tmp_path: Path) -> None:
    replay = _make_replay(tmp_path, [_make_spawn(1, _nested_params("adult"))])
    params = replay.get_spawn_params(1)
    assert params is not None
    assert params.name == "adult"
    assert params.desired_velocity == pytest.approx(1.1)
    assert params.agent_radius == pytest.approx(0.25)
    assert params.perception.vision_range == pytest.approx(7.5)
    assert params.perception.vision_fov == pytest.approx(200.0)
    assert params.local_planner_params["relaxation_time"] == pytest.approx(0.7)
    assert params.local_planner_params["repulsion_strength"] == pytest.approx(3.0)
    assert params.local_planner_params["repulsion_range"] == pytest.approx(0.4)
    assert params.local_planner_params["anisotropy"] == pytest.approx(0.6)
    assert params.perception_stack == ("default",)
    assert params.local_planner == "sfm"
    assert params.stair_effort == pytest.approx(2.2)


def test_get_spawn_params_flat_legacy_fallback(tmp_path: Path) -> None:
    replay = _make_replay(tmp_path, [_make_spawn(2, _flat_legacy_params("child"))])
    params = replay.get_spawn_params(2)
    assert params is not None
    assert params.name == "child"
    assert params.perception.vision_range == pytest.approx(4.0)
    assert params.perception.vision_fov == pytest.approx(120.0)
    assert params.local_planner_params["relaxation_time"] == pytest.approx(0.6)
    assert params.local_planner_params["repulsion_strength"] == pytest.approx(2.5)
    assert params.local_planner_params["repulsion_range"] == pytest.approx(0.35)
    assert params.local_planner_params["anisotropy"] == pytest.approx(0.55)
    assert params.stair_effort == 1.0


def test_get_spawn_params_missing_returns_none(tmp_path: Path) -> None:
    replay = _make_replay(tmp_path, [_make_spawn(1, _nested_params())])
    assert replay.get_spawn_params(999) is None


def test_get_agents_at_tick_parses_pose_and_velocity(tmp_path: Path) -> None:
    agents = {
        "1": {
            "pose": {"x": 1.5, "y": -0.5, "theta": 0.7},
            "velocity": {"vx": 0.3, "vy": -0.1},
        },
        "7": {
            "pose": {"x": -2.0, "y": 3.0, "theta": 1.1},
            "velocity": {"vx": 0.0, "vy": 0.5},
        },
    }
    replay = _make_replay(tmp_path, [_tick_record(10, agents=agents)])

    out = replay.get_agents_at_tick(10)
    assert set(out.keys()) == {1, 7}

    a1 = out[1]
    assert isinstance(a1, AgentState)
    assert a1.agent_id == 1
    assert a1.pose.x == pytest.approx(1.5)
    assert a1.pose.y == pytest.approx(-0.5)
    assert a1.pose.theta == pytest.approx(0.7)
    assert a1.velocity[0] == pytest.approx(0.3)
    assert a1.velocity[1] == pytest.approx(-0.1)

    assert replay.get_agents_at_tick(999) == {}


def test_get_commands_at_tick_parses_target_pose(tmp_path: Path) -> None:
    commands = {
        "1": {
            "type": 0,
            "target_pose": {"x": 5.0, "y": 2.5, "theta": 0.25},
            "interaction_target": -1,
        },
        "4": {
            "type": 1,
            "target_pose": {"x": -1.0, "y": 0.0, "theta": 0.0},
            "interaction_target": 99,
        },
    }
    replay = _make_replay(tmp_path, [_tick_record(3, commands=commands)])

    out = replay.get_commands_at_tick(3)
    assert set(out.keys()) == {1, 4}

    c1 = out[1]
    assert c1.agent_id == 1
    assert c1.type == 0
    assert c1.target_pose.x == pytest.approx(5.0)
    assert c1.target_pose.y == pytest.approx(2.5)
    assert c1.target_pose.theta == pytest.approx(0.25)
    assert c1.interaction_target == -1

    c4 = out[4]
    assert c4.type == 1
    assert c4.interaction_target == 99

    assert replay.get_commands_at_tick(999) == {}


def _state(aid: int, x: float = 0.0, y: float = 0.0, theta: float = 0.0, vx: float = 0.0, vy: float = 0.0) -> AgentState:
    return AgentState(agent_id=aid, pose=Pose2D(x=x, y=y, theta=theta), velocity=(vx, vy))


def test_compare_agent_states_identical_returns_none() -> None:
    actual = {1: _state(1, 1.0, 2.0, 0.5, 0.1, 0.2)}
    expected = {1: _state(1, 1.0, 2.0, 0.5, 0.1, 0.2)}
    assert _compare_agent_states(actual, expected, tick=4) is None


def test_compare_agent_states_accepts_last_bit_differences() -> None:
    """A replay on another CPU family drifts in the last bits (CI saw 2e-13 m / 1e-12 rad after 193 ticks)."""
    actual = {1: _state(1, -1.1925549249988792, -0.49339679083479454, 6.565278789123273, 0.3 + 1e-12, 0.1)}
    expected = {1: _state(1, -1.1925549249988154, -0.49339679083502924, 6.565278789122241, 0.3, 0.1)}
    assert _compare_agent_states(actual, expected, tick=193) is None


def test_compare_agent_states_tracks_the_largest_error() -> None:
    errors = StateError()
    _compare_agent_states({1: _state(1, x=1e-12, theta=3e-12)}, {1: _state(1)}, tick=1, max_error=errors)
    _compare_agent_states({1: _state(1, y=-2e-12, vy=4e-12)}, {1: _state(1)}, tick=2, max_error=errors)
    assert (errors.position_m, errors.heading_rad, errors.velocity_mps) == (2e-12, 3e-12, 4e-12)


def test_compare_agent_states_detects_missing_from_actual() -> None:
    actual: dict[int, AgentState] = {}
    expected = {1: _state(1)}
    div = _compare_agent_states(actual, expected, tick=4)
    assert div is not None
    assert div.tick == 4
    assert div.agent_id == 1
    assert "missing from actual" in div.detail


def test_compare_agent_states_detects_extra_in_actual() -> None:
    actual = {2: _state(2)}
    expected: dict[int, AgentState] = {}
    div = _compare_agent_states(actual, expected, tick=9)
    assert div is not None
    assert div.tick == 9
    assert div.agent_id == 2
    assert "missing from expected" in div.detail


def test_compare_agent_states_detects_pose_divergence() -> None:
    actual = {1: _state(1, x=1.0, y=0.0)}
    expected = {1: _state(1, x=1.0001, y=0.0)}
    div = _compare_agent_states(actual, expected, tick=2)
    assert div is not None
    assert div.agent_id == 1
    assert "pose mismatch: off by 0.0001 m" in div.detail


def test_compare_agent_states_detects_heading_divergence_just_above_tolerance() -> None:
    div = _compare_agent_states({1: _state(1, theta=0.5 + 1e-8)}, {1: _state(1, theta=0.5)}, tick=3)
    assert div is not None
    assert "pose mismatch" in div.detail


def test_compare_agent_states_detects_velocity_divergence() -> None:
    actual = {1: _state(1, vx=0.5, vy=0.0)}
    expected = {1: _state(1, vx=0.5, vy=0.001)}
    div = _compare_agent_states(actual, expected, tick=7)
    assert div is not None
    assert div.agent_id == 1
    assert "velocity mismatch" in div.detail
