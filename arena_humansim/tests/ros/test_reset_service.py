from __future__ import annotations

import pytest

from tests.ros._helpers import (
    RemoveAgents,
    ResetSimulation,
    RosTestSystem,
    SpawnAgents,
    make_agent_msg,
    make_remove_request,
    make_reset_request,
    make_spawn_request,
)

pytestmark = pytest.mark.ros


@pytest.fixture(scope="module")
def system(ros_system: RosTestSystem) -> RosTestSystem:
    ros_system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    return ros_system


def test_reset_empties_pool(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    specs = [{"x": float(i), "y": 0.0} for i in range(3)]
    system.call(SpawnAgents, "spawn_agents", make_spawn_request(specs))

    resp = system.call(ResetSimulation, "reset", make_reset_request())
    assert resp.success is True
    assert system.manager._pool.n == 0
    assert len(system.manager._agents) == 0
    assert system.manager._tick_count == 0


def test_reset_reseeds_ids(system: RosTestSystem) -> None:
    system.call(ResetSimulation, "reset", make_reset_request())
    first = system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0}]),
    )
    first_ids = list(first.spawned_ids)

    system.call(ResetSimulation, "reset", make_reset_request())
    second = system.call(
        SpawnAgents,
        "spawn_agents",
        make_spawn_request([{"x": 0.0, "y": 0.0}]),
    )
    second_ids = list(second.spawned_ids)

    assert first_ids == second_ids


def _named_params(system: RosTestSystem, *names: str) -> dict[str, object]:
    msgs = []
    for name in names:
        msg = make_agent_msg()
        msg.name = name
        msgs.append(msg)
    resp = system.call(SpawnAgents, "spawn_agents", SpawnAgents.Request(agents=msgs))
    assert resp.success
    return {name: system.manager._agents[aid].params for name, aid in zip(names, resp.spawned_ids, strict=True)}


def test_remove_all_gives_the_next_crowd_fresh_streams(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    first = _named_params(system, "p")
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    assert _named_params(system, "p") == first


def test_named_agents_draw_by_name_whatever_their_id(system: RosTestSystem) -> None:
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    first = _named_params(system, "env_0/p", "env_0/q")
    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    swapped = _named_params(system, "env_1/q", "env_1/p")
    assert swapped["env_1/q"] == first["env_0/q"]
    assert swapped["env_1/p"] == first["env_0/p"]
    assert first["env_0/p"] != first["env_0/q"]


def test_reset_deterministic_trajectory(system: RosTestSystem) -> None:
    def run_one_pass() -> list[tuple[float, float]]:
        system.call(ResetSimulation, "reset", make_reset_request())
        system.call(
            SpawnAgents,
            "spawn_agents",
            make_spawn_request([{"x": 0.0, "y": 0.0, "waypoints": [(5.0, 0.0)]}]),
        )
        poses: list[tuple[float, float]] = []
        for _ in range(10):
            system.tick_manager(1)
            pool = system.manager._pool
            if pool.n > 0:
                poses.append((float(pool.pos[0, 0]), float(pool.pos[0, 1])))
        return poses

    first = run_one_pass()
    second = run_one_pass()

    assert first == second, "reset+reseed must produce identical trajectory"


@pytest.mark.parametrize("wipe", [False, True], ids=["remove_all", "reset"])
def test_a_fresh_crowd_starts_without_the_old_crowds_paths(system: RosTestSystem, wipe: bool) -> None:
    """Ids restart at 1, so a path or goal cached for the old agent 1 would steer the new agent 1."""
    import numpy as np

    from arena_humansim.utils.types import Pose2D

    system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    system.call(SpawnAgents, "spawn_agents", make_spawn_request([{"x": 0.0, "y": 0.0, "waypoints": [(5.0, 0.0)]}]))
    system.tick_manager(5)
    planners = system.manager._module_pool["global_planner"].values()
    for planner in planners:
        planner._path_cache[1] = ((5.0, 0.0), [Pose2D(x=5.0, y=0.0)], np.array([[5.0, 0.0]]), 0)
        planner._cached_results = {1: Pose2D(x=5.0, y=0.0)}
    if wipe:
        system.call(ResetSimulation, "reset", make_reset_request())
    else:
        system.call(RemoveAgents, "remove_agents", make_remove_request([]))
    for planner in planners:
        assert planner.get_cached_paths() == {}
        assert planner.get_cached_goals() == {}
