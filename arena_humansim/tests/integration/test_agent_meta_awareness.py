from __future__ import annotations

from collections.abc import Callable

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import (
    AgentTemplateModel,
    FlowScenarioConfig,
    ModuleConfig,
    RateKeyframeModel,
    ScenarioConfig,
    ShapeModel,
    SimulationParams,
    SinkScenarioConfig,
    SourceScenarioConfig,
)
from arena_humansim.utils.types import Pose2D
from arena_humansim_msgs.msg import AgentMeta as AgentMetaMsg
from rclpy.executors import SingleThreadedExecutor

from ._capture import LATCHED, settle, spin_until, subscribe

TICKS = 200
NEUTRAL = (0.0, 0.0, 1.0, 1.0, False)
BLIND = (0.0, 0.0, 1.0, 1.0, True)

# (range offset m, fov offset deg, repulsion gain, reaction gain, blind) of one agent
Awareness = tuple[float, float, float, float, bool]


def _crowd(awareness: str) -> ScenarioConfig:
    source = SourceScenarioConfig(
        pose=Pose2D(x=-6.0, y=0.0, theta=0.0),
        shape=ShapeModel(type="circle", radius=0.8),
        rate_profile=[RateKeyframeModel(t=0.0, rate=1.5)],
        max_concurrent=4,
        agent_template=AgentTemplateModel(desired_velocity_min=1.0, desired_velocity_max=1.3, agent_radius=0.25, robot_awareness=awareness),
    )
    sink = SinkScenarioConfig(pose=Pose2D(x=7.0, y=0.0, theta=0.0), shape=ShapeModel(type="circle", radius=1.0), absorption_radius=1.0)
    return ScenarioConfig(
        name=f"meta_{awareness}",
        simulation=SimulationParams(seed=5, dt=0.05, max_ticks=TICKS),
        modules=ModuleConfig(local_planner="sfm"),
        flow=FlowScenarioConfig(sources=[source], sinks=[sink]),
    )


def _pool_awareness(mgr: AgentManager) -> dict[int, Awareness]:
    pool = mgr._pool
    return {
        int(pool.agent_ids[row]): (
            float(pool.robot_range_offset[row]),
            float(pool.robot_fov_offset[row]),
            float(pool.robot_repulsion_gain[row]),
            float(pool.robot_reaction_gain[row]),
            bool(pool.robot_blind[row]),
        )
        for row in mgr._own_rows()
    }


def _last_published(mgr: AgentManager) -> tuple[AgentMetaMsg, dict[int, Awareness]]:
    """The last agent_meta of a run, with the pool values of its agents at the moment it was published."""
    metas = subscribe(mgr, AgentMetaMsg, "agent_meta", LATCHED)
    executor = SingleThreadedExecutor()
    executor.add_node(mgr)
    settle(executor)
    pool_at_publish: dict[int, Awareness] = {}
    for _ in range(TICKS):
        published = mgr._meta_published
        mgr.tick()
        if mgr._meta_published is not published:
            count = len(metas) + 1
            spin_until(executor, lambda count=count: len(metas) == count)
            pool_at_publish = _pool_awareness(mgr)
    executor.remove_node(mgr)
    mgr.destroy_node()
    return metas[-1], pool_at_publish


def _assert_meta_carries_pool_values(meta: AgentMetaMsg, pool: dict[int, Awareness]) -> None:
    assert [int(aid) for aid in meta.agent_id] == list(pool)
    for i, aid in enumerate(meta.agent_id):
        range_offset, fov_offset, repulsion_gain, reaction_gain, blind = pool[int(aid)]
        assert meta.robot_range_offset[i] == pytest.approx(range_offset)
        assert meta.robot_fov_offset[i] == pytest.approx(fov_offset)
        assert meta.robot_repulsion_gain[i] == pytest.approx(repulsion_gain)
        assert meta.robot_reaction_gain[i] == pytest.approx(reaction_gain)
        assert meta.robot_blind[i] == blind


@pytest.mark.parametrize(("awareness", "expected"), [("normal", NEUTRAL), ("blind", BLIND)])
def test_agent_meta_publishes_the_robot_values_of_each_crowd_agent(manager_factory: Callable[..., AgentManager], awareness: str, expected: Awareness) -> None:
    meta, pool = _last_published(manager_factory(_crowd(awareness), node_name=f"meta_{awareness}"))
    assert len(meta.agent_id) > 0
    assert set(pool.values()) == {expected}
    _assert_meta_carries_pool_values(meta, pool)


def test_agent_meta_publishes_a_scared_crowd_with_wider_fov_and_more_repulsion(manager_factory: Callable[..., AgentManager]) -> None:
    meta, pool = _last_published(manager_factory(_crowd("scared"), node_name="meta_scared"))
    assert len(meta.agent_id) > 0
    for _, fov_offset, repulsion_gain, reaction_gain, blind in pool.values():
        assert fov_offset > 0.0
        assert repulsion_gain > 1.0
        assert reaction_gain > 1.0
        assert not blind
    _assert_meta_carries_pool_values(meta, pool)
