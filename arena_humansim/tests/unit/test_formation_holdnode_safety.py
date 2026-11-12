from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

py_trees = pytest.importorskip("py_trees")

from arena_humansim.core.agents import BaseAgent
from arena_humansim.core.agents.types import AgentType, GoToStepDef, ParamDist, RobotStepDef, SequenceDef, StepDef
from arena_humansim.core.behavior.compiler import BehaviorTreeFactory
from arena_humansim.core.behavior.nodes import CancelNode, HoldNode, SeekNode
from arena_humansim.core.behavior.nodes.primitives import ClearOutcomeNode
from arena_humansim.core.world_knowledge import WorldKnowledge
from arena_humansim.utils.event_bus import EventBus
from arena_humansim.utils.scenario import load_scenario
from arena_humansim.utils.types import BehaviorTreeMovement, CommandType

_SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "config" / "scenarios"
_SCENARIO_PATHS = sorted(_SCENARIOS_DIR.glob("*.yaml"))


def _hold_node_step(step: StepDef | GoToStepDef | RobotStepDef) -> bool:
    if isinstance(step, (GoToStepDef, RobotStepDef)):
        return False
    if step.autonomous:
        return False
    if step.interaction is not None:
        return False
    return step.duration is not None


@pytest.mark.parametrize("scenario_path", _SCENARIO_PATHS, ids=lambda p: p.name)
def test_hold_node_steps_never_carry_interaction(scenario_path: Path) -> None:
    scenario = load_scenario(str(scenario_path))
    for atype_name, atype in scenario.agent_types.items():
        for seq_name, seq in atype.sequences.items():
            for step_name, step in seq.steps.items():
                if not _hold_node_step(step):
                    continue
                assert isinstance(step, StepDef)
                assert step.interaction is None, f"{scenario_path.name}:{atype_name}.{seq_name}.{step_name} compiles to a HoldNode but carries interaction={step.interaction!r}; a HoldNode emission would race formation binding"


def _compile_step(agent: BaseAgent, step: StepDef | GoToStepDef, world: WorldKnowledge, rng: np.random.Generator) -> py_trees.behaviour.Behaviour:
    atype = AgentType(
        name="t",
        mode="behavior_tree",
        sequences={"default": SequenceDef(steps={"s": step})},
    )
    factory = BehaviorTreeFactory(atype)
    bt = factory.build(agent, world, EventBus(), rng, 0.05)
    sm = bt.root
    return sm._sequences["default"]  # type: ignore[attr-defined]


def _leaves(root: py_trees.behaviour.Behaviour) -> list[py_trees.behaviour.Behaviour]:
    out: list[py_trees.behaviour.Behaviour] = []
    stack: list[py_trees.behaviour.Behaviour] = [root]
    while stack:
        node = stack.pop()
        children = getattr(node, "children", None)
        if children:
            stack.extend(children)
        else:
            out.append(node)
    return out


def test_pure_wait_step_contains_no_seek_or_cancel_leaves(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=1)
    agent.movement = BehaviorTreeMovement()
    step = StepDef(duration=ParamDist(0.5))
    root = _compile_step(agent, step, WorldKnowledge(), np.random.default_rng(0))
    leaf_types = {type(leaf) for leaf in _leaves(root)}
    assert HoldNode in leaf_types
    assert SeekNode not in leaf_types
    assert CancelNode not in leaf_types


def test_hold_node_emits_only_navigate(agent_factory: Callable[..., BaseAgent]) -> None:
    agent = agent_factory(agent_id=2, x=1.0, y=2.0)
    agent.movement = BehaviorTreeMovement()
    node = HoldNode("hold", agent, duration_source=ParamDist(1.0), rng=np.random.default_rng(0), dt=0.5)
    node.setup()
    seen_types: set[CommandType] = set()
    for _ in range(5):
        node.tick_once()
        cmd = agent.movement.command  # type: ignore[union-attr]
        if cmd is not None:
            seen_types.add(cmd.type)
    assert seen_types == {CommandType.NAVIGATE}


def test_go_to_literal_step_contains_no_seek(agent_factory: Callable[..., BaseAgent]) -> None:
    from arena_humansim.utils.types import Pose2D

    agent = agent_factory(agent_id=3)
    agent.movement = BehaviorTreeMovement()
    step = GoToStepDef(target_pose=Pose2D(x=5.0, y=5.0))
    root = _compile_step(agent, step, WorldKnowledge(), np.random.default_rng(0))
    leaf_types = {type(leaf) for leaf in _leaves(root)}
    assert SeekNode not in leaf_types
    assert CancelNode not in leaf_types
    assert ClearOutcomeNode in leaf_types
