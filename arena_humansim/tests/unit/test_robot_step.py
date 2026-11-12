from __future__ import annotations

import pytest
from arena_humansim.core.agents.types import RobotStepDef, RobotTrigger
from arena_humansim.utils.scenario import _structure_manual
from arena_humansim.utils.scenario_loader import converter


def _scenario(step: dict) -> dict:
    return {
        "name": "t",
        "simulation": {"seed": 1, "dt": 0.05, "max_ticks": 1},
        "modules": {},
        "agent_types": {"w": {"mode": "behavior_tree", "sequences": {"s": {"steps": {"r": step}}}}},
    }


def test_robot_step_structures_trigger_and_act() -> None:
    scn = _structure_manual(_scenario({"kind": "robot", "when": {"ahead": 10.0, "lateral": 2.0}, "act": "stand", "face": "robot", "gesture": "halt", "duration": {"mean": 5.0}}))
    step = scn.agent_types["w"].sequences["s"].steps["r"]
    assert isinstance(step, RobotStepDef)
    assert step.when == RobotTrigger(ahead=10.0, lateral=2.0)
    assert (step.act, step.face, step.gesture) == ("stand", "robot", "halt")
    assert step.duration is not None and step.duration.mean == 5.0


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"kind": "robot", "act": "dance"}, "act"),
        ({"kind": "robot", "when": {"lateral": 3.0}}, "at least one of"),
        ({"kind": "robot", "when": {"ahead": -1.0}}, "> 0"),
        ({"kind": "robot", "when": {"nearby": 1.0}}, "unknown robot trigger"),
        ({"kind": "robot", "act": "stand", "gesture": "wave"}, "gesture"),
        ({"kind": "robot", "face": "nobody", "act": "stand"}, "face"),
        ({"kind": "robot", "face": 90.0}, "no face or gesture"),
        ({"kind": "robot", "target": "bench"}, "unknown robot step fields"),
    ],
)
def test_robot_step_rejects_malformed_fields(step: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        converter.structure(step, RobotStepDef)
