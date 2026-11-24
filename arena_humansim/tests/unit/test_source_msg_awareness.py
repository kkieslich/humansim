from __future__ import annotations

import pytest
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.agents.types import AWARENESS_PRESETS, AwarenessMixture
from arena_humansim_msgs.msg import SourceConfig as SourceConfigMsg


def _source_msg(robot_awareness: str) -> SourceConfigMsg:
    msg = SourceConfigMsg()
    msg.name = "crowd"
    msg.agent.agent_type = "adult"
    msg.agent.robot_awareness = robot_awareness
    return msg


def test_source_with_empty_awareness_text_keeps_the_agent_type_awareness() -> None:
    config = AgentManager._source_msg_to_config(_source_msg(""))
    assert config.agent.robot_awareness is None


def test_source_awareness_text_can_name_a_preset() -> None:
    config = AgentManager._source_msg_to_config(_source_msg("scared"))
    assert config.agent.robot_awareness == AWARENESS_PRESETS["scared"]


def test_source_awareness_text_can_carry_a_crowd_mixture_with_distributions() -> None:
    text = '[{"weight": 0.3, "awareness": {"detection": {"mean": -1.0, "std": 0.5, "clip_low": -2.0, "clip_high": 0.0}}}, {"weight": 0.7, "awareness": "normal"}]'
    config = AgentManager._source_msg_to_config(_source_msg(text))
    assert isinstance(config.agent.robot_awareness, AwarenessMixture)
    assert [weight for weight, _ in config.agent.robot_awareness.components] == [0.3, 0.7]


def test_source_awareness_text_that_does_not_parse_is_rejected() -> None:
    with pytest.raises(ValueError):
        AgentManager._source_msg_to_config(_source_msg('{"detection": '))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('{"detection": {"mean": 0.0, "sd": 1.0}}', "unknown robot awareness distribution field 'sd'"),
        ('{"detection": {"mean": "far"}}', "distribution values must be numbers"),
        ('{"detection": {"mean": 0.0, "std": -1.0}}', "std must be non-negative"),
        ('[{"weight": null, "awareness": "normal"}]', "mixture weights must be numbers"),
        ('{"blind": "false"}', "blind must be true or false"),
    ],
    ids=["typo_key", "text_value", "negative_std", "null_weight", "text_blind"],
)
def test_source_awareness_text_with_malformed_values_is_a_value_error(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AgentManager._source_msg_to_config(_source_msg(text))
