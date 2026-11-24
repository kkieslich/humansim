from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim.core.agents.types import AWARENESS_PRESETS, AwarenessMixture, AwarenessShift, as_awareness_spec, awareness_from_text, sample_agent_type
from arena_humansim.utils.scenario import _structure_manual

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "agent_types"


def test_agent_type_yaml_accepts_a_preset_name(tmp_path: Path) -> None:
    path = tmp_path / "pal.yaml"
    path.write_text("name: pal\nextends: adult\nperception:\n  robot_awareness: distracted\n", encoding="utf-8")
    atype = load_agent_type_from_file(path)
    assert atype.perception.robot_awareness == AWARENESS_PRESETS["distracted"]


def test_agent_type_yaml_accepts_a_shift_mapping(tmp_path: Path) -> None:
    path = tmp_path / "wary.yaml"
    path.write_text(
        "name: wary\nextends: adult\nperception:\n  robot_awareness:\n    detection: -1.5\n    reaction: {mean: 0.5, std: 0.5, clip_low: -1.0, clip_high: 1.0}\n",
        encoding="utf-8",
    )
    spec = load_agent_type_from_file(path).perception.robot_awareness
    assert spec.detection.mean == -1.5
    assert spec.reaction.std == 0.5
    assert spec.fov.mean == 0.0


def test_bool_alias_maps_to_the_normal_and_blind_presets() -> None:
    assert as_awareness_spec(True) == AWARENESS_PRESETS["normal"]
    assert as_awareness_spec(False) == AWARENESS_PRESETS["blind"]


def test_unknown_preset_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown robot awareness preset 'sleepy'"):
        as_awareness_spec("sleepy")


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown robot awareness field 'range'"):
        as_awareness_spec({"range": -1.0})


def test_list_is_a_weighted_mixture() -> None:
    spec = as_awareness_spec([{"weight": 0.6, "awareness": "normal"}, {"weight": 0.4, "awareness": "scared"}])
    assert isinstance(spec, AwarenessMixture)
    assert [weight for weight, _ in spec.components] == [0.6, 0.4]
    assert spec.components[1][1] == AWARENESS_PRESETS["scared"]


def test_mixture_weights_must_sum_above_zero() -> None:
    with pytest.raises(ValueError, match="mixture weights"):
        as_awareness_spec([{"weight": 0.0, "awareness": "normal"}])


def test_shipped_archetypes_resolve_to_their_presets() -> None:
    for name in ("normal", "interested", "scared", "blind", "aggressive"):
        atype = load_agent_type_from_file(CONFIG_DIR / f"{name}.yaml")
        assert atype.perception.robot_awareness == AWARENESS_PRESETS[name], name


def test_distracted_archetype_keeps_its_own_small_shifts() -> None:
    atype = load_agent_type_from_file(CONFIG_DIR / "distracted.yaml")
    assert atype.perception.robot_awareness == AwarenessShift(detection=-0.75, repulsion=-3.0, reaction=-3.0)


def test_scenario_manual_structure_keeps_awareness_values() -> None:
    scenario_dict = {
        "name": "test_awareness_scenario",
        "agents": [
            {"agent_id": 1, "agent_type": "adult", "robot_awareness": "distracted"},
            {"agent_id": 2, "agent_type": "adult", "robot_aware": False},
            {"agent_id": 3, "agent_type": "adult", "robot_awareness": {"detection": -1.0}},
        ],
        "flow": {
            "sources": [
                {
                    "pose": {"x": 0.0, "y": 0.0},
                    "agent_template": {
                        "agent_type": "adult",
                        "robot_awareness": [{"weight": 1.0, "awareness": "blind"}],
                    },
                }
            ]
        },
    }

    config = _structure_manual(scenario_dict)

    assert len(config.agents) == 3
    assert config.agents[0].robot_awareness == "distracted"
    assert config.agents[1].robot_aware is False
    assert config.agents[2].robot_awareness == {"detection": -1.0}
    assert config.flow is not None
    assert config.flow.sources[0].agent_template.robot_awareness == [{"weight": 1.0, "awareness": "blind"}]


def test_awareness_text_empty_means_no_override() -> None:
    assert awareness_from_text("  ") is None


def test_awareness_text_preset_name_is_the_preset() -> None:
    assert awareness_from_text("scared") == AWARENESS_PRESETS["scared"]


def test_awareness_text_json_mapping_keeps_straight_values_and_distributions() -> None:
    spec = awareness_from_text('{"detection": -2, "repulsion": {"mean": -1.0, "std": 0.5, "clip_low": -2.0, "clip_high": 0.0}}')
    assert spec.detection.mean == -2.0
    assert spec.repulsion.std == 0.5
    assert spec.reaction.mean == 0.0


def test_awareness_text_json_list_is_a_weighted_mixture() -> None:
    spec = awareness_from_text('[{"weight": 0.25, "awareness": "normal"}, {"weight": 0.75, "awareness": {"fov": 1.0}}]')
    assert isinstance(spec, AwarenessMixture)
    assert [weight for weight, _ in spec.components] == [0.25, 0.75]


def test_awareness_text_that_is_broken_json_is_rejected() -> None:
    with pytest.raises(ValueError):
        awareness_from_text('{"detection": ')


def test_awareness_text_with_an_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown robot awareness preset 'sleepy'"):
        awareness_from_text("sleepy")


def test_robot_type_samples_fixed_sensor_and_planner_values() -> None:
    params = sample_agent_type(load_agent_type_from_file(CONFIG_DIR / "robot.yaml"), np.random.default_rng(7))
    perception = params.perception
    assert (perception.vision_range_sd, perception.vision_fov_sd, perception.repulsion_rel_sd, perception.reaction_rel_sd) == (0.0, 0.0, 0.0, 0.0)
    assert params.local_planner_params["repulsion_strength"] == 2.1
    assert params.local_planner_params["repulsion_range"] == 0.3
