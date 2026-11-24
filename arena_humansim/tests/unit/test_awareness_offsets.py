from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim.core.agents.types import (
    AWARENESS_PRESETS,
    AwarenessShift,
    SampledPerception,
    sample_agent_type,
    sample_awareness,
    with_robot_awareness,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "agent_types"


def _base_perception() -> SampledPerception:
    return SampledPerception(
        vision_range=5.0,
        vision_fov=180.0,
        proximity_sense=1.0,
        vision_range_sd=0.5,
        vision_fov_sd=10.0,
        repulsion_rel_sd=0.1,
        reaction_rel_sd=0.1,
    )


def test_normal_awareness_adds_no_robot_offset() -> None:
    draw = sample_awareness(AWARENESS_PRESETS["normal"], np.random.default_rng(1))
    perception = with_robot_awareness(_base_perception(), draw)
    assert perception.robot_range_offset == 0.0
    assert perception.robot_fov_offset == 0.0
    assert perception.robot_repulsion_gain == 1.0
    assert perception.robot_reaction_gain == 1.0
    assert perception.robot_blind is False


def test_distracted_moves_each_dimension_by_its_own_sigma() -> None:
    draw = sample_awareness(AWARENESS_PRESETS["distracted"], np.random.default_rng(1))
    perception = with_robot_awareness(_base_perception(), draw)
    assert perception.robot_range_offset == pytest.approx(-5.0 * 0.5)
    assert perception.robot_fov_offset == pytest.approx(-8.0 * 10.0)
    assert perception.robot_repulsion_gain == pytest.approx(1.0 - 3.0 * 0.1)
    assert perception.robot_reaction_gain == pytest.approx(1.0 - 3.0 * 0.1)


def test_aggressive_tolerates_robots_closer_and_reacts_later() -> None:
    draw = sample_awareness(AWARENESS_PRESETS["aggressive"], np.random.default_rng(1))
    perception = with_robot_awareness(_base_perception(), draw)
    assert perception.robot_repulsion_gain == pytest.approx(1.0 - 4.0 * 0.1)
    assert perception.robot_reaction_gain == pytest.approx(1.0 - 2.0 * 0.1)


def test_repulsion_gain_clips_at_zero_for_a_shift_past_zero() -> None:
    draw = sample_awareness(AwarenessShift(repulsion=-20.0), np.random.default_rng(1))
    assert with_robot_awareness(_base_perception(), draw).robot_repulsion_gain == 0.0


def test_blind_preset_sets_the_blind_flag() -> None:
    draw = sample_awareness(AWARENESS_PRESETS["blind"], np.random.default_rng(1))
    assert with_robot_awareness(_base_perception(), draw).robot_blind is True


def test_override_awareness_scales_by_the_sampled_type_spreads() -> None:
    adult = load_agent_type_from_file(CONFIG_DIR / "adult.yaml")
    params = sample_agent_type(adult, np.random.default_rng(3))
    rng = np.random.default_rng(3)
    perception = with_robot_awareness(params.perception, sample_awareness(AWARENESS_PRESETS["scared"], rng))
    assert perception.vision_range_sd == pytest.approx(0.5)
    assert perception.repulsion_rel_sd == pytest.approx(0.2 / 2.1)
    assert perception.robot_repulsion_gain == pytest.approx(1.0 + 6.0 * 0.2 / 2.1)
    assert perception.robot_reaction_gain == pytest.approx(1.0 + 3.0 * params.perception.reaction_rel_sd)


def test_sample_agent_type_uses_the_type_awareness() -> None:
    adult = load_agent_type_from_file(CONFIG_DIR / "adult.yaml")
    default = sample_agent_type(adult, np.random.default_rng(4))
    assert default.perception.robot_range_offset == 0.0
    distracted = load_agent_type_from_file(CONFIG_DIR / "distracted.yaml")
    shifted = sample_agent_type(distracted, np.random.default_rng(4))
    assert shifted.perception.robot_range_offset == pytest.approx(-0.75 * 0.4)


def test_sample_agent_type_takes_the_awareness_from_the_type_file(tmp_path: Path) -> None:
    path = tmp_path / "wary.yaml"
    path.write_text("name: wary\nextends: adult\nperception:\n  robot_awareness: scared\n", encoding="utf-8")
    params = sample_agent_type(load_agent_type_from_file(path), np.random.default_rng(5))
    assert params.perception.robot_repulsion_gain == pytest.approx(1.0 + 6.0 * params.perception.repulsion_rel_sd)
