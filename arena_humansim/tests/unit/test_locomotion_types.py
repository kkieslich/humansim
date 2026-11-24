from __future__ import annotations

import json
from pathlib import Path

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents import BUILTIN_AGENTS, sample_agent_type
from arena_humansim.core.agents.loader import load_agent_type_from_file, resolve_extends
from arena_humansim.core.agents.types import AgentType, LocomotionDist, ParamDist, SampledLocomotion, _sample_locomotion
from arena_humansim.utils.scenario_loader import converter

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "agent_types"

NESTED = """
name: cart
locomotion:
  kinematics: along_heading
  cadence: {base: 0.6, per_speed: 0.4, min: 0.6, max: 1.4}
  phase_warp: {split: 0.4}
  speed_profile:
    harmonics: [[0.12, 0.0], [0.05, 1.1]]
    amplitude_scale: {mean: 1.0, std: 0.15, clip_low: 0.5, clip_high: 1.5}
  lateral_profile:
    harmonics: [[0.02, 0.3]]
  footprint_length: 1.1
  recovery: {stall_after_s: 3.0, reverse_m: 0.3}
local_planner_params:
  heading_source: total
interaction_class: cart
assets: [human::mobility/cart]
pose: {walk: {base: seated}}
"""


def _write(tmp_path: Path, text: str, name: str = "cart.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_nested_locomotion_yaml_maps_onto_flat_fields(tmp_path: Path) -> None:
    at = load_agent_type_from_file(_write(tmp_path, NESTED))
    loc = at.locomotion
    assert at.locomotion_active is True
    assert loc.kinematics == "along_heading"
    assert (loc.cadence.base.mean, loc.cadence.per_speed.mean, loc.cadence.min.mean, loc.cadence.max.mean) == (0.6, 0.4, 0.6, 1.4)
    assert loc.phase_warp_split.mean == 0.4
    assert loc.speed_profile == ((0.12, 0.0), (0.05, 1.1))
    assert loc.speed_amplitude_scale == ParamDist(1.0, 0.15, 0.5, 1.5)
    assert loc.lateral_profile == ((0.02, 0.3),)
    assert loc.footprint_length.mean == 1.1
    assert loc.recovery_stall_after_s.mean == 3.0
    assert loc.recovery_reverse_m.mean == 0.3
    assert at.local_planner_params["heading_source"] == ParamDist(1.0, 0.0, 1.0, 1.0)
    assert at.interaction_class == "cart"
    assert at.assets == ("human::mobility/cart",)
    assert at.pose == {"walk": {"base": "seated"}}


def test_heading_source_words_map_to_numbers() -> None:
    at = converter.structure({"name": "x", "local_planner_params": {"heading_source": "attraction"}}, AgentType)
    assert at.local_planner_params["heading_source"].mean == 0.0
    with pytest.raises(ValueError, match="heading_source"):
        converter.structure({"name": "x", "local_planner_params": {"heading_source": "sideways"}}, AgentType)


def test_locomotion_rejects_unknown_keys_bad_kinematics_and_too_many_harmonics() -> None:
    with pytest.raises(ValueError, match="unknown locomotion fields"):
        converter.structure({"name": "x", "locomotion": {"candence": {}}}, AgentType)
    with pytest.raises(ValueError, match="unknown locomotion cadence fields"):
        converter.structure({"name": "x", "locomotion": {"cadence": {"bass": 1.0}}}, AgentType)
    with pytest.raises(ValueError):
        LocomotionDist(kinematics="sideways")
    with pytest.raises(ValueError, match="harmonics"):
        LocomotionDist(speed_profile=((0.1, 0.0),) * 4)


def test_empty_locomotion_section_activates_defaults() -> None:
    at = converter.structure({"name": "x", "locomotion": {}}, AgentType)
    assert at.locomotion_active is True
    assert at.locomotion == LocomotionDist()
    bare = converter.structure({"name": "x", "locomotion": None}, AgentType)
    assert bare.locomotion_active is True


def test_extends_merges_locomotion_sub_dicts_field_by_field() -> None:
    parent = {"name": "parent", "locomotion": {"kinematics": "along_heading", "cadence": {"base": 0.6, "per_speed": 0.4, "min": 0.6, "max": 1.4}, "recovery": {"stall_after_s": 3.0, "reverse_m": 0.3}}, "pose": {"walk": {"base": "seated", "symmetric": False}}}
    child = {"name": "child", "extends": "parent", "locomotion": {"cadence": {"max": 1.2}, "recovery": {"reverse_m": 0.0}}, "pose": {"walk": {"symmetric": True}}}
    merged = resolve_extends({"child": child}, {"parent": parent})["child"]
    assert merged["locomotion"]["cadence"] == {"base": 0.6, "per_speed": 0.4, "min": 0.6, "max": 1.2}
    assert merged["locomotion"]["recovery"] == {"stall_after_s": 3.0, "reverse_m": 0.0}
    assert merged["locomotion"]["kinematics"] == "along_heading"
    assert merged["pose"] == {"walk": {"base": "seated", "symmetric": True}}
    at = converter.structure(merged, AgentType)
    assert at.locomotion.cadence.max.mean == 1.2
    assert at.locomotion.cadence.base.mean == 0.6


def test_type_without_section_samples_identically_to_all_default_section() -> None:
    plain = BUILTIN_AGENTS["adult"]
    explicit = converter.structure({"name": "adult", "locomotion": {}}, AgentType)
    explicit = attrs.evolve(explicit, **{f.name: getattr(plain, f.name) for f in attrs.fields(AgentType) if f.name not in ("locomotion", "locomotion_active")})
    assert plain.locomotion_active is False
    assert explicit.locomotion_active is True
    a = sample_agent_type(plain, np.random.default_rng(5))
    b = sample_agent_type(explicit, np.random.default_rng(5))
    assert a.locomotion.active is False
    assert b.locomotion.active is True
    assert attrs.evolve(a, locomotion=SampledLocomotion()) == attrs.evolve(b, locomotion=SampledLocomotion())


def test_std_zero_locomotion_fields_never_draw_from_the_rng() -> None:
    fixed = converter.structure({"name": "x", "locomotion": {"cadence": {"base": 0.7}, "footprint_length": 1.0}}, AgentType)
    rng = np.random.default_rng(3)
    before = rng.bit_generator.state
    loc = _sample_locomotion(fixed, rng)
    assert rng.bit_generator.state == before
    assert (loc.cadence_base, loc.footprint_length) == (0.7, 1.0)

    noisy = converter.structure({"name": "x", "locomotion": {"speed_profile": {"amplitude_scale": {"mean": 1.0, "std": 0.2}}}}, AgentType)
    _sample_locomotion(noisy, rng)
    assert rng.bit_generator.state != before


def test_sampled_locomotion_survives_the_spawn_log_round_trip() -> None:
    at = converter.structure(json.loads(json.dumps({"name": "x"})) | {"locomotion": {"kinematics": "along_heading", "speed_profile": {"harmonics": [[0.12, 0.0], [0.05, 1.1]]}, "lateral_profile": {"harmonics": [[0.02, 0.3]]}}}, AgentType)
    params = sample_agent_type(at, np.random.default_rng(0))
    loc = json.loads(json.dumps(attrs.asdict(params)))["locomotion"]
    assert SampledLocomotion(**loc) == params.locomotion
    assert params.locomotion.kinematics == 1


def test_shipped_wheelchair_manual_profile() -> None:
    at = load_agent_type_from_file(CONFIG_DIR / "wheelchair_manual.yaml")
    assert at.name == "wheelchair_manual"
    assert at.extends is None
    assert at.locomotion_active is True
    assert at.locomotion.kinematics == "along_heading"
    assert at.locomotion.footprint_length.mean == 1.1
    assert at.locomotion.recovery_stall_after_s.mean == 3.0
    assert at.interaction_class == "wheelchair"
    assert at.assets == ("human::mobility::wheelchair",)
    assert at.local_planner == "hsfm"
    assert at.local_planner_params["heading_source"].mean == 1.0
    assert at.local_planner_params["lateral_gain"].mean == 0.0
    assert at.local_planner_params["relaxation_time"] == BUILTIN_AGENTS["adult"].local_planner_params["relaxation_time"]
    assert at.pose["walk"]["base"] == "seated_wheelchair"
    assert at.pose["walk"]["gain"] == {"min": 1.0, "max": 1.0}
    assert at.pose["idle"] == {"base": "seated_wheelchair"}
    assert at.pose["run"] == {"same_as": "walk", "gain": {"factor": 1.0}}
    params = sample_agent_type(at, np.random.default_rng(1))
    assert params.agent_radius == 0.325
    assert params.locomotion.active is True
    assert params.locomotion.speed_profile == ((0.12, 0.0), (0.05, 1.1))
    assert params.interaction_class == "wheelchair"


def test_shipped_wheelchair_electric_profile() -> None:
    at = load_agent_type_from_file(CONFIG_DIR / "wheelchair_electric.yaml")
    assert at.name == "wheelchair_electric"
    assert at.locomotion_active is True
    assert at.locomotion.kinematics == "along_heading"
    assert at.locomotion.footprint_length.mean == 1.1
    assert at.interaction_class == "wheelchair"
    assert at.assets == ("human::propulsion::electric",)
    assert at.climbs == ()
    assert at.local_planner == "hsfm"
    assert at.local_planner_params["lateral_gain"].mean == 0.0
    assert at.pose["walk"] == {"base": "seated_wheelchair_electric"}
    assert at.pose["idle"] == {"base": "seated_wheelchair_electric"}
    assert at.pose["run"] == {"same_as": "walk", "gain": {"factor": 1.0}}
    params = sample_agent_type(at, np.random.default_rng(1))
    assert params.climbs == ()
    assert params.locomotion.speed_profile == ()
    assert params.locomotion.lateral_profile == ()


def test_shipped_adult_limp_right_profile() -> None:
    limp = load_agent_type_from_file(CONFIG_DIR / "adult_limp_right.yaml")
    assert limp.locomotion_active
    assert limp.locomotion.kinematics == "holonomic"
    assert limp.locomotion.phase_warp_split.mean == 0.58
    assert limp.locomotion.speed_profile == ((0.10, 1.6),)
    assert limp.locomotion.lateral_profile == ((0.04, 1.6),)
    assert limp.interaction_class == ""
    assert limp.pose["walk"]["symmetric"] is False
    sampled = sample_agent_type(limp, np.random.default_rng(3))
    assert sampled.locomotion.active
    assert sampled.locomotion.footprint_length == 0.0
    assert sampled.locomotion.recovery_stall_after_s == 0.0


@pytest.mark.parametrize("name", ["wheelchair_manual", "wheelchair_electric", "adult_limp_right"])
def test_builtin_registry_resolves_extends_like_the_file_loader(name: str) -> None:
    from_file = load_agent_type_from_file(CONFIG_DIR / f"{name}.yaml")
    assert attrs.evolve(BUILTIN_AGENTS[name], source_path=from_file.source_path) == from_file
    assert BUILTIN_AGENTS[name].perception == BUILTIN_AGENTS["adult"].perception


@pytest.mark.parametrize(("name", "kind"), [("blind_white_cane", "cane"), ("blind_rolling_cane", "cane"), ("elder_rollator", "rollator"), ("patient_walking_frame", "walking_frame"), ("patient_iv_pole", "iv_stand")])
def test_shipped_device_blocks_pass_through(name: str, kind: str) -> None:
    assert load_agent_type_from_file(CONFIG_DIR / f"{name}.yaml").device["kind"] == kind


def test_a_device_block_is_inherited_whole_and_defaults_to_none() -> None:
    device = {"kind": "cane", "bone": "cane"}
    merged = resolve_extends({"child": {"name": "child", "extends": "parent", "perception": {"robot_awareness": 0.0}}}, {"parent": {"name": "parent", "device": device}})
    assert merged["child"]["device"] == device
    assert AgentType(name="plain").device == {}
