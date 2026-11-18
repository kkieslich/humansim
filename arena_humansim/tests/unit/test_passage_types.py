"""The `wall_trailing` and `passage` blocks of an agent type: parsed, inherited, off by default, and shipped on the
long-cane and manual-wheelchair types."""

from __future__ import annotations

import json
from pathlib import Path

import attrs
import numpy as np
import pytest

from arena_humansim.core.agents import BUILTIN_AGENTS
from arena_humansim.core.agents.loader import load_agent_type_from_file, resolve_extends
from arena_humansim.core.agents.types import AgentType, Passage, SampledParams, WallTrailing, sample_agent_type
from arena_humansim.utils.scenario_loader import converter

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "agent_types"


def test_both_blocks_are_off_unless_a_type_sets_them() -> None:
    adult = BUILTIN_AGENTS["adult"]
    assert adult.wall_trailing == WallTrailing()
    assert adult.passage == Passage()
    assert not adult.wall_trailing.active
    assert not adult.passage.active
    params = sample_agent_type(adult, np.random.default_rng(1))
    assert not params.wall_trailing.active
    assert not params.passage.active


def test_the_blocks_draw_nothing_from_the_type_s_random_stream() -> None:
    plain = converter.structure({"name": "x"}, AgentType)
    with_blocks = converter.structure({"name": "x", "wall_trailing": {"offset": 0.42, "reach": 4.5}, "passage": {"lane_width": 0.915, "corridor_width": 3.6}}, AgentType)
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    left, right = sample_agent_type(plain, a), sample_agent_type(with_blocks, b)
    assert a.bit_generator.state == b.bit_generator.state
    assert attrs.evolve(right, wall_trailing=WallTrailing(), passage=Passage()) == left


def test_a_block_parses_its_fields_and_refuses_unknown_ones() -> None:
    at = converter.structure({"name": "x", "wall_trailing": {"offset": 0.4, "reach": 3}, "passage": {"two_way_width": 1.8, "hold_back": 1.5, "queue_space": 1.22}}, AgentType)
    assert at.wall_trailing == WallTrailing(offset=0.4, reach=3.0)
    assert at.wall_trailing.active
    assert at.passage.takes_turns and at.passage.queues and not at.passage.lanes
    with pytest.raises(ValueError, match="unknown wall_trailing fields"):
        converter.structure({"name": "x", "wall_trailing": {"distance": 0.4}}, AgentType)
    with pytest.raises(ValueError, match="unknown passage fields"):
        converter.structure({"name": "x", "passage": {"width": 1.8}}, AgentType)
    with pytest.raises(ValueError):
        converter.structure({"name": "x", "passage": {"lane_width": -1.0}}, AgentType)


def test_a_block_with_its_values_at_zero_is_off() -> None:
    assert not WallTrailing(offset=0.0, reach=4.5).active
    assert not WallTrailing(offset=0.42, reach=0.0).active
    assert not Passage(lane_width=0.915).lanes
    assert not Passage(two_way_width=1.8).takes_turns
    assert not Passage().active


def test_a_child_type_inherits_the_blocks_and_overrides_single_fields() -> None:
    parent = {"name": "p", "wall_trailing": {"offset": 0.42, "reach": 4.5}, "passage": {"lane_width": 0.915, "corridor_width": 3.6}}
    child = {"name": "c", "extends": "p", "wall_trailing": {"offset": 0.0}}
    resolved = resolve_extends({"c": child}, {"p": parent})["c"]
    at = converter.structure(resolved, AgentType)
    assert at.wall_trailing == WallTrailing(offset=0.0, reach=4.5)
    assert at.passage == Passage(lane_width=0.915, corridor_width=3.6)


def test_sampled_blocks_survive_the_spawn_log_round_trip() -> None:
    at = converter.structure({"name": "x", "wall_trailing": {"offset": 0.42, "reach": 4.5}, "passage": {"two_way_width": 1.8, "hold_back": 1.5}}, AgentType)
    params = sample_agent_type(at, np.random.default_rng(0))
    logged = json.loads(json.dumps(attrs.asdict(params)))
    assert SampledParams(**{**attrs.asdict(params, recurse=False), "wall_trailing": logged["wall_trailing"], "passage": logged["passage"]}) == params


def test_shipped_long_cane_type_trails_walls() -> None:
    at = load_agent_type_from_file(CONFIG_DIR / "blind_white_cane.yaml")
    assert at.wall_trailing == WallTrailing(offset=0.42, reach=4.5)
    assert not at.passage.active
    assert BUILTIN_AGENTS["blind_white_cane"].wall_trailing == at.wall_trailing


def test_shipped_manual_wheelchair_takes_passages_by_rule() -> None:
    at = load_agent_type_from_file(CONFIG_DIR / "wheelchair_manual.yaml")
    assert at.passage == Passage(lane_width=0.915, corridor_width=3.6, two_way_width=1.8, hold_back=1.5, queue_space=1.22)
    assert not at.wall_trailing.active
    params = sample_agent_type(at, np.random.default_rng(1))
    assert params.passage == at.passage
    assert at.passage.lane_width >= 2.0 * params.agent_radius
    assert at.passage.queue_space >= params.locomotion.footprint_length
    assert BUILTIN_AGENTS["wheelchair_manual"].passage == at.passage


def test_the_other_shipped_types_are_as_before() -> None:
    for name, agent_type in BUILTIN_AGENTS.items():
        if name not in ("blind_white_cane", "wheelchair_manual"):
            assert not agent_type.wall_trailing.active, name
            assert not agent_type.passage.active, name
