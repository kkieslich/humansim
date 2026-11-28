"""A world object's capacity reaches the crowd: needs-driven walkers use a capacity-2 fountain two at a time."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("rclpy")

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import load_scenario

_SCENARIO = """
name: object_capacity
simulation: {{seed: 4, dt: 0.05, bt_tick_interval: 1}}
world_objects:
  - object_id: fountain
    type: fountain
    pose: {{x: 0.0, y: 0.0, theta: 0.0}}
    capacity: {capacity}
    satisfies: {{thirst: 100.0}}
    formation: {{type: line, params: {{base_step: 1.0, front_offset: 0.8}}}}
agent_types:
  drinker:
    extends: adult
    mode: behavior_tree
    needs:
      thirst: {{initial: {{mean: 25.0, std: 0.0}}, decay_rate: {{mean: 0.0, std: 0.0}}}}
    utility_weights: {{thirst: 1.0}}
    initial_sequence: idle
    sequences:
      idle:
        steps:
          wait: {{kind: go_to, target_pose: {{x: -8.0, y: 0.0, theta: 0.0}}}}
        then: idle
        transitions:
          - when: {{thirst: {{below: 30.0}}}}
            goto: drink
      drink:
        steps:
          use: {{target: fountain, interaction: USE, duration: {{mean: 6.0, std: 0.0}}, patience: {{mean: 200.0, std: 0.0}}, satisfies: {{thirst: 100.0}}}}
        then: idle
        on_failure: idle
agents:
  - {{agent_id: 1, agent_type: drinker, spawn_pose: {{x: -4.0, y: 0.5, theta: 0.0}}}}
  - {{agent_id: 2, agent_type: drinker, spawn_pose: {{x: -4.0, y: -0.5, theta: 0.0}}}}
  - {{agent_id: 3, agent_type: drinker, spawn_pose: {{x: -5.0, y: 0.0, theta: 0.0}}}}
"""


def _peak(manager_factory: Callable[..., AgentManager], tmp_path: Path, capacity: int) -> tuple[int, int]:
    path = tmp_path / f"object_capacity_{capacity}.yaml"
    path.write_text(_SCENARIO.format(capacity=capacity))
    mgr = manager_factory(load_scenario(str(path)), node_name=f"test_object_capacity_{capacity}")
    im = mgr._interaction_manager
    peak_held = peak_queue = 0
    for _ in range(int(40.0 / 0.05)):
        mgr.tick()
        for interaction in im.interactions.values():
            if interaction.object_id and interaction.object_id.endswith("fountain"):
                formation = interaction.contract.formation
                held = sum(1 for p in interaction.participants if formation is None or formation.arrived(p)) if im.is_holding(interaction) else 0
                peak_held = max(peak_held, held)
                peak_queue = max(peak_queue, len(interaction.contract.queue))
    return peak_held, peak_queue


def test_a_capacity_two_fountain_serves_two_walkers_at_once(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    assert _peak(manager_factory, tmp_path, 2) == (2, 1)


def test_a_capacity_one_fountain_serves_one_and_queues_the_others(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    held, queue = _peak(manager_factory, tmp_path, 1)
    assert held == 1 and queue == 2
