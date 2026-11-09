"""A source-released visitor walks through its own sink on its way in and stays; once it heads for the sink, the sink takes it."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import yaml
from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.utils.scenario import load_scenario

SINK = (2.0, 0.0)
STOP = (5.0, 0.0)
PATIENCE = {"mean": 120.0, "std": 0.0, "clip_low": 120.0, "clip_high": 120.0}


def _scenario(tmp_path: Path) -> Path:
    def go_to(x: float, y: float) -> dict[str, object]:
        return {"kind": "go_to", "target_pose": {"x": x, "y": y}, "patience": PATIENCE}

    corners = [(-2.0, -3.0), (8.0, -3.0), (8.0, 3.0), (-2.0, 3.0)]
    raw = {
        "name": "sink_on_the_way",
        "simulation": {"seed": 3, "dt": 0.05},
        "walls": [{"name": f"w{i}", "start": {"x": ax, "y": ay}, "end": {"x": bx, "y": by}} for i, ((ax, ay), (bx, by)) in enumerate(zip(corners, corners[1:] + corners[:1], strict=True))],
        "agent_types": {
            "visitor": {
                "extends": "adult",
                "mode": "behavior_tree",
                "initial_sequence": "visit",
                "sequences": {
                    "visit": {"steps": {"to_stop": go_to(*STOP)}, "then": "leave"},
                    "leave": {"steps": {"exit": go_to(*SINK)}, "then": "leave"},
                },
            },
        },
        "flow": {
            "sources": [
                {
                    "pose": {"x": 0.0, "y": 0.0},
                    "shape": {"type": "circle", "radius": 0.1},
                    "type": "poisson",
                    "rate_profile": [{"t": 0.0, "rate": 20.0}],
                    "max_concurrent": 1,
                    "max_total": 1,
                    "agent_template": {"agent_type": "visitor", "desired_velocity_min": 1.2, "desired_velocity_max": 1.2, "agent_radius": 0.25, "sink_affinity": [{"sink_idx": 0, "weight": 1.0}]},
                },
            ],
            "sinks": [{"pose": {"x": SINK[0], "y": SINK[1]}, "shape": {"type": "circle", "radius": 0.5}, "absorption_radius": 0.5}],
        },
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_visitor_passes_its_sink_on_the_way_in_and_leaves_by_it(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    mgr = manager_factory(load_scenario(str(_scenario(tmp_path))), node_name="test_sink_on_the_way")
    aid = None
    for _ in range(40):
        mgr.tick()
        if mgr._last_spawned_ids:
            aid = mgr._last_spawned_ids[0]
            break
    assert aid is not None, "the source released no visitor"

    through_sink, at_stop, last = False, False, None
    for _ in range(1200):
        mgr.tick()
        if aid not in mgr._agents:
            break
        pose = mgr._agents[aid].state.pose
        last = (pose.x, pose.y)
        through_sink |= not at_stop and math.dist(last, SINK) < 0.5
        at_stop |= math.dist(last, STOP) < 0.6
    assert through_sink, "the visitor's way in never ran through its sink"
    assert at_stop, "the visitor was taken before it reached its stop"
    assert aid not in mgr._agents, "the visitor never left by its sink"
    assert last is not None and math.dist(last, SINK) < 0.5 + 0.1, f"the visitor vanished at {last}, away from its sink"
