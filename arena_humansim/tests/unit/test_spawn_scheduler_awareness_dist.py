from __future__ import annotations

import numpy as np
from arena_humansim.core.agents.types import AWARENESS_PRESETS, as_awareness_spec, sample_awareness
from arena_humansim.core.spawn_scheduler import SpawnScheduler
from arena_humansim.utils.types import AgentTemplate, Pose2D, Shape, SourceConfig


def test_preset_draw_is_the_preset_shift() -> None:
    draw = sample_awareness(AWARENESS_PRESETS["distracted"], np.random.default_rng(1))
    assert (draw.detection, draw.fov, draw.repulsion, draw.reaction, draw.blind) == (-5.0, -8.0, -3.0, -3.0, False)


def test_normal_distribution_shift_stays_inside_its_clip() -> None:
    spec = as_awareness_spec({"detection": {"mean": 0.0, "std": 3.0, "clip_low": -1.0, "clip_high": 1.0}})
    rng = np.random.default_rng(2)
    draws = [sample_awareness(spec, rng).detection for _ in range(500)]
    assert min(draws) >= -1.0
    assert max(draws) <= 1.0
    assert len(set(draws)) > 50


def test_mixture_draws_follow_their_weights() -> None:
    spec = as_awareness_spec([{"weight": 0.25, "awareness": "normal"}, {"weight": 0.75, "awareness": "blind"}])
    rng = np.random.default_rng(3)
    blind_share = sum(sample_awareness(spec, rng).blind for _ in range(4000)) / 4000
    assert 0.70 < blind_share < 0.80


def test_spawn_request_carries_the_template_awareness() -> None:
    scheduler = SpawnScheduler(rng=np.random.default_rng(42))
    source = SourceConfig(
        name="source_0",
        pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        shape=Shape(type="circle", radius=1.0),
        agent=AgentTemplate(agent_type="adult", robot_awareness="scared"),
    )
    scheduler.add_source(source)
    req = scheduler._sample_spawn_request(source, tick_count=0, seq=1)
    assert req.robot_awareness == "scared"
