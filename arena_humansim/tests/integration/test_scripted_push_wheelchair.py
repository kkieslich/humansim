from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from arena_humansim.core.agent_manager import AgentManager
from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim.utils.scenario import load_scenario

_SCENARIO = Path(__file__).resolve().parents[2] / "config" / "scenarios" / "push_wheelchair.yaml"
RIDER = 1
PUSHER = 2
HANDLE_OFFSET = 0.715


def _pose(mgr: AgentManager, aid: int) -> tuple[float, float, float]:
    idx = mgr._pool.idx(aid)
    return float(mgr._pool.pos[idx, 0]), float(mgr._pool.pos[idx, 1]), float(mgr._pool.theta[idx])


def _pushing(mgr: AgentManager, aid: int) -> bool:
    active = mgr._interaction_manager.active_interaction(aid)
    return active is not None and active[1] == InteractionType.PUSH_WHEELCHAIR


def test_caregiver_takes_the_handles_pushes_parks_and_walks_off(manager_factory: Callable[..., AgentManager]) -> None:
    scenario = load_scenario(str(_SCENARIO))
    mgr = manager_factory(scenario, node_name="test_scripted_push_wheelchair")

    held_ticks = 0
    released_at: tuple[float, float] | None = None
    rider_start = _pose(mgr, RIDER) if RIDER in mgr._pool._id_to_idx else None
    for _ in range(scenario.simulation.max_ticks):
        mgr.tick()
        if rider_start is None:
            rider_start = _pose(mgr, RIDER)
        pushing = _pushing(mgr, PUSHER)
        assert pushing == _pushing(mgr, RIDER), "the pair shows the push together"
        if RIDER in mgr._carrier.held():
            held_ticks += 1
            px, py, pt = _pose(mgr, PUSHER)
            rx, ry, rt = _pose(mgr, RIDER)
            if mgr._carrier._holds[(PUSHER, RIDER)].t > 1.0:  # the chair has rolled to her hands
                ahead = (rx - px) * math.cos(pt) + (ry - py) * math.sin(pt)
                side = -(rx - px) * math.sin(pt) + (ry - py) * math.cos(pt)
                assert abs(ahead - HANDLE_OFFSET) < 1e-6 and abs(side) < 1e-6 and abs(math.remainder(rt - pt, math.tau)) < 1e-6
            released_at = None
        elif held_ticks and released_at is None:
            rx, ry, _ = _pose(mgr, RIDER)
            released_at = (rx, ry)

    assert held_ticks * scenario.simulation.dt > 5.0, "the chair was pushed"
    assert released_at is not None, "she let go"
    rx, ry, _ = _pose(mgr, RIDER)
    assert math.hypot(rx - (8.0 + HANDLE_OFFSET), ry - 2.35) < 0.4, f"parked beside the wall, got ({rx:.2f}, {ry:.2f})"
    assert math.hypot(rx - released_at[0], ry - released_at[1]) < 0.05, "the parked chair stays where she left it"
    px, py, _ = _pose(mgr, PUSHER)
    assert math.hypot(px - 4.0, py + 3.0) < 0.5, f"she walked off, got ({px:.2f}, {py:.2f})"
    assert not _pushing(mgr, PUSHER) and not _pushing(mgr, RIDER)


def test_after_letting_go_she_walks_around_the_chair_not_through_it(manager_factory: Callable[..., AgentManager], tmp_path: Path) -> None:
    # her way off runs straight through the parked chair (beside the north wall, so she goes round it on the south)
    text = _SCENARIO.read_text().replace("walk_off: {kind: go_to, target_pose: {x: 4.0, y: -3.0, theta: 0.0}}", "walk_off: {kind: go_to, target_pose: {x: 12.0, y: 2.2, theta: 0.0}}")
    assert "x: 12.0" in text
    path = tmp_path / "push_through.yaml"
    path.write_text(text)
    scenario = load_scenario(str(path))
    mgr = manager_factory(scenario, node_name="test_scripted_push_wheelchair_round")
    pool = mgr._pool

    released_gap: float | None = None
    parked: tuple[float, float] | None = None
    worst = math.inf
    for _ in range(scenario.simulation.max_ticks):
        mgr.tick()
        if RIDER not in pool._id_to_idx:
            continue
        held = RIDER in mgr._carrier.held()
        if held:
            continue
        gap = mgr._carrier._gap(pool, PUSHER, RIDER)[0]
        if released_gap is None and mgr._carrier._released:  # the first tick after she let go
            released_gap = gap
            parked = _pose(mgr, RIDER)[:2]
        if released_gap is not None:
            worst = min(worst, gap)

    assert released_gap is not None and parked is not None, "she let go"
    assert worst >= released_gap - 1e-6, f"she got deeper into the chair after letting go: {worst:.3f} < {released_gap:.3f}"
    px, py, _ = _pose(mgr, PUSHER)
    assert math.hypot(px - 12.0, py - 2.2) < 0.5, f"she got past the chair, got ({px:.2f}, {py:.2f})"
    rx, ry, _ = _pose(mgr, RIDER)
    assert math.hypot(rx - parked[0], ry - parked[1]) < 0.05, "the parked chair stays where she left it"
    assert mgr._carrier._released == {}, "they count as apart once she is clear"
