from __future__ import annotations

import numpy as np
from arena_humansim.utils.lanes import build_lane, build_track, resolve_clearance, segments_array, wall_distance
from arena_humansim.utils.types import LaneConfig, LaneSide
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._lane_layouts import corridor, straight

CLEARANCE = resolve_clearance(0.38, 0.2)


@settings(max_examples=60, deadline=None)
@given(
    pillar_x=st.floats(min_value=3.0, max_value=17.0),
    pillar_y=st.floats(min_value=-0.3, max_value=0.3),
    offset=st.floats(min_value=-1.2, max_value=1.2),
    q=st.floats(min_value=-1.0, max_value=1.0),
    s0=st.floats(min_value=0.0, max_value=1.0),
)
def test_every_waypoint_stays_outside_every_wall_clearance(pillar_x: float, pillar_y: float, offset: float, q: float, s0: float) -> None:
    walls = corridor(4.0, 20.0) + [((pillar_x, pillar_y - 0.3), (pillar_x, pillar_y + 0.3))]
    config = LaneConfig(centerlines=straight(20.0), side=LaneSide.RIGHT, offset=offset)
    lane = build_lane(config, walls, flow="prop", agent_radius=0.25, clearance=CLEARANCE, source_extent=1.5)
    track = build_track(lane, q=q, s0=s0)
    points = np.array([[p.x, p.y] for p in track.points])
    assert float(wall_distance(points, segments_array(walls)).min()) > CLEARANCE
